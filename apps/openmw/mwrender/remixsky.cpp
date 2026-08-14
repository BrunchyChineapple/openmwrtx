#include "remixsky.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>

#include <osg/Quat>
#include <osg/Vec3f>

#include <components/debug/debuglog.hpp>
#include <components/settings/values.hpp>
#include <components/remixrt/runtime.hpp>

namespace
{
    constexpr float kRadToDeg = 57.29577951308232f;

    /// World units per metre, as Remix computes it: worldUnitsPerKm = 100000 * sceneScale, and this
    /// project runs sceneScale 1.43 to match the Morrowind Remix project. Kept as a constant rather than
    /// read back from the runtime because there is no getter for a config variable on this API, and the
    /// two have to agree or fog lands orders of magnitude out.
    constexpr float kUnitsPerMetre = 100.0f * 1.43f;

    /// Scattering-to-absorption ratio for the fog medium.
    ///
    /// Near one because that is what fog is: droplets scatter light, they do not swallow it. Anything much
    /// below this turns the medium into an absorber that darkens the scene instead of fogging it, which is
    /// the failure this constant exists to make hard to reintroduce.
    constexpr float kFogAlbedo = 0.99f;

    /// Transmittance level at the measurement distance, before the fog's hue is applied.
    ///
    /// This is a reference point rather than a look: it and the measurement distance together define the
    /// density, and fixing one lets the other be derived. A half is convenient because it puts the
    /// measurement distance at a recognisable "half the light is gone by here".
    constexpr float kFogTransmittance = 0.5f;

    /// How opaque the view distance should be when OpenMW asks for full fog depth.
    ///
    /// Five percent surviving reads as fully fogged without being pure flat colour, which leaves distant
    /// landmarks faintly legible the way Morrowind's own fog does.
    constexpr float kFogEndTransmittance = 0.05f;

    /// How much of the fog colour's hue reaches transmittanceColor.
    ///
    /// Partial on purpose. What looks coloured about fog is mostly the light it scatters, not a tint on what
    /// passes through, and the scattered part is already handled by the albedo picking up the sun and sky.
    constexpr float kFogTint = 0.5f;

    float envFloat(const char* name, float fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        char* end = nullptr;
        const float parsed = std::strtof(value, &end);
        return end != value ? parsed : fallback;
    }

    /// Whether OpenMW drives the fog at all.
    ///
    /// Exists because the host writing these options every time the weather moves is indistinguishable, from
    /// the other side of the developer menu, from the menu being broken: a value typed there is overwritten
    /// within seconds by the next fog colour change. Setting OPENMW_REMIX_FOG=0 hands the medium over to the
    /// menu entirely so it can be tuned by hand, which is the only way to find numbers worth hard-coding.
    bool fogEnabled()
    {
        static const bool value = []() -> bool {
            // Setting first, environment override second, matching every other knob in [Remix]. Absent
            // variable defers to the setting; present means the variable decides, and "0" disables.
            const char* env = std::getenv("OPENMW_REMIX_FOG");
            if (env == nullptr)
                return Settings::remix().mFog.get();
            return *env != '\0' && *env != '0';
        }();
        return value;
    }

    /// Morrowind's ten weather IDs, mapped to the preset names the runtime's weather blender knows.
    ///
    /// Indexed by the weather's script ID, which is Morrowind's own numbering and is what OpenMW reports
    /// through getCurrentWeatherScriptId. Ash and Blight both map to sandstorm: the runtime has no blight
    /// preset, and a sandstorm is the closer of what it does have.
    const char* const kWeatherPresets[] = {
        "clear", // 0 Clear
        "partlyCloudy", // 1 Cloudy
        "foggy", // 2 Foggy
        "overcast", // 3 Overcast
        "rainstorm", // 4 Rain
        "thunderstorm", // 5 Thunderstorm
        "sandstorm", // 6 Ashstorm
        "sandstorm", // 7 Blight
        "snow", // 8 Snow
        "blizzard", // 9 Blizzard
    };
    constexpr int kWeatherPresetCount = static_cast<int>(std::size(kWeatherPresets));

    /// Direction toward a moon, from the two rotations OpenMW carries in MoonState.
    ///
    /// Deliberately the same expression as Moon::setState in skyutil.cpp rather than an equivalent one.
    /// The quaternion composition order and the choice of (0,1,0) as the reference axis are conventions,
    /// not derivations, and reproducing them literally is what guarantees Remix's moon ends up where
    /// OpenMW's billboard is. If OpenMW's changes, this has to change with it.
    osg::Vec3f moonDirection(const MWRender::MoonState& state)
    {
        const float radsX = state.mRotationFromHorizon / kRadToDeg;
        const float radsZ = state.mRotationFromNorth / kRadToDeg;
        const osg::Quat rotX(radsX, osg::Vec3f(1.0f, 0.0f, 0.0f));
        const osg::Quat rotZ(radsZ, osg::Vec3f(0.0f, 0.0f, 1.0f));
        return rotX * rotZ * osg::Vec3f(0.0f, 1.0f, 0.0f);
    }

    /// Remix's phase parameter runs 0 = new, 0.5 = full, 1 = new again. OpenMW's Phase enum distinguishes
    /// waxing from waning; MoonState::phaseToInt does not -- it folds the two crescents onto 1 and the two
    /// quarters onto 2, because the vanilla moon texture is symmetric and the renderer only needs to pick a
    /// sprite. Keeping the distinction costs nothing, so the honest value goes across.
    ///
    /// It is not what mirrors the moon, though. The runtime places the terminator at cos(phase * 2pi), which
    /// is symmetric about full, so 0.125 and 0.875 light the same fraction of the disk and always will. What
    /// separates the two halves of the month on screen is that the terminator is oriented toward the sun,
    /// and the sun sits on the other side of the sky in each.
    float remixMoonPhase(MWRender::MoonState::Phase phase)
    {
        switch (phase)
        {
            case MWRender::MoonState::Phase::New:
                return 0.0f;
            case MWRender::MoonState::Phase::WaxingCrescent:
                return 0.125f;
            case MWRender::MoonState::Phase::FirstQuarter:
                return 0.25f;
            case MWRender::MoonState::Phase::WaxingGibbous:
                return 0.375f;
            case MWRender::MoonState::Phase::Full:
                return 0.5f;
            case MWRender::MoonState::Phase::WaningGibbous:
                return 0.625f;
            case MWRender::MoonState::Phase::ThirdQuarter:
                return 0.75f;
            case MWRender::MoonState::Phase::WaningCrescent:
                return 0.875f;
            case MWRender::MoonState::Phase::Unspecified:
                break;
        }
        // Full rather than new. An unspecified phase is missing information, and a new moon is not a
        // neutral default for it -- it is unlit, which reads on screen as the moon having vanished.
        return 0.5f;
    }
}

namespace MWRender
{
    RemixSky::RemixSky(RemixRT::Runtime& runtime)
        : mRuntime(runtime)
    {
    }

    void RemixSky::pushFloat(const char* key, float value, float& last, int decimals)
    {
        // Compared against what was last SENT, not against a tolerance on the source value, so a slow
        // drift still eventually crosses a printed digit and gets written. Quantising to the printed
        // precision first means a value that formats identically is never sent twice.
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, static_cast<double>(value));
        const float quantised = static_cast<float>(std::atof(buffer));
        if (quantised == last)
            return;
        last = quantised;
        mRuntime.setConfigVariable(key, buffer);
    }

    void RemixSky::pushBool(const char* key, bool value, int& last)
    {
        const int encoded = value ? 1 : 0;
        if (encoded == last)
            return;
        last = encoded;
        mRuntime.setConfigVariable(key, value ? "True" : "False");
    }

    void RemixSky::update(const SkyManager::State& sky, bool exterior, float waterPlaneHeight, int weatherId,
        int nextWeatherId,
        float weatherTransition, float viewDistance)
    {
        if (!mLoggedOnce)
        {
            mLoggedOnce = true;
            Log(Debug::Info)
                << "Remix sky: driving rtx.atmosphere.* and the weather blender from OpenMW's weather "
                   "state. The runtime owns the sun and moons as distant lights; this only positions "
                   "them.";
        }

        // The atmosphere runs in every cell, interiors included. There is deliberately no interior branch
        // here any more.
        //
        // What was here drove the sun to -90 degrees elevation at zero intensity, zeroed the stars and the
        // night sky, switched off the clouds and both moons, thinned the fog medium to nothing and parked
        // the weather blender. That was carried over from the MGE-XE build, where it earned its place: a
        // sun left at its last exterior elevation goes on lighting through every crack and seam, and
        // Morrowind's interiors are assembled from pieces rather than sealed volumes, so the leak was
        // immediately visible.
        //
        // Three things changed. This engine's occlusion is good enough that the leak is no longer the
        // dominant artefact it was there. The Interiors Project places distant land inside interiors to
        // fake a seamless world, so those cells need a real sky behind that geometry rather than a black
        // void. And under path tracing the sky is the scene's dominant area light, not a backdrop -- with
        // it switched off, an interior is lit only by whatever point lights the cell happens to carry,
        // which is why interiors read as flat and stage-lit and why the material response looked wrong
        // indoors.
        //
        // It also explains Mournhold showing no sky while some other interiors did. Morrowind's
        // quasi-exterior flag is what opted individual large interiors back in, and the MGE-XE build
        // relied on that flag being set on its side; nothing sets it here, so every interior took this
        // branch regardless of size or whether it had a sightline out.
        //
        // The parameter is retained rather than removed so the caller and the sky state it passes stay
        // untouched, but nothing reads it now. It should come out of the signature on a tidying pass.
        static_cast<void>(exterior);

        // Runs once, and now only to establish the initial state rather than to undo an interior branch.
        if (mExterior != 1)
        {
            mExterior = 1;
            pushBool("rtx.atmosphere.cloudEnabled", true, mCloudEnabled);
        }

        if (sky.mHaveSunDirection)
        {
            const osg::Vec3f& dir = sky.mSunDirection;

            // Normalised before asin, which the vector handed over is not.
            //
            // WeatherManager builds the orbit in Morrowind's own units and RenderingManager rewrites the
            // height to 400 - |x| (renderingmanager.cpp, setSunDirection), so this arrives with components
            // in the hundreds. Passing that to asin meant the clamp did all the work: z stays above 1 for
            // every part of the orbit except the last quarter of a percent at each end, so the elevation
            // pushed to the atmosphere was pinned at exactly 90 degrees -- the sun directly overhead -- for
            // the whole day and, mirrored, the whole night. It then slewed through the entire real range
            // inside the thin window where z finally drops below 1.
            //
            // On screen that is a sun which holds still all day and then lurches through sunrise and
            // sunset, appearing to rise a second time as the mirror below takes effect. Normalising restores
            // the curve the geometry already describes: zero at both horizons, near-overhead at midday, and
            // smooth between.
            //
            // Only the elevation was wrong. The rotation below is an azimuth in the horizontal plane, where
            // atan2 needs no normalisation and the constant -75 simply offsets the arc.
            const float length = dir.length();
            float elevation = length > 0.0f
                ? std::asin(std::clamp(dir.z() / length, -1.0f, 1.0f)) * kRadToDeg
                : 0.0f;
            const float rotation = std::atan2(dir.x(), dir.y()) * kRadToDeg;

            // OpenMW's sun never descends. WeatherManager builds the orbit as (-400*orbit, 75, -100) and
            // RenderingManager rewrites the height to 400 - |x|, which is zero at both horizons and
            // maximal in between -- so the same curve is traced during the night as during the day, and
            // at midnight the sun reads as overhead. OpenMW does not care because it swaps in the night
            // skybox and stops drawing the disc, but Remix's atmosphere takes the elevation literally and
            // would light the world at midnight. Mirroring it below the horizon is what the night flag is
            // for.
            if (sky.mNight)
                elevation = -elevation;

            pushFloat("rtx.atmosphere.sunElevation", elevation, mSunElevation);
            pushFloat("rtx.atmosphere.sunRotation", rotation, mSunRotation);

            // The sun, and nothing else, is dimmed in cells that are not outdoors.
            //
            // Morrowind's interiors are pieces fitted together rather than sealed volumes, so a sun at its
            // last exterior elevation shines through every crack and seam. The symptom is boiling light
            // indoors during the day that is absent at night, which is what names the sun rather than the sky
            // or the ambient: the sky dome and the point lights do not change between the two.
            //
            // sunIntensity is the lever for the same reason rtx.volumetrics.enable is the lever below: the
            // runtime's weather blender interpolates sunIlluminance every frame from its preset table
            // (rtx_fork_weather.h), so anything this host wrote there would be overwritten within the frame.
            // It never writes sunIntensity, and rtx_atmosphere.cpp forms the final value as
            // sunIlluminance * sunIntensity, so scaling here survives whatever the blender does to the
            // colour. The option is already declared NoSave and game-drivable per frame, which is exactly
            // this use.
            //
            // Only the sun, deliberately. The interior branch this file used to carry also switched off the
            // sky, stars, clouds, moons and the fog medium, and the reasons it was removed all still hold --
            // the Interiors Project puts distant land inside interiors, the sky dome is the dominant area
            // light under path tracing, and this engine occludes better than the MGE-XE build the branch came
            // from. Leaving the dome lit and taking away the direct beam is the part that was wanted.
            //
            // mHaveWeather is the gate, which is WeatherResult::mOutdoorAtmosphere, which is
            // isCellExterior() || isCellQuasiExterior(). So every cell Morrowind marks as behaving like an
            // exterior keeps its full sun -- Mournhold and the Vivec cantons among them -- and the decision
            // comes from the game's own data rather than a list maintained here. The removed branch keyed on
            // an `exterior` parameter that nothing on this side ever set, which is why it caught every
            // interior indiscriminately and why Mournhold lost its sky.
            static const float interiorSun = std::clamp(
                envFloat("OPENMW_REMIX_INTERIOR_SUN", Settings::remix().mInteriorSunIntensity), 0.0f, 1.0f);
            const float sunIntensity = sky.mHaveWeather ? 1.0f : interiorSun;

            const int dimmed = sunIntensity < 1.0f ? 1 : 0;
            if (dimmed != mLoggedSunDimmed)
            {
                mLoggedSunDimmed = dimmed;
                Log(Debug::Info) << "Remix sky: sun intensity " << sunIntensity
                                 << (dimmed ? " (ordinary interior, so the direct sun is held back while the"
                                              " sky keeps lighting the cell)"
                                            : " (cell is exterior or quasi-exterior)");
            }

            pushFloat("rtx.atmosphere.sunIntensity", sunIntensity, mSunIntensity);

            // Logged on every day/night changeover, because the mirror above is the one claim in this file
            // that has never been checked against a running game.
            //
            // OpenMW's sun does not descend -- RenderingManager rewrites its height to 400 - |x|, so the
            // orbit is a dome traced twice a day -- and the comment above asserts that negating the elevation
            // puts it correctly below the horizon at night. If that is wrong, the runtime lights the sky with
            // a sun at or near the horizon all night, and no night-sky option can be seen against it: airglow
            // and stars are small additive terms next to sun-scattered sky, which is exactly what "the slider
            // does nothing" looks like from the outside.
            //
            // Printing the inputs alongside the result so the arithmetic can be checked rather than trusted:
            // a night elevation that is not strongly negative is the bug, and the raw direction says whether
            // the fault is the mirror or the direction feeding it.
            const int night = sky.mNight ? 1 : 0;
            if (night != mLoggedNight)
            {
                mLoggedNight = night;
                Log(Debug::Info) << "Remix sky: " << (sky.mNight ? "NIGHT" : "DAY")
                                 << " -- sunDirection (" << dir.x() << ", " << dir.y() << ", " << dir.z()
                                 << ") len " << length << "; elevation " << elevation << " deg, rotation "
                                 << rotation << " deg"
                                 << (sky.mNight && elevation > -10.0f
                                            ? "  <-- sun is NOT below the horizon at night; the sky is being"
                                              " lit as twilight and no night-sky option can be seen against it"
                                            : "");
            }
        }

        // Secunda is moon0 and Masser is moon1, matching the Morrowind Remix project's assignment so the
        // per-moon appearance settings in rtx.conf -- size, colour, brightness, surface style -- describe
        // the moon they were authored for.
        struct MoonBinding
        {
            const MoonState* mState;
            bool mHave;
            const char* mElevationKey;
            const char* mRotationKey;
            const char* mPhaseKey;
            const char* mEnabledKey;
        };
        const MoonBinding moons[2] = {
            { &sky.mSecunda, sky.mHaveSecunda, "rtx.atmosphere.moon0.elevation0",
                "rtx.atmosphere.moon0.rotation0", "rtx.atmosphere.moon0.phase0",
                "rtx.atmosphere.moon0.enabled0" },
            { &sky.mMasser, sky.mHaveMasser, "rtx.atmosphere.moon1.elevation1",
                "rtx.atmosphere.moon1.rotation1", "rtx.atmosphere.moon1.phase1",
                "rtx.atmosphere.moon1.enabled1" },
        };

        for (int i = 0; i < 2; ++i)
        {
            if (!moons[i].mHave)
                continue;

            const osg::Vec3f dir = moonDirection(*moons[i].mState);
            const float length = dir.length();
            if (length <= 0.0001f)
                continue;
            const osg::Vec3f unit = dir / length;

            pushFloat(moons[i].mElevationKey,
                std::asin(std::clamp(unit.z(), -1.0f, 1.0f)) * kRadToDeg, mMoonElevation[i]);
            pushFloat(moons[i].mRotationKey, std::atan2(unit.x(), unit.y()) * kRadToDeg,
                mMoonRotation[i]);
            pushFloat(moons[i].mPhaseKey, remixMoonPhase(moons[i].mState->mPhase), mMoonPhase[i], 4);

            // OpenMW fades a moon out by alpha as it approaches the horizon and when the sky brightens,
            // rather than by moving it. Enabling on that alpha keeps Remix's moon light from persisting
            // through the day, when OpenMW's own moon has faded to nothing.
            pushBool(moons[i].mEnabledKey, moons[i].mState->mMoonAlpha > 0.01f, mMoonEnabled[i]);
        }

        // Stars and airglow follow OpenMW's own day/night decision rather than the sun's elevation, so
        // they change over at the same moment the game's sky does.
        // Night brightness is a value here, not a flag. starBrightness multiplies the star field and the
        // blue cloud night-glow coupled to it, and its tuned default is 0.5 -- which no rtx.conf in this
        // project or the Morrowind Remix reference overrides. Writing 1.0 to mean "stars are on" therefore
        // doubled the night sky against the very default both configurations rely on, and that was the whole
        // of this host reading brighter at night. Day still writes 0, because off is off rather than a
        // brightness.
        pushFloat("rtx.atmosphere.starBrightness",
            sky.mNight ? envFloat("OPENMW_REMIX_NIGHT_STARS", Settings::remix().mNightStarBrightness) : 0.0f,
            mStarBrightness);

        // nightSkyBrightness is deliberately not written from here, and this is now measured rather than
        // argued.
        //
        // The runtime's weather blender assigns it from its per-preset table every frame with no varies-gate
        // (rtx_fork_weather.cpp), so a host write is discarded within the same frame. The confirmation came
        // from the game rather than the source: the Night Sky Brightness slider in Remix's own developer menu
        // has no effect at either extreme, which is what an option being rewritten every frame looks like from
        // the outside. It is owned by rtx.conf as rtx.weather.preset.<name>.<name>_nightSkyBrightness, set to
        // 0.001 across all twelve presets here and in the Morrowind Remix reference alike.
        //
        // Restored briefly on the grounds that the removal was untested, then removed again once the slider
        // settled it. Writing it changed the value the runtime held at night from 0.001 to 0.008 for as long
        // as that lasted.

        // Fog, driven into the volumetric medium directly.
        //
        // The Morrowind Remix project routes fog through rtx.volumetrics.enableFogRemap, which reads the
        // D3D9 fixed-function fog state. An API host sets no such state, so that path has nothing to read
        // and this drives the medium's own parameters instead -- which is the better description anyway,
        // since Remix's volumetrics is a participating medium rather than a distance blend.
        //
        // The water surface, so the runtime can split its medium at it.
        //
        // Remix decides underwater per froxel in the shader by comparing altitude against this plane, rather
        // than being handed a submerged flag, which is what makes the surface itself look right from either
        // side. The plane goes in world units along the up axis -- Z here, matching rtx.zUp -- and a value
        // far below anything real means the cell has no water and leaves the split inert.
        pushFloat("rtx.volumetrics.waterPlaneWorldZ", waterPlaneHeight, mWaterPlane);

        // Fog density is decoupled from fog colour for this host, permanently.
        //
        // Left coupled, extinction comes out as -ln(fogColour)/measurementDistance, so Morrowind's dark
        // weather fog colours both thicken the fog and eat the daylight -- one number doing two jobs, with no
        // setting of it that gives thick fog and a lit scene at once. Decoupled, the colour only tints and
        // the density comes from the reference transmittance, which is what the per-weather day/night values
        // drive. Published once at startup because it describes the integration rather than the weather.
        if (!mFogModeSet)
        {
            mFogModeSet = true;
            mRuntime.setConfigVariable("rtx.volumetrics.fogDensityDecoupleFromColor", "True");
        }

        // Three parameters, three separate jobs, and an earlier version of this conflated the first two.
        //
        // singleScatteringAlbedo is NOT a colour. The option's own documentation calls it "the ratio of
        // scattering to absorption... more of a mathematical albedo... treated as linearly encoded data
        // (not gamma)". Feeding OpenMW's fog colour into it therefore does not tint the fog, it decides how
        // much light the medium destroys instead of scattering: a dark fog colour makes a nearly pure
        // ABSORBER, which eats the scene and returns nothing. That is why daylight went black while the fog
        // itself stayed invisible, and why the density had to be pushed to a few metres before anything
        // showed -- that was an absorber being forced to in-scatter. Fog scatters almost everything it
        // interacts with, so this belongs near white and stays there.
        //
        // transmittanceColor is the colour slot, documented as sRGB and gamma encoded, so OpenMW's fog
        // colour goes here. Its magnitude is part of the density though, not just its hue: transmittance
        // falls as transmittanceColor raised to (distance / measurementDistance). Handing it a raw fog
        // colour would therefore smuggle a density in with the tint, so the hue is taken and the level is
        // set deliberately below.
        //
        // Where the day/night/sunrise/sunset variation comes from: OpenMW already blends it. This colour is
        // weather.mFogColor, which the weather system has already interpolated between Morrowind's four
        // per-phase fog colours, so the phases arrive for free once the value reaches a slot that tints
        // rather than one that absorbs. Scattered sunlight does the rest -- a white-albedo medium under a
        // midday sun reads bright and under stars reads dark, without either being asked for.
        if (sky.mHaveWeather && fogEnabled())
        {
            // Written once and left alone, and written per channel rather than as one grey level.
            //
            // The paragraph above argues this belongs near white because fog scatters rather than absorbs.
            // That is true of daylight and it is what made a value near one look right, but it is only half
            // the story, and the missing half is what produced a night nothing could darken.
            //
            // Albedo is how much light survives each scatter event. At 0.99 the medium is very nearly
            // lossless, so light entering it bounces many times before being absorbed and multiple scattering
            // smears it into an even veil over the whole scene. With Remix's own sky feeding the medium, that
            // veil is the floor on how dark night can get: turning off the moons, the stars and the airglow
            // changes what enters the fog, not the fog's willingness to keep passing it around. Which is
            // exactly what "there is just some glow everywhere" looks like, and why no sky control touched it.
            //
            // The Morrowind Remix reference sets rtx.volumetrics.singleScatteringAlbedo to
            // (0.90, 0.92, 0.96) and its nights are the ones being matched. That is four to ten times more
            // absorption per bounce than 0.99, and it is tilted: red is absorbed most and blue least, which is
            // what makes those nights read cold rather than washed. A single grey level cannot express that,
            // so the setting is the blue channel -- the largest -- and the other two keep the reference's
            // ratios against it. At the default of 0.96 this reproduces the reference exactly.
            if (!mAlbedoSet)
            {
                mAlbedoSet = true;
                const float albedo
                    = std::clamp(envFloat("OPENMW_REMIX_FOG_ALBEDO", Settings::remix().mFogAlbedo), 0.0f, 1.0f);
                // 0.90 / 0.96 and 0.92 / 0.96, so one knob moves the level and the tilt rides along.
                char value[64];
                std::snprintf(value, sizeof(value), "%.4f, %.4f, %.4f",
                    static_cast<double>(albedo * 0.9375f), static_cast<double>(albedo * 0.958333f),
                    static_cast<double>(albedo));
                mRuntime.setConfigVariable("rtx.volumetrics.singleScatteringAlbedo", value);
            }

            // Hue without level: the fog colour is normalised by its own largest component so it carries
            // only its tint, then blended toward white. Fog is barely coloured in transmission -- what
            // looks coloured about it is the light it scatters -- so a partial tint is the honest amount.
            const float tintStrength
                = std::clamp(envFloat("OPENMW_REMIX_FOG_TINT", Settings::remix().mFogTint), 0.0f, 1.0f);
            const float level
                = std::clamp(envFloat("OPENMW_REMIX_FOG_TRANSMITTANCE", Settings::remix().mFogTransmittance), 0.01f, 0.99f);
            const float peak = std::max({ sky.mFogColor.x(), sky.mFogColor.y(), sky.mFogColor.z(), 1e-4f });

            float rgb[3] = { sky.mFogColor.x(), sky.mFogColor.y(), sky.mFogColor.z() };
            for (float& c : rgb)
            {
                const float hue = c / peak;
                c = level * (1.0f - tintStrength + tintStrength * hue);
            }

            // The option is a three-component vector, so it goes as one string and is compared as one
            // value. Formatting first and comparing the formatted result means a colour that rounds to the
            // same four decimals is never resent.
            char colour[64];
            std::snprintf(colour, sizeof(colour), "%.4f, %.4f, %.4f", static_cast<double>(rgb[0]),
                static_cast<double>(rgb[1]), static_cast<double>(rgb[2]));
            if (std::strcmp(colour, mFogColour) != 0)
            {
                std::snprintf(mFogColour, sizeof(mFogColour), "%s", colour);
                mRuntime.setConfigVariable("rtx.volumetrics.transmittanceColor", colour);
            }

            // Density, derived rather than calibrated by eye.
            //
            // A uniform medium has no start distance, so OpenMW's fogDepth cannot be honoured as one:
            // FogManager reads it as landFogStart = viewDistance * (1 - fogDepth) with the end always at
            // viewDistance, which is a ramp position, not a density. What survives the translation is how
            // opaque the far plane should be -- fogDepth of 1 means fully fogged by the view distance and a
            // tenth of that means a light haze -- so optical depth at the view distance is made
            // proportional to fogDepth.
            //
            // Solving transmittance = level^(distance / measurement) for the measurement distance that puts
            // kFogEndTransmittance at the view distance gives the ratio below. Written as the two named
            // constants rather than the number they produce, so changing either stays consistent.
            const float endTransmittance = std::clamp(
                envFloat("OPENMW_REMIX_FOG_END_TRANSMITTANCE", Settings::remix().mFogEndTransmittance), 0.001f, 0.99f);
            const float opticalDepths = std::log(endTransmittance) / std::log(level);
            const float viewMetres = viewDistance / kUnitsPerMetre;

            // A fogDepth of zero means "no fog" in OpenMW, which as a density would be infinitely thick
            // rather than absent. Substituting a very long measurement distance is the same statement in
            // the medium's own terms.
            const float distance = sky.mFogDepth > 0.001f && opticalDepths > 0.001f
                ? viewMetres / (opticalDepths * sky.mFogDepth)
                : 100000.0f;
            pushFloat("rtx.volumetrics.transmittanceMeasurementDistanceMeters", distance, mFogDistance);

            // How far the medium actually exists, which the density above says nothing about.
            //
            // The froxel grid is what carries volumetric scattering, and rtx.volumetrics
            // .froxelMaxDistanceMeters defaults to 20 metres -- 2860 units at this scene scale, against a
            // Morrowind view distance of tens of thousands. Everything past that bubble has no participating
            // medium at all, whatever density it was given, which is why the fog read as present but barely
            // volumetric: only the ground at the player's feet was inside it.
            //
            // Costed before being widened, because the obvious worry is that a bigger volume is a more
            // expensive one. It is not: the grid's resolution is fixed by froxelGridResolutionScale and
            // froxelDepthSlices, so this changes what each cell covers rather than how many there are. The
            // trade is near-field detail for far-field coverage, and froxelDepthSliceDistributionExponent is
            // 2.0, which already biases slices toward the camera and softens that trade.
            //
            // Held just inside the far plane rather than at it. The option's own documentation warns the
            // grid clips to the far plane if it reaches past it, and OpenMW's far plane is the view
            // distance, so asking for exactly that invites the clip this is trying to avoid.
            // Everything above is why the volume was widened. What it missed is what widening it costs.
            //
            // The grid's cell count is fixed, so a bigger volume is not more expensive -- that part was
            // measured and holds. But it is emphatically not free: the froxel grid is where volumetric
            // in-scattering accumulates, so its depth is how far light travels through participating medium
            // before reaching the eye. Deriving it from the view distance gave
            // 65000 * 0.95 / 143 -- about 430 metres against the runtime's 20 metre default, and the Morrowind
            // Remix project never overrides that default at all. Twenty-one times the medium depth, with the
            // sky as the light entering it, is a uniform glow across the entire scene.
            //
            // It is also invisible to every control that looks like it should govern it. nightSkyBrightness,
            // starBrightness and the moon gains change what light enters the medium; none of them changes how
            // far it then travels through it. That is why maxing or zeroing each of them in turn did nothing,
            // and why switching Remix's sky off produced a black frame instead of a dimmer one -- with no sky
            // there is nothing entering the medium to be smeared.
            //
            // Default is the runtime's own 20 metres, which is what the reference renders with. Set the
            // setting higher to trade night darkness for distant volumetric fog; 0 restores the old
            // view-distance derivation for anyone who wants it back.
            static const float froxelSetting
                = envFloat("OPENMW_REMIX_FROXEL_DISTANCE", Settings::remix().mFroxelDistance);
            constexpr float kFroxelCoverage = 0.95f;
            const float froxelMetres = froxelSetting > 0.0f
                ? froxelSetting
                : std::max(20.0f, viewDistance * kFroxelCoverage / kUnitsPerMetre);
            pushFloat("rtx.volumetrics.froxelMaxDistanceMeters", froxelMetres, mFroxelDistance);
        }

        // Volumetric fog follows whether the cell has weather at all, because otherwise it follows nothing.
        //
        // The block above only writes the medium while sky.mHaveWeather holds. It had no else, so on stepping
        // into a shop the host simply stopped updating and the runtime kept the last exterior's medium --
        // density, colour and all -- which is why interiors had outdoor fog standing in them.
        //
        // mHaveWeather is the right test rather than an explicit interior flag: a cell that behaves as an
        // exterior, Mournhold being the case that matters, reports weather and therefore keeps its fog, while
        // an ordinary interior reports none and loses it. That is the distinction wanted, and it comes from
        // the game's own data rather than a list maintained here.
        //
        // rtx.volumetrics.enable is the lever because it is one of the few volumetric options the runtime's
        // weather blender never writes -- it interpolates density, colour, anisotropy and the rest every
        // frame, so anything this host set among those would be overwritten within the frame. The enable is
        // ours to own.
        //
        // Caves and other interiors that would genuinely suit fog are not served by this: they report no
        // weather, so they get none. Giving them their own medium means driving it from cell data rather than
        // from weather, which is a separate change and wants its own setting.
        {
            const int wantVolumetrics = (sky.mHaveWeather && fogEnabled()) ? 1 : 0;
            if (wantVolumetrics != mVolumetricsEnabled)
            {
                mVolumetricsEnabled = wantVolumetrics;
                mRuntime.setConfigVariable("rtx.volumetrics.enable", wantVolumetrics ? "True" : "False");
                Log(Debug::Info) << "Remix sky: volumetric fog "
                                 << (wantVolumetrics
                                            ? "ON (cell is exterior or quasi-exterior)"
                                            : "OFF (ordinary interior, so no outdoor medium stands in it)");
            }
        }

        // The weather preset drives the runtime's own blender, which interpolates between presets on its
        // own clock. Only the target is handed over, and only when it changes -- writing it every frame
        // restarts the interpolation and the weather would never finish transitioning.
        //
        // Once OpenMW is part way through a transition, the effective target is the weather it is heading
        // toward rather than the one it is leaving.
        int target = weatherId;
        if (weatherTransition > 0.01f && nextWeatherId >= 0 && nextWeatherId < kWeatherPresetCount)
            target = nextWeatherId;
        if (target < 0 || target >= kWeatherPresetCount)
            target = 0;

        if (target != mWeatherTarget)
        {
            mWeatherTarget = target;
            mRuntime.setGameValue("__weather.blend_seconds", "20.0");
            mRuntime.setGameValue("__weather.target", kWeatherPresets[target]);
            Log(Debug::Info) << "Remix sky: weather target -> " << kWeatherPresets[target];
        }
    }
}
