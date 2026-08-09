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
            const char* env = std::getenv("OPENMW_REMIX_FOG");
            return env == nullptr || (*env != '\0' && *env != '0');
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

    /// Remix's phase parameter runs 0 = new, 0.5 = full, 1 = new again, so it needs to know waxing from
    /// waning. OpenMW's Phase enum distinguishes them; MoonState::phaseToInt does not -- it folds the two
    /// crescents onto 1 and the two quarters onto 2, because the vanilla moon texture is symmetric and the
    /// renderer only needs to pick a sprite. Using that would light both halves of the month identically.
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
            pushFloat("rtx.atmosphere.sunIntensity", 1.0f, mSunIntensity);
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
        pushFloat("rtx.atmosphere.starBrightness", sky.mNight ? 1.0f : 0.0f, mStarBrightness);
        pushFloat("rtx.atmosphere.nightSkyBrightness", sky.mNight ? 0.008f : 0.0f, mNightSkyBrightness);

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
            // Written once and left alone. Fog scatters; it does not absorb.
            if (!mAlbedoSet)
            {
                mAlbedoSet = true;
                const float albedo = std::clamp(envFloat("OPENMW_REMIX_FOG_ALBEDO", kFogAlbedo), 0.0f, 1.0f);
                char value[64];
                std::snprintf(value, sizeof(value), "%.4f, %.4f, %.4f", static_cast<double>(albedo),
                    static_cast<double>(albedo), static_cast<double>(albedo));
                mRuntime.setConfigVariable("rtx.volumetrics.singleScatteringAlbedo", value);
            }

            // Hue without level: the fog colour is normalised by its own largest component so it carries
            // only its tint, then blended toward white. Fog is barely coloured in transmission -- what
            // looks coloured about it is the light it scatters -- so a partial tint is the honest amount.
            const float tintStrength
                = std::clamp(envFloat("OPENMW_REMIX_FOG_TINT", kFogTint), 0.0f, 1.0f);
            const float level
                = std::clamp(envFloat("OPENMW_REMIX_FOG_TRANSMITTANCE", kFogTransmittance), 0.01f, 0.99f);
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
                envFloat("OPENMW_REMIX_FOG_END_TRANSMITTANCE", kFogEndTransmittance), 0.001f, 0.99f);
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
            constexpr float kFroxelCoverage = 0.95f;
            const float froxelMetres = std::max(20.0f, viewDistance * kFroxelCoverage / kUnitsPerMetre);
            pushFloat("rtx.volumetrics.froxelMaxDistanceMeters", froxelMetres, mFroxelDistance);
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
