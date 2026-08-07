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

    void RemixSky::update(const SkyManager::State& sky, bool exterior, int weatherId, int nextWeatherId,
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
        // singleScatteringAlbedo is what the medium scatters, so that is where OpenMW's fog COLOUR
        // belongs: it is the colour distant geometry fades toward, which is exactly what in-scattering
        // looks like. transmittanceMeasurementDistanceMeters carries the density, because that is the
        // distance over which transmittanceColor survives -- shorter means thicker.
        //
        // fogDepth is a fraction of view distance, not a distance: FogManager derives
        // landFogStart = viewDistance * (1 - fogDepth) with the end always at viewDistance. So the span
        // over which the scene goes from clear to fully fogged is viewDistance * fogDepth, and that span
        // is what sets the density. This mapping is a calibration rather than a derivation and is the
        // first thing to adjust if fog reads too thick or too thin.
        if (sky.mHaveWeather)
        {
            // The option is a three-component vector, so it goes as one string and is compared as one
            // value. Formatting first and comparing the formatted result means a colour that rounds to the
            // same four decimals is never resent.
            char colour[64];
            std::snprintf(colour, sizeof(colour), "%.4f, %.4f, %.4f",
                static_cast<double>(sky.mFogColor.x()), static_cast<double>(sky.mFogColor.y()),
                static_cast<double>(sky.mFogColor.z()));
            if (std::strcmp(colour, mFogColour) != 0)
            {
                std::snprintf(mFogColour, sizeof(mFogColour), "%s", colour);
                mRuntime.setConfigVariable("rtx.volumetrics.singleScatteringAlbedo", colour);
            }

            const float span = viewDistance * sky.mFogDepth;
            // A fogDepth of zero means "no fog" in OpenMW, which as a density would be infinitely thick
            // rather than absent. Substituting a very long measurement distance is the same statement in
            // the medium's own terms.
            const float distance = span > 1.0f ? span / kUnitsPerMetre : 100000.0f;
            pushFloat("rtx.volumetrics.transmittanceMeasurementDistanceMeters", distance, mFogDistance);
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
