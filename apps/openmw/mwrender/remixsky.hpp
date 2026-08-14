#ifndef OPENMW_MWRENDER_REMIXSKY_H
#define OPENMW_MWRENDER_REMIXSKY_H

#include <limits>

#include "sky.hpp"

namespace RemixRT
{
    class Runtime;
}

namespace MWRender
{
    /// Drives Remix's own sky, sun, moons and fog from OpenMW's weather state.
    ///
    /// Remix has a complete atmosphere of its own -- a physical sky model, procedural clouds, stars,
    /// and real distant lights for the sun and each moon, built inside the runtime by
    /// fhSyncAtmosphereDistantLights when rtx.skyMode is Numos. None of that needs geometry submitted to
    /// it and none of it can be driven through the mesh/instance API. What it needs is numbers.
    ///
    /// So this class submits nothing. It pushes configuration: a handful of rtx.atmosphere.* values
    /// through SetConfigVariable and a weather preset name through SetGameValue, both of which are
    /// ordinary entry points the host already has. Without it those options keep whatever rtx.conf
    /// happened to set, which is why the sun sat frozen at a fixed elevation regardless of the time of
    /// day -- the light was always there, nothing was moving it.
    ///
    /// Everything read here comes from SkyManager::getState(), which retains what OpenMW already
    /// computes and blends for its own sky. Nothing is recomputed and no game state is read twice.
    class RemixSky
    {
    public:
        explicit RemixSky(RemixRT::Runtime& runtime);

        /// Pushes anything that has changed since the previous call. Cheap to call every frame.
        ///
        /// Values are only written when they actually move, for two reasons. Formatting and parsing
        /// strings for a dozen options every frame is waste, but more importantly re-pushing the weather
        /// target resets the runtime's blender mid-interpolation, so a preset written unconditionally
        /// would hold the weather permanently at the start of its transition.
        void update(const SkyManager::State& sky, bool exterior, float waterPlaneHeight, int weatherId,
            int nextWeatherId,
            float weatherTransition, float viewDistance);

    private:
        /// Writes \a key only when \a value differs from what was last written for it.
        void pushFloat(const char* key, float value, float& last, int decimals = 2);
        void pushBool(const char* key, bool value, int& last);

        RemixRT::Runtime& mRuntime;

        /// Last written values. Sentinels are deliberately unreachable so the first update always writes:
        /// a NaN compares unequal to everything including itself, and -1 is outside the bool encoding.
        float mSunElevation = std::numeric_limits<float>::quiet_NaN();
        float mSunRotation = std::numeric_limits<float>::quiet_NaN();
        float mSunIntensity = std::numeric_limits<float>::quiet_NaN();
        float mMoonElevation[2] = { std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN() };
        float mMoonRotation[2] = { std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN() };
        float mMoonPhase[2] = { std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN() };
        float mStarBrightness = std::numeric_limits<float>::quiet_NaN();
        float mFogDistance = std::numeric_limits<float>::quiet_NaN();
        float mFroxelDistance = std::numeric_limits<float>::quiet_NaN();
        float mWaterPlane = std::numeric_limits<float>::quiet_NaN();
        /// The density/colour decouple is a one-time statement about how this host drives fog, not a
        /// per-frame value, so it is published once rather than fought over with the developer menu.
        bool mFogModeSet = false;
        /// The scattering albedo is a property of fog rather than of the weather, so it is written once.
        /// Resending it per frame would also fight anyone tuning it in the developer menu.
        bool mAlbedoSet = false;
        /// Held as the formatted string rather than three floats, because the option is written as one
        /// vector and comparing the formatted form is what decides whether a write is needed. Empty means
        /// nothing written yet, which no formatted colour can equal.
        char mFogColour[64] = { '\0' };
        int mMoonEnabled[2] = { -1, -1 };
        int mCloudEnabled = -1;
        int mExterior = -1;
        /// Day/night state at the last changeover log. -1 so the first update always reports.
        int mLoggedNight = -1;
        /// Whether the sun was last reported dimmed for an interior (1), at full strength (0), or never (-1).
        int mLoggedSunDimmed = -1;

        /// Whether rtx.volumetrics.enable was last pushed on (1), off (0), or never (-1).
        ///
        /// Tri-state so the first frame always writes, whichever way it goes. The runtime's weather blender
        /// does not touch this option, so a value pushed here stays put until the cell type changes.
        int mVolumetricsEnabled = -1;
        /// Weather preset last handed to the blender. -1 rather than 0, because 0 is Clear.
        int mWeatherTarget = -1;
        bool mLoggedOnce = false;
    };
}

#endif
