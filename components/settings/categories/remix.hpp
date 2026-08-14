#ifndef OPENMW_COMPONENTS_SETTINGS_CATEGORIES_REMIX_H
#define OPENMW_COMPONENTS_SETTINGS_CATEGORIES_REMIX_H

#include <components/settings/sanitizerimpl.hpp>
#include <components/settings/settingvalue.hpp>

#include <string>
#include <string_view>

namespace Settings
{
    /// Settings for the RTX Remix path-traced scene submission.
    ///
    /// These were environment variables first, which was the right shape while each one existed for a single
    /// afternoon's experiment and the wrong shape once they became the knobs the renderer is actually tuned
    /// with. An environment variable cannot be discovered, cannot be documented where anyone will read it,
    /// and cannot be changed by whoever ends up playing this rather than building it.
    ///
    /// Every one of these is still overridable by its OPENMW_REMIX_* variable, and deliberately so: an
    /// override that does not touch settings.cfg is what makes an A/B comparison cheap, and the launch
    /// scripts already rely on them. The setting is the default, the variable wins when present.
    ///
    /// Note that a setting declared here MUST also appear in files/settings-default.cfg under [Remix].
    /// Settings::StaticValues::initDefaults throws "Default setting [Remix] <name> is not initialized" at
    /// startup otherwise, so a typo here is a failure to launch rather than a silently wrong value.
    struct RemixCategory : WithIndex
    {
        using WithIndex::WithIndex;

        // Submission features. Each of these switches off a whole class of geometry, which is how a
        // suspected culprit gets isolated without a rebuild.
        SettingValue<bool> mTerrain{ mIndex, "Remix", "terrain" };
        SettingValue<bool> mTerrainLayers{ mIndex, "Remix", "terrain layers" };
        SettingValue<bool> mTerrainComposite{ mIndex, "Remix", "terrain composite" };
        SettingValue<bool> mCompositeCompress{ mIndex, "Remix", "composite compress" };
        SettingValue<float> mCompositeBudgetMs{ mIndex, "Remix", "composite budget ms",
            makeMaxSanitizerFloat(0) };
        SettingValue<bool> mGroundcover{ mIndex, "Remix", "groundcover" };
        SettingValue<bool> mPackMatching{ mIndex, "Remix", "pack matching" };
        SettingValue<bool> mBudgetNearestFirst{ mIndex, "Remix", "budget nearest first" };

        // Culling. A path tracer needs geometry it cannot see -- for shadows and reflections -- so these are
        // a quality/cost trade rather than a straight saving, and worth reaching without a build.
        SettingValue<bool> mCull{ mIndex, "Remix", "cull" };
        SettingValue<float> mCullMargin{ mIndex, "Remix", "cull margin", makeMaxSanitizerFloat(0) };
        SettingValue<float> mMinAngularRadius{ mIndex, "Remix", "min angular radius",
            makeMaxSanitizerFloat(0) };

        // Materials and lighting.
        SettingValue<bool> mMaterialConstants{ mIndex, "Remix", "material constants" };
        SettingValue<bool> mUntexturedEmissive{ mIndex, "Remix", "untextured emissive" };
        SettingValue<float> mGlowIntensity{ mIndex, "Remix", "glow intensity", makeMaxSanitizerFloat(0) };
        SettingValue<float> mParticleEmissive{ mIndex, "Remix", "particle emissive",
            makeMaxSanitizerFloat(0) };
        SettingValue<float> mParallaxDepth{ mIndex, "Remix", "parallax depth", makeMaxSanitizerFloat(0) };
        SettingValue<float> mLightRadius{ mIndex, "Remix", "light radius", makeMaxSanitizerFloat(0) };
        SettingValue<float> mLightIntensity{ mIndex, "Remix", "light intensity", makeMaxSanitizerFloat(0) };

        // Volumetric fog. mFog is not a visual toggle: it decides whether the host drives the runtime's fog
        // medium from the weather at all. Leaving it on means anything typed into Remix's developer menu is
        // overwritten by the next fog colour change, so it has to be reachable in order to tune the rest.
        //
        // The clamp ranges match the ones the call sites already apply, so a bad value in settings.cfg is
        // corrected rather than obeyed. The call-site clamps stay regardless, because an environment override
        // does not pass through a sanitizer.
        // Brightness of Remix's star field at night, and of the cloud night-glow coupled to it. Matches the
        // runtime's own default rather than the 1.0 this host used to send, because nothing in rtx.conf
        // overrides the option and sending 1.0 doubled the night sky against that default.
        SettingValue<float> mNightStarBrightness{ mIndex, "Remix", "night star brightness",
            makeClampSanitizerFloat(0, 4) };

        // Scale on the emissive a NIF's NiMaterialProperty asks for, on ordinary geometry. 1.0 reproduces
        // what OpenMW's raster adds; 0 restores the previous behaviour of ignoring it outside particles.
        SettingValue<float> mMaterialEmissiveScale{ mIndex, "Remix", "material emissive",
            makeClampSanitizerFloat(0, 16) };

        // Submit OpenMW's own sky dome, sun and moon billboards. Off, because RemixSky drives Remix's
        // procedural atmosphere and Remix turns submitted sky geometry into the environment light -- leaving
        // both on lights the scene from two skies at once. Turn on only if Remix's own sky is disabled.
        SettingValue<bool> mSkyGeometry{ mIndex, "Remix", "sky geometry" };

        // Depth in metres of the froxel grid, which is how far volumetric in-scattering accumulates. The
        // runtime default of 20 is what the Morrowind Remix reference renders with; deriving it from the view
        // distance instead gave ~430 and made the fog a scene-wide glow no sky control could reach. 0 restores
        // that derivation.
        SettingValue<float> mFroxelDistance{ mIndex, "Remix", "froxel distance",
            makeClampSanitizerFloat(0, 4000) };

        SettingValue<bool> mFog{ mIndex, "Remix", "fog" };
        SettingValue<float> mFogAlbedo{ mIndex, "Remix", "fog albedo", makeClampSanitizerFloat(0, 1) };
        SettingValue<float> mFogTint{ mIndex, "Remix", "fog tint", makeClampSanitizerFloat(0, 1) };
        SettingValue<float> mFogTransmittance{ mIndex, "Remix", "fog transmittance",
            makeClampSanitizerFloat(0.01f, 0.99f) };
        SettingValue<float> mFogEndTransmittance{ mIndex, "Remix", "fog end transmittance",
            makeClampSanitizerFloat(0.001f, 0.99f) };

        // Diagnostics and logging. The two log limits exist because the per-material and per-texture lines
        // are how a wrong-looking surface gets traced back to a file, and also the bulk of a multi-megabyte
        // openmw.log. Zero switches each off; the defaults keep the behaviour that shipped.
        SettingValue<bool> mProbeQuad{ mIndex, "Remix", "probe quad" };
        SettingValue<int> mSceneLogFrames{ mIndex, "Remix", "scene log frames", makeMaxSanitizerInt(1) };
        SettingValue<int> mMaterialLogLimit{ mIndex, "Remix", "material log limit", makeMaxSanitizerInt(0) };
        SettingValue<int> mTextureLogLimit{ mIndex, "Remix", "texture log limit", makeMaxSanitizerInt(0) };

        /// Extra texture path fragments to classify as non-surface, comma or semicolon separated.
        ///
        /// Exists so the next mod that submits a post-process pass as ordinary geometry can be neutralised
        /// from a config line instead of a rebuild -- see kNonSurfaceTexturePatterns in remixscene.cpp, and
        /// the Sun's Dusk liquid effect that motivated it.
        SettingValue<std::string> mSkipTextures{ mIndex, "Remix", "skip textures" };
    };
}

#endif
