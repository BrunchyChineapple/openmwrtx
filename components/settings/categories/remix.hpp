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
        // The additive counterpart. Separate because the two mean different things: this is a flat radiance
        // handed to a surface on the grounds that additive blending means it emits, where "particle emissive"
        // multiplies an emission the asset actually declares. Reachable because its default stacks with the
        // emissive blend type the material already carries -- it predates that blend type and was not
        // rechecked against it -- so the figure that holds for both flames and smoke has to be found against
        // the screen. See submitParticles for the provenance.
        SettingValue<float> mParticleAdditiveEmissive{ mIndex, "Remix", "particle additive emissive",
            makeMaxSanitizerFloat(0) };
        // Whether a surface whose texcoords are driven by a controller gets real transparency instead of the
        // cutout that is substituted for alpha blending otherwise. Morrowind authors smoke, steam, mist and
        // waterfalls as UV-scrolled tri-shapes rather than particles, and a cutout renders those as opaque
        // fragments of their densest texels. Reachable because it is one signal standing in for a judgement
        // about a whole class of assets. See materialFor.
        SettingValue<bool> mBlendAnimatedUv{ mIndex, "Remix", "blend animated uv" };
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
        //
        // The clamp's ceiling is well above 1.0 on purpose. Reproducing the raster is not the same as looking
        // right under a path tracer, and measured on a lit Glow in the Dahrk pane, 1.0 reads dim. The slider
        // in the settings window stops at 10; this accepts 16 so a hand-edited settings.cfg has room above
        // the slider. Keep the slider's SettingMax at or below this or its top end silently does nothing.
        SettingValue<float> mMaterialEmissiveScale{ mIndex, "Remix", "material emissive",
            makeClampSanitizerFloat(0, 16) };

        // Submit OpenMW's own sky dome, sun and moon billboards. Off, because RemixSky drives Remix's
        // procedural atmosphere and Remix turns submitted sky geometry into the environment light -- leaving
        // both on lights the scene from two skies at once. Turn on only if Remix's own sky is disabled.
        SettingValue<bool> mSkyGeometry{ mIndex, "Remix", "sky geometry" };

        // What rtx.atmosphere.sunIntensity is driven to in cells the engine does not flag as outdoors.
        //
        // Morrowind's interiors are assembled from pieces rather than sealed volumes, so a sun at its last
        // exterior elevation shines through every crack and seam. Under path tracing that reads as boiling
        // light indoors during the day and is absent at night, which is what identifies the sun as the source
        // rather than the sky or the ambient.
        //
        // Deliberately the sun alone. An earlier interior branch in RemixSky switched off the sky, stars,
        // clouds, moons and fog medium as well, and was removed for three reasons that all still hold: the
        // Interiors Project places distant land inside interiors and those cells need a real sky behind it;
        // under path tracing the sky dome is the scene's dominant area light, so removing it leaves interiors
        // flat and stage-lit; and this engine's occlusion makes the leak less dominant than it was in the
        // MGE-XE build the branch came from. Scaling the sun keeps the dome lighting the room and takes away
        // only the direct beam.
        //
        // The gate is the engine's own exterior flag, not a cell list, so anything Morrowind marks as
        // behaving like an exterior -- Mournhold and the Vivec cantons -- keeps its full sun. That is also
        // what the removed branch got wrong: it keyed on a parameter nothing set, so every interior took it
        // regardless of size or whether it had a sightline out.
        SettingValue<float> mInteriorSunIntensity{ mIndex, "Remix", "interior sun intensity",
            makeClampSanitizerFloat(0, 1) };

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
        // How many phases of a UV animation the stage bake resolves into a sprite sheet, when a mesh's
        // texture stages cannot all be left to move with the animation and so have to be flattened.
        //
        // The sheet is laid out in one row, so this multiplies the baked image's width and is bounded by an
        // 8,192 budget: raising it narrows each frame rather than growing the sheet. Eight at the bake's
        // default 1,024 cap comes to exactly that budget, which is why the default costs no resolution at all
        // against a single frozen frame. One disables the sheet and freezes the animation.
        SettingValue<int> mStagePhases{ mIndex, "Remix", "stage phases", makeClampSanitizerInt(1, 255) };

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
