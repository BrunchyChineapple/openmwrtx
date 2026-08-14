#include "remixscene.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <chrono>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <osg/Camera>
#include <osg/FrameStamp>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osg/Polytope>
#include <osg/TriangleIndexFunctor>

#include <osg/AlphaFunc>
#include <osg/BlendFunc>
#include <osg/Image>
#include <osg/Light>
#include <osg/Material>
#include <osg/FrameStamp>
#include <osg/StateSet>
#include <osg/TexMat>
#include <osg/Texture2D>

#include <components/debug/debuglog.hpp>
#include <components/sceneutil/lightmanager.hpp>
#include <components/sceneutil/statesetupdater.hpp>
#include <components/nifosg/controller.hpp>
#include <osgParticle/Particle>
#include <osgParticle/ParticleSystem>

#include <components/remixrt/assethash.hpp>
#include <components/sceneutil/morphgeometry.hpp>
#include <components/sceneutil/riggeometry.hpp>
#include <components/sceneutil/texturetype.hpp>
#include <components/settings/values.hpp>
// terraindrawable.hpp holds an osg::ref_ptr to a forward-declared CompositeMapRenderer, and ref_ptr's
// destructor needs the complete type. Included for that reason alone; nothing here uses it.
#include <components/terrain/compositemaprenderer.hpp>
#include <components/terrain/terraindrawable.hpp>

#include "vismask.hpp"

namespace
{
    /// Frames a cached mesh may go unused before it is released.
    ///
    /// Generous on purpose. Terrain repaging and object paging bring geometry in and out constantly, and
    /// destroying a mesh the moment it leaves view would mean recreating it as soon as the player turns
    /// around -- and recreation is the expensive direction, since the API has no update path.
    // Short, because nothing legitimate idles in this cache. Culling currently rejects nothing, so every
    // drawable in a loaded cell is resubmitted every frame and its mesh is refreshed every frame. What does
    // go idle is geometry from an unloaded cell, and animation: a drawable whose vertices change fails the
    // identity memo's modified-count gate, re-hashes to a different hash, and gets a new mesh, because the
    // runtime offers createMesh with no update counterpart. The superseded mesh is unreachable the moment
    // its successor exists, so holding it for hundreds of frames retains one acceleration structure per
    // animated drawable per frame -- the population grows as drawables times the window, which is why it
    // compounded with every cell entered.
    constexpr std::uint64_t kMeshEvictionFrames = 30;

    /// Ceiling on destroyMesh calls in one frame. See evictStaleMeshes for why a ceiling is needed at all.
    ///
    /// Has to be read together with kMeshBuildsPerFrame, because the two are the inflow and the outflow of
    /// the same cache and a mismatch is unbounded growth rather than a slower steady state. Measured over a
    /// long session: around 250 meshes built every frame -- morph geometry and particles re-hash constantly,
    /// since the API offers createMesh with no update counterpart -- against 64 retired. The cache climbed
    /// from nothing to 393,689 entries and was still gaining 26,940 per report at the end, dipping
    /// occasionally but never catching up.
    ///
    /// Matched to the build ceiling rather than set to some fraction of it. A destroy releases an
    /// acceleration structure where a build constructs one, so retirement is the cheaper direction; there is
    /// no reason for it to be the narrower one. The device-stall concern that put a ceiling here in the first
    /// place is about retiring thousands at once with no frame submitted in between, which this still bounds.
    constexpr unsigned int kMeshDestroysPerFrame = 512;

    /// Ceiling on createMesh calls in one frame.
    ///
    /// The counterpart to the destroy ceiling, and the more important of the two, because creation is the
    /// expensive direction: each call builds an acceleration structure. meshFor described itself as bounded
    /// per frame for a long time while nothing actually bounded it, and the cost of that showed up as whole
    /// frames spent building:
    ///
    ///     2147 meshes -> 487 ms      551 -> 111 ms      283 -> 53 ms      126 -> 31 ms
    ///
    /// which is a flat 0.2 ms per build and says the traversal's own walk is almost free by comparison --
    /// culling fifteen thousand drawables costs a few milliseconds, building two thousand meshes costs half
    /// a second. Bursts like that are geometry coming into view at once, on a cell change or a corner.
    ///
    /// Deferring costs the overflow one frame of freshness: nothing is destroyed, no memo is written, the
    /// caller simply does not submit that instance, and the next frame retries. New geometry therefore
    /// appears a frame or two late under load, which is a much better failure than a half-second freeze --
    /// and better than the failure the old unbounded form risked, where a device that cannot retire the
    /// builds fails its fence sync and reports a bare device loss with no fault behind it.
    ///
    /// 128 was chosen against the measured standing load of about 33 builds a frame, which it clears
    /// comfortably. What it does not clear is arrival: walking into a town measured up to 1,476 builds
    /// deferred in a single frame, non-zero in twelve of eighteen sampled frames. At 128 a frame that
    /// backlog takes a dozen frames to drain, and everything still queued is not merely late -- meshFor
    /// returns nothing, so the instance is not submitted and the object is absent. The visible result was
    /// buildings and actors blinking in a few at a time on entering Pelagiad, and it read as memory pressure
    /// because that is what missing geometry usually is.
    ///
    /// 512 at the measured 0.2 ms a build puts the worst arrival frame near 100 ms rather than 26. That is a
    /// deliberate trade of a brief hitch on arrival for geometry that is present when it should be: a stutter
    /// is honest about being one frame, whereas absent geometry looks like a defect. The standing load is far
    /// below either figure, so this still never binds during ordinary play.
    ///
    /// Raising the eviction window would be the wrong companion change. Arrival is new geometry rather than
    /// geometry being rebuilt after eviction, so a longer window buys nothing here, and kMeshEvictionFrames
    /// is short on purpose -- see its own note on superseded meshes from animated drawables accumulating one
    /// acceleration structure per drawable per frame.
    ///
    /// Overridable through OPENMW_REMIX_MESH_BUILD_BUDGET, which is how the figures above were obtained and
    /// how a device that cannot sustain 512 can be brought back down without a rebuild.
    constexpr unsigned int kMeshBuildsPerFrame = 512;


    /// Most instances handed over in a single frame.
    ///
    /// A safety rail, not a quality setting. There is no culling in this traversal yet, and OpenMW's
    /// viewing distance is user-configurable into the hundreds of thousands of units, so paged terrain
    /// and object paging can present far more geometry than is sane to convert and submit each frame.
    /// Hitting this draws a partial scene and says so, which is a much better failure than hanging.
    constexpr unsigned int kMaxInstancesPerFrame = 20000;

    /// Most groundcover copies handed over in a single frame.
    ///
    /// Its own allowance rather than a share of kMaxInstancesPerFrame, because grass is instanced in the
    /// thousands per chunk across every chunk within Groundcover/"rendering distance". Drawn from the
    /// general budget it would crowd the world out of its own frame -- architecture and actors dropped so
    /// that more blades could be submitted. When this runs out the remaining blades are skipped and
    /// everything else is submitted as usual.
    constexpr unsigned int kMaxGroundcoverCopies = 8000;

    /// Vertex attribute slots OpenMW's groundcover keeps per-copy placement in.
    ///
    /// 6 is (position.xyz, scale) relative to the chunk, 7 is an Euler rotation. Set in
    /// Groundcover::InstancingVisitor and read by files/shaders/compatibility/groundcover.vert, both as
    /// bare literals -- there is no shared symbol to reference, so these have to be kept in step by hand.
    constexpr unsigned int kGroundcoverOffsetAttrib = 6;
    constexpr unsigned int kGroundcoverRotationAttrib = 7;

    /// Reads a boolean environment variable, defaulting to \a fallback when unset or empty.
    bool envFlag(const char* name, bool fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        return *value != '0';
    }

    /// Reads a millisecond budget from an environment variable, defaulting to \a fallback when unset,
    /// empty or not a number. Negative values are clamped to zero, which the callers read as "no limit".
    double envMilliseconds(const char* name, double fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        return std::max(0.0, std::atof(value));
    }

    /// Emission strength for world geometry while lights are still missing. Low enough to read as flat
    /// lighting rather than a white-out, high enough to survive auto-exposure against a daylight sky.
    constexpr float kWorldEmissive = 2.0f;

    /// Distance in front of the eye at which the bisect quad sits, and its half-extent there. Sized to
    /// cover roughly the middle third of a 90-degree vertical field of view, so it is unmistakable
    /// without hiding the rest of the frame.
    constexpr float kProbeDistance = 100.0f;
    constexpr float kProbeHalfSize = 25.0f;

    /// Emitting radius for a converted OpenMW light, in OpenMW units.
    ///
    /// OpenMW's lights are points with an attenuation curve and no physical size. A path tracer needs
    /// one, because a true point light gives razor-sharp shadows and unbounded radiance at the source.
    ///
    /// This matches the value the Morrowind Remix work settled on: Remix's own legacy light conversion
    /// computes lightConversionSphereLightFixedRadius * sceneScale, and 0.45 * 1.43 is 0.6435 units, a
    /// touch under a centimetre. That option cannot reach these lights -- it only applies to lights the
    /// runtime converts from D3D9, and these are created directly through the API -- so the number has
    /// to be reproduced here.
    ///
    /// An earlier value of twenty units was far too large, and the failure is instructive: a sphere light
    /// is a volume, so an emitter that big placed against a wall has part of itself on the far side, and
    /// light pours through solid geometry into the next room.
    ///
    /// Set to one unit rather than that reproduced 0.6435, chosen by eye against this build's lighting.
    /// Keep it in step with the rtx.externalLight.radius default in the runtime's rtx_light_manager.h: this
    /// value is published into that option at startup, so the two disagreeing only shows up as the
    /// developer menu's slider sitting somewhere other than where the lighting actually is.
    constexpr float kLightRadiusDefault = 1.0f;

    /// Floor for the emitter radius, whatever the menu or environment asks for.
    ///
    /// The radiance derivation divides by the square of this, so zero is not a small light, it is a
    /// division by zero that propagates into every light in the scene at once.
    constexpr float kLightRadiusMinimum = 0.001f;

    /// Frames of no further change before live light tuning is written to the log.
    ///
    /// Long enough that dragging a slider produces one line instead of one per frame, short enough that
    /// the line has appeared by the time anyone goes looking for it.
    constexpr std::uint64_t kLightTuningSettleFrames = 30;

    /// How often the scene handover summary is repeated, in frames.
    ///
    /// Slow enough not to fill the log over a session, often enough that a walk between two areas produces
    /// several samples to compare. Overridable because 600 frames is 20-40 seconds at path-traced frame
    /// times, which is too coarse to catch a transient state -- holding third person for fifteen seconds
    /// produced no sample at all, and the camera values in that line were the whole point of looking.
    std::uint64_t sceneLogInterval()
    {
        if (const char* value = std::getenv("OPENMW_REMIX_SCENE_LOG_FRAMES"); value != nullptr)
        {
            const long parsed = std::strtol(value, nullptr, 10);
            if (parsed > 0)
                return static_cast<std::uint64_t>(parsed);
        }
        return static_cast<std::uint64_t>(std::max(1, Settings::remix().mSceneLogFrames.get()));
    }

    /// Game-state store keys the runtime publishes live light tuning on.
    ///
    /// Must match kExternalLightRadiusKey and kExternalLightIntensityKey in the runtime's
    /// rtx_light_manager.h. They are the return path for the two sliders under "External (API) Light
    /// settings": the menu writes an rtx.externalLight.* option, its change callback mirrors the value
    /// here, and this side polls. The store is used rather than a config getter because reading an option
    /// back would need a new API entry point, and a new vtable slot plus an ABI bump is a lot to spend on
    /// two floats.
    constexpr const char* kExternalLightRadiusKey = "__externalLight.radius";
    constexpr const char* kExternalLightIntensityKey = "__externalLight.intensityFactor";

    /// The rtx.* options behind those keys, written once at startup so the menu opens showing the values
    /// actually in use rather than its own defaults. Both are NoSave in the runtime, so writing them does
    /// not leave anything behind in the user's configuration.
    constexpr const char* kExternalLightRadiusOption = "rtx.externalLight.radius";
    constexpr const char* kExternalLightIntensityOption = "rtx.externalLight.intensityFactor";

    /// Radiance multiplier at a radius of one unit, i.e. radiance * radius^2.
    ///
    /// Expressed this way so the radius above can be changed without re-tuning brightness. Radiance for
    /// a fixed total power goes as the inverse square of the radius, so a bare radiance constant silently
    /// couples the two -- shrink the emitter and the scene goes dark by the square of the change.
    ///
    /// The value itself is not physically derived and cannot be: OpenMW's lighting is an artist-tuned
    /// attenuation curve evaluated per vertex with no defined relationship to radiometric units. It is
    /// chosen to preserve the brightness the previous radius produced, so this change alters shadow
    /// sharpness and light leakage without altering exposure.
    /// Radiance a converted light is expected to have fallen to at the far end of its reach.
    ///
    /// Matches kNewLightEndValue in the runtime's rtx_lights.h. It is the threshold the conversion solves
    /// against, so it has to be the same number on both sides or lights submitted through the API land in
    /// a different range from lights the runtime converts itself.
    constexpr float kLightEndValue = 0.01f;

    /// Default for OPENMW_REMIX_LIGHT_INTENSITY, and the value published into
    /// rtx.externalLight.intensityFactor at startup.
    ///
    /// Duplicated rather than read back from the runtime because the API still exposes no getter for an
    /// option, which means the two can drift. Keep it in step with the rtx.externalLight.intensityFactor
    /// default in the runtime's rtx_light_manager.h.
    ///
    /// No longer tracks rtx.lightConversionIntensityFactor, which it was originally set to mirror. That
    /// option applies only to lights the runtime converts from legacy D3D9 draws and cannot reach lights
    /// created through the API, so tying this to it was misleading -- this is the exposure knob for this
    /// build's own lights, tuned by eye.
    constexpr float kLightIntensityFactorDefault = 2.5f;

    /// The attenuation value a legacy light is considered to have faded out at, 1/255.
    ///
    /// Matches kLegacyLightEndValue in the runtime. One part in 255 is where an eight-bit framebuffer can
    /// no longer represent the contribution, which is what makes it the point the original game's lighting
    /// effectively stopped.
    constexpr float kLegacyLightEndValue = 1.0f / 255.0f;

    /// Emissive radiance scale for ADDITIVELY blended particle quads.
    ///
    /// Applied only where SurfaceState::mAdditive says the blend function accumulates -- flames, glows,
    /// sparks, magic. Those are self-lit by construction and come through as grey cardboard without it,
    /// which is what a path tracer makes of a non-emissive quad.
    ///
    /// It used to be applied to every particle unconditionally, on the reasoning that a faintly glowing
    /// smoke puff was the smaller error. That was wrong twice over: at 6.0 over a pale texture the puff is
    /// not faintly glowing but saturated solid white, and an emitter is outside the lighting solution
    /// altogether -- it takes no shadow and picks up no bounce colour, so smoke stops behaving like smoke
    /// rather than merely looking too bright.
    constexpr float kParticleEmissive = 6.0f;
    /// Radiance scale applied to an ALPHA-blended particle's OWN emissive colour.
    ///
    /// Different in both value and meaning from kParticleEmissive. That one is a flat radiance handed to a
    /// surface that emits by construction. This one multiplies the emission the asset actually declares, so a
    /// mesh carrying none still gets none and only what Morrowind marked as self-lit is affected.
    ///
    /// 1.5 because both neighbouring failures are visible on screen: at kParticleEmissive's 6.0 a pale smoke
    /// texture saturates to solid white, and at 0 -- which is what shipped -- smoke and steam are lit only by
    /// whatever reaches a thin billboard in a dim interior, which is nearly nothing. A modest term leaves the
    /// surface in the lighting solution, still taking shadow and bounce, with a floor under it so the effect
    /// is legible at all.
    ///
    /// Tunable through OPENMW_REMIX_PARTICLE_EMISSIVE for the same reason as the glow scale: the defensible
    /// number depends on the exposure the rest of the scene settles at, and that is a judgement to make
    /// against the screen rather than in a header.
    constexpr float kParticleAlphaEmissive = 1.5f;
    /// Radiance scale for a surface carrying a glow map.
    ///
    /// Well below kParticleEmissive, because the two are doing different jobs. That figure has to make a
    /// flat quad read as fire across its whole area; a glow map is already black everywhere that should not
    /// emit, so this only scales the texels that do -- a lantern's glass, a rune, the gills of a glowing
    /// plant. Overstating it does not make those brighter so much as it turns small bright regions into
    /// light sources that wash the room and drag auto-exposure down with them.
    ///
    /// Tunable through OPENMW_REMIX_GLOW_INTENSITY, because the defensible value depends on the exposure the
    /// rest of the scene settles at and that is a judgement to make against the screen, not in a header.
    constexpr float kGlowEmissive = 2.0f;

    /// Remix BlendType values, from BlendType in the runtime's surface_shared.h.
    ///
    /// Passed as plain ints because the enum lives in a shader-shared header the API does not export. The
    /// two used here are the ones the runtime itself derives from the D3D9 blend factors Morrowind's
    /// particles amount to: kAlpha for SRC_ALPHA/ONE_MINUS_SRC_ALPHA, kAlphaEmissive for SRC_ALPHA/ONE.
    /// Naming them after that derivation rather than after the effect keeps the mapping checkable against
    /// InstanceManager::calculateAlphaState, which is the only place the meaning is actually defined.
    constexpr int kBlendTypeAlpha = 0;
    constexpr int kBlendTypeAlphaEmissive = 1;

    /// Sentinel for createTexturedMaterial's blend type meaning "do not blend at all".
    constexpr int kBlendTypeNone = -1;

    /// A 2D affine as { m00, m01, m10, m11, m30, m31 }, applied to a texcoord as a row vector -- the same
    /// six-float layout SurfaceState uses for mTexMat and mCoverageTexMat, and the same convention osg's
    /// TexMat uses, so a matrix read out of a state set drops straight in.
    using TexAffine = std::array<float, 6>;

    constexpr TexAffine kIdentityAffine = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };

    /// Solves for the affine taking \a from's output space to \a to's, i.e. inverse(from) composed with
    /// \a to, and writes it as two rows of { a, b, c } evaluated as a*u + b*v + c.
    ///
    /// This is what lets the coverage mask be sampled correctly from a mesh that carries only one texcoord
    /// set. OpenMW draws a terrain layer with the diffuse tiled many times across the chunk and the blend
    /// map stretched once through an inset of its own, and the texcoords submitted to Remix have the
    /// *diffuse* matrix already baked in, because a Remix vertex has nowhere else to put it. Going from
    /// those texcoords back to blend map UVs therefore means undoing one matrix and applying the other.
    ///
    /// @return false if \a from is singular, in which case there is no such map and the caller has no
    ///         business claiming one.
    bool solveTexAffineBetween(const TexAffine& from, const TexAffine& to, float (&outU)[3], float (&outV)[3])
    {
        const float det = from[0] * from[3] - from[1] * from[2];
        // Relative to the coefficients themselves: a terrain tiling factor is in the tens, so this is only
        // ever true for a genuinely degenerate matrix rather than for a small one.
        const float magnitude = std::max(
            { std::abs(from[0]), std::abs(from[1]), std::abs(from[2]), std::abs(from[3]), 1.0f });
        if (std::abs(det) < 1e-9f * magnitude * magnitude)
            return false;

        const float invDet = 1.0f / det;

        // Linear part of inverse(from), row-vector convention.
        const float i00 = from[3] * invDet;
        const float i01 = -from[1] * invDet;
        const float i10 = -from[2] * invDet;
        const float i11 = from[0] * invDet;

        // ...and its translation, which is the original translation carried through the inverted linear
        // part and negated. Dropping this term is a silent half-texel shift, so it is spelled out.
        const float it0 = -(from[4] * i00 + from[5] * i10);
        const float it1 = -(from[4] * i01 + from[5] * i11);

        // Compose with `to`.
        outU[0] = i00 * to[0] + i01 * to[2];
        outU[1] = i10 * to[0] + i11 * to[2];
        outU[2] = it0 * to[0] + it1 * to[2] + to[4];

        outV[0] = i00 * to[1] + i01 * to[3];
        outV[1] = i10 * to[1] + i11 * to[3];
        outV[2] = it0 * to[1] + it1 * to[3] + to[5];
        return true;
    }

    /// Distance at which \a light has faded to imperceptibility, by the runtime's own derivation.
    ///
    /// This is the number the radiance conversion needs, and it is emphatically not the light's radius.
    /// OpenMW's radius is the cutoff for which objects are lit; the attenuation curve keeps going well past
    /// it. Remix says as much in LightUtils::calculateIntensity -- "the calculated max distance may be
    /// greater than the Light's original Range value" -- because older games used a small range with a
    /// deliberately bright colour and a slow curve as a culling optimisation, and the physical light has to
    /// reflect the curve rather than the cull distance. Using the radius directly made every light roughly
    /// thirty times too dim, and worse, too dim by a radius-dependent amount.
    ///
    /// The five-sample least-squares fit is Remix's leastSquareIntensity rather than its closed-form
    /// solver, and that choice matters for Morrowind specifically. Solving `brightness / (a d^2 + b d + c)
    /// = 1/255` exactly is fine for an inverse-square curve, but Morrowind's default falloff is *linear*,
    /// and a linear curve reaches 1/255 only at an absurd distance -- of the order of twenty thousand units
    /// for a lantern, which then squares into a radiance in the millions. Fitting an inverse-square curve
    /// over the light's actual range instead gives a distance that reflects where the light really stops
    /// mattering.
    /// Takes SceneUtil::Light rather than osg::Light because upstream 28fefe86d9 ("remove osg::Light*")
    /// replaced the osg type with its own. Only the three attenuation terms are read, and they carry the
    /// same names and meanings on the new type, so the fit below is unchanged.
    float lightEndDistance(const SceneUtil::Light& light, float range, float brightness)
    {
        const double a = light.getQuadraticAttenuation();
        const double b = light.getLinearAttenuation();
        const double c = light.getConstantAttenuation();

        // Fit intensity/d^2 to brightness/(a*d^2 + b*d + c) over five samples across the range, choosing
        // the intensity that minimises the squared error. Double precision because the 1/d^4 term spans
        // several orders of magnitude across the samples and loses badly in single.
        constexpr int kSamples = 5;
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 1; i <= kSamples; ++i)
        {
            const double distance = static_cast<double>(i) / kSamples * range;
            const double distanceSq = distance * distance;
            const double attenuation = a * distanceSq + b * distance + c;
            if (!(attenuation > 0.0) || !(distanceSq > 0.0))
                continue;
            numerator += (brightness / attenuation) / distanceSq;
            denominator += 1.0 / (distanceSq * distanceSq);
        }

        // No usable samples means the attenuation is degenerate -- all zero, which is a light that never
        // falls off. The range is the only defensible answer left.
        if (!(denominator > 0.0) || !(numerator > 0.0))
            return range;

        return static_cast<float>(std::sqrt((numerator / denominator) / kLegacyLightEndValue));
    }

    /// Smallest change in a light's position that justifies recreating it, in OpenMW units.
    ///
    /// The API has no light-update call, so a change means recreating. OpenMW's lights jitter
    /// continuously -- carried lights follow animated bones, and many flicker every frame -- so an exact
    /// comparison would recreate almost every light every frame.
    constexpr float kLightMoveEpsilon = 1.0f;

    /// Smallest *relative* change in radiance that justifies recreating a light.
    ///
    /// Relative, not absolute: radiance scales with the inverse square of the emitter radius, so at a
    /// sub-centimetre radius the values run into the hundreds of thousands and any fixed epsilon is
    /// either meaninglessly tight or absurdly loose. One percent is below the visible threshold in a
    /// path-traced frame at any radius.
    constexpr float kLightRadianceRelativeEpsilon = 0.01f;

    /// Reads a float environment variable, keeping \a fallback when unset or unparseable.
    ///
    /// Both light constants are exposed this way because they are tuning values, and tuning through a
    /// rebuild is not tuning. Remix's own light options cannot be used for it -- they act on the legacy
    /// conversion path, which this integration does not go through.
    float envFloat(const char* name, float fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        // Zero is a value, not a synonym for unset.
        //
        // This used to end `return parsed > 0.0f ? parsed : fallback`, which meant OPENMW_REMIX_*=0 was read,
        // rejected, and silently replaced by the configured value. Every one of these knobs -- glow intensity,
        // material emissive, light intensity, the fog terms -- has zero as its meaningful "switch this
        // contribution off" setting, and several already default to 0.0, so the variable could not even
        // express its own default. A test that appears to run and changes nothing is worse than one that
        // fails: it reads as evidence that the thing under test does not matter.
        //
        // Rejecting only unparseable input keeps the original intent, which was to ignore junk rather than to
        // ignore zero. strtof reports that by leaving `end` at the start of the string.
        char* end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end == value || !std::isfinite(parsed) || parsed < 0.0f)
            return fallback;
        return parsed;
    }

    /// Reads a non-negative integer environment variable, keeping \a fallback when unset or unparseable.
    ///
    /// Zero is honoured rather than rejected: for a budget it is the meaningful "submit none of this"
    /// value, which is the same reason envByte exists alongside envFloat.
    ///
    /// \a maximum is explicit at every call site and has no default, deliberately. It used to be
    /// hardcoded to kMaxInstancesPerFrame, which was right for the one caller that existed -- a
    /// groundcover *instance* budget -- and silently wrong for the next one. A triangle budget in the
    /// tens of millions read back as 20,000, which turned away 85% of the scene's instances and looked
    /// exactly like a rendering bug. Worse, it only bit when the variable was *set*: unset took this
    /// function's fallback path and returned the correct value unclamped, so the default was fine and
    /// writing the default down explicitly broke it. A clamp whose bound belongs to one caller must not
    /// be reachable by another without saying so.
    unsigned int envUInt(const char* name, unsigned int fallback, unsigned int maximum)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        char* end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (end == value || parsed < 0)
            return fallback;
        return static_cast<unsigned int>(std::min<long long>(parsed, maximum));
    }

    /// Reads a 0..255 environment variable, keeping \a fallback only when unset or unparseable.
    ///
    /// Separate from envFloat because that one treats zero as unparseable, which is the right call for a
    /// radius or a brightness but wrong for a threshold: zero is the meaningful "switch this off" value.
    unsigned char envByte(const char* name, unsigned char fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value)
            return fallback;
        return static_cast<unsigned char>(std::clamp<long>(parsed, 0, 255));
    }

    /// Samples an image's alpha at \a s, \a t with bilinear filtering and returns it as 0..255.
    ///
    /// Goes through osg::Image::getColor rather than indexing the data directly, because OpenMW's blend
    /// maps are not one format: they arrive as GL_ALPHA, GL_LUMINANCE_ALPHA or GL_RGBA depending on how the
    /// chunk was built, and getColor already knows how to read each. This runs once per vertex when a
    /// terrain layer's mesh is built, not per frame, so four reads instead of one is not on a hot path.
    ///
    /// Filtered by hand because OSG has no filtered accessor. osg::Image::getColor(Vec2) scales the
    /// texcoord by the dimension and truncates, so it returns the nearest texel and nothing else -- the only
    /// underlying reader is getColor(unsigned, unsigned, unsigned), which takes integer texel coordinates.
    ///
    /// That mattered, and it is the whole reason terrain layers looked unblended. A cell's blend map is 17
    /// texels across against the terrain's 65 vertices, so the blend map is four times coarser than the grid
    /// sampling it: with nearest sampling every four-by-four block of vertices read one identical coverage
    /// value, and the ground came out as flat blocks with hard creases along the triangle edges between
    /// them. It reads exactly like coverage that was never applied, which is what it was mistaken for --
    /// twice, and the second time a whole GPU-readback compositor was built to work around it.
    ///
    /// The engine's own renderer never had this problem because it samples the blend map on the GPU with
    /// GL_LINEAR. This is that filter, done where the coverage is baked into vertex alpha instead.
    ///
    /// Uses OSG's own texel-centre convention -- a texcoord of 0 and 1 land on texel 0 and n-1 -- so the
    /// endpoints stay where getColor put them and this only adds interpolation between them.
    ///
    /// Clamped rather than wrapped: a blend map covers its chunk exactly once, so a texcoord landing
    /// outside it means the vertex is on the chunk's own edge, and wrapping there would fetch coverage from
    /// the opposite side of the chunk.
    unsigned int sampleImageAlpha(const osg::Image& image, float s, float t)
    {
        if (image.data() == nullptr || image.s() <= 0 || image.t() <= 0)
            return 0xFFu;

        const int width = image.s();
        const int height = image.t();
        const float cs = std::clamp(s, 0.0f, 1.0f) * static_cast<float>(std::max(width - 1, 0));
        const float ct = std::clamp(t, 0.0f, 1.0f) * static_cast<float>(std::max(height - 1, 0));

        const int x0 = std::clamp(static_cast<int>(std::floor(cs)), 0, width - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(ct)), 0, height - 1);
        const int x1 = std::min(x0 + 1, width - 1);
        const int y1 = std::min(y0 + 1, height - 1);
        const float fx = cs - static_cast<float>(x0);
        const float fy = ct - static_cast<float>(y0);

        const auto alphaAt = [&image](int x, int y) {
            return image.getColor(static_cast<unsigned int>(x), static_cast<unsigned int>(y)).a();
        };

        const float top = alphaAt(x0, y0) + (alphaAt(x1, y0) - alphaAt(x0, y0)) * fx;
        const float bottom = alphaAt(x0, y1) + (alphaAt(x1, y1) - alphaAt(x0, y1)) * fx;
        const float filtered = top + (bottom - top) * fy;

        return static_cast<unsigned int>(std::clamp(filtered, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    /// Whether an image carries any non-zero alpha at all.
    ///
    /// Worth asking because a terrain chunk lists every land texture used anywhere across its area, and a
    /// large chunk spans many cells -- so most of its layers cover none of it. Measured at 104 of 227
    /// layers entirely zero, and a layer that covers nothing still costs an instance, a mesh and its share
    /// of acceleration-structure work.
    ///
    /// Scans the bytes directly for GL_ALPHA, which is what OpenMW's blend maps are, and bails at the first
    /// non-zero one. The cost is therefore near nothing for a layer that does contribute, and a full 16 KB
    /// scan only for one that does not -- which is the case being eliminated, so it pays for itself. Other
    /// formats fall back to a sparse grid through getColor rather than assuming a layout.
    bool imageHasAnyAlpha(const osg::Image& image)
    {
        if (image.data() == nullptr || image.s() <= 0 || image.t() <= 0)
            return true; // Unknown rather than empty; submitting is the safe answer.

        // Spelled out rather than using the kGl* constants below, which are declared after this function.
        // 0x1906 is GL_ALPHA, 0x1401 is GL_UNSIGNED_BYTE -- what the log reports for every blend map.
        if (image.getPixelFormat() == 0x1906u && image.getDataType() == 0x1401u)
        {
            for (int row = 0; row < image.t(); ++row)
            {
                const unsigned char* line = image.data(0, row);
                if (line == nullptr)
                    return true;
                for (int column = 0; column < image.s(); ++column)
                {
                    if (line[column] != 0)
                        return true;
                }
            }
            return false;
        }

        for (int gy = 0; gy <= 16; ++gy)
        {
            for (int gx = 0; gx <= 16; ++gx)
            {
                if (sampleImageAlpha(image, gx / 16.0f, gy / 16.0f) != 0)
                    return true;
            }
        }
        return false;
    }

    /// GL pixel and internal formats, spelled out rather than taken from OSG's headers.
    ///
    /// osg/GL and osg/Texture define some of these and not others depending on the GL headers in use,
    /// and this file must not depend on which. Same approach as the interop code, for the same reason.
    constexpr unsigned int kGlRed = 0x1903;
    constexpr unsigned int kGlAlpha = 0x1906;
    constexpr unsigned int kGlRgb = 0x1907;
    constexpr unsigned int kGlRgba = 0x1908;
    constexpr unsigned int kGlLuminance = 0x1909;
    constexpr unsigned int kGlLuminanceAlpha = 0x190A;
    constexpr unsigned int kGlRg = 0x8227;
    constexpr unsigned int kGlBgr = 0x80E0;
    constexpr unsigned int kGlBgra = 0x80E1;
    constexpr unsigned int kGlUnsignedByte = 0x1401;

    /// One entry per block-compressed layout the runtime can take.
    ///
    /// A table rather than a switch because three separate things have to stay in step for each format --
    /// the Remix enumerant, the bytes per 4x4 block, and the name used in diagnostics -- and a switch
    /// spreads them across three places that can drift apart. Adding a format is one line here.
    ///
    /// Both the plain and the sRGB GL spellings appear for each layout, and both map to the same entry.
    /// The GL enum says how the *source* was tagged; it does not say what the data means. A DDS file is
    /// almost always tagged with the plain variant whether or not its contents are colour, so trusting the
    /// tag would leave most albedo textures uncorrected. The choice of the sRGB Remix format is made per
    /// layout on what the layout is used for, which for everything except BC5 is colour.
    struct CompressedFormat
    {
        unsigned int mGlInternalFormat;
        RemixRT::Runtime::TextureFormat mColour; ///< sRGB-decoding variant, for albedo and emission.
        RemixRT::Runtime::TextureFormat mLinear; ///< For data: normals, roughness, masks.
        unsigned int mBlockBytes;
        const char* mName;
    };

    /// Shorthand for the table below only. Runtime is a class, not a namespace, so this has to be a type
    /// alias rather than a namespace alias.
    using Fmt = RemixRT::Runtime;

    constexpr CompressedFormat kCompressedFormats[] = {
        // S3TC / DXT, GL_EXT_texture_compression_s3tc plus its sRGB counterpart.
        { 0x83F0, Fmt::Format_BC1_RGB, Fmt::Format_BC1_RGB_Linear, 8, "BC1_RGB" }, // GL_COMPRESSED_RGB_S3TC_DXT1
        { 0x8C4C, Fmt::Format_BC1_RGB, Fmt::Format_BC1_RGB_Linear, 8, "BC1_RGB(srgb)" },
        { 0x83F1, Fmt::Format_BC1_RGBA, Fmt::Format_BC1_RGBA_Linear, 8, "BC1_RGBA" }, // one-bit alpha
        { 0x8C4D, Fmt::Format_BC1_RGBA, Fmt::Format_BC1_RGBA_Linear, 8, "BC1_RGBA(srgb)" },
        { 0x83F2, Fmt::Format_BC2, Fmt::Format_BC2_Linear, 16, "BC2" }, // DXT3, explicit four-bit alpha
        { 0x8C4E, Fmt::Format_BC2, Fmt::Format_BC2_Linear, 16, "BC2(srgb)" },
        { 0x83F3, Fmt::Format_BC3, Fmt::Format_BC3_Linear, 16, "BC3" }, // DXT5, interpolated alpha
        { 0x8C4F, Fmt::Format_BC3, Fmt::Format_BC3_Linear, 16, "BC3(srgb)" },
        // RGTC, GL_ARB_texture_compression_rgtc. Two channels and no colour meaning at all, so the same
        // linear format serves both columns. The signed spelling is listed because the loader can produce
        // it, not because a signed normal map would be interpreted correctly here.
        { 0x8DBD, Fmt::Format_BC5, Fmt::Format_BC5, 16, "BC5" }, // GL_COMPRESSED_RG_RGTC2
        { 0x8DBE, Fmt::Format_BC5, Fmt::Format_BC5, 16, "BC5(signed)" },
        // BPTC, GL_ARB_texture_compression_bptc. What modern high-resolution replacement packs use.
        { 0x8E8C, Fmt::Format_BC7, Fmt::Format_BC7_Linear, 16, "BC7" }, // GL_COMPRESSED_RGBA_BPTC_UNORM
        { 0x8E8D, Fmt::Format_BC7, Fmt::Format_BC7_Linear, 16, "BC7(srgb)" },
    };

    /// Looks up \a glInternalFormat, or null if the runtime has no format for it.
    ///
    /// Deliberately not a fallback to something plausible. Uploading data as the wrong block layout does
    /// not produce a slightly wrong image, it produces garbage of the right size, which is far harder to
    /// recognise than an untextured surface.
    const CompressedFormat* compressedFormatFor(unsigned int glInternalFormat)
    {
        for (const CompressedFormat& candidate : kCompressedFormats)
            if (candidate.mGlInternalFormat == glInternalFormat)
                return &candidate;
        return nullptr;
    }

    /// Bytes one BC1 mip level of the given extent occupies.
    ///
    /// Rounded up to whole 4x4 blocks, which is why the levels below 4x4 all cost the same eight bytes.
    /// Deliberately the same expression the compressed upload path uses to validate a source mip chain:
    /// CreateTexture memcpys exactly the byte count it is handed, so producer and consumer disagreeing
    /// about a level's size is a buffer overrun inside the runtime rather than a wrong-looking texture.
    constexpr unsigned long long bc1LevelBytes(int width, int height)
    {
        return ((static_cast<unsigned long long>(width) + 3ull) / 4ull)
            * ((static_cast<unsigned long long>(height) + 3ull) / 4ull) * 8ull;
    }

    /// Packs 8-bit RGB into RGB565, the endpoint format a BC1 block stores.
    constexpr unsigned short packRgb565(int red, int green, int blue)
    {
        return static_cast<unsigned short>(((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3));
    }

    /// Expands RGB565 back to 8 bits, replicating the high bits downwards exactly as a BC1 decoder does.
    ///
    /// Reproducing the decoder's reconstruction rather than keeping the original 8-bit endpoints is what
    /// makes index selection below agree with the colours the GPU will actually produce. Selecting against
    /// unquantised endpoints picks a visibly worse index for texels near a palette boundary.
    void unpackRgb565(unsigned short packed, int& red, int& green, int& blue)
    {
        const int red5 = (packed >> 11) & 0x1F;
        const int green6 = (packed >> 5) & 0x3F;
        const int blue5 = packed & 0x1F;
        red = (red5 << 3) | (red5 >> 2);
        green = (green6 << 2) | (green6 >> 4);
        blue = (blue5 << 3) | (blue5 >> 2);
    }

    /// sRGB byte to linear and back, tabulated, for gamma-correct mip generation.
    ///
    /// Averaging sRGB-encoded bytes directly biases every level dark, and the bias compounds down the
    /// chain. Only minified ground samples the small levels, so in-game that reads as the distance at
    /// which the terrain suddenly gets darker -- a moving band across the ground as the camera travels,
    /// which is worse than the shimmer the mip chain is being added to fix. Both directions are tables
    /// because this runs over about a third again as many pixels as the composite has, per composite.
    struct SrgbTransfer
    {
        std::uint16_t mToLinear[256]; ///< sRGB byte to linear, full 16-bit range.
        unsigned char mToSrgb[4096]; ///< Linear, quantised to 12 bits, back to an sRGB byte.

        SrgbTransfer()
        {
            for (int i = 0; i < 256; ++i)
            {
                const double encoded = i / 255.0;
                const double linear = encoded <= 0.04045 ? encoded / 12.92
                                                         : std::pow((encoded + 0.055) / 1.055, 2.4);
                mToLinear[i] = static_cast<std::uint16_t>(std::lround(linear * 65535.0));
            }
            for (int i = 0; i < 4096; ++i)
            {
                // Bucket centre rather than edge, so the round trip of an exactly representable value
                // lands back on itself instead of one step low.
                const double linear = (i + 0.5) / 4096.0;
                const double encoded = linear <= 0.0031308 ? linear * 12.92
                                                           : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
                mToSrgb[i] = static_cast<unsigned char>(
                    std::clamp(std::lround(encoded * 255.0), 0L, 255L));
            }
        }
    };

    const SrgbTransfer& srgbTransfer()
    {
        static const SrgbTransfer transfer;
        return transfer;
    }

    /// Splits [0, count) across hardware threads, running \a body(begin, end) on each span.
    ///
    /// The compressor below is the largest synchronous cost in texture submission, and it runs on the
    /// frame thread. Serially it is tens of milliseconds for one 1024-square composite -- more than a
    /// whole frame at 60 Hz -- so every chunk the terrain composites lands as a hitch. That is
    /// independent of replacements, which is why it shows up walking through empty countryside.
    ///
    /// The work parallelises exactly: BC1 block encoding reads one 4x4 texel neighbourhood and writes
    /// eight bytes at a fixed offset, and the box filter reads two source rows and writes one. No span
    /// touches another's output, so no synchronisation is needed beyond the join.
    ///
    /// Threads are created per call rather than pooled. A composite is infrequent and the call is already
    /// tens of milliseconds, so creation cost is noise against it, and a pool here would have to coexist
    /// with OpenMW's own work queues for no benefit.
    ///
    /// \a minimumPerSpan keeps small mip levels serial: below it the thread handshake costs more than the
    /// work, and the chain descends to 1x1.
    template <typename Body>
    void parallelSpans(int count, int minimumPerSpan, Body body)
    {
        const int available = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        const int worthwhile = count / std::max(1, minimumPerSpan);
        const int spans = std::clamp(worthwhile, 1, available);
        if (spans <= 1)
        {
            body(0, count);
            return;
        }

        const int perSpan = (count + spans - 1) / spans;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(spans) - 1);
        for (int span = 1; span < spans; ++span)
        {
            const int begin = span * perSpan;
            const int end = std::min(count, begin + perSpan);
            if (begin >= end)
                break;
            workers.emplace_back([&body, begin, end] { body(begin, end); });
        }
        // The calling thread takes the first span rather than waiting, so one span costs no handshake.
        body(0, std::min(count, perSpan));
        for (std::thread& worker : workers)
            worker.join();
    }

    /// Encodes one 4x4 RGBA8 block, given as 16 tightly packed texels, into the eight bytes of a BC1 block.
    ///
    /// Range fit: the endpoints are the corners of the block's colour bounding box, inset slightly, and
    /// each texel takes the nearest of the four palette entries, found by projecting onto the endpoint
    /// axis rather than by comparing against all four. That is the standard fast encoder rather than an
    /// exhaustive search, which is the right trade here -- this runs once per chunk on a frame thread, and
    /// what it encodes is ground albedo whose whole purpose is to be seen at a distance.
    ///
    /// Alpha is discarded. The only caller is the terrain composite, whose render target has no alpha
    /// channel at all and whose alpha is forced opaque on readback.
    void encodeBc1Block(const unsigned char* block, unsigned char* out)
    {
        int low[3] = { 255, 255, 255 };
        int high[3] = { 0, 0, 0 };
        for (int texel = 0; texel < 16; ++texel)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                const int value = block[texel * 4 + channel];
                low[channel] = std::min(low[channel], value);
                high[channel] = std::max(high[channel], value);
            }
        }

        // Inset the bounding box by a sixteenth of its extent on each side. Its corners are outliers by
        // construction, so endpoints placed exactly on them spend both of the two exactly-representable
        // palette entries on the two most extreme texels and leave the bulk of the block to the
        // interpolated ones. Pulling in lowers the total error; this is the inset stb_dxt applies.
        for (int channel = 0; channel < 3; ++channel)
        {
            const int margin = (high[channel] - low[channel]) >> 4;
            low[channel] = std::min(low[channel] + margin, 255);
            high[channel] = std::max(high[channel] - margin, 0);
        }

        unsigned short packedHigh = packRgb565(high[0], high[1], high[2]);
        unsigned short packedLow = packRgb565(low[0], low[1], low[2]);
        // BC1 selects its four-colour opaque mode on the first endpoint comparing greater than the second.
        // The other ordering means three colours plus a punch-through slot, which is not what an opaque
        // ground albedo wants -- and the two orderings are not interchangeable, so this is a correctness
        // step rather than a preference.
        if (packedHigh < packedLow)
            std::swap(packedHigh, packedLow);

        int endpointHigh[3];
        int endpointLow[3];
        unpackRgb565(packedHigh, endpointHigh[0], endpointHigh[1], endpointHigh[2]);
        unpackRgb565(packedLow, endpointLow[0], endpointLow[1], endpointLow[2]);

        int axis[3];
        int axisLengthSquared = 0;
        for (int channel = 0; channel < 3; ++channel)
        {
            axis[channel] = endpointHigh[channel] - endpointLow[channel];
            axisLengthSquared += axis[channel] * axis[channel];
        }

        std::uint32_t indices = 0;
        // A zero-length axis means the whole block quantised to one colour. Index 0 is that colour under
        // either mode, so leaving every index at zero is correct and the palette below is irrelevant.
        if (axisLengthSquared > 0)
        {
            // BC1 numbers its palette high, low, two-thirds-high, one-third-high, so walking the axis
            // upwards from the low endpoint does not walk the indices in order.
            static constexpr std::uint32_t kIndexForStep[4] = { 1, 3, 2, 0 };
            for (int texel = 0; texel < 16; ++texel)
            {
                int projection = 0;
                for (int channel = 0; channel < 3; ++channel)
                    projection += (block[texel * 4 + channel] - endpointLow[channel]) * axis[channel];

                // Nearest of the four evenly spaced points on the axis. The division is folded into the
                // rounding so this stays in integers: step = round(3 * projection / axisLengthSquared).
                // A texel outside the inset box projects negative, which truncates towards zero and is
                // then clamped -- the clamp is load-bearing, not defensive.
                const int step
                    = std::clamp((projection * 6 + axisLengthSquared) / (axisLengthSquared * 2), 0, 3);
                indices |= kIndexForStep[step] << (texel * 2);
            }
        }

        out[0] = static_cast<unsigned char>(packedHigh & 0xFF);
        out[1] = static_cast<unsigned char>(packedHigh >> 8);
        out[2] = static_cast<unsigned char>(packedLow & 0xFF);
        out[3] = static_cast<unsigned char>(packedLow >> 8);
        out[4] = static_cast<unsigned char>(indices & 0xFF);
        out[5] = static_cast<unsigned char>((indices >> 8) & 0xFF);
        out[6] = static_cast<unsigned char>((indices >> 16) & 0xFF);
        out[7] = static_cast<unsigned char>((indices >> 24) & 0xFF);
    }

    /// Compresses \a pixels to BC1 and appends a full mip chain, largest first, to \a out.
    ///
    /// Returns the number of levels written, which is the count the upload has to declare. Levels run all
    /// the way to 1x1: that is what a DDS mip chain does, and the byte arithmetic above already accounts
    /// for the sub-block levels costing a whole block each.
    ///
    /// \a pixels is tightly packed RGBA8 of \a width by \a height.
    ///
    /// Span thresholds: a block row is four pixel rows, so 16 of them is 64 pixel rows of encoding, and a
    /// level under 256 rows tall stays serial. Both are set so the smaller mip levels -- which are cheap
    /// and numerous, the chain running to 1x1 -- do not pay a thread handshake to save microseconds.
    constexpr int kMinBlockRowsPerSpan = 16;
    constexpr int kMinFilterRowsPerSpan = 64;

    unsigned int compressBc1WithMips(
        const unsigned char* pixels, int width, int height, std::vector<unsigned char>& out)
    {
        const SrgbTransfer& transfer = srgbTransfer();

        // One parallel region for the whole chain, not one per level.
        //
        // The first version parallelised each mip level's encode and each downsample separately, which for
        // a 1024-square composite spawned roughly sixty threads across the eleven levels. Thread creation
        // on Windows costs tens of microseconds, so the handshake came to milliseconds and dominated the
        // work it was hiding: measured cost per composite stayed at about 7 ms, and a burst of 82 in one
        // frame produced the 678 ms scene-submit spike that the frame breakdown attributed the freeze to.
        //
        // So the chain is built first -- each downsample depends on the previous level, so that part is
        // inherently sequential and is memory-bound rather than compute-bound -- and then every block of
        // every level is encoded in a single parallel pass over a flat list of block rows. That is one
        // thread-spawn set per composite instead of sixty.
        struct MipLevel
        {
            std::vector<unsigned char> pixels;
            int width = 0;
            int height = 0;
            int blocksAcross = 0;
            int blocksDown = 0;
            std::size_t outOffset = 0;
        };

        std::vector<MipLevel> chain;
        {
            MipLevel base;
            base.pixels.assign(
                pixels, pixels + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
            base.width = width;
            base.height = height;
            chain.push_back(std::move(base));
        }

        while (chain.back().width > 1 || chain.back().height > 1)
        {
            const MipLevel& source = chain.back();
            const int levelWidth = source.width;
            const int levelHeight = source.height;

            // 2x2 box filter in linear light. An odd extent repeats its surviving column or row rather
            // than dropping it; composites are square powers of two, so that path is defensive only.
            MipLevel next;
            next.width = std::max(1, levelWidth / 2);
            next.height = std::max(1, levelHeight / 2);
            next.pixels.resize(
                static_cast<std::size_t>(next.width) * static_cast<std::size_t>(next.height) * 4);

            for (int y = 0; y < next.height; ++y)
            {
                const int top = std::min(y * 2, levelHeight - 1);
                const int bottom = std::min(y * 2 + 1, levelHeight - 1);
                for (int x = 0; x < next.width; ++x)
                {
                    const int left = std::min(x * 2, levelWidth - 1);
                    const int right = std::min(x * 2 + 1, levelWidth - 1);
                    const unsigned char* corner[4] = {
                        source.pixels.data() + (static_cast<std::size_t>(top) * levelWidth + left) * 4,
                        source.pixels.data() + (static_cast<std::size_t>(top) * levelWidth + right) * 4,
                        source.pixels.data() + (static_cast<std::size_t>(bottom) * levelWidth + left) * 4,
                        source.pixels.data() + (static_cast<std::size_t>(bottom) * levelWidth + right) * 4,
                    };
                    unsigned char* destination
                        = next.pixels.data() + (static_cast<std::size_t>(y) * next.width + x) * 4;
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        const unsigned int sum = transfer.mToLinear[corner[0][channel]]
                            + transfer.mToLinear[corner[1][channel]] + transfer.mToLinear[corner[2][channel]]
                            + transfer.mToLinear[corner[3][channel]];
                        destination[channel] = transfer.mToSrgb[(sum / 4u) >> 4];
                    }
                    // Alpha carries no transfer function, so it averages directly.
                    destination[3] = static_cast<unsigned char>(
                        (corner[0][3] + corner[1][3] + corner[2][3] + corner[3][3] + 2) / 4);
                }
            }

            chain.push_back(std::move(next));
        }

        // Lay the levels out largest first, which is the order a DDS mip chain and the upload both expect.
        std::size_t total = 0;
        for (MipLevel& level : chain)
        {
            level.blocksAcross = (level.width + 3) / 4;
            level.blocksDown = (level.height + 3) / 4;
            level.outOffset = total;
            total += static_cast<std::size_t>(bc1LevelBytes(level.width, level.height));
        }
        out.resize(total);
        unsigned char* const outBase = out.data();

        // Flat list of (level, block row) so one parallel range covers every block in the chain. A
        // 1024-square composite comes to about 511 rows, so the index costs nothing to build.
        std::vector<std::pair<unsigned int, int>> rows;
        rows.reserve(512);
        for (unsigned int levelIndex = 0; levelIndex < chain.size(); ++levelIndex)
        {
            for (int row = 0; row < chain[levelIndex].blocksDown; ++row)
                rows.emplace_back(levelIndex, row);
        }

        parallelSpans(static_cast<int>(rows.size()), kMinBlockRowsPerSpan, [&](int beginRow, int endRow) {
            for (int index = beginRow; index < endRow; ++index)
            {
                const MipLevel& level = chain[rows[index].first];
                const int row = rows[index].second;
                const int levelWidth = level.width;
                const int levelHeight = level.height;
                unsigned char* cursor = outBase + level.outOffset
                    + static_cast<std::size_t>(row) * level.blocksAcross * 8;
                const int blockY = row * 4;

                for (int blockX = 0; blockX < levelWidth; blockX += 4)
                {
                    unsigned char block[64];
                    for (int y = 0; y < 4; ++y)
                    {
                        // Levels smaller than 4x4 still occupy a whole block, so the last row and column
                        // repeat to fill it. Clamping rather than zero-filling keeps the block's colour
                        // range honest; padding with black would drag an endpoint to black.
                        const int sourceY = std::min(blockY + y, levelHeight - 1);
                        for (int x = 0; x < 4; ++x)
                        {
                            const int sourceX = std::min(blockX + x, levelWidth - 1);
                            const unsigned char* source = level.pixels.data()
                                + (static_cast<std::size_t>(sourceY) * levelWidth + sourceX) * 4;
                            std::memcpy(block + (y * 4 + x) * 4, source, 4);
                        }
                    }
                    encodeBc1Block(block, cursor);
                    cursor += 8;
                }
            }
        });

        return static_cast<unsigned int>(chain.size());
    }

    /// Appends mip levels 1..N to \a out, which must already hold level 0 as tightly packed RGBA8, and
    /// returns the total level count.
    ///
    /// Every texture that is not block-compressed on disk arrived here with one mip level and left with one,
    /// because the uncompressed path converts level 0 and nothing else. That has two costs, and the second is
    /// the one that hurt.
    ///
    /// The first is ordinary: minification has nothing to fall back on, so those surfaces shimmer at
    /// distance. The second is that a single-level texture cannot be partially resident, so the runtime's
    /// texture manager has no way to demote it under pressure -- its only move is to evict the whole thing.
    /// The same reasoning is already written down for terrain composites a few hundred lines below; it
    /// applies to every other texture just as much, and those are the majority by count.
    ///
    /// It also travels: a capture exports a DDS with exactly the level count the live image has, so a
    /// one-level upload becomes a one-level DDS, which is what the toolkit complains about and what makes it
    /// hold every captured texture at full resolution at once.
    ///
    /// Unlike compressBc1WithMips this re-encodes nothing -- level 0 is left exactly as it was handed over.
    /// That is deliberate and load-bearing: level 0's bytes are the texture's public identity, so a tag in
    /// rtx.conf and a replacement authored against it keep working. Only levels the runtime had no way to
    /// sample before are added.
    unsigned int appendRgba8Mips(std::vector<unsigned char>& out, int width, int height, bool colour)
    {
        const SrgbTransfer& transfer = srgbTransfer();

        std::size_t sourceOffset = 0;
        int sourceWidth = width;
        int sourceHeight = height;
        unsigned int levels = 1;

        while (sourceWidth > 1 || sourceHeight > 1)
        {
            const int levelWidth = std::max(1, sourceWidth / 2);
            const int levelHeight = std::max(1, sourceHeight / 2);
            const std::size_t destinationOffset = out.size();
            out.resize(destinationOffset
                + static_cast<std::size_t>(levelWidth) * static_cast<std::size_t>(levelHeight) * 4);

            // Taken after the resize, because that is what invalidates them.
            const unsigned char* const source = out.data() + sourceOffset;
            unsigned char* const destination = out.data() + destinationOffset;

            for (int y = 0; y < levelHeight; ++y)
            {
                // An odd extent repeats its surviving row or column rather than dropping it.
                const int top = std::min(y * 2, sourceHeight - 1);
                const int bottom = std::min(y * 2 + 1, sourceHeight - 1);
                for (int x = 0; x < levelWidth; ++x)
                {
                    const int left = std::min(x * 2, sourceWidth - 1);
                    const int right = std::min(x * 2 + 1, sourceWidth - 1);
                    const unsigned char* const corner[4] = {
                        source + (static_cast<std::size_t>(top) * sourceWidth + left) * 4,
                        source + (static_cast<std::size_t>(top) * sourceWidth + right) * 4,
                        source + (static_cast<std::size_t>(bottom) * sourceWidth + left) * 4,
                        source + (static_cast<std::size_t>(bottom) * sourceWidth + right) * 4,
                    };
                    unsigned char* const pixel
                        = destination + (static_cast<std::size_t>(y) * levelWidth + x) * 4;

                    for (int channel = 0; channel < 3; ++channel)
                    {
                        if (colour)
                        {
                            // Averaged in linear light. Averaging sRGB values directly darkens every mip,
                            // which reads as distant surfaces getting muddier the further away they are --
                            // the same correction compressBc1WithMips makes for composites.
                            const unsigned int sum = transfer.mToLinear[corner[0][channel]]
                                + transfer.mToLinear[corner[1][channel]]
                                + transfer.mToLinear[corner[2][channel]]
                                + transfer.mToLinear[corner[3][channel]];
                            pixel[channel] = transfer.mToSrgb[(sum / 4u) >> 4];
                        }
                        else
                        {
                            // Linear data -- normal maps, masks -- carries no transfer function, so it
                            // averages directly. Normals are not renormalised: a box-filtered normal map is
                            // the conventional result and the shader normalises what it samples anyway.
                            pixel[channel] = static_cast<unsigned char>((corner[0][channel]
                                                                            + corner[1][channel]
                                                                            + corner[2][channel]
                                                                            + corner[3][channel] + 2)
                                / 4);
                        }
                    }

                    // Alpha has no transfer function either way.
                    pixel[3] = static_cast<unsigned char>(
                        (corner[0][3] + corner[1][3] + corner[2][3] + corner[3][3] + 2) / 4);
                }
            }

            sourceOffset = destinationOffset;
            sourceWidth = levelWidth;
            sourceHeight = levelHeight;
            ++levels;
        }

        return levels;
    }

    /// Alpha threshold used when OpenMW asks for a cutout but its own reference value is unusable.
    ///
    /// OpenMW expresses alpha testing through osg::AlphaFunc with a float reference; a NIF that enables
    /// alpha testing without a sensible value is common enough that a default is needed. Half way is the
    /// conventional cutout point.
    constexpr unsigned char kDefaultAlphaTestReference = 128;

    /// Roughness for a surface no rule matched.
    ///
    /// Fairly rough, because most of Morrowind is stone, wood and cloth, and because a wrong guess in the
    /// rough direction reads as an unremarkable surface while a wrong guess in the smooth direction reads
    /// as wet plastic.
    constexpr float kTexturedRoughness = 0.65f;

    /// Surface response by texture name, and why it has to be done this way.
    ///
    /// The principled source would be the NIF's own NiMaterialProperty, which carries a glossiness and a
    /// specular colour. It cannot be used: NifOsg::Loader zeroes both for every Morrowind-era file
    /// (nifloader.cpp, "Morrowind has its support disabled"), so nothing survives to read. And the reason
    /// it zeroes them is the deeper problem -- Morrowind's own renderer ignored specular, so the values in
    /// the files were never authored to mean anything and harvesting them before they are cleared would
    /// mostly collect defaults.
    ///
    /// What is left is the texture name, and Morrowind's naming is systematic enough to carry real signal:
    /// tx_<material>_<variant>. This is a classification, not a measurement, and it is the same judgement
    /// a Remix mod author makes by hand for each texture -- the difference is that it is written down here
    /// where it can be reviewed and corrected.
    ///
    /// Matched as a substring against the diffuse texture's path, first rule wins, so more specific
    /// patterns must come first. Metallic is set only where Morrowind's own art is unambiguously metal;
    /// a false positive there is far more visible than a roughness that is slightly off, because a
    /// metallic surface takes its colour entirely from what it reflects.
    struct SurfaceRule
    {
        const char* mPattern;
        float mRoughness;
        float mMetallic;
    };

    constexpr SurfaceRule kSurfaceRules[] = {
        // Metals. Before the generic armour and weapon patterns, which would otherwise claim them.
        { "metal", 0.35f, 1.0f },
        { "_silver", 0.22f, 1.0f },
        { "_gold", 0.25f, 1.0f },
        { "_ebony", 0.20f, 1.0f },
        { "_steel", 0.30f, 1.0f },
        { "_iron", 0.45f, 1.0f },
        { "_dwrv", 0.38f, 1.0f }, // Dwemer
        { "_dwemer", 0.38f, 1.0f },
        { "_mithril", 0.28f, 1.0f },
        { "_daedric", 0.32f, 1.0f },
        // Smooth dielectrics.
        { "glass", 0.10f, 0.0f },
        { "_glaze", 0.18f, 0.0f },
        { "ice", 0.12f, 0.0f },
        { "_polish", 0.20f, 0.0f },
        { "_marble", 0.25f, 0.0f },
        { "_gem", 0.12f, 0.0f },
        { "_pearl", 0.20f, 0.0f },
        // Organic and worked surfaces.
        { "_wood", 0.62f, 0.0f },
        { "wood", 0.62f, 0.0f },
        { "_bark", 0.85f, 0.0f },
        { "_leaf", 0.60f, 0.0f },
        { "_leaves", 0.60f, 0.0f },
        { "_bone", 0.55f, 0.0f },
        { "_chitin", 0.40f, 0.0f },
        { "_leather", 0.60f, 0.0f },
        { "_hide", 0.72f, 0.0f },
        { "_fur", 0.90f, 0.0f },
        { "_cloth", 0.85f, 0.0f },
        { "_fabric", 0.85f, 0.0f },
        { "_rug", 0.90f, 0.0f },
        { "_silk", 0.55f, 0.0f },
        { "_paper", 0.80f, 0.0f },
        { "_parchment", 0.80f, 0.0f },
        { "_skin", 0.55f, 0.0f },
        { "_flesh", 0.60f, 0.0f },
        { "_hair", 0.70f, 0.0f },
        // Rough mineral surfaces. Last, because "stone" and "rock" appear inside many other names.
        { "_plaster", 0.80f, 0.0f },
        { "_stucco", 0.82f, 0.0f },
        { "_brick", 0.78f, 0.0f },
        { "_stone", 0.75f, 0.0f },
        { "stone", 0.75f, 0.0f },
        { "_rock", 0.80f, 0.0f },
        { "rock", 0.80f, 0.0f },
        { "_sand", 0.88f, 0.0f },
        { "_dirt", 0.90f, 0.0f },
        { "_mud", 0.75f, 0.0f },
        { "_ash", 0.92f, 0.0f },
        { "_grass", 0.85f, 0.0f },
        { "_moss", 0.88f, 0.0f },
        { "_snow", 0.70f, 0.0f },
    };

    /// Texture path fragments whose draws are not surfaces at all.
    ///
    /// Separate from kSurfaceRules because that table answers "what is this material like"; this answers
    /// "is this a material at all". OpenMW's shader framework submits post-process style passes as ordinary
    /// geometry, and a path tracer has nothing to do with them: SunsDusk's liquid effect binds an animated
    /// sequence under omw_distortion/ that is meant to warp what is behind it, and handing that to the
    /// runtime as an albedo draws it as a solid surface in front of the drink.
    ///
    /// Matched on path rather than texture hash because the sequence is twenty-four plus frames with a hash
    /// each, so a hash list would need all of them and would break when the mod's frame count changed.
    constexpr const char* kNonSurfaceTexturePatterns[] = {
        "omw_distortion",
    };

    /// Extra patterns from OPENMW_REMIX_SKIP_TEXTURES, comma or semicolon separated.
    ///
    /// Exists so the next mod that does this can be identified and neutralised from a config line instead
    /// of a rebuild -- the failure is recognisable in the log (an opaque draw with a suspicious texture
    /// path and no rule matched) long before anyone can patch the source.
    const std::vector<std::string>& extraNonSurfacePatterns()
    {
        static const std::vector<std::string> patterns = [] {
            std::vector<std::string> out;
            // Setting first, environment override second, same precedence as every other knob here.
            const std::string fromSettings = Settings::remix().mSkipTextures.get();
            const char* fromEnv = std::getenv("OPENMW_REMIX_SKIP_TEXTURES");
            const std::string source = fromEnv != nullptr ? std::string(fromEnv) : fromSettings;
            if (source.empty())
                return out;
            const char* raw = source.c_str();

            std::string current;
            for (const char* c = raw;; ++c)
            {
                if (*c == '\0' || *c == ',' || *c == ';')
                {
                    if (!current.empty())
                    {
                        std::transform(current.begin(), current.end(), current.begin(),
                            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                        out.push_back(current);
                        current.clear();
                    }
                    if (*c == '\0')
                        break;
                }
                else
                    current.push_back(*c);
            }
            return out;
        }();

        return patterns;
    }

    /// True when \a path names a texture whose draw should be dropped rather than turned into a surface.
    bool isNonSurfaceTexture(std::string_view path)
    {
        if (path.empty())
            return false;

        std::string lowered(path);
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        for (const char* pattern : kNonSurfaceTexturePatterns)
            if (lowered.find(pattern) != std::string::npos)
                return true;

        for (const std::string& pattern : extraNonSurfacePatterns())
            if (lowered.find(pattern) != std::string::npos)
                return true;

        return false;
    }

    /// Classifies \a path, or null when no rule matched.
    const SurfaceRule* surfaceRuleFor(std::string_view path)
    {
        // Lower cased once here rather than per rule. Morrowind's own data is inconsistently cased and
        // mods more so, and a case-sensitive match would silently classify half a texture set.
        std::string lowered(path);
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        for (const SurfaceRule& rule : kSurfaceRules)
            if (lowered.find(rule.mPattern) != std::string::npos)
                return &rule;
        return nullptr;
    }

    /// Reads the alpha-test threshold OpenMW asks for, or 0 for none.
    ///
    /// Deliberately not gated on GL_ALPHA_TEST being enabled. OpenMW is a shader-based renderer:
    /// Shader::ShaderVisitor moves the test into the fragment shader and substitutes a
    /// Shader::RemovedAlphaFunc for the attribute, specifically so the fixed-function mode is *never*
    /// switched on. Requiring the mode therefore rejected every cutout in the game, and foliage,
    /// lattices and railings all came through as solid polygons.
    unsigned char alphaTestReferenceFor(const osg::StateSet& stateSet)
    {
        const auto* alphaFunc
            = dynamic_cast<const osg::AlphaFunc*>(stateSet.getAttribute(osg::StateAttribute::ALPHAFUNC));
        if (alphaFunc == nullptr)
            return 0;

        // Only the "keep what is more opaque than this" comparisons map onto a path tracer's cutout.
        // This also excludes ALWAYS, which is how OpenMW spells "no test at all", so a surface that
        // wants to be solid stays solid.
        const auto function = alphaFunc->getFunction();
        if (function != osg::AlphaFunc::GREATER && function != osg::AlphaFunc::GEQUAL)
            return 0;

        // The substituted attribute carries the comparison but a placeholder reference of 1.0; the real
        // threshold travels alongside it as an "alphaRef" uniform, because that is what the shader reads.
        // Trusting the attribute's own value would give 1.0, and "keep only alpha greater than 1.0" keeps
        // nothing -- cutout geometry would disappear altogether rather than merely stay solid.
        float reference = alphaFunc->getReferenceValue();
        if (const osg::Uniform* uniform = stateSet.getUniform("alphaRef"))
        {
            float value = 0.0f;
            if (uniform->get(value))
                reference = value;
        }

        if (!(reference > 0.0f))
            return kDefaultAlphaTestReference;

        const int scaled = static_cast<int>(reference * 255.0f + 0.5f);
        return static_cast<unsigned char>(std::clamp(scaled, 1, 255));
    }

    /// Collects triangle indices from any primitive set OSG can hold.
    ///
    /// Used through osg::TriangleIndexFunctor, which decomposes strips, fans and quads into triangles
    /// for us. Doing that by hand over the primitive sets is the obvious approach and gets the less
    /// common modes wrong; OpenMW's meshes are not all plain triangle lists.
    struct TriangleCollector
    {
        std::vector<unsigned int>* mIndices = nullptr;
        unsigned int mVertexCount = 0;

        void operator()(unsigned int i1, unsigned int i2, unsigned int i3)
        {
            // Degenerate triangles are dropped: they contribute nothing and a path tracer can spend real
            // time on zero-area geometry.
            if (i1 == i2 || i2 == i3 || i1 == i3)
                return;
            if (i1 >= mVertexCount || i2 >= mVertexCount || i3 >= mVertexCount)
                return;
            mIndices->push_back(i1);
            mIndices->push_back(i2);
            mIndices->push_back(i3);
        }
    };

    /// Maps an OSG node mask to the Remix instance categories that describe it.
    ///
    /// This is the whole reason no texture-hash tagging is needed: OpenMW already knows what every
    /// subgraph is, so Remix can be told outright instead of inferring it from what a draw looks like.
    /// Every bit VisMask actually defines.
    ///
    /// Used to tell a meaningful mask from the default one. This has to be kept in step with vismask.hpp;
    /// a bit added there and forgotten here means that subgraph stops being classified.
    constexpr unsigned int kKnownVisMaskBits = MWRender::Mask_UpdateVisitor | MWRender::Mask_Effect
        | MWRender::Mask_Debug | MWRender::Mask_Actor | MWRender::Mask_Player | MWRender::Mask_Sky
        | MWRender::Mask_Water | MWRender::Mask_SimpleWater | MWRender::Mask_Terrain
        | MWRender::Mask_FirstPerson | MWRender::Mask_Object | MWRender::Mask_Static | MWRender::Mask_Sun
        | MWRender::Mask_WeatherParticles | MWRender::Mask_Scene | MWRender::Mask_GUI
        | MWRender::Mask_ParticleSystem | MWRender::Mask_RenderToTexture | MWRender::Mask_PreCompile
        | MWRender::Mask_Lighting | MWRender::Mask_Groundcover;

    unsigned int categoriesFor(unsigned int nodeMask)
    {
        // A node mask only identifies what a subgraph *is* when OpenMW set it deliberately. The OSG
        // default is 0xffffffff -- every bit -- which carries no information at all, and reading it as
        // categories claims the node is simultaneously sky, terrain, water and a particle system.
        //
        // That is not a hypothetical: it is what this function did, and because the flags were OR'd down
        // the tree it labelled the entire world as sky. Remix duly rendered it as sky, at infinity and
        // occluding nothing, so the player saw nothing but sky even indoors while instance counts looked
        // perfectly healthy.
        //
        // Bits outside the VisMask set mean this is a general-purpose mask rather than an identity.
        if ((nodeMask & ~kKnownVisMaskBits) != 0)
            return 0;

        unsigned int flags = 0;
        if (nodeMask & (MWRender::Mask_Sky | MWRender::Mask_Sun | MWRender::Mask_WeatherParticles))
            flags |= RemixRT::Runtime::Category_Sky;
        if (nodeMask & MWRender::Mask_Terrain)
            flags |= RemixRT::Runtime::Category_Terrain;
        if (nodeMask & (MWRender::Mask_Water | MWRender::Mask_SimpleWater))
            flags |= RemixRT::Runtime::Category_AnimatedWater;
        if (nodeMask & MWRender::Mask_ParticleSystem)
            flags |= RemixRT::Runtime::Category_Particle;
        return flags;
    }

    /// Walks the scene graph accumulating world transforms and submitting drawables.
    class SubmitVisitor : public osg::NodeVisitor
    {
    public:
        SubmitVisitor(MWRender::RemixScene& scene, unsigned int skipMask, const osg::Vec3f& eye,
            const osg::Vec3f& right, const osg::Vec3f& up, const osg::Polytope& frustum)
            : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ACTIVE_CHILDREN)
            , mScene(scene)
            , mEye(eye)
            , mRight(right)
            , mUp(up)
            , mFrustum(frustum)
        {
            // Tunable without a rebuild, because the right value is a judgement about how much off-screen
            // geometry the lighting needs rather than something derivable. Raise it if shadows or
            // reflections lose contributors at the edge of view; lower it to cull harder.
            if (const char* value = std::getenv("OPENMW_REMIX_CULL_MARGIN");
                value != nullptr && *value != '\0')
            {
                const double parsed = std::atof(value);
                if (parsed >= 0.0)
                    mCullMargin = static_cast<float>(parsed);
            }
            // Same reasoning for the sub-pixel test: the threshold is a judgement about how small is too
            // small to resolve, and the honest way to settle it is to look. Zero switches the test off, which
            // is the control to compare against.
            if (const char* value = std::getenv("OPENMW_REMIX_MIN_ANGULAR_RADIUS");
                value != nullptr && *value != '\0')
            {
                const double parsed = std::atof(value);
                if (parsed >= 0.0)
                    mMinAngularRadius = static_cast<float>(parsed);
            }
            // The traversal mask is the whole mask mechanism as far as this visitor is concerned. OSG
            // tests traversalMask & nodeMask in Node::accept, so a subgraph masked exclusively as GUI or
            // render-to-texture is excluded here, while ordinary geometry with the default all-bits mask
            // passes.
            setTraversalMask(~skipMask);

            // Declared an intersection visitor, and this is load-bearing rather than cosmetic.
            //
            // Terrain::QuadTreeWorld::accept opens with:
            //     if (!isCullVisitor && getVisitorType() != INTERSECTION_VISITOR) return;
            // A plain NodeVisitor is therefore refused outright, and OpenMW's exteriors are built almost
            // entirely behind that gate: the quadtree owns the terrain chunks *and* ObjectPaging's
            // batched statics, which is every building and rock. Nothing below it was ever visited, so
            // exteriors came through as loose interactive props -- doors, crates, actors, birds --
            // floating over an empty world with no ground and no architecture.
            //
            // Cull is the other accepted type and is not an option: it means being an
            // osgUtil::CullVisitor, which the code casts to. Intersection is the honest fit anyway --
            // this walk asks "what geometry is here", which is the same question a ray cast asks.
            // Switchable, because this is the one change that can take exteriors down with it: everything
            // it unlocks lives behind OpenMW's terrain machinery, which is not designed to be driven from
            // outside the cull traversal. OPENMW_REMIX_TERRAIN=0 restores the plain-NodeVisitor
            // behaviour -- exteriors without ground or architecture, but running -- and makes the
            // question answerable by bisection rather than by argument.
            if (envFlag("OPENMW_REMIX_TERRAIN", Settings::remix().mTerrain))
                setVisitorType(osg::NodeVisitor::INTERSECTION_VISITOR);

            // The quadtree picks level of detail from getEyePoint() and keys its cached view data on it.
            // NodeVisitor's default is the origin, which would select detail for a point the player is
            // nowhere near.
            //
            // No frame stamp, deliberately, and this one was learned the hard way. QuadTreeWorld::accept
            // ends with:
            //     if (referenceTime != 0.0) { vd->setLastUsageTimeStamp(referenceTime);
            //                                 mViewDataMap->clearUnusedViews(referenceTime); }
            // and clearUnusedViews expires a view by comparing its stored timestamp against that
            // reference time. Supplying a stamp therefore hands this traversal's clock to a global
            // expiry sweep -- and it is a *different* clock. osgViewer measures reference time from the
            // tick it records when the viewer is constructed; osg::Timer::instance()->time_s() measures
            // from process start, seconds earlier. The value is uniformly ahead of every timestamp OSG
            // has written, so every view looks expired, and the sweep recycles the ViewData that the cull
            // traversal is holding a pointer to. Exteriors died on the spot; interiors, having no
            // quadtree, were unaffected.
            //
            // With no stamp the reference time reads as zero and the whole block is skipped, which is the
            // correct behaviour for a caller that is not the frame's owner. The cost is that this
            // traversal's view is not marked as used and OpenMW's own sweep retires it -- but the reuse
            // path then finds the cull traversal's view a better match and copies it, so the LOD
            // selection is inherited rather than recomputed. Cheaper than the stamp was, and safe.
        }

        osg::Vec3 getEyePoint() const override { return mEye; }
        osg::Vec3 getViewPoint() const override { return mEye; }

        // No manual node-mask test in any of these. OSG already applies the traversal mask in
        // Node::accept via validNodeMask, which tests traversalMask & nodeMask -- so setTraversalMask
        // above is sufficient and correct.
        //
        // Testing it by hand is worse than redundant, it is inverted: nearly every OSG node carries the
        // default mask 0xffffffff, so "skip if nodeMask & skipMask" is true for almost everything and
        // aborts the traversal at the first ordinary group. That is what submitted zero instances while
        // looking like a healthy walk.
        void apply(osg::Node& node) override
        {
            // LightSource derives straight from osg::Node, not from Transform or Drawable, so this is
            // where it arrives. Checked by dynamic_cast rather than by node mask, because a light source
            // carries no distinguishing mask -- the lighting mask marks what *receives* light.
            if (const auto* source = dynamic_cast<SceneUtil::LightSource*>(&node))
                mScene.submitLight(*source, mMatrix(3, 0), mMatrix(3, 1), mMatrix(3, 2));

            pushState(node);
            traverse(node);
            popState();
        }

        // Most cameras terminate the walk, because a camera generally means its subtree is drawn
        // somewhere other than in place: render-to-texture for the local and global maps, character
        // preview, shadow maps, water ripples and terrain compositing, and -- the one that forced this
        // override -- the POST_RENDER overlay that paints Remix's own output back over the frame.
        // Submitting that would feed last frame's path-traced image back in as world geometry.
        //
        // The exception is a nested, non-RTT camera, which is drawn inline with everything around it and
        // is therefore ordinary content. OpenMW's sky is built that way, so excluding it wholesale would
        // change behaviour beyond what this override is for.
        //
        // osg::Camera derives from osg::Transform, so without this cameras fall through to the Transform
        // override below and are traversed like any other node -- which is what they did before.
        void apply(osg::Camera& node) override
        {
            if (node.getRenderOrder() == osg::Camera::NESTED_RENDER && !node.isRenderToTextureCamera())
                apply(static_cast<osg::Transform&>(node));
        }

        void apply(osg::Transform& node) override
        {
            const osg::Matrix saved = mMatrix;
            // computeLocalToWorldMatrix pre-multiplies, so accumulating downwards from identity yields
            // the local-to-world matrix for the node we end up at.
            node.computeLocalToWorldMatrix(mMatrix, this);
            pushState(node);
            traverse(node);
            popState();
            mMatrix = saved;
        }

        // Drawable, not Geometry. osg::Drawable::accept calls nv.apply(*this) where *this has static
        // type Drawable&, so overriding apply(osg::Geometry&) alone never fires -- dispatch lands on
        // apply(Drawable&), whose default forwards to apply(Node&). That silently submitted nothing at
        // all while the traversal otherwise looked healthy.
        void apply(osg::Drawable& drawable) override
        {
            // Frustum reject first, ahead of every other cost in this function: state merging, material
            // resolution, mesh cache lookup and the handover itself.
            //
            // Particle systems are exempt. Their bounds are rebuilt as the particles move and are not
            // dependable enough to reject on, and there are only tens of systems, so the saving would not
            // pay for a flame that vanishes.
            // Distance from the eye to this drawable's world-space bounding centre, for the submission
            // budget's nearest-first admission. Computed here, ahead of and independently of the cull,
            // because the budget needs it for every drawable that reaches the handover -- including when
            // culling is switched off and including particle systems, which the cull below exempts.
            //
            // From the bound rather than from the transform's translation, and that distinction is
            // load-bearing. ObjectPaging flattens static transforms into the vertices of a merged chunk, so
            // the very instances the budget exists to weigh -- the large merged distant chunks -- carry a
            // translation that says nothing about where their geometry is. Sorting on that would put the
            // heaviest geometry in an arbitrary place in the order.
            float distanceSquared = 0.0f;
            {
                const osg::BoundingSphere& bound = drawable.getBound();
                const osg::Vec3f worldCentre = bound.valid()
                    ? bound.center() * mMatrix
                    : osg::Vec3f(static_cast<float>(mMatrix(3, 0)), static_cast<float>(mMatrix(3, 1)),
                          static_cast<float>(mMatrix(3, 2)));
                distanceSquared = (worldCentre - mEye).length2();
            }

            if (!mFrustum.getPlaneList().empty() && dynamic_cast<osgParticle::ParticleSystem*>(&drawable) == nullptr)
            {
                const osg::BoundingSphere& local = drawable.getBound();
                if (local.valid())
                {
                    // Largest row length, so a non-uniform scale inflates the radius rather than shrinking
                    // it. Erring large here keeps geometry that should not have been culled.
                    const double sx = osg::Vec3d(mMatrix(0, 0), mMatrix(0, 1), mMatrix(0, 2)).length();
                    const double sy = osg::Vec3d(mMatrix(1, 0), mMatrix(1, 1), mMatrix(1, 2)).length();
                    const double sz = osg::Vec3d(mMatrix(2, 0), mMatrix(2, 1), mMatrix(2, 2)).length();
                    const double scale = std::max(sx, std::max(sy, sz));

                    const osg::Vec3f center = local.center() * mMatrix;
                    const float radius = static_cast<float>(local.radius() * scale) + mCullMargin;

                    if (!mFrustum.contains(osg::BoundingSphere(center, radius)))
                    {
                        // Rejected for this frame, not gone. Without this its mesh stops being touched,
                        // expires kMeshEvictionFrames later and has its acceleration structure destroyed
                        // -- so turning away and back rebuilds the BLAS of everything behind the camera.
                        mScene.retainCulledDrawable(drawable);
                        ++mCulled;
                        return;
                    }

                    // Then reject on apparent size. Being inside the frustum says nothing about being
                    // visible: at a far plane of 81920 units most of the world projects to less than a pixel
                    // while still carrying every one of its triangles into the acceleration structure. A
                    // measured session reached 143,316,362 primitives against a hard ceiling of 67,108,863 --
                    // past that the runtime's primitive index wraps, the NEE cache and prefix-sum lookups
                    // read garbage, and the result is a GPU reset rather than a visual artefact. That is the
                    // AppHangTransient plus nvlddmkm reset pair, not a crash.
                    //
                    // Compared as an angular radius so no viewport is needed: one pixel at 1440p across a 90
                    // degree field is roughly 0.0014 radians, so the default threshold is a little over one
                    // pixel of radius. Geometry that small cannot be told apart from the pixel it occupies,
                    // which is what makes this free in a way that shortening the far plane is not -- the
                    // silhouette of the distant landscape is unchanged, only objects too small to resolve
                    // stop being traced.
                    //
                    // The cull margin is deliberately excluded from this test. It exists to retain
                    // off-screen geometry for reflections, and something under a pixel contributes nothing
                    // to a reflection either.
                    if (mMinAngularRadius > 0.0f)
                    {
                        const float distance = (center - mEye).length();
                        if (distance > 1.0f
                            && static_cast<float>(local.radius() * scale) / distance < mMinAngularRadius)
                        {
                            // Same reasoning as the frustum reject above. This one is distance-keyed
                            // rather than direction-keyed, so what it churned was every object sitting
                            // near the threshold as the player walks -- a smaller herd than a spin, and
                            // more often.
                            mScene.retainCulledDrawable(drawable);
                            ++mCulled;
                            return;
                        }
                    }
                }
            }

            osg::Geometry* geometry = drawable.asGeometry();

            // Skinned geometry arrives as a RigGeometry, which is a Drawable and not a Geometry, so
            // asGeometry() returns null and the drawable was simply dropped -- that is why actors were
            // missing every part of themselves that moves. What it does hold is the bind pose plus the
            // influence data needed to deform it, which is exactly what a path tracer wants: the mesh
            // goes over once and only the bone transforms are resubmitted per frame.
            auto* rig = dynamic_cast<SceneUtil::RigGeometry*>(&drawable);
            if (rig != nullptr)
            {
                geometry = rig->getSourceGeometry().get();

                // Bring the pose up to date before anything reads it.
                //
                // RigGeometry::accept only refreshes the skin for a cull or an update visitor; every other
                // visitor, this one included, falls through to a plain apply() that touches nothing. So the
                // bone matrices read later in the frame were whatever OpenMW's own render last computed --
                // and it stops computing them for a skeleton it has marked inactive, or one its cull did
                // not reach. The symptom is an actor whose animation runs only while OpenMW's camera
                // happens to be refreshing it and freezes the moment it does not, which tracks camera
                // angle and distance rather than anything the animation is doing.
                //
                // The node path goes with it because a rig that has never been bound to a skeleton has to be
                // bound before its pose means anything, and the path is all that takes. Without that the
                // skin reported not-ready, drawSubmitted dropped the instance, and a freshly rebuilt actor
                // -- one that just changed equipment, or whose cell just loaded -- was invisible until
                // OpenMW's own cull reached it. Worn items that are not skinned kept rendering throughout,
                // which is what made it look like bodies were losing their parts.
                rig->refreshPose(getNodePath(), static_cast<unsigned int>(mScene.frameNumber()));
            }

            // Particles, the third Drawable that is not a Geometry, and the last of the invisible ones.
            //
            // osgParticle::ParticleSystem derives from osg::Drawable and builds its quads on the fly in
            // drawImplementation, so there is no geometry to find and the traversal dropped it -- the same
            // way it dropped RigGeometry. Every flame, smoke plume, fog effect and spell effect in the game
            // went missing this way, which is why candles had no flame and chimneys no smoke.
            //
            // Handled separately rather than through the geometry path because a particle system has to be
            // turned into geometry first, and that geometry is different every frame.
            if (auto* particles = dynamic_cast<osgParticle::ParticleSystem*>(&drawable))
            {
                MWRender::RemixScene::SurfaceState particleSurface = currentSurface();
                if (const osg::StateSet* stateSet = drawable.getStateSet())
                    mergeState(*stateSet, particleSurface);
                // Particles get real transparency rather than the cutout materialFor substitutes for
                // blending. That substitution is calibrated for foliage, whose alpha is effectively binary,
                // and particles are the opposite case: smoke, fog and dust are soft the whole way through,
                // with most of the puff below half alpha. Any threshold either erases the effect or leaves
                // it as discrete blobs with visibly hard edges.
                //
                // It also cost more than the edges. The runtime only sets its own isParticle flag inside
                // the branch where blending is enabled -- see InstanceManager::calculateAlphaState -- so a
                // cutout particle was never treated as a particle at all, whatever category it was tagged
                // with, and never reached the unordered TLAS where transparency accumulates.
                //
                // mAlphaBlend and mAdditive are deliberately left to mergeState, which reads the actual
                // BlendFunc. A particle system that genuinely is not blended should stay opaque.
                particleSurface.mPreferBlend = true;
                mScene.submitParticles(*particles, particleSurface, mMatrix, mRight, mUp,
                    currentCategories() | categoriesFor(drawable.getNodeMask()));
                return;
            }

            // Morph geometry, the other Drawable that hides a Geometry, is submitted with its animated
            // vertices.
            //
            // The Remix API has no morph targets, so there is nothing to hand the runtime the way bone
            // transforms are handed over for a skin. But OSG has already done the work: MorphGeometry::cull
            // blends the targets on the CPU into one of two frame-parity buffers, and getMorphedGeometry
            // returns whichever holds the last completed result. So this costs a mesh rebuild on the frames
            // the vertices actually change and nothing on the others -- meshFor decides that by watching
            // the vertex array's modified count.
            //
            // This used to submit getSourceGeometry(), the bind pose, on the reasoning that a face that
            // does not move is a smaller error than a missing one. True as far as it went, but the visible
            // consequence was that no NPC's mouth moved while talking, which is most of what Morrowind uses
            // morphs for.
            if (const auto* morph = dynamic_cast<const SceneUtil::MorphGeometry*>(&drawable))
            {
                // const_cast because meshFor needs a mutable reference to run a TriangleIndexFunctor over
                // the geometry, which is a read that OSG's visitor interface does not express as const.
                // Nothing downstream modifies it. Same reason the surrounding traversal takes drawables by
                // non-const reference despite only reading them.
                geometry = const_cast<osg::Geometry*>(morph->getMorphedGeometry());
            }

            if (geometry == nullptr)
                return;

            // A hard ceiling on how much is handed over in one frame. OpenMW's viewing distance is a
            // user setting and can be enormous -- this machine reports 811008 units -- and with paged
            // terrain plus object paging an unbounded traversal can produce enough geometry to wedge the
            // machine. Better to render a partial scene and say so than to hang.
            if (mInstances >= kMaxInstancesPerFrame)
            {
                // Retained for the same reason as the culls: being over the per-frame instance ceiling
                // says nothing about whether this geometry is still in the scene, and letting it expire
                // would mean the frame after a clamp pays to rebuild what it just declined to draw.
                mScene.retainGeometry(*geometry);
                mClamped = true;
                return;
            }

            // The drawable's own state set is combined with what it inherits, and it wins -- that is
            // OSG's own override-free precedence, and in OpenMW the leaf is usually where the NIF's
            // texture ends up.
            // A drawable's own node mask contributes categories too. It was being ignored, which had one
            // very visible consequence: OpenMW sets Mask_Water on the water quad itself rather than on a
            // parent, so water was never recognised as water and took the opaque grey fallback material.
            // That quad is CellSizeInUnits * 150 across -- over a million units -- so it read as a flat
            // grey plane lidding the world out to the horizon, which is easy to mistake for the terrain.
            const unsigned int categories
                = currentCategories() | categoriesFor(drawable.getNodeMask());

            MWRender::RemixScene::SurfaceState surface = currentSurface();
            surface.mIsWater = (categories & RemixRT::Runtime::Category_AnimatedWater) != 0;

            // A source geometry is a detached template rather than a node in the graph, so its state set
            // is not on the path and has to be folded in explicitly. Merged before the drawable's own so
            // that the one actually in the graph still wins.
            if (geometry != &drawable && geometry->getStateSet() != nullptr)
                mergeState(*geometry->getStateSet(), surface);

            if (const osg::StateSet* stateSet = drawable.getStateSet())
                mergeState(*stateSet, surface);

            // Terrain keeps its texture where a scene-graph walk cannot see it.
            //
            // Terrain::TerrainDrawable holds one state set per texture layer as a member and does the
            // multi-pass draw itself, so nothing about its appearance is reachable through the graph --
            // which is why terrain came out as flat grey while everything else was textured.
            //
            // Only the first pass is used. The remaining layers are alpha-blended over it through
            // per-layer blend maps, and a path-traced surface has one material: reproducing the blend
            // would mean compositing the layers into a per-chunk texture on the CPU. The base layer is
            // the dominant one, so this is the right approximation to start from rather than the final
            // answer.
            const auto* terrain = dynamic_cast<const Terrain::TerrainDrawable*>(&drawable);
            if (terrain != nullptr)
            {
                const auto& passes = terrain->getPasses();
                if (!passes.empty() && passes.front() != nullptr)
                {
                    mergeState(*passes.front(), surface);

                    // Assigned only when the pass actually has one, matching how mergeState treats a tagged
                    // normal, so a layer without a normal map keeps whatever the graph above it established
                    // rather than being forced back to none.
                    if (const osg::Texture2D* normal = terrainNormalMap(*passes.front()))
                        surface.mNormalMap = normal;
                }

                // Distant chunks need one more step, and this is what was putting white patches along
                // the horizon.
                //
                // Past a threshold chunk size (Terrain settings, "composite map level") a chunk is not
                // drawn from its layer textures at all. Its layers are rendered once into a per-chunk
                // render target and the chunk gets a single pass bound to that target. The pass merged
                // above therefore names an osg::Texture2D with no osg::Image behind it -- the pixels only
                // ever existed on the GPU -- so materialFor fell through to the untextured fallback, and
                // while that fallback was self-lit the sun blew it out to white.
                //
                // The layer textures are still reachable: CompositeMap keeps the quads it composites
                // from, and each carries the same kind of state set as a near chunk's pass. Taking the
                // first one is the same base-layer approximation already made above.
                if (const Terrain::CompositeMap* composite = terrain->getCompositeMap())
                    mergeCompositeLayer(*composite, surface);

                // Terrain is never a cutout. Its passes enable GL_BLEND because that is how the layers
                // are composited over one another, not because the ground has holes in it -- and
                // materialFor turns blending into a cutout, which would punch the land through wherever
                // a layer texture happened to carry alpha.
                surface.mAlphaBlend = false;

                if (terrain->getCompositeMap() != nullptr)
                    ++sChunksComposite;
                else
                    ++sChunksLayered;
                if (surface.mNormalMap != nullptr)
                    ++sChunksBaseNormal;

                const unsigned int chunks = sChunksComposite + sChunksLayered;
                if (chunks >= sNextTerrainReport)
                {
                    // Multiplied rather than incremented, so a long session reports a handful of times
                    // instead of once per fixed block of chunks -- terrain is re-submitted every frame, so a
                    // fixed interval would fill the log.
                    sNextTerrainReport = chunks * 4;
                    const auto percent = [chunks](unsigned int n) { return chunks > 0 ? 100 * n / chunks : 0; };
                    Log(Debug::Info)
                        << "[Remix terrain] " << chunks << " chunk submissions; composite "
                        << sChunksComposite << " (" << percent(sChunksComposite) << "%), per-layer "
                        << sChunksLayered << " (" << percent(sChunksLayered) << "%); base normal map "
                        << sChunksBaseNormal << " (" << percent(sChunksBaseNormal) << "%); overlay layers "
                        << sLayersSubmitted << ", of those with a normal map " << sLayersWithNormal;
                }
            }

            const unsigned long long mesh = mScene.submitGeometry(*geometry, surface, rig);
            if (mesh == 0)
                return;

            // Groundcover is the one thing OpenMW draws instanced, and instancing is invisible to a
            // scene-graph walk: the geometry holds a single blade, and the thousands of copies exist only
            // as a primitive-set instance count plus two vertex attributes the vertex shader unpacks.
            // Reading the geometry alone therefore yields one blade standing at its chunk's origin, which
            // is exactly what came through -- grass loaded and was traversed, and the world got one sprig
            // per chunk.
            //
            // Worth expanding rather than approximating because Remix's model is the same shape as OSG's
            // here: one mesh, many instance transforms. The blade goes over once and each copy costs a
            // transform, which is what the runtime wants anyway -- every copy shares one BLAS.
            if (submitGroundcoverCopies(*geometry, mesh, categories))
                return;

            float transform[12];
            writeTransform(mMatrix, transform);

            // Double-sided unconditionally for now. OpenMW's winding after the accumulated transforms
            // has not been verified against what Remix expects, and a wrong answer there makes geometry
            // vanish rather than look wrong -- which is much harder to diagnose than the cost of
            // disabling backface culling.
            //
            // The instance transform is the same for a skinned mesh as for a static one. The bone
            // matrices take a bind-pose vertex to the drawable's local space and the accumulated path
            // matrix takes it from there to the world, which is the same division of labour OSG uses --
            // RigGeometry's own skin-to-skeleton matrix exists precisely to cancel the transforms the
            // path already accounts for.
            //
            // The rig is passed on only if the mesh actually carries skinning. A rig whose skin this code
            // refused still produced a perfectly good static mesh from its bind pose, and sending bone
            // transforms for it would be asking the runtime to deform vertices that have no weights.
            // mInstances + 1 as the picking identity: it is already a per-frame running count, and the
            // one thing the runtime requires is that no two draws in a frame share a value. Zero means
            // "not pickable", hence the offset -- the first instance of a frame would otherwise opt
            // itself out.
            mScene.drawSubmitted(mesh, transform, categories, true,
                mScene.lastMeshIsSkinned() ? rig : nullptr, mInstances + 1, distanceSquared);
            mScene.noteInstancePosition(mMatrix(3, 0), mMatrix(3, 1), mMatrix(3, 2));
            ++mInstances;

            // The base layer is submitted; the rest of the ground goes over it.
            if (terrain != nullptr)
                submitTerrainLayers(*terrain, *geometry, categories, transform, distanceSquared);
        }

        /// Submits a terrain chunk's overlaid layers, one draw each, over the base layer already sent.
        ///
        /// This is what makes blended ground possible at all. OpenMW's terrain is multi-pass -- one pass
        /// per texture layer, each alpha-blended over the last through its own blend map -- and a
        /// path-traced surface has one material, so merging only the first pass gave every chunk a single
        /// layer and produced the hard rectangular edges where neighbouring chunks chose different base
        /// layers.
        ///
        /// Each layer carries its own diffuse at its own tiling and its coverage in the vertex alpha, so
        /// the runtime blends them the same way it blends a multi-pass terrain from a D3D9 game. Which is
        /// the point: that path is well travelled, and the alternative -- baking the layers into a texture
        /// -- was tried and is both slower and blurrier.
        void submitTerrainLayers(const Terrain::TerrainDrawable& terrain, osg::Geometry& geometry,
            unsigned int categories, const float (&baseTransform)[12], float distanceSquared)
        {
            static const bool enabled = envFlag("OPENMW_REMIX_TERRAIN_LAYERS", Settings::remix().mTerrainLayers);
            if (!enabled)
                return;

            const auto& passes = terrain.getPasses();

            // One-shot diagnostic over the first few chunks. Which of the several ways this can silently
            // do nothing is not guessable from the result: a chunk with one pass, a blend map whose CPU
            // image was released, or coverage that arrives and is then ignored all look identical on
            // screen -- hard-edged ground.
            static unsigned int logged = 0;
            const bool logThis = logged < 24;
            if (logThis)
            {
                ++logged;
                // Composite presence is the fact that matters. A chunk drawn from a composite map has one
                // pass whose texture is a render target OpenMW already blended on the GPU -- so there are
                // no layers here to blend, and the hard edges come from approximating that target by its
                // first layer. Raising Terrain/"composite map level" pushes chunks back onto the
                // per-layer path, which is where the layers this code needs actually exist.
                Log(Debug::Info) << "Remix terrain chunk: " << passes.size() << " passes, composite "
                                 << (terrain.getCompositeMap() != nullptr ? "YES" : "no");
            }

            if (passes.size() < 2)
                return;

            for (std::size_t pass = 1; pass < passes.size(); ++pass)
            {
                if (mInstances >= kMaxInstancesPerFrame)
                {
                    mClamped = true;
                    return;
                }
                if (passes[pass] == nullptr)
                    continue;

                MWRender::RemixScene::SurfaceState layer = currentSurface();
                mergeState(*passes[pass], layer);

                // Each overlay layer has its own normal map, at unit 2 of its own pass. Without this the
                // overlays would inherit the base layer's normal -- or none at all -- and the ground would
                // keep the base layer's relief wherever another texture was painted over it.
                if (const osg::Texture2D* normal = terrainNormalMap(*passes[pass]))
                    layer.mNormalMap = normal;

                // Unit 1 is the blend map -- see components/terrain/material.cpp, which binds the layer
                // diffuse at unit 0 and the blend map at unit 1 with its own texture matrix. Without an
                // image behind it there is no coverage to read and the layer would cover the whole chunk
                // opaquely, hiding everything below it, so skipping is the safe failure.
                const auto* blendTexture = dynamic_cast<const osg::Texture2D*>(
                    passes[pass]->getTextureAttribute(1, osg::StateAttribute::TEXTURE));

                if (logThis)
                {
                    const osg::Image* img = blendTexture != nullptr ? blendTexture->getImage(0) : nullptr;
                    Log(Debug::Info) << "  layer " << pass << ": blendTexture "
                                     << (blendTexture != nullptr ? "yes" : "NO") << ", image "
                                     << (img != nullptr ? "yes" : "NO")
                                     << (img != nullptr
                                             ? " " + std::to_string(img->s()) + "x" + std::to_string(img->t())
                                                 + " fmt 0x" + std::to_string(img->getPixelFormat())
                                                 + " data " + (img->data() != nullptr ? "yes" : "NO")
                                             : std::string())
                                     << ", texmat "
                                     << (passes[pass]->getTextureAttribute(1, osg::StateAttribute::TEXMAT)
                                                != nullptr
                                             ? "yes"
                                             : "no");
                }

                if (blendTexture == nullptr || blendTexture->getImage(0) == nullptr)
                    continue;

                // A layer that covers none of this chunk is not submitted at all. Skipping it is free
                // correctness -- a fully transparent surface contributes nothing to the image either way --
                // and it removes close to half of the terrain instances, because a chunk enumerates every
                // land texture used anywhere in its area while covering only some of it.
                if (!imageHasAnyAlpha(*blendTexture->getImage(0)))
                {
                    if (logThis)
                        Log(Debug::Info) << "    skipped: covers none of this chunk";
                    continue;
                }

                layer.mCoverageImage = blendTexture->getImage(0);
                layer.mCoverageLayer = static_cast<unsigned int>(pass);

                if (logThis)
                {
                    // A blend map that reads as uniformly 255 means the coverage is not the gradient it is
                    // supposed to be, which would make every layer cover its chunk completely.
                    unsigned int lo = 255;
                    unsigned int hi = 0;
                    for (int gy = 0; gy <= 4; ++gy)
                    {
                        for (int gx = 0; gx <= 4; ++gx)
                        {
                            const unsigned int a = sampleImageAlpha(
                                *layer.mCoverageImage, gx * 0.25f, gy * 0.25f);
                            lo = std::min(lo, a);
                            hi = std::max(hi, a);
                        }
                    }
                    Log(Debug::Info) << "    coverage alpha range " << lo << ".." << hi;
                }

                if (const auto* blendTexMat = dynamic_cast<const osg::TexMat*>(
                        passes[pass]->getTextureAttribute(1, osg::StateAttribute::TEXMAT)))
                {
                    const osg::Matrix& m = blendTexMat->getMatrix();
                    layer.mHasCoverageTexMat = true;
                    layer.mCoverageTexMat[0] = static_cast<float>(m(0, 0));
                    layer.mCoverageTexMat[1] = static_cast<float>(m(0, 1));
                    layer.mCoverageTexMat[2] = static_cast<float>(m(1, 0));
                    layer.mCoverageTexMat[3] = static_cast<float>(m(1, 1));
                    layer.mCoverageTexMat[4] = static_cast<float>(m(3, 0));
                    layer.mCoverageTexMat[5] = static_cast<float>(m(3, 1));
                }

                // Real transparency rather than the cutout materialFor substitutes for blending. A cutout
                // would quantise the coverage to on or off and reinstate hard edges in a new place -- the
                // gradient between layers is the entire content of a blend map.
                layer.mAlphaBlend = true;
                layer.mPreferBlend = true;
                layer.mAlphaTestReference = 0;

                // Not additive, whatever the blend function says. This is the line that made terrain refuse
                // to blend at all.
                //
                // mergeState sets mAdditive from `destination == GL_ONE`, and a terrain overlay pass really
                // does use SRC_ALPHA/ONE (Terrain::BlendFunc in components/terrain/material.cpp). But it
                // uses it to accumulate, not to glow: getBlendmaps gives the base layer whatever coverage
                // the overlays do not claim, so the weights sum to one across a chunk. The first pass writes
                // base * alpha with destination ZERO and each overlay adds layer * alpha, which is a
                // weighted average spelled as a sum.
                //
                // mAdditive exists to recognise flames and glows, and materialFor turns it into
                // kBlendTypeAlphaEmissive -- a surface that emits light rather than contributing albedo.
                // Terrain tripped that test for an unrelated reason and every overlay was handed to the
                // runtime as an emissive blend, which adds its colour on top of the ground instead of
                // mixing with it. No amount of correct coverage can look blended through that.
                layer.mAdditive = false;

                // Counted here rather than where the normal map is read, so these describe layers that were
                // actually handed to the runtime -- the reads above happen before the coverage tests, which
                // discard close to half of them.
                ++sLayersSubmitted;
                if (layer.mNormalMap != nullptr)
                    ++sLayersWithNormal;

                const unsigned long long layerMesh = mScene.submitGeometry(geometry, layer, nullptr);
                if (layerMesh == 0)
                    continue;

                // The base layer's distance, because that is what this is: the same chunk, submitted again
                // with different coverage. A layer sorting away from its own base layer would let the
                // budget admit one and reject the other, which is worse than dropping the chunk outright.
                mScene.drawSubmitted(
                    layerMesh, baseTransform, categories, true, nullptr, mInstances + 1, distanceSquared);
                ++mInstances;
            }
        }

        unsigned int instances() const { return mInstances; }
        bool clamped() const { return mClamped; }
        unsigned int culled() const { return mCulled; }

    private:
        /// Writes an OSG matrix as the twelve floats Remix takes for an instance transform.
        ///
        /// OSG is row-vector (p * M); the Remix transform is applied as p' = M * p with translation in the
        /// last column. Hence the transpose of the 3x3 and the translation read from OSG's fourth row. If
        /// instances come out rotated or mirrored, this is the function to question.
        static void writeTransform(const osg::Matrix& matrix, float (&out)[12])
        {
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                    out[row * 4 + col] = static_cast<float>(matrix(col, row));
                out[row * 4 + 3] = static_cast<float>(matrix(3, row));
            }
        }

        /// Rebuilds the per-copy rotation that groundcover.vert derives from its aRotation attribute.
        ///
        /// Transcribed from that shader's rotation() rather than derived independently, because the two
        /// have to agree exactly or grass leans one way in the path-traced view and another in OpenMW's
        /// own. GLSL's mat4 constructor takes columns and OSG applies matrices to row vectors, so each of
        /// the shader's columns becomes a row here; that transpose is the whole of the conversion.
        static osg::Matrix groundcoverRotation(const osg::Vec3f& angle)
        {
            const double sinX = std::sin(angle.x());
            const double cosX = std::cos(angle.x());
            const double sinY = std::sin(angle.y());
            const double cosY = std::cos(angle.y());
            const double sinZ = std::sin(angle.z());
            const double cosZ = std::cos(angle.z());

            osg::Matrix rotation;
            rotation(0, 0) = cosZ * cosY + sinX * sinY * sinZ;
            rotation(0, 1) = -sinZ * cosX;
            rotation(0, 2) = cosZ * sinY + sinZ * sinX * cosY;
            rotation(1, 0) = sinZ * cosY + cosZ * sinX * sinY;
            rotation(1, 1) = cosZ * cosX;
            rotation(1, 2) = sinZ * sinY - cosZ * sinX * cosY;
            rotation(2, 0) = -sinY * cosX;
            rotation(2, 1) = sinX;
            rotation(2, 2) = cosX * cosY;
            return rotation;
        }

        /// Hands over one Remix instance per groundcover copy.
        ///
        /// Returns false when \a geometry is not instanced, meaning the caller should submit it the
        /// ordinary way. Returns true once it has taken responsibility for the geometry -- including when
        /// grass is switched off, because the alternative is submitting the base blade, and a lone sprig
        /// at each chunk's origin is worse than no grass at all.
        ///
        /// Per copy the local transform is scale, then rotation, then translation, matching
        /// groundcover.vert:
        ///     position = aOffset.xyz;  scale = aOffset.w;  rotation = rotation(aRotation);
        ///     displacedVertex = rotation * scale * gl_Vertex;  displacedVertex.xyz += position;
        /// The offsets are chunk-relative and the accumulated path matrix carries the chunk into the
        /// world, so composing the two is the whole placement.
        ///
        /// What is deliberately not reproduced is groundcoverDisplacement, the wind sway. That is a
        /// per-vertex displacement with no equivalent in an instance transform, so path-traced grass
        /// stands still. Animating it would mean rebuilding every blade's vertices every frame, which is
        /// the one thing this integration avoids everywhere else.
        bool submitGroundcoverCopies(
            const osg::Geometry& geometry, unsigned long long mesh, unsigned int categories)
        {
            unsigned int copies = 0;
            for (unsigned int i = 0; i < geometry.getNumPrimitiveSets(); ++i)
            {
                const osg::PrimitiveSet* set = geometry.getPrimitiveSet(i);
                if (set != nullptr && set->getNumInstances() > 0)
                    copies = std::max(copies, static_cast<unsigned int>(set->getNumInstances()));
            }

            // One instance is what every ordinary drawable reports, so this is the not-instanced exit.
            if (copies <= 1)
                return false;

            const auto* offsets = dynamic_cast<const osg::Vec4Array*>(
                geometry.getVertexAttribArray(kGroundcoverOffsetAttrib));
            const auto* rotations = dynamic_cast<const osg::Vec3Array*>(
                geometry.getVertexAttribArray(kGroundcoverRotationAttrib));

            // Instanced by something other than groundcover, or by a future version of it that places its
            // copies differently. Submitting the base geometry once is then the honest fallback: it is
            // where it belongs, there is simply one of it.
            if (offsets == nullptr || offsets->size() < copies)
                return false;

            // Report which grass model this mesh is, once per mesh.
            //
            // The hash is what a replacement pack binds to and the model path is what a person can recognise;
            // pairing them is the difference between authoring a region's grass and guessing at hashes. The
            // path is tagged onto the cloned subtree by Groundcover::createChunk, so it sits above this
            // drawable in the traversal's node path.
            for (auto it = getNodePath().rbegin(); it != getNodePath().rend(); ++it)
            {
                std::string model;
                if ((*it)->getUserValue("remixGroundcoverModel", model))
                {
                    // The blade's own size, measured over its vertices.
                    //
                    // NOT from getBoundingBox(): Groundcover::InstancingVisitor deliberately expands the
                    // geometry's bound to enclose every copy in the chunk so that culling works on the
                    // instanced set as a whole. Reading it back reported a single blade as 4530 x 4405 x 1062
                    // units -- over half a cell wide -- which is the bound of a few hundred blades scattered
                    // across the chunk, not one of them. The vertex array is the only place the blade's own
                    // dimensions still exist.
                    float extent[3] = { 0.0f, 0.0f, 0.0f };
                    float base = 0.0f;
                    if (const auto* verts = dynamic_cast<const osg::Vec3Array*>(geometry.getVertexArray());
                        verts != nullptr && !verts->empty())
                    {
                        osg::Vec3f low = (*verts)[0];
                        osg::Vec3f high = (*verts)[0];
                        for (const osg::Vec3f& v : *verts)
                        {
                            low.x() = std::min(low.x(), v.x());
                            low.y() = std::min(low.y(), v.y());
                            low.z() = std::min(low.z(), v.z());
                            high.x() = std::max(high.x(), v.x());
                            high.y() = std::max(high.y(), v.y());
                            high.z() = std::max(high.z(), v.z());
                        }
                        extent[0] = high.x() - low.x();
                        extent[1] = high.y() - low.y();
                        extent[2] = high.z() - low.z();
                        // Carried too, because a replacement inherits the blade's pivot: OpenMW's blades sit
                        // on the origin and grow up, and an asset that straddles zero sinks by the difference.
                        base = low.z();
                    }
                    mScene.noteGroundcoverModel(mesh, model, copies, extent, base);
                    break;
                }
            }

            static const bool enabled = envFlag("OPENMW_REMIX_GROUNDCOVER", Settings::remix().mGroundcover);
            // Bounded by the per-frame instance ceiling, because that is what this budget counts: one
            // grass copy is one instance, and the loop below is already stopped by kMaxInstancesPerFrame.
            static const unsigned int budget = envUInt(
                "OPENMW_REMIX_GROUNDCOVER_BUDGET", kMaxGroundcoverCopies, kMaxInstancesPerFrame);
            if (!enabled || budget == 0)
                return true;

            // The same distance the shader fades grass out at, so the path-traced extent matches
            // OpenMW's own rather than being a second, unrelated draw distance. Zero means the setting
            // imposes no limit, and then only the budget bounds this.
            const float fadeEnd = static_cast<float>(Settings::groundcover().mRenderingDistance);

            for (unsigned int i = 0; i < copies; ++i)
            {
                if (mInstances >= kMaxInstancesPerFrame)
                {
                    mClamped = true;
                    break;
                }
                if (mGroundcoverCopies >= budget)
                    break;

                const osg::Vec4f& offset = (*offsets)[i];
                const osg::Vec3f position(offset.x(), offset.y(), offset.z());
                const float scale = offset.w();

                osg::Matrix copyMatrix = osg::Matrix::scale(scale, scale, scale);
                if (rotations != nullptr && i < rotations->size())
                    copyMatrix = copyMatrix * groundcoverRotation((*rotations)[i]);
                copyMatrix = copyMatrix * osg::Matrix::translate(position);

                const osg::Matrix world = copyMatrix * mMatrix;
                const osg::Vec3f at(static_cast<float>(world(3, 0)), static_cast<float>(world(3, 1)),
                    static_cast<float>(world(3, 2)));

                const float copyDistanceSquared = (at - mEye).length2();
                if (fadeEnd > 0.0f && copyDistanceSquared > fadeEnd * fadeEnd)
                    continue;

                float transform[12];
                writeTransform(world, transform);

                // Per copy rather than per blade geometry, and here the translation is the right source:
                // each copy is a real instance transform, so unlike a merged chunk its origin is where its
                // geometry is.
                mScene.drawSubmitted(
                    mesh, transform, categories, true, nullptr, mInstances + 1, copyDistanceSquared);
                mScene.noteInstancePosition(world(3, 0), world(3, 1), world(3, 2));
                ++mInstances;
                ++mGroundcoverCopies;
            }

            return true;
        }

        unsigned int currentCategories() const
        {
            return mCategoryStack.empty() ? 0u : mCategoryStack.back();
        }

        MWRender::RemixScene::SurfaceState currentSurface() const
        {
            return mSurfaceStack.empty() ? MWRender::RemixScene::SurfaceState{} : mSurfaceStack.back();
        }

        /// Substitutes a composited chunk's base layer texture for its render target.
        ///
        /// The tiling correction is the interesting part. A chunk larger than the "max composite geometry
        /// size" setting is composited from a mosaic of sub-quads, and the retained pass carries a texture
        /// matrix sized for one sub-quad's footprint rather than the chunk's. Applying it unchanged to the
        /// chunk's own full-chunk texcoords tiles the layer once per chunk instead of once per sub-quad,
        /// which reads as a stretched, low-frequency smear -- so it is scaled back up by the number of
        /// sub-quads across, which is what CompositeMap::mBaseLayerTiling holds.
        static void mergeCompositeLayer(
            const Terrain::CompositeMap& composite, MWRender::RemixScene::SurfaceState& surface)
        {
            // The composited result when it is available, which is the whole point: it already carries every
            // layer blended by OpenMW's own compositor, with the layer tiling and the half-texel nudge
            // material.cpp applies to match vanilla. The base-layer path below is the fallback for a chunk
            // that has not finished compositing yet, and it is what makes the ground look like flat
            // rectangles with hard edges.
            //
            // No texture matrix goes with it. A composited chunk is drawn with a single UV set spanning the
            // chunk exactly once, so the composite is sampled 1:1 and the sub-quad tiling correction below
            // would be actively wrong here.
            // Taken as a reference under the composite's lock. The image is published from the draw thread,
            // and holding a reference is what stops its chunk releasing it out from under the upload below.
            if (osg::ref_ptr<osg::Image> readback = composite.readback())
            {
                surface.mExplicitImage = readback;
                surface.mHasTexMat = false;
                return;
            }

            if (composite.mBaseLayerPass == nullptr)
                return;

            mergeState(*composite.mBaseLayerPass, surface);

            if (composite.mBaseLayerTiling > 0.0f && composite.mBaseLayerTiling != 1.0f)
            {
                // The 2x2 linear part only. The layer matrix is a pure scale, so there is no translation
                // to carry along.
                for (int i = 0; i < 4; ++i)
                    surface.mTexMat[i] *= composite.mBaseLayerTiling;
                surface.mHasTexMat = true;
            }
        }

        /// What the ground is actually made of, counted over the whole session.
        ///
        /// These exist because the per-chunk diagnostic below logs the first two dozen chunks it happens to
        /// see and then stops, which is too small and too early a sample to conclude anything from: scene
        /// traversal order decides which chunks those are, and a run in which all of them came back
        /// composite says nothing about the ground the player is standing on. Session-long counters separate
        /// the two explanations that look identical on screen -- terrain having no normal maps to give, and
        /// terrain having them but not passing them on.
        static inline unsigned int sChunksComposite = 0;
        static inline unsigned int sChunksLayered = 0;
        static inline unsigned int sChunksBaseNormal = 0;
        static inline unsigned int sLayersSubmitted = 0;
        static inline unsigned int sLayersWithNormal = 0;
        static inline unsigned int sNextTerrainReport = 4000;

        /// Terrain's normal map, which no role tag identifies.
        ///
        /// Terrain::createPasses builds its state sets by hand instead of going through
        /// Shader::ShaderVisitor, so it attaches no SceneUtil::TextureType to anything -- and mergeState
        /// identifies maps only by tag, so it finds no normal map on any terrain pass and never has. That is
        /// the ground looking flat while objects picked their normals up.
        ///
        /// The unit is fixed by construction rather than guessed at: components/terrain/material.cpp binds
        /// the layer diffuse at unit 0, the blend map at unit 1 and the normal map at unit 2, and binds
        /// nothing else at any unit. Reading unit 2 is therefore exact for terrain and only for terrain,
        /// which is why this is kept out of mergeState -- unit 2 of an object state set is whatever
        /// ShaderVisitor happened to put there, and assuming otherwise is how the normal-as-albedo defect
        /// happened.
        ///
        /// Composite passes never carry one, and cannot: they are drawn by the terrain_composite program
        /// from a render target that was already blended on the GPU, and createPasses skips the normal-map
        /// branch entirely when building them. Distant chunks therefore stay as they are.
        static const osg::Texture2D* terrainNormalMap(const osg::StateSet& pass)
        {
            return dynamic_cast<const osg::Texture2D*>(
                pass.getTextureAttribute(2, osg::StateAttribute::TEXTURE));
        }

        /// Folds one state set into \a surface. Later calls override earlier ones, which matches OSG's
        /// precedence for attributes that carry no override flag.
        static void mergeState(const osg::StateSet& stateSet, MWRender::RemixScene::SurfaceState& surface)
        {
            // Every unit is examined for a role tag, and the unit index is deliberately not assumed.
            // Shader::ShaderVisitor binds an auto-detected normal map at `texAttributes.size()` -- the
            // next free unit, whatever that happens to be for this state set -- and records what it is by
            // attaching a SceneUtil::TextureType naming it. The tag is the only reliable identification.
            //
            // The albedo is now resolved the same way, and used not to be. It was read from unit 0
            // unconditionally, which is wrong for precisely the reason above: "the next free unit" is unit 0
            // for a state set that binds no diffuse, so an auto-detected normal map lands there and was then
            // taken as the base colour. That is a tangent-space normal map rendered as albedo -- the
            // iridescent purple surface, in the toolkit and in game alike -- and the very same texture was
            // also bound as the normal, which is what made it look as though the normal had been assigned
            // twice.
            //
            // An untagged unit 0 is still accepted, and has to be: fixed-function state sets carry no tags
            // at all and their unit 0 genuinely is the diffuse. What is no longer accepted is a unit 0 whose
            // tag says it is something else. Nothing is assigned in that case, so the surface keeps the
            // albedo inherited from an outer state set -- which is the right answer, since the parent's
            // diffuse is still the diffuse.
            //
            // Four roles are consumed: diffuseMap, normalMap/normalHeightMap, specularMap and emissiveMap.
            // The rest are recognised by OpenMW and deliberately left alone -- see the note by glossMap at
            // the end of the loop for why picking up a map because its name sounds right is how the
            // normal-as-albedo defect happened in the first place.
            const unsigned int units = static_cast<unsigned int>(stateSet.getTextureAttributeList().size());

            const osg::Texture2D* taggedDiffuse = nullptr;
            const osg::Texture2D* taggedNormal = nullptr;
            const osg::Texture2D* taggedSpecular = nullptr;
            const osg::Texture2D* taggedEmissive = nullptr;
            const SceneUtil::TextureType* unitZeroRole = nullptr;

            for (unsigned int unit = 0; unit < units; ++unit)
            {
                const auto* type = dynamic_cast<const SceneUtil::TextureType*>(
                    stateSet.getTextureAttribute(unit, SceneUtil::TextureType::AttributeType));
                if (unit == 0)
                    unitZeroRole = type;
                if (type == nullptr)
                    continue;

                const auto* texture = dynamic_cast<const osg::Texture2D*>(
                    stateSet.getTextureAttribute(unit, osg::StateAttribute::TEXTURE));
                if (texture == nullptr)
                    continue;

                const std::string& role = type->getName();

                if (role == "diffuseMap")
                {
                    // First tagged diffuse wins, so a later unit cannot displace it.
                    if (taggedDiffuse == nullptr)
                        taggedDiffuse = texture;
                }
                // "normalHeightMap" is a normal map with height in alpha. The normal half is what Remix is
                // being given; the height half would need Remix's separate heightTexture slot and a
                // channel split, which is not done here.
                else if (role == "normalMap" || role == "normalHeightMap")
                {
                    taggedNormal = texture;
                    // Only the tag distinguishes the two. The alpha of a plain normalMap means nothing, so
                    // the height split has to be gated on what OpenMW called it rather than on whether an
                    // alpha channel happens to be present.
                    surface.mHasNormalHeight = (role == "normalHeightMap");
                }
                else if (role == "specularMap")
                {
                    taggedSpecular = texture;
                }
                // The one PBR-ish map vanilla Morrowind actually ships. It comes from a NiTexturingProperty
                // glow slot rather than a filename pattern, so unlike the normal and specular maps it needs
                // no texture pack to be present -- lanterns, runes, glowing plants and enchanted items all
                // carry one in the base game.
                else if (role == "emissiveMap")
                {
                    taggedEmissive = texture;
                }

                // "glossMap" is deliberately not read, despite the name. OpenMW multiplies it into the
                // environment-map term (objects.frag: envEffect *= texture2D(glossMap, ...).xyz), so it is
                // an env-map mask rather than a glossiness map, and feeding it to roughness would be wrong
                // in a way that looks plausible. darkMap is now read, above, having had that decision
                // made for it -- it is a colour multiply, and the emissive path is where it earns its keep.
                // in a way that looks plausible. Nor are darkMap, detailMap, decalMap, bumpMap or envMap
                // consumed: each needs its own decision about where it belongs, and guessing is what
                // produced the normal-as-albedo defect above.
            }

            if (taggedDiffuse != nullptr)
            {
                surface.mTexture = taggedDiffuse;
            }
            else if (unitZeroRole == nullptr)
            {
                if (const auto* texture = dynamic_cast<const osg::Texture2D*>(
                        stateSet.getTextureAttribute(0, osg::StateAttribute::TEXTURE)))
                {
                    surface.mTexture = texture;
                }
            }

            if (taggedNormal != nullptr)
            {
                surface.mNormalMap = taggedNormal;
            }

            if (taggedSpecular != nullptr)
            {
                surface.mSpecularMap = taggedSpecular;
            }

            if (taggedEmissive != nullptr)
            {
                surface.mEmissiveMap = taggedEmissive;
            }


            if (const auto* texMat = dynamic_cast<const osg::TexMat*>(
                    stateSet.getTextureAttribute(0, osg::StateAttribute::TEXMAT)))
            {
                // Row-vector convention, matching the fixed-function texture matrix OSG is emulating:
                // s' = s*m00 + t*m10 + m30, and likewise for t. The third row and column are dropped
                // because texcoords here are 2D.
                const osg::Matrix& m = texMat->getMatrix();
                surface.mTexMat[0] = static_cast<float>(m(0, 0));
                surface.mTexMat[1] = static_cast<float>(m(0, 1));
                surface.mTexMat[2] = static_cast<float>(m(1, 0));
                surface.mTexMat[3] = static_cast<float>(m(1, 1));
                surface.mTexMat[4] = static_cast<float>(m(3, 0));
                surface.mTexMat[5] = static_cast<float>(m(3, 1));
                surface.mHasTexMat = surface.mTexMat[0] != 1.0f || surface.mTexMat[1] != 0.0f
                    || surface.mTexMat[2] != 0.0f || surface.mTexMat[3] != 1.0f
                    || surface.mTexMat[4] != 0.0f || surface.mTexMat[5] != 0.0f;
            }

            // Recomputed rather than inherited when the state set says anything about alpha testing, so
            // that a subgraph turning the test off is respected as well as one turning it on.
            if (stateSet.getMode(GL_ALPHA_TEST) != osg::StateAttribute::INHERIT
                || stateSet.getAttribute(osg::StateAttribute::ALPHAFUNC) != nullptr)
            {
                surface.mAlphaTestReference = alphaTestReferenceFor(stateSet);
            }

            // Same treatment for blending, and for the same reason: a state set that switches GL_BLEND
            // off has to be able to override a parent that switched it on.
            const unsigned int blendMode = stateSet.getMode(GL_BLEND);
            if (blendMode != osg::StateAttribute::INHERIT)
                surface.mAlphaBlend = (blendMode & osg::StateAttribute::ON) != 0;

            // Additive blending, which is what separates a flame from a smoke plume.
            //
            // The destination factor is the whole test. GL_ONE means the fragment is ADDED to what is
            // already in the framebuffer, so the surface only ever brightens the scene -- that is emission,
            // and it covers both (ONE, ONE) and (SRC_ALPHA, ONE). Ordinary alpha blending is
            // (SRC_ALPHA, ONE_MINUS_SRC_ALPHA), which replaces rather than accumulates, so the surface
            // occludes what is behind it and must be lit rather than lit-from-within.
            if (const auto* blendFunc = dynamic_cast<const osg::BlendFunc*>(
                    stateSet.getAttribute(osg::StateAttribute::BLENDFUNC)))
            {
                surface.mAdditive = blendFunc->getDestination() == GL_ONE;
            }

            // Material emission, which is where Morrowind keeps the brightness of smoke and steam.
            //
            // mAdditive above answers "does this surface accumulate", which correctly identifies flames and
            // does not identify smoke. Vanilla smoke and steam are SRC_ALPHA/ONE_MINUS_SRC_ALPHA and really
            // do occlude, yet they read bright in raster because NiMaterialProperty carries an emissive
            // colour the fixed-function pipeline adds regardless of scene lighting. nifloader forwards it
            // (setEmission from matprop->mEmissive), and where a NiVertexColorProperty asks for emissive
            // vertex colours it sets colour mode EMISSION with emission white, leaving the per-particle tint
            // to the vertex colours already submitted.
            //
            // Reading it here is what lets an alpha-blended particle be given the emission its asset asked
            // for instead of a constant guessed at in this file. Nothing consumed this before, which is why
            // [Remix PBR] reported emissive on 0% of materials while steam arrived nearly black.
            if (const auto* material = dynamic_cast<const osg::Material*>(
                    stateSet.getAttribute(osg::StateAttribute::MATERIAL)))
            {
                const osg::Vec4f emission = material->getEmission(osg::Material::FRONT_AND_BACK);
                surface.mEmissiveColor[0] = emission.r();
                surface.mEmissiveColor[1] = emission.g();
                surface.mEmissiveColor[2] = emission.b();
                // Max component, not luma, to agree with how the runtime splits emission into a scale and a
                // unit colour. See the field's comment.
                surface.mMaterialEmissive
                    = std::max(emission.r(), std::max(emission.g(), emission.b()));

                // Diffuse carries both the tint and the material's own alpha: nifloader writes
                // setDiffuse(FRONT_AND_BACK, Vec4f(matprop->mDiffuse, matprop->mAlpha)), so the opacity is
                // in w. Both multiply what the albedo texture supplies.
                const osg::Vec4f diffuse = material->getDiffuse(osg::Material::FRONT_AND_BACK);
                surface.mDiffuseColor[0] = diffuse.r();
                surface.mDiffuseColor[1] = diffuse.g();
                surface.mDiffuseColor[2] = diffuse.b();
                surface.mMaterialAlpha = diffuse.a();
            }
        }

        void pushState(osg::Node& node)
        {
            mCategoryStack.push_back(currentCategories() | categoriesFor(node.getNodeMask()));

            MWRender::RemixScene::SurfaceState surface = currentSurface();
            if (const osg::StateSet* stateSet = node.getStateSet())
                mergeState(*stateSet, surface);

            // State that animates, which until now only existed inside OpenMW's cull traversal.
            //
            // NIF state set animation -- scrolling UVs, texture flipbooks, alpha and material colour fades --
            // is driven by SceneUtil::StateSetUpdater. For a node carrying AnimFlag_AutoPlay, which is what
            // every always-running ambient effect is, nifloader attaches that updater as a *cull* callback
            // (see nifloader.cpp, the AutoPlay branch). StateSetUpdater::applyCull then writes the result
            // into a state set it keeps per CullVisitor and pushes onto that visitor's stack -- it never
            // touches node->getStateSet(). This traversal is not a CullVisitor and reads the node's own state
            // set, so it saw only the identity matrix setDefaults installed and nothing after it. Every such
            // effect arrived frozen on its first frame, and an effect whose entire motion IS the animated
            // state arrived completely static: candle smoke is a plane that moves only by scrolling its UVs.
            //
            // Evaluated into a state set this scene owns. The alternatives are worse: the updater's per-cull
            // map is private, and invoking the callback the way a non-cull visitor would drives its *update*
            // path, which assigns to the node -- mutating the graph from here would fight the traversal that
            // legitimately owns it.
            // Guarded on the frame stamp rather than assuming one: SceneUtil::FrameTimeSource::getValue
            // dereferences it without checking, so evaluating a controller without one is a crash, not a
            // frame of missing animation.
            NifOsg::FlipController* flip = nullptr;
            for (osg::Callback* callback = getFrameStamp() != nullptr ? node.getCullCallback() : nullptr;
                 callback != nullptr; callback = callback->getNestedCallback())
            {
                auto* updater = dynamic_cast<SceneUtil::StateSetUpdater*>(callback);
                if (updater == nullptr)
                    continue;

                // Merged after the node's own state set so the animated value wins, which is the precedence
                // the cull path produces by pushing it on top.
                osg::StateSet& animated = mScene.animatedStateFor(*updater);
                updater->apply(&animated, this);
                mergeState(animated, surface);

                // A texture flipbook can be handed to the runtime as a sprite sheet instead, which is one
                // material and one texture rather than one of each per frame. Found here because this is
                // where the controllers are already in hand; resolved after the loop so the atlas overrides
                // the single frame apply() just wrote into the merged state.
                if (flip == nullptr)
                {
                    if (auto* direct = dynamic_cast<NifOsg::FlipController*>(updater))
                    {
                        flip = direct;
                    }
                    else if (auto* composite = dynamic_cast<SceneUtil::CompositeStateSetUpdater*>(updater))
                    {
                        // nifloader wraps a node's controllers in a composite, so a flipbook is usually a
                        // child of one rather than the callback itself.
                        for (size_t i = 0; i < composite->getNumControllers() && flip == nullptr; ++i)
                            flip = dynamic_cast<NifOsg::FlipController*>(composite->getController(i));
                    }
                }
            }

            // A sheet replaces the per-frame albedo, so this has to come after every merge above.
            //
            // mDelta of zero means the controller picks frames through an interpolator rather than at a
            // constant rate, which a sprite sheet cannot express -- those keep the per-frame path rather than
            // being animated at a rate that is merely plausible.
            if (flip != nullptr && flip->getDelta() > 0.0f)
            {
                const std::vector<osg::ref_ptr<osg::Texture2D>>& frames = flip->getTextures();
                const long rate = std::lround(1.0f / flip->getDelta());
                if (frames.size() >= 2 && frames.size() <= 255 && rate >= 1)
                {
                    if (const osg::Image* sheet = mScene.spriteSheetFor(frames))
                    {
                        surface.mExplicitImage = sheet;
                        surface.mSpriteSheetCols = static_cast<unsigned char>(frames.size());
                        surface.mSpriteSheetFps
                            = static_cast<unsigned char>(std::min<long>(rate, 255));
                    }
                }
            }

            mSurfaceStack.push_back(surface);
        }

        void popState()
        {
            mCategoryStack.pop_back();
            mSurfaceStack.pop_back();
        }

        MWRender::RemixScene& mScene;
        osg::Vec3f mEye;
        /// Camera right and up in world space, for building camera-facing particle quads. A particle has a
        /// position and a size but no orientation, so the quad has to be oriented against the viewer, and
        /// that cannot be derived from the scene graph.
        osg::Vec3f mRight;
        osg::Vec3f mUp;
        osg::Matrix mMatrix;

        /// World-space view frustum, empty when culling is switched off. Planes point inward.
        osg::Polytope mFrustum;

        /// Slack added to every bounding radius before the frustum test, which is arithmetically the same as
        /// pushing all of the frustum's planes outward by this distance.
        ///
        /// A path tracer legitimately needs some geometry that is not directly visible: it casts shadows into
        /// view and shows up in reflections. Rejecting strictly on the frustum would take those with it, so
        /// the default keeps a wide band of off-screen geometry and still discards everything behind the
        /// camera, which is where most of the saving is. In OpenMW units, where a cell is 8192 across.
        float mCullMargin = 2048.0f;

        /// Smallest angular radius a drawable may subtend before it is rejected as too small to resolve.
        ///
        /// Roughly one pixel of radius at 1440p across a 90 degree field. Raise it to cull harder, set it to
        /// zero to switch the test off entirely as a control.
        float mMinAngularRadius = 0.0015f;

        unsigned int mCulled = 0;
        std::vector<unsigned int> mCategoryStack;
        std::vector<MWRender::RemixScene::SurfaceState> mSurfaceStack;
        unsigned int mInstances = 0;
        /// Groundcover copies submitted this frame, counted separately so grass is bounded by its own
        /// allowance rather than by whatever is left of the general one.
        unsigned int mGroundcoverCopies = 0;
        bool mClamped = false;
    };
}

namespace MWRender
{
    RemixScene::RemixScene(RemixRT::Runtime& runtime)
        : mRuntime(runtime)
    {
        // One shared material until textures are wired up. Mid-grey and fairly rough, because a
        // featureless white surface makes it impossible to tell shading from blown-out exposure, and a
        // mirror-smooth one shows the sky instead of the geometry.
        //
        // Not self-lit any more, and the change matters. It was self-lit so that geometry submitted
        // before any lights existed could be told apart from geometry that was not submitted at all --
        // black and absent look identical. Lights are submitted now, so the crutch has a cost and no
        // benefit: every surface that falls back to this material glows, and the sun then blows it out to
        // white. Distant terrain is the visible case, because a terrain chunk far enough out is drawn
        // from a composite render target whose osg::Image is null, so it lands here -- and a field of
        // white patches along the horizon is the result. OPENMW_REMIX_EMISSIVE=1 brings it back for
        // diagnosis.
        const bool emissive = envFlag("OPENMW_REMIX_EMISSIVE", Settings::remix().mUntexturedEmissive);
        constexpr unsigned long long kDefaultMaterialHash = 0x0B7A5E'0000'0001ull;
        mDefaultMaterial = mRuntime.createFlatMaterial(
            kDefaultMaterialHash, 0.6f, 0.6f, 0.6f, 0.7f, 0.0f, emissive ? kWorldEmissive : 0.0f);
        if (mDefaultMaterial == 0)
            Log(Debug::Error) << "Remix scene: could not create the default material; geometry submission "
                                 "will not work";
        else
            Log(Debug::Info) << "Remix scene: default material is " << (emissive ? "self-lit" : "unlit")
                             << " (OPENMW_REMIX_EMISSIVE=1 to make untextured geometry glow)";

        // Foliage cutout threshold. See materialFor for why blending has to become a cutout at all.
        mBlendCutout = envByte("OPENMW_REMIX_BLEND_CUTOUT", kDefaultAlphaTestReference);
        Log(Debug::Info) << "Remix scene: alpha-blended surfaces are cut out at "
                         << static_cast<unsigned int>(mBlendCutout)
                         << "/255 (OPENMW_REMIX_BLEND_CUTOUT=0 leaves them solid)";

        // Ask the terrain compositor to keep a CPU copy of what it composites.
        //
        // Off by default because it is pure cost for anyone only rasterising -- one glReadPixels and 1MB
        // retained per chunk at the default 512x512. This is the only route to a real terrain albedo: the
        // composited result exists solely as a render target, and taking the base layer instead is what
        // made the ground read as flat rectangles with hard edges between chunks.
        //
        // Only chunks at or above Terrain/"composite map level" are composited at all, so that setting has
        // to be low enough to cover near chunks or they keep the base-layer approximation.
        // Off by default, because it is a trade rather than an improvement.
        //
        // Reading the composite back gives correctly blended ground: every layer combined by OpenMW's own
        // compositor, no hard rectangles where neighbouring chunks pick different base layers. But the
        // composite is one 512x512 texture for a whole chunk, where the base-layer path tiles its texture
        // many times across the same chunk. At cell scale that is roughly 16 world units per texel against
        // vanilla's 2, so the ground goes soft and low-frequency -- which reads worse than the wrong
        // blending did, particularly on large-chunk terrain like Tamriel Rebuilt.
        //
        // Neither option is right. Sharpness needs the layers tiled, correctness needs them blended, and one
        // baked texture per chunk cannot do both at any resolution that fits in memory: matching vanilla
        // would take 4096x4096 a chunk. The real answer is Remix's own TerrainBaker, which composites
        // multi-pass terrain at a resolution it manages -- see rtx_terrain_baker.cpp. Until that is wired to
        // the API path this stays opt-in so it can be compared rather than assumed.
        const bool terrainComposite = envFlag("OPENMW_REMIX_TERRAIN_COMPOSITE", Settings::remix().mTerrainComposite);
        Terrain::CompositeMap::sReadbackEnabled = terrainComposite;
        Log(Debug::Info) << "Remix scene: terrain composite readback "
                         << (terrainComposite ? "ENABLED -- ground is correctly blended but softer, since a "
                                                "chunk's whole albedo is one 512x512 composite"
                                              : "off; ground uses the tiled base layer, which is sharp but "
                                                "shows hard edges where chunks pick different layers "
                                                "(OPENMW_REMIX_TERRAIN_COMPOSITE=1 to compare)");

        // Logged rather than silent because it changes what a capture contains and what binds. Two captures
        // of the same cell are only comparable if it is recorded which mode each was taken in.
        mPackMatching = envFlag("OPENMW_REMIX_PACK_MATCH", Settings::remix().mPackMatching);

        // Read once here rather than per line. These bound the two log streams that identify which file a
        // wrong-looking surface came from, and are also most of a large openmw.log, so they are the knobs
        // most likely to be reached for -- and 0 for either switches that stream off entirely.
        mMaterialLogLimit = static_cast<unsigned int>(std::max(0, Settings::remix().mMaterialLogLimit.get()));
        mTextureLogLimit = static_cast<unsigned int>(std::max(0, Settings::remix().mTextureLogLimit.get()));

        // Instrumentation manifest, host side.
        //
        // Printed for discoverability rather than diagnosis. Every stream named here can be switched without a
        // rebuild, but only by someone who knows it exists, and that knowledge otherwise leaves with whoever
        // wrote it. Putting the inventory in the log means any run, read by anyone, later, indexes its own
        // tooling. The second line points at the runtime's half deliberately: the two sets live in different
        // config files behind different UIs, so whichever log gets opened first has to lead to both.
        {
            const auto onOff = [](bool value) { return value ? "on" : "off"; };
            const std::string skip = Settings::remix().mSkipTextures.get();
            Log(Debug::Info) << "Remix instrumentation: materialLog=" << mMaterialLogLimit
                             << " textureLog=" << mTextureLogLimit
                             << " sceneInterval=" << sceneLogInterval() << "f"
                             << " probeQuad=" << onOff(Settings::remix().mProbeQuad)
                             << " untexturedEmissive=" << onOff(Settings::remix().mUntexturedEmissive)
                             << " parallax=" << Settings::remix().mParallaxDepth.get()
                             << " skipTextures=" << (skip.empty() ? std::string("<none>") : skip);
            Log(Debug::Info) << "Remix instrumentation: set these in settings.cfg [Remix] or the launcher's "
                                "Testing tab; either takes effect on the next launch. Runtime-side streams "
                                "(scatter submit, mesh and material lookups, heavy assets, frame spikes, NEE "
                                "overflow) are rtx.fork.log.* in rtx.conf and are listed in remix-dxvk.log.";
        }
        Log(Debug::Info) << "Remix scene: identities use "
                         << (mPackMatching
                                 ? "Remix's D3D9 formulation for both meshes and textures, so a pack "
                                   "authored against a Morrowind capture can bind"
                                 : "native stable identities -- PACK MATCHING OFF, nothing authored against "
                                   "a Morrowind capture can bind, though Remix and replacement loading are "
                                   "unaffected (OPENMW_REMIX_PACK_MATCH=1 to restore)");

        mLightRadius = envFloat("OPENMW_REMIX_LIGHT_RADIUS", Settings::remix().mLightRadius);
        mLightIntensityFactor
            = envFloat("OPENMW_REMIX_LIGHT_INTENSITY", Settings::remix().mLightIntensity);
        Log(Debug::Info) << "Remix scene: light emitter radius " << mLightRadius
                         << " units, intensity factor " << mLightIntensityFactor
                         << " (OPENMW_REMIX_LIGHT_RADIUS and OPENMW_REMIX_LIGHT_INTENSITY override both; "
                            "the latter mirrors rtx.lightConversionIntensityFactor). Radiance is derived "
                            "per light from where its attenuation curve fades out, not from its radius";

        // Publish the starting values so the developer menu's "External (API) Light settings" sliders open
        // on what is actually in use. Without this an environment override would leave the menu showing the
        // runtime's compiled-in defaults, and the first touch of a slider would jump the lighting.
        {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "%.5f", mLightRadius);
            mRuntime.setConfigVariable(kExternalLightRadiusOption, buffer);
            snprintf(buffer, sizeof(buffer), "%.5f", mLightIntensityFactor);
            mRuntime.setConfigVariable(kExternalLightIntensityOption, buffer);
        }

        // Water. The transmittance distance is in OpenMW units, which is why it looks large: one metre
        // is about seventy of them, so this absorbs over roughly four metres of depth. Morrowind's water
        // is a murky green-brown, so red is absorbed hardest.
        constexpr unsigned long long kWaterMaterialHash = 0x0B7A5E'0000'0002ull;
        const float waterTransmittance[3] = { 0.35f, 0.75f, 0.60f };
        mWaterMaterial = mRuntime.createTranslucentMaterial(kWaterMaterialHash, 1.33f,
            waterTransmittance, 280.0f);
        if (mWaterMaterial == 0)
            Log(Debug::Warning) << "Remix scene: could not create the water material; water will use the "
                                   "untextured fallback and read as a solid plane";

        // Strongly emissive so it cannot be confused with a dim surface or lost to auto-exposure.
        constexpr unsigned long long kProbeMaterialHash = 0x0B7A5E'0000'0010ull;
        mProbeMaterial
            = mRuntime.createFlatMaterial(kProbeMaterialHash, 1.0f, 0.1f, 0.6f, 0.9f, 0.0f, 40.0f);
        if (mProbeMaterial == 0)
            Log(Debug::Warning) << "Remix scene: could not create the bisect quad's material";
    }

    RemixScene::~RemixScene()
    {
        for (const auto& [key, cached] : mMeshes)
            mRuntime.destroyMesh(cached.mHandle);
        mMeshes.clear();
        for (const auto& [key, cached] : mLights)
            mRuntime.destroyLight(cached.mHandle);
        mLights.clear();
        if (mProbeMesh != 0)
            mRuntime.destroyMesh(mProbeMesh);

        // Materials before textures: a material naming a released texture is the more dangerous of the
        // two orderings. The default material can appear as several map values when a textured material
        // failed to create, so it is skipped here and released once afterwards.
        for (const auto& [key, handle] : mMaterials)
        {
            if (handle != mDefaultMaterial)
                mRuntime.destroyMaterial(handle);
        }
        mMaterials.clear();
        mMaterialAlbedoHashes.clear();
        // Walked by identity rather than by cache entry, because mTextures is keyed per image and several
        // images can name one identity -- iterating it destroyed a shared texture once per image that
        // referenced it. Harmless at process exit, but this loop is the obvious model for anyone adding a
        // mid-session sweep, and there it would not be.
        for (const auto& [hash, claims] : mUploadedTextures)
            mRuntime.destroyTexture(hash);
        mUploadedTextures.clear();
        mTextureBytes.clear();
        mTextureBytesResident = 0;
        mTextures.clear();
        if (mProbeMaterial != 0)
            mRuntime.destroyMaterial(mProbeMaterial);
        if (mWaterMaterial != 0)
            mRuntime.destroyMaterial(mWaterMaterial);
        if (mDefaultMaterial != 0)
            mRuntime.destroyMaterial(mDefaultMaterial);
    }

    void RemixScene::noteInstancePosition(double x, double y, double z)
    {
        const double p[3] = { x, y, z };
        if (!mHaveExtent)
        {
            mHaveExtent = true;
            for (int i = 0; i < 3; ++i)
                mMin[i] = mMax[i] = p[i];
            return;
        }
        for (int i = 0; i < 3; ++i)
        {
            mMin[i] = std::min(mMin[i], p[i]);
            mMax[i] = std::max(mMax[i], p[i]);
        }
    }

    void RemixScene::submitLight(const SceneUtil::LightSource& source, double x, double y, double z)
    {
        // Deliberately non-const: getLight takes a frame index because the light is double buffered for
        // the draw thread. Reading either buffer is fine here -- this traversal runs on the update
        // thread, before the draw -- and the frame counter is this class's own, so it only has to be
        // stable, not aligned with OSG's.
        //
        // SceneUtil::Light rather than osg::Light: upstream 28fefe86d9 ("remove osg::Light*") replaced
        // the osg type with its own lightweight osg::Referenced-derived one. The two are unrelated
        // types, so this stopped compiling on merge rather than changing behaviour quietly. Every
        // accessor read below -- diffuse and the three attenuation terms -- carries the same name and
        // meaning on the new type, so nothing here had to be reinterpreted.
        auto& mutableSource = const_cast<SceneUtil::LightSource&>(source);
        const SceneUtil::Light* light = mutableSource.getLight(static_cast<size_t>(mFrame));
        if (light == nullptr || source.getEmpty())
            return;

        const osg::Vec4f& diffuse = light->getDiffuse();

        // Radiance derived from the light's own reach, using the runtime's conversion rather than a
        // constant.
        //
        // The previous version multiplied a fixed power by the light's colour, which is wrong in two
        // separate ways and produced sphere lights of around 30000 intensity where a candle wants tens.
        //
        // First, brightness has to come from how far the light reaches, not from a global constant. Remix
        // derives it in LightUtils::calculateIntensity by asking what radiance a sphere emitter of a given
        // size needs in order to fall to a just-perceptible threshold at the original light's end
        // distance:
        //
        //   radiance = endValue * endDistance^2 * intensityFactor / (pi * emitterRadius^2)
        //
        // A candle and a bonfire then differ by the square of their reach, which is the whole point --
        // with a constant they came out identical.
        //
        // Second, the colour is normalised to its largest component and the brightness carried entirely by
        // the intensity, which is what LightUtils::calculateRadiance does. Multiplying radiance by the raw
        // colour instead makes a deep red light dimmer than a white one of the same reach, when what the
        // colour is meant to describe is hue.
        //
        // Following the same formula matters beyond just being less wrong: it is what the legacy D3D9 path
        // does, so lights here land in the same range as the ones RTXremixMW's MGE-XE setup produces, and
        // rtx.lightConversionIntensityFactor tunes both the same way.
        const float range = source.getRadius();
        const float brightest = std::max({ diffuse.r(), diffuse.g(), diffuse.b() });
        if (!(range > 0.0f) || !(brightest > 0.0f))
            return;

        const float endDistance = lightEndDistance(*light, range, brightest);

        // Actor fade is OpenMW's way of dimming a carried light as its owner fades out. Ignoring it
        // leaves lights at full strength on invisible actors.
        const float fade = source.getActorFade();
        const float intensity = kLightEndValue * endDistance * endDistance * mLightIntensityFactor
            / (osg::PI * mLightRadius * mLightRadius) * (fade > 0.0f ? fade : 1.0f);
        if (!(intensity > 0.0f))
            return;

        const float radiance[3] = { diffuse.r() / brightest * intensity,
            diffuse.g() / brightest * intensity, diffuse.b() / brightest * intensity };

        const float position[3]
            = { static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) };

        // A handful of real lights, described once. The conversion depends on the attenuation curve as much
        // as on the radius, so a formula in the log says very little about whether the numbers are sane --
        // and "the Toolkit says 30000, is that a lot" was the question that found the last two bugs here.
        if (mLightsLogged < kLightLogLimit)
        {
            ++mLightsLogged;
            Log(Debug::Info) << "Remix light " << mLightsLogged << ": radius " << range
                             << " units, attenuation " << light->getConstantAttenuation() << " + "
                             << light->getLinearAttenuation() << "d + " << light->getQuadraticAttenuation()
                             << "d^2 -> fades out at " << endDistance << " units -> radiance " << intensity
                             << (mLightsLogged == kLightLogLimit ? " (last of these)" : "");
        }

        const int key = source.getId();
        // Identity carried over from the cached entry when this submission has not changed it.
        //
        // A colour change alone leaves the hash alone, and the recreate below has to reuse the exact value
        // the handle was made with so that it lands on the runtime's update path. Recomputing it from the
        // current position would not be equivalent: the move test tolerates drift up to an epsilon, so a
        // light trembling below that threshold keeps its handle while its recomputed hash would wander,
        // and every such frame would quietly orphan a light.
        unsigned long long reuseHash = 0;
        auto found = mLights.find(key);
        if (found != mLights.end())
        {
            const CachedLight& cached = found->second;
            const bool moved = std::abs(cached.mPosition[0] - position[0]) > kLightMoveEpsilon
                || std::abs(cached.mPosition[1] - position[1]) > kLightMoveEpsilon
                || std::abs(cached.mPosition[2] - position[2]) > kLightMoveEpsilon;
            // Relative comparison, against the larger of the two so a light switching on from zero
            // always counts as changed.
            const auto changedBy = [](float before, float after) {
                const float scale = std::max({ std::abs(before), std::abs(after), 1e-6f });
                return std::abs(before - after) / scale > kLightRadianceRelativeEpsilon;
            };
            const bool recoloured = changedBy(cached.mRadiance[0], radiance[0])
                || changedBy(cached.mRadiance[1], radiance[1])
                || changedBy(cached.mRadiance[2], radiance[2]);
            // Radius is compared as well as radiance even though the two move together today -- radiance
            // is derived from the radius, so retuning one shows up in the other. Leaving it out would make
            // the cache silently wrong the moment that stops being true, for no saving worth having.
            const bool resized = cached.mRadius != mLightRadius;

            if (!moved && !recoloured && !resized)
            {
                found->second.mLastUsedFrame = mFrame;
                if (mRuntime.drawLight(cached.mHandle))
                    ++mLastLightCount;
                return;
            }

            // Position and radius are what the hash is made of, so these two are exactly the changes
            // that rename the light. Anything else -- colour, flicker, an actor fading out -- keeps its
            // identity, and the recreate below has to reuse it rather than recompute it.
            const bool reidentified = moved || resized;
            if (!reidentified)
                reuseHash = cached.mHash;

            // Recreated in place, and emphatically NOT destroyed first, whenever the identity is
            // unchanged. Creating with an existing hash *is* the runtime's update path. Destroying first
            // cannot work: destroys are queued and drained late in the frame, while creates go straight
            // into the command stream, and the drain builds a tombstone set from the queued destroys that
            // suppresses any create sharing a handle with one. So destroy-then-recreate reliably ends with
            // the light gone -- and gone for good, because this cache then believes it exists and never
            // rebuilds it. That is what made every light die a frame or two after it first moved.
            //
            // A move or a resize is the exception, and only became one when the hash started coming from
            // position and radius. Such a light is a different light as far as the runtime is concerned, so
            // the create below cannot update the old handle -- and the old handle is about to become
            // unreachable, because this cache entry is the only thing that refers to it and the end-of-frame
            // sweep only visits entries still in the map. Left alone it would leak one light per movement
            // per frame, which for actor-carried torches is every torch every frame.
            //
            // Destroying here is safe precisely because the identity differs: the tombstone set suppresses
            // a create that shares a handle with a queued destroy, and these two do not share one.
            if (reidentified)
                mRuntime.destroyLight(found->second.mHandle);
            mLights.erase(found);
        }

        // The hash is the identity and the handle both, so it has to be stable for a given light -- and
        // the previous formulation was not.
        //
        // It derived the hash from OpenMW's light-source id, which is a counter handed out as cells
        // stream in. That satisfies "distinct from every other live light", which is all the old comment
        // here claimed, but it is not identity: the same light gets a different id on the next run, and a
        // different one again after its cell unloads and reloads within a run. Since this value is the
        // name the toolkit stores light edits under, every edit went dead almost immediately -- which is
        // the reported symptom, and it was a property of the hash rather than of the toolkit.
        //
        // AssetHash::d3d9SphereLight reproduces the runtime's own RtSphereLight hash over position and
        // radius, so identity now follows the light rather than the order cells happened to load in.
        // Deliberately excluding radiance, as the runtime does, keeps the identity stable while a light
        // flickers, dims with its owner's fade, or shifts colour with the time of day.
        //
        // avoidZero rather than the old `| 1`: forcing the low bit would move the value off the formula
        // and defeat the whole point, whereas zero is the one value the runtime treats as no light at
        // all. avoidZero perturbs only that single case.
        const unsigned long long hash = reuseHash != 0
            ? reuseHash
            : RemixRT::AssetHash::avoidZero(RemixRT::AssetHash::d3d9SphereLight(position, mLightRadius));

        const unsigned long long handle
            = mRuntime.createSphereLight(hash, position, radiance, mLightRadius);
        if (handle == 0)
            return;

        CachedLight cached;
        cached.mHandle = handle;
        cached.mLastUsedFrame = mFrame;
        std::copy(std::begin(position), std::end(position), std::begin(cached.mPosition));
        std::copy(std::begin(radiance), std::end(radiance), std::begin(cached.mRadiance));
        cached.mRadius = mLightRadius;
        cached.mHash = hash;
        mLights.emplace(key, cached);

        if (mRuntime.drawLight(handle))
            ++mLastLightCount;
    }

    void RemixScene::pollLightTuning()
    {
        // Seeded with the current values so a key the menu has never written leaves them alone. That is
        // the normal case for a whole session in which nobody opens the light panel.
        float radius = mLightRadius;
        float intensity = mLightIntensityFactor;
        mRuntime.getGameValueFloat(kExternalLightRadiusKey, radius);
        mRuntime.getGameValueFloat(kExternalLightIntensityKey, intensity);

        // Clamped here as well as in the menu. The menu's minimum constrains its own slider, but these
        // arrive as text from a UI on the far side of an API boundary, and a zero radius divides by zero
        // in the radiance derivation below -- one bad value would black out or blow out every light.
        radius = std::max(radius, kLightRadiusMinimum);
        intensity = std::max(intensity, 0.0f);

        if (radius != mLightRadius || intensity != mLightIntensityFactor)
        {
            mLightRadius = radius;
            mLightIntensityFactor = intensity;

            // No explicit cache invalidation. Both values feed every light's radiance, and submitLight
            // compares cached radiance and radius against what it just derived, so each light rebuilds
            // itself as it comes back round this frame.
            mLightTuningChangedFrame = mFrame;
            mLightTuningPendingLog = true;
        }

        // Logged once the value settles rather than on every change. A slider being dragged changes it
        // every frame, and the only number worth recording is the one that was settled on -- it is the one
        // to carry back into OPENMW_REMIX_LIGHT_RADIUS, since these options are deliberately not saved.
        if (mLightTuningPendingLog && mFrame - mLightTuningChangedFrame >= kLightTuningSettleFrames)
        {
            mLightTuningPendingLog = false;
            Log(Debug::Info) << "Remix scene: light tuning set live to emitter radius " << mLightRadius
                             << " units, intensity factor " << mLightIntensityFactor;
        }
    }

    void RemixScene::releaseStaleLights()
    {
        // No grace period, unlike meshes. A light is cheap to recreate -- there is no vertex data to
        // convert or upload -- and keeping a stale one alive means an unexplained light in a cell the
        // player has left, which is far worse than recreating it on re-entry.
        for (auto it = mLights.begin(); it != mLights.end();)
        {
            if (it->second.mLastUsedFrame != mFrame)
            {
                mRuntime.destroyLight(it->second.mHandle);
                it = mLights.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void RemixScene::ensureProbeAssets()
    {
        if (mProbeMesh != 0 || mProbeMaterial == 0)
            return;

        // A unit quad in the object XY plane. The instance transform maps object X to the camera's right,
        // object Y to its up and object Z to its forward, so the -Z normal ends up pointing back at the
        // eye and the quad always faces the viewer whatever the camera does.
        RemixRT::Runtime::Vertex vertices[4] = {};
        const float corners[4][2] = { { -1.0f, -1.0f }, { 1.0f, -1.0f }, { 1.0f, 1.0f }, { -1.0f, 1.0f } };
        const float uvs[4][2] = { { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f } };
        for (int i = 0; i < 4; ++i)
        {
            vertices[i].mPosition[0] = corners[i][0];
            vertices[i].mPosition[1] = corners[i][1];
            vertices[i].mPosition[2] = 0.0f;
            vertices[i].mNormal[2] = -1.0f;
            vertices[i].mTexcoord[0] = uvs[i][0];
            vertices[i].mTexcoord[1] = uvs[i][1];
            vertices[i].mColor = 0xFFFFFFFFu;
        }
        const unsigned int indices[6] = { 0, 1, 2, 0, 2, 3 };

        constexpr unsigned long long kProbeMeshHash = 0x0B7A5E'0000'0011ull;
        mProbeMesh = mRuntime.createMesh(kProbeMeshHash, vertices, 4, indices, 6, mProbeMaterial);
        if (mProbeMesh == 0)
            Log(Debug::Error) << "Remix scene: could not create the bisect quad's mesh";
    }

    void RemixScene::submitProbeQuad(
        const double* eye, const double* forward, const double* up, const double* right)
    {
        ensureProbeAssets();
        if (mProbeMesh == 0)
            return;

        float transform[12];
        for (int row = 0; row < 3; ++row)
        {
            transform[row * 4 + 0] = static_cast<float>(right[row]) * kProbeHalfSize;
            transform[row * 4 + 1] = static_cast<float>(up[row]) * kProbeHalfSize;
            transform[row * 4 + 2] = static_cast<float>(forward[row]);
            transform[row * 4 + 3] = static_cast<float>(eye[row] + forward[row] * kProbeDistance);
        }

        mRuntime.drawInstance(mProbeMesh, transform, 0, true);
    }

    void RemixScene::noteGroundcoverModel(unsigned long long mesh, const std::string& model,
        unsigned int copies, const float (&extent)[3], float base)
    {
        if (!mLoggedGroundcoverModels.insert(mesh).second)
            return;

        Log(Debug::Info) << "[Remix grass] " << model << " -> mesh 0x" << std::hex << mesh << std::dec
                         << ", " << copies << " copies in the chunk that introduced it; blade extent "
                         << extent[0] << " x " << extent[1] << " x " << extent[2] << " units, base z "
                         << base;
    }

    osg::StateSet& RemixScene::animatedStateFor(SceneUtil::StateSetUpdater& updater)
    {
        AnimatedState& entry = mAnimatedStates[&updater];
        if (entry.mStateSet == nullptr)
        {
            entry.mStateSet = new osg::StateSet;
            updater.setDefaults(entry.mStateSet);
        }
        entry.mLastUsedFrame = mFrame;
        return *entry.mStateSet;
    }

    void RemixScene::pruneAnimatedStates()
    {
        for (auto it = mAnimatedStates.begin(); it != mAnimatedStates.end();)
        {
            if (it->second.mLastUsedFrame == mFrame)
                ++it;
            else
                it = mAnimatedStates.erase(it);
        }
    }

    unsigned int RemixScene::submit(osg::Node* sceneRoot, const osg::Camera& camera, osg::FrameStamp* frameStamp)
    {
        mLastInstanceCount = 0;
        mLastLightCount = 0;
        mSkinnedInstances = 0;
        mCulled = 0;
        // Reset per frame, unlike mMeshes.size() which is the cache's population. The distinction is the
        // whole point of the counter: 500 cached meshes reused every frame is nearly free, and 500 rebuilt
        // every frame is 500 acceleration structure builds. Those are indistinguishable from the cache
        // size alone, and the difference is the dominant term in the frame.
        mMeshesCreated = 0;
        mMeshBuildsDeferred = 0;
        mMeshesRetained = 0;
        mPrimitivesSubmitted = 0;
        mInstancesOverBudget = 0;
        mCompositeEncodeMs = 0.0;
        mCompositesDeferred = 0;
        // Cleared rather than shrunk. flushSubmissions empties it at the end of every frame, so this only
        // matters on the paths below that return early -- but leaving a previous frame's instances queued
        // would submit them again with stale transforms.
        mPendingInstances.clear();
        mSkinnedDropped = 0;
        mLastParticleCount = 0;
        mParticleInstances = 0;
        mHaveExtent = false;
        if (sceneRoot == nullptr || mDefaultMaterial == 0)
            return 0;

        ++mFrame;

        // Before any light is submitted, so a slider moved this frame takes effect this frame.
        pollLightTuning();

        // Camera first, as parameters rather than matrices. Deriving eye and basis from the inverse view
        // matrix, and the frustum from OpenMW's own settings, keeps handedness, matrix majorness and
        // reversed-Z out of the picture entirely -- all three would otherwise be load-bearing at once,
        // with no way to tell which was wrong from the result.
        const osg::Matrixd inverseView = osg::Matrixd::inverse(camera.getViewMatrix());
        const osg::Vec3d eye = inverseView.getTrans();
        // Rows of the inverse view matrix are the camera basis in world space.
        const osg::Vec3d right(inverseView(0, 0), inverseView(0, 1), inverseView(0, 2));
        const osg::Vec3d up(inverseView(1, 0), inverseView(1, 1), inverseView(1, 2));
        // Row 2 points backwards along the view direction, hence the negation.
        const osg::Vec3d forward(-inverseView(2, 0), -inverseView(2, 1), -inverseView(2, 2));

        const osg::Viewport* viewport = camera.getViewport();
        const float aspect = (viewport != nullptr && viewport->height() > 0)
            ? static_cast<float>(viewport->width() / viewport->height())
            : 1.7777f;

        const float eyeF[3] = { static_cast<float>(eye.x()), static_cast<float>(eye.y()),
            static_cast<float>(eye.z()) };
        const float forwardF[3] = { static_cast<float>(forward.x()), static_cast<float>(forward.y()),
            static_cast<float>(forward.z()) };
        const float upF[3]
            = { static_cast<float>(up.x()), static_cast<float>(up.y()), static_cast<float>(up.z()) };
        const float rightF[3] = { static_cast<float>(right.x()), static_cast<float>(right.y()),
            static_cast<float>(right.z()) };

        // Taken from settings rather than decomposed out of the projection matrix: OpenMW's projection is
        // reversed-Z, so getPerspective would either fail or hand back a plausible-looking lie.
        const float fov = static_cast<float>(Settings::camera().mFieldOfView);
        const float nearClip = static_cast<float>(Settings::camera().mNearClip);
        const float farClip = static_cast<float>(Settings::camera().mViewingDistance);

        mRuntime.setupCameraParameterized(eyeF, forwardF, upF, rightF, fov, aspect, nearClip, farClip);

        // Kept so the loops that present without submitting can repeat this camera. See resubmitCamera.
        std::copy(std::begin(eyeF), std::end(eyeF), std::begin(mLastCameraEye));
        std::copy(std::begin(forwardF), std::end(forwardF), std::begin(mLastCameraForward));
        std::copy(std::begin(upF), std::end(upF), std::begin(mLastCameraUp));
        std::copy(std::begin(rightF), std::end(rightF), std::begin(mLastCameraRight));
        mLastCameraFov = fov;
        mLastCameraAspect = aspect;
        mLastCameraNear = nearClip;
        mLastCameraFar = farClip;
        mHaveLastCamera = true;

        // Off by default now that it has served its purpose. It answered the question it existed for --
        // whether the mesh, material, instance and camera path worked, separately from the world
        // transforms -- and the answer was yes, which localised the fault to the material's alpha test.
        // It is a strongly emissive panel a short distance from the eye, so leaving it on floods the
        // whole scene with magenta bounce light.
        if (envFlag("OPENMW_REMIX_PROBE_QUAD", Settings::remix().mProbeQuad))
        {
            const double eyeD[3] = { eye.x(), eye.y(), eye.z() };
            const double forwardD[3] = { forward.x(), forward.y(), forward.z() };
            const double upD[3] = { up.x(), up.y(), up.z() };
            const double rightD[3] = { right.x(), right.y(), right.z() };
            submitProbeQuad(eyeD, forwardD, upD, rightD);
        }

        // Everything OpenMW draws that is not part of the world it wants path traced. GUI is composited
        // by OpenMW itself; render-to-texture subgraphs are inputs to effects rather than scene content,
        // and submitting them would place their geometry in the world twice.
        // Mask_SimpleWater is excluded as well: it is a second, flat-shaded copy of the water quad that
        // exists only for the local map, so submitting it puts a million-unit duplicate surface in the
        // world coincident with the real one.
        //
        // Mask_FirstPerson is deliberately NOT excluded. It was, without a stated reason, and the effect
        // was that the player's own arms and weapon never reached the path tracer at all -- invisible in
        // first person while everything else rendered. The viewmodel is ordinary skinned geometry with
        // ordinary world transforms as far as this traversal is concerned.
        //
        // Worth knowing if it looks wrong rather than missing: OpenMW draws the viewmodel through a
        // separate camera with its own narrower projection so it cannot clip into walls. Nothing here
        // reproduces that, so the geometry is placed by its world transform like any other actor. If it
        // intersects nearby scenery, that projection difference is the reason, not the transform.
        const unsigned int skipMask = Mask_GUI | Mask_RenderToTexture | Mask_Debug | Mask_SimpleWater;

        // Frustum culling, and the clamp warning below is what asked for it: this traversal submitted every
        // instance in the loaded world every frame, roughly 12,000 outdoors against 860 indoors. That cost
        // lands twice -- CPU here building and handing over instances, GPU in the runtime building and
        // tracing them -- and it is the only reason the instance ceiling was ever reachable.
        //
        // Built from the same camera that is handed to the runtime, so no disagreement between the two can
        // cull something that is on screen.
        //
        // Near and far planes are left out deliberately. Far would impose a view distance this traversal has
        // no business choosing, since OpenMW already limits it; near would drop geometry pressed against the
        // camera, which is exactly where the first-person viewmodel sits.
        static const bool cullEnabled = envFlag("OPENMW_REMIX_CULL", Settings::remix().mCull);

        osg::Polytope frustum;
        if (cullEnabled)
        {
            frustum.setToUnitFrustum(false, false);
            frustum.transformProvidingInverse(camera.getViewMatrix() * camera.getProjectionMatrix());
        }

        SubmitVisitor visitor(*this, skipMask,
            osg::Vec3f(static_cast<float>(eye.x()), static_cast<float>(eye.y()),
                static_cast<float>(eye.z())),
            osg::Vec3f(static_cast<float>(right.x()), static_cast<float>(right.y()),
                static_cast<float>(right.z())),
            osg::Vec3f(static_cast<float>(up.x()), static_cast<float>(up.y()), static_cast<float>(up.z())),
            frustum);
        // The state set controllers evaluated in pushState read their time from this. An AutoPlay
        // controller's source is a SceneUtil::FrameTimeSource, which takes simulation time straight off the
        // visitor's frame stamp, so without one there is nothing to evaluate them against.
        visitor.setFrameStamp(frameStamp);
        const auto traversalStart = std::chrono::steady_clock::now();
        sceneRoot->accept(visitor);
        const auto flushStart = std::chrono::steady_clock::now();
        // Everything the traversal queued is handed over here, nearest-first and under the triangle budget.
        // Must run before the counters below are read: mPrimitivesSubmitted and mInstancesOverBudget are
        // decided by the admission pass, not by the traversal.
        flushSubmissions();
        const auto flushEnd = std::chrono::steady_clock::now();
        mLastInstanceCount = visitor.instances();
        mCulled = visitor.culled();

        // Keep the worst submit of the window together with the counters from the same frame. Recording
        // them separately would pair a spike with whichever frame's counts happened to be logged next,
        // which is the mistake that had an opacity micromap message blamed for a stall earlier.
        {
            const double traversalMs
                = std::chrono::duration<double, std::milli>(flushStart - traversalStart).count();
            const double flushMs = std::chrono::duration<double, std::milli>(flushEnd - flushStart).count();
            if (traversalMs + flushMs > mWorstSubmit.mTotalMs)
            {
                mWorstSubmit.mTotalMs = traversalMs + flushMs;
                mWorstSubmit.mTraversalMs = traversalMs;
                mWorstSubmit.mFlushMs = flushMs;
                mWorstSubmit.mCompositeEncodeMs = mCompositeEncodeMs;
                mWorstSubmit.mInstances = mLastInstanceCount;
                mWorstSubmit.mMeshesCreated = mMeshesCreated;
                mWorstSubmit.mPrimitives = mPrimitivesSubmitted;
                mWorstSubmit.mCulled = mCulled;
                mWorstSubmit.mFrame = mFrame;
            }
        }

        if (visitor.clamped() && !mLoggedClamp)
        {
            mLoggedClamp = true;
            Log(Debug::Warning) << "Remix scene: hit the " << kMaxInstancesPerFrame
                                << " instance ceiling, so the submitted scene is incomplete. There is no "
                                   "culling in this traversal yet and the viewing distance is "
                                << farClip << " units.";
        }

        evictStaleMeshes();
        releaseStaleLights();
        releaseStaleParticleMeshes();
        pruneAnimatedStates();
        // Runs after the traversal, never during it. A destroy issued mid-traversal could be followed by a
        // create of the same hash in the same frame, which the runtime's tombstone set suppresses rather
        // than honours -- that is why the previous attempt at this turned surfaces permanently white.
        releaseOrphanedTextures();

        // Log the first submission, and then again whenever the scene goes from empty to populated or
        // back. A one-shot on frame 1 was actively misleading: the first frame happens before the world
        // is up, so it reported zero instances and then never spoke again, which read as "the traversal
        // never works" when it only meant "there was nothing there yet".
        // Also on a slow interval, not only on the populated/unpopulated edge. The edge alone meant this
        // reported the main menu -- one instance, one mesh -- and then never again for the whole session,
        // so the counts that actually matter were never visible.
        const bool populated = mLastInstanceCount > 0;
        static const std::uint64_t logInterval = sceneLogInterval();
        if (!mLoggedFirstSubmit || populated != mWasPopulated || mFrame % logInterval == 0)
        {
            mLoggedFirstSubmit = true;
            mWasPopulated = populated;
            Log(Debug::Info) << "Remix scene: handed over " << mLastInstanceCount << " instances from "
                             << mMeshes.size() << " cached meshes, " << mMeshesCreated
                             << " BUILT THIS FRAME (" << mMeshesShared
                             << " geometries shared an existing mesh, " << mMeshBuildsDeferred
                             << " builds deferred to a later frame), " << mLastLightCount << " lights and "
                             << mTexturesUploaded << " textures (" << mTexturesShared
                             << " uploads avoided, content already present); " << mSkinnedInstances
                             << " instances were skinned, " << mCulled
                             << " drawables outside the frustum were culled (" << mMeshesRetained
                             << " of them held resident rather than left to expire), and " << mSkinnedDropped
                             << " skins were not ready; " << mLastParticleCount << " particles from "
                             << mParticleMeshes.size() << " systems"
                             // Bytes, not counts. A count could not be weighed against video memory at all:
                             // one 2048-square terrain composite is 16.8 MB and an icon is a few kilobytes,
                             // and both moved this number by one. Resident is what the runtime is holding
                             // now; peak is the high-water mark, so the two diverging is the proof that
                             // release is working rather than merely that growth has slowed.
                             << "; textures resident " << (mTextureBytesResident >> 20) << " MiB (peak "
                             << (mTextureBytesPeak >> 20) << " MiB) over " << mUploadedTextures.size()
                             << " identities, " << mTexturesReleased << " released and " << mMaterialsReleased
                             << " materials with them"
                             // The number that decides whether the runtime's NEE cache prefix-sum index
                             // holds. Note the real ceiling is 16,777,214 and not the 67,108,863 the runtime
                             // logs -- see kPrimitiveBudget for why. Instances turned away being non-zero
                             // means the budget is binding, and since admission is nearest-first what is
                             // missing is the far field; it staying zero means the scene fits and this cost
                             // nothing.
                             << "; " << mPrimitivesSubmitted << " triangles submitted of "
                             << primitiveBudget() << " budgeted, " << mInstancesOverBudget
                             << " instances turned away"
                             << "; composite encode " << mCompositeEncodeMs << " ms, "
                             << mCompositesDeferred << " deferred to a later frame"
                             << "; camera eye " << eye.x() << ", " << eye.y()
                             << ", " << eye.z() << " looking " << forward.x() << ", " << forward.y()
                             << ", " << forward.z() << " up " << up.x() << ", " << up.y() << ", "
                             << up.z() << " fov " << fov << " near " << nearClip << " far " << farClip
                             << (populated ? "" : " -- nothing found, so Remix has nothing to raytrace");
            // The spike, not the average. See WorstSubmit.
            Log(Debug::Info) << "Remix scene: worst submit since the last report " << mWorstSubmit.mTotalMs
                             << " ms on frame " << mWorstSubmit.mFrame << " -- traversal "
                             << mWorstSubmit.mTraversalMs << " ms, hand-over and mesh builds "
                             << mWorstSubmit.mFlushMs << " ms (composite encode "
                             << mWorstSubmit.mCompositeEncodeMs << " ms of that); " << mWorstSubmit.mInstances
                             << " instances, " << mWorstSubmit.mMeshesCreated << " meshes built, "
                             << mWorstSubmit.mPrimitives << " triangles, " << mWorstSubmit.mCulled
                             << " culled";
            mWorstSubmit = WorstSubmit{};
            if (mHaveExtent)
            {
                // The camera sits inside this box when geometry really is around the player. If it does
                // not, the instance transforms are wrong and no amount of lighting or material work will
                // put anything on screen.
                Log(Debug::Info) << "Remix scene: instance origins span " << mMin[0] << ".." << mMax[0]
                                 << ", " << mMin[1] << ".." << mMax[1] << ", " << mMin[2] << ".."
                                 << mMax[2];
            }

            // The runtime's own VRAM accounting, per category, on the same interval as the report above.
            //
            // A separate line rather than more fields on that one: it answers a different question, it
            // wants grepping on its own, and the scene line is long enough already.
            //
            // Every field is measured by the runtime's allocator, not estimated here, and the categories
            // are worth more read against each other than summed:
            //
            // - `accel` is the BVH. It is the only number that reflects geometry the *runtime* substituted:
            //   a replacement swaps a mesh for a heavier one without moving the triangle count this host
            //   reports, so `accel` rising while `triangles submitted` holds steady localises the growth to
            //   the replacement pack.
            // - `material textures` is the pool `rtx.texturemanager.fixedBudgetMiB` bounds. It does *not*
            //   include what this host uploads through createTexture -- those are reported as `textures
            //   resident` on the line above, are a separate pool, and cannot be demoted, only released.
            // - `retained` is the allocator holding freed chunks rather than returning them to the driver.
            //   Retention, not consumption. Reading a rising total without this field is how a
            //   high-water-mark allocator gets mistaken for a leak.
            // - `driver` minus `allocated` is everything outside the runtime's allocator: DLSS and NGX
            //   working memory, raytracing pipeline state, bindless descriptor pools, NRC. No other number
            //   here can see it, and it is not small.
            RemixRT::Runtime::VramStats vram;
            if (mRuntime.vramStats(vram))
            {
                const auto mib = [](unsigned long long bytes) { return bytes >> 20; };
                Log(Debug::Info) << "Remix VRAM: driver " << mib(vram.mDriverAllocated) << " of "
                                 << mib(vram.mDriverBudget) << " MiB budget; runtime allocator "
                                 << mib(vram.mTotalAllocated) << " allocated, " << mib(vram.mTotalUsed)
                                 << " used, " << mib(vram.mPoolRetained) << " retained; accel "
                                 << mib(vram.mAccelerationStructure) << ", replacement geometry "
                                 << mib(vram.mReplacementGeometry) << ", material textures "
                                 << mib(vram.mMaterialTextures) << ", buffers " << mib(vram.mBuffers)
                                 << ", render targets " << mib(vram.mRenderTargets)
                                 << ", opacity micromap " << mib(vram.mOpacityMicromap)
                                 << " MiB; runtime texture cache " << vram.mTextureCacheCount
                                 << " entries";
            }
        }

        return mLastInstanceCount;
    }

    unsigned long long RemixScene::submitGeometry(
        osg::Geometry& geometry, const SurfaceState& surface, const SceneUtil::RigGeometry* rig)
    {
        mLastMeshBonesPerVertex = 0;
        return meshFor(geometry, materialFor(surface), surface, rig);
    }

    unsigned long long RemixScene::roughnessTextureFor(const osg::Image& specular)
    {
        if (auto found = mDerivedRoughness.find(&specular); found != mDerivedRoughness.end())
            return found->second != nullptr ? textureFor(*found->second, false) : 0;

        // Only uncompressed sources are converted, because the conversion is per-texel and this code does
        // not decode block formats. A specular map is small and mods ship them uncompressed far more often
        // than not; a compressed one simply keeps the constant roughness from the surface rule, which is
        // the behaviour that existed before any of this. Recorded as a null entry so the same image is not
        // examined again every frame.
        const GLenum pixelFormat = specular.getPixelFormat();
        const bool convertible = specular.getDataType() == GL_UNSIGNED_BYTE
            && (pixelFormat == GL_RGBA || pixelFormat == GL_BGRA)
            && specular.data() != nullptr && specular.s() > 0 && specular.t() > 0;
        if (!convertible)
        {
            mDerivedRoughness.emplace(&specular, osg::ref_ptr<osg::Image>());
            return 0;
        }

        osg::ref_ptr<osg::Image> derived = new osg::Image;
        derived->allocateImage(specular.s(), specular.t(), 1, GL_RGBA, GL_UNSIGNED_BYTE);
        derived->setInternalTextureFormat(GL_RGBA8);
        // Named so the surface-rule classifier and any log line have something meaningful to show; the
        // suffix keeps it from colliding with the source in a rule match.
        derived->setFileName(specular.getFileName() + "@roughness");

        const unsigned char* src = specular.data();
        unsigned char* dst = derived->data();
        const int texels = specular.s() * specular.t();
        // Alpha is the last byte in both RGBA and BGRA, which is the only channel read here, so the two
        // formats need no separate handling.
        for (int i = 0; i < texels; ++i)
        {
            // objects.frag reads a normalised alpha and multiplies by 255 to recover the exponent, so the
            // stored byte is already the shininess and no rescaling is needed.
            const float shininess = static_cast<float>(src[i * 4 + 3]);
            // Blinn-Phong exponent to GGX roughness. Clamped away from zero so a fully smooth texel does
            // not produce a perfectly specular surface, which reads as a mirror and fireflies badly.
            const float roughness = std::clamp(std::sqrt(2.0f / (shininess + 2.0f)), 0.03f, 1.0f);
            const auto value = static_cast<unsigned char>(std::lround(roughness * 255.0f));
            dst[i * 4 + 0] = value; // red: the channel the runtime samples
            dst[i * 4 + 1] = value;
            dst[i * 4 + 2] = value;
            dst[i * 4 + 3] = 255;
        }

        const osg::Image* stored = mDerivedRoughness.emplace(&specular, derived).first->second.get();
        return textureFor(*stored, false);
    }

    unsigned long long RemixScene::heightTextureFor(const osg::Image& normalHeight)
    {
        if (auto found = mDerivedHeight.find(&normalHeight); found != mDerivedHeight.end())
            return found->second != nullptr ? textureFor(*found->second, false) : 0;

        // Uncompressed only, for the same reason as the roughness conversion: this is per-texel and does not
        // decode block formats. A compressed normal-height map keeps its normal and simply gets no
        // displacement, which is the behaviour that existed before this. Cached as null so the same image is
        // not examined again every frame.
        const GLenum pixelFormat = normalHeight.getPixelFormat();
        const bool convertible = normalHeight.getDataType() == GL_UNSIGNED_BYTE
            && (pixelFormat == GL_RGBA || pixelFormat == GL_BGRA)
            && normalHeight.data() != nullptr && normalHeight.s() > 0 && normalHeight.t() > 0;
        if (!convertible)
        {
            mDerivedHeight.emplace(&normalHeight, osg::ref_ptr<osg::Image>());
            return 0;
        }

        osg::ref_ptr<osg::Image> derived = new osg::Image;
        derived->allocateImage(normalHeight.s(), normalHeight.t(), 1, GL_RGBA, GL_UNSIGNED_BYTE);
        derived->setInternalTextureFormat(GL_RGBA8);
        // Suffixed so it cannot collide with the source in a surface-rule match, and so a log line naming it
        // is recognisable as derived rather than as an asset on disk.
        derived->setFileName(normalHeight.getFileName() + "@height");

        const unsigned char* src = normalHeight.data();
        unsigned char* dst = derived->data();
        const int texels = normalHeight.s() * normalHeight.t();
        // Alpha is the last byte in both RGBA and BGRA, so the two formats need no separate handling.
        //
        // Copied straight across with no inversion or rescaling. The runtime treats 1.0 as the outer surface
        // with lower values displacing inward (rtx_terrain_baker.cpp: "a value of 1.f will have 0
        // displacement"), and OpenMW's height alpha is high where the surface is raised. Those agree.
        for (int i = 0; i < texels; ++i)
        {
            const unsigned char height = src[i * 4 + 3];
            dst[i * 4 + 0] = height; // red: the channel the runtime samples
            dst[i * 4 + 1] = height;
            dst[i * 4 + 2] = height;
            dst[i * 4 + 3] = 255;
        }

        const osg::Image* stored = mDerivedHeight.emplace(&normalHeight, derived).first->second.get();
        return textureFor(*stored, false);
    }

    const osg::Image* RemixScene::spriteSheetFor(const std::vector<osg::ref_ptr<osg::Texture2D>>& frames)
    {
        if (frames.size() < 2 || frames.size() > 255 || frames.front() == nullptr)
            return nullptr;

        const osg::Image* first = frames.front()->getImage();
        if (first == nullptr)
            return nullptr;

        if (auto found = mSpriteSheets.find(first); found != mSpriteSheets.end())
            return found->second.get();

        // Every refusal below is cached as a null entry, so a sequence this cannot assemble is examined once
        // rather than on every draw for the rest of the session.
        const auto refuse = [&]() -> const osg::Image* {
            mSpriteSheets.emplace(first, osg::ref_ptr<osg::Image>());
            return nullptr;
        };

        const int frameWidth = first->s();
        const int frameHeight = first->t();
        const int count = static_cast<int>(frames.size());
        const unsigned int internalFormat = first->getInternalTextureFormat();
        const GLenum pixelFormat = first->getPixelFormat();

        // 1 x N, so the sheet is as wide as every frame laid side by side. Refused past 8192 because that is
        // the width a sheet of many frames runs into first, and a texture the driver refuses to create is a
        // worse outcome than leaving the flipbook on the per-frame path.
        if (frameWidth <= 0 || frameHeight <= 0 || frameWidth * count > 8192)
            return refuse();

        const CompressedFormat* compressed = compressedFormatFor(internalFormat);
        const bool uncompressed = compressed == nullptr && first->getDataType() == GL_UNSIGNED_BYTE
            && (pixelFormat == GL_RGBA || pixelFormat == GL_BGRA);
        if (compressed == nullptr && !uncompressed)
            return refuse();
        // Block formats can only be concatenated on block boundaries. Every BC texture is a multiple of four
        // in practice, but a sequence that is not would be silently sheared by the copy below.
        if (compressed != nullptr && (frameWidth % 4 != 0 || frameHeight % 4 != 0))
            return refuse();

        // Every frame has to agree, because the copy derives its offsets from the first frame alone.
        for (const osg::ref_ptr<osg::Texture2D>& texture : frames)
        {
            const osg::Image* image = texture != nullptr ? texture->getImage() : nullptr;
            if (image == nullptr || image->data() == nullptr || image->s() != frameWidth
                || image->t() != frameHeight
                // getInternalTextureFormat returns GLint; the local is unsigned because that is what
                // compressedFormatFor takes. Cast rather than compare across the signedness.
                || static_cast<unsigned int>(image->getInternalTextureFormat()) != internalFormat
                || image->getPixelFormat() != pixelFormat)
                return refuse();
        }

        osg::ref_ptr<osg::Image> atlas = new osg::Image;
        const int atlasWidth = frameWidth * count;

        if (compressed != nullptr)
        {
            // Block copy. Each 4x4 block is self-contained in BC1/BC2/BC3, so a frame's block rows land
            // whole at a column offset and nothing is recompressed.
            const std::size_t blockBytes = static_cast<std::size_t>(compressed->mBlockBytes);
            const std::size_t frameBlocksX = static_cast<std::size_t>(frameWidth) / 4u;
            const std::size_t blockRows = static_cast<std::size_t>(frameHeight) / 4u;
            const std::size_t atlasBlocksX = frameBlocksX * static_cast<std::size_t>(count);

            auto* data = new unsigned char[atlasBlocksX * blockRows * blockBytes];
            for (int frame = 0; frame < count; ++frame)
            {
                const unsigned char* src = frames[frame]->getImage()->data();
                for (std::size_t row = 0; row < blockRows; ++row)
                {
                    const std::size_t dstBlock
                        = row * atlasBlocksX + static_cast<std::size_t>(frame) * frameBlocksX;
                    std::memcpy(data + dstBlock * blockBytes, src + row * frameBlocksX * blockBytes,
                        frameBlocksX * blockBytes);
                }
            }
            atlas->setImage(atlasWidth, frameHeight, 1, internalFormat, pixelFormat, GL_UNSIGNED_BYTE, data,
                osg::Image::USE_NEW_DELETE);
        }
        else
        {
            const std::size_t rowBytes = static_cast<std::size_t>(frameWidth) * 4u;
            const std::size_t atlasRowBytes = rowBytes * static_cast<std::size_t>(count);

            auto* data = new unsigned char[atlasRowBytes * static_cast<std::size_t>(frameHeight)];
            for (int frame = 0; frame < count; ++frame)
            {
                const unsigned char* src = frames[frame]->getImage()->data();
                for (int row = 0; row < frameHeight; ++row)
                {
                    std::memcpy(data + static_cast<std::size_t>(row) * atlasRowBytes
                            + static_cast<std::size_t>(frame) * rowBytes,
                        src + static_cast<std::size_t>(row) * rowBytes, rowBytes);
                }
            }
            atlas->setImage(atlasWidth, frameHeight, 1, internalFormat, pixelFormat, GL_UNSIGNED_BYTE, data,
                osg::Image::USE_NEW_DELETE);
        }

        // Keeps the first frame's path with a suffix, so both the surface-rule classifier and the
        // non-surface classifier -- substring matches, both of them -- reach the same verdict they would for
        // a single frame. That is what keeps an omw_distortion sequence dropped instead of atlased and drawn.
        atlas->setFileName(first->getFileName() + "@sprites");

        return mSpriteSheets.emplace(first, atlas).first->second.get();
    }

    unsigned long long RemixScene::textureFor(const osg::Image& image, bool colour)
    {
        // Keyed on the image *and* the colour interpretation. The same bytes uploaded as sRGB and as
        // linear are two different textures to the runtime, and one image used both ways -- a diffuse map
        // that some other mesh binds as a normal map -- would otherwise silently get whichever
        // interpretation was asked for first.
        const unsigned long long key
            = (reinterpret_cast<unsigned long long>(&image) << 1) | (colour ? 1ull : 0ull);
        if (auto found = mTextures.find(key); found != mTextures.end())
        {
            // A hit is only trustworthy if this entry was built from the image in hand. The key is the
            // image's address, and OpenMW reuses addresses once it frees an image, so a hit can be a new
            // image wearing a dead one's address -- which would silently return the previous image's
            // texture. Nothing caught this before because nothing was ever released, so the dead entry
            // stayed correct for as long as it existed.
            osg::ref_ptr<const osg::Image> alive;
            if (!found->second.mImage.lock(alive) || alive.get() != &image)
            {
                // Give up this entry's claim on the identity, but do not destroy anything here. The sweep at
                // the end of the frame destroys identities nothing claims, which matters: if this new image
                // holds the same content, the branch below re-claims the identity and the destroy correctly
                // never happens. Destroying mid-frame and re-creating the same hash immediately afterwards
                // is the pattern the runtime's tombstone set turns into a permanently missing texture.
                releaseTextureClaim(found->second);
                mTextures.erase(found);
            }
            else
            {
                // Stamped on every hit, so "unused" means no drawable resolved this texture recently rather
                // than merely that it was uploaded a while ago. This runs per drawable per frame, hence the
                // stamp going inside the existing lookup rather than adding a second one.
                found->second.mLastUsedFrame = mFrame;
                return found->second.mUsable ? found->second.mHash : 0ull;
            }
        }

        // Cached even on failure, so an unsupported format is diagnosed once rather than per drawable
        // per frame. Written before every early return below.
        CachedTexture cached;
        // Stamped here so it applies to every path below, including the negative results cached for
        // unusable formats. A texture uploaded this frame must not be eligible for eviction immediately.
        cached.mLastUsedFrame = mFrame;
        // Recorded on every path for the same reason, and observed rather than held so that caching an
        // image does not extend its life. This is what release is decided on: OpenMW freeing the image is
        // the only dependable evidence that the entry is garbage.
        cached.mImage = &image;

        const int width = image.s();
        const int height = image.t();
        const unsigned char* data = image.data();

        if (width <= 0 || height <= 0 || data == nullptr)
        {
            mTextures.emplace(key, cached);
            return 0;
        }

        std::vector<unsigned char> converted;
        const void* uploadData = nullptr;
        unsigned long long uploadSize = 0;
        // Bytes of mip level 0 alone. Tracked only to test whether Remix's own D3D9 texture hash can be
        // reproduced here: that hash is XXH3 over the staging buffer of subresource 0, which is mip 0 and
        // nothing else -- no dimensions, no format, no mip chain. Zero when the branch taken cannot say.
        unsigned long long mip0Size = 0;
        // Bytes the texture's identity is hashed over, which is not necessarily everything uploaded.
        //
        // Kept separate from uploadSize so that appending a mip chain cannot move an identity. The fallback
        // hash below used to read uploadSize directly, which was the same number until mips were generated
        // for the uncompressed path -- at which point every tag in rtx.conf and every replacement authored
        // against one of those textures would have gone dead, silently, for a change that adds nothing to
        // level 0. Each branch sets this to exactly the span it hashed before.
        unsigned long long identityBytes = 0;
        unsigned int mipLevels = 1;
        RemixRT::Runtime::TextureFormat format = RemixRT::Runtime::Format_RGBA8;

        const unsigned int internalFormat = static_cast<unsigned int>(image.getInternalTextureFormat());
        const unsigned int pixelFormat = static_cast<unsigned int>(image.getPixelFormat());
        const unsigned int dataType = static_cast<unsigned int>(image.getDataType());

        if (image.isCompressed())
        {
            // Passed through untouched. The block layouts are identical to the Vulkan BC formats, so
            // decompressing to upload would cost time and quality for nothing. Mip chains come through
            // as well: OSG stores them contiguously, largest first, which is exactly what the upload
            // expects -- and mips matter more here than usual, since without them minification has
            // nothing to fall back on and textured surfaces shimmer at distance.
            const CompressedFormat* compressed = compressedFormatFor(internalFormat);
            if (compressed == nullptr)
            {
                Log(Debug::Warning) << "Remix scene: no Remix format for compressed GL internal format 0x"
                                    << std::hex << internalFormat << std::dec
                                    << "; using the untextured material for it. Add it to "
                                       "kCompressedFormats if the runtime has a format for it.";
                mTextures.emplace(key, cached);
                return 0;
            }
            format = colour ? compressed->mColour : compressed->mLinear;
            cached.mFormat = compressed->mName;

            // Every byte count below is derived from the format and the extent, never taken from OSG, and
            // this is a correctness requirement rather than a preference.
            //
            // CreateTexture memcpys exactly the size it is given into a staging buffer sized from that
            // same number, so an overstated value walks the copy off the end of the source mapping and
            // takes the runtime down inside memcpy, with nothing on the stack to say which texture did
            // it. And OSG's own size accessors cannot be used for the check: for a compressed image with
            // no mip chain, getTotalSizeInBytesIncludingMipmaps() reports the *uncompressed* size, which
            // for a 256x256 BC3 texture is 262144 bytes against a real 65536. That 4:1 overstatement is
            // what was pushing textures onto the base-level-only path for no reason.
            const unsigned long long blockBytes = compressed->mBlockBytes;
            const auto levelBytes = [&](unsigned int level) -> unsigned long long {
                const unsigned long long levelWidth = std::max(1, width >> level);
                const unsigned long long levelHeight = std::max(1, height >> level);
                return ((levelWidth + 3ull) / 4ull) * ((levelHeight + 3ull) / 4ull) * blockBytes;
            };

            const unsigned int osgLevels
                = std::max(1u, static_cast<unsigned int>(image.getNumMipmapLevels()));

            // OSG's mipmap *offsets* are trustworthy where its sizes are not: they come from the loader
            // walking the file, not from a pixel-size calculation. Checking the computed layout against
            // them catches a chain that is not contiguous-largest-first, which is the one assumption the
            // upload makes about the memory it is handed.
            unsigned long long chainBytes = levelBytes(0);
            unsigned int usableLevels = 1;
            for (unsigned int level = 1; level < osgLevels; ++level)
            {
                if (static_cast<unsigned long long>(image.getMipmapOffset(level)) != chainBytes)
                {
                    Log(Debug::Verbose)
                        << "Remix scene: mip level " << level << " of a " << width << "x" << height << " "
                        << compressed->mName << " texture starts at " << image.getMipmapOffset(level)
                        << " where the format implies " << chainBytes
                        << "; uploading the levels up to that point only";
                    break;
                }
                chainBytes += levelBytes(level);
                ++usableLevels;
            }

            mipLevels = usableLevels;
            uploadSize = chainBytes;
            uploadData = data;
            // Largest-first and contiguous, verified just above, so mip 0 is the leading levelBytes(0).
            mip0Size = levelBytes(0);
            // Whole chain, which is what this path has always hashed.
            identityBytes = chainBytes;
        }
        else if (dataType == kGlUnsignedByte)
        {
            // Everything uncompressed is widened to RGBA8 rather than matching the source layout. Remix
            // accepts only four-channel 8-bit formats, so three- and one-channel sources have to be
            // expanded anyway, and having one path for all of them removes a class of channel-order bug
            // that shows up as a blue-for-red swap nobody notices until a screenshot.
            int red = -1;
            int green = -1;
            int blue = -1;
            int alpha = -1;
            int channels = 0;
            switch (pixelFormat)
            {
                case kGlRgb:
                    red = 0, green = 1, blue = 2, channels = 3;
                    break;
                case kGlRgba:
                    red = 0, green = 1, blue = 2, alpha = 3, channels = 4;
                    break;
                case kGlBgr:
                    red = 2, green = 1, blue = 0, channels = 3;
                    break;
                case kGlBgra:
                    red = 2, green = 1, blue = 0, alpha = 3, channels = 4;
                    break;
                case kGlLuminance:
                case kGlRed:
                    red = 0, green = 0, blue = 0, channels = 1;
                    break;
                case kGlLuminanceAlpha:
                case kGlRg:
                    red = 0, green = 0, blue = 0, alpha = 1, channels = 2;
                    break;
                case kGlAlpha:
                    // Alpha-only: white with the source as the cutout. Treating the single channel as
                    // luminance instead would render these as black shapes.
                    alpha = 0, channels = 1;
                    break;
                default:
                    Log(Debug::Verbose) << "Remix scene: unsupported texture pixel format 0x" << std::hex
                                        << pixelFormat << std::dec
                                        << "; using the untextured material for it";
                    mTextures.emplace(key, cached);
                    return 0;
            }

            const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
            // Only mip 0 is converted. OSG's uncompressed mip chain would have to be widened level by
            // level, and OpenMW's uncompressed textures are the minority; compressed ones, which are the
            // majority and the large ones, keep their mips above.
            const std::size_t available = static_cast<std::size_t>(image.getImageSizeInBytes());
            if (available < pixels * static_cast<std::size_t>(channels))
            {
                mTextures.emplace(key, cached);
                return 0;
            }

            converted.resize(pixels * 4);
            for (std::size_t i = 0; i < pixels; ++i)
            {
                const unsigned char* source = data + i * static_cast<std::size_t>(channels);
                unsigned char* destination = converted.data() + i * 4;
                destination[0] = red >= 0 ? source[red] : 0xFF;
                destination[1] = green >= 0 ? source[green] : 0xFF;
                destination[2] = blue >= 0 ? source[blue] : 0xFF;
                destination[3] = alpha >= 0 ? source[alpha] : 0xFF;
            }

            uploadData = converted.data();
            uploadSize = converted.size();
            format = colour ? RemixRT::Runtime::Format_RGBA8 : RemixRT::Runtime::Format_RGBA8_Linear;
            // Distinguished in the log, because a source with no alpha channel gets 255 written into it
            // and any cutout against the result is a no-op.
            cached.mFormat = alpha >= 0 ? "RGBA8" : "RGBA8(opaque)";
            // Level 0 alone, taken before anything is appended to it. This is the span this path has always
            // hashed, so recording it here is what keeps the identity fixed across the mip generation below.
            identityBytes = converted.size();

            // Terrain composites, and only terrain composites, are block-compressed and given a mip chain.
            //
            // They are the one class of texture here that is both very large and generated rather than
            // loaded: a whole chunk's blended albedo read back from a render target at Terrain/"composite
            // map resolution". At 2048 that is 16.8 MB each, and an exterior session composites hundreds of
            // them, which makes this the largest single call on texture memory in the game. BC1 is a 6x
            // reduction and is what Morrowind's own ground textures already are, so it asks nothing of this
            // art that it did not already accept.
            //
            // The mip chain matters twice over, and the second reason is the less obvious one. Without it
            // minification has nothing to fall back on and the ground shimmers. But a single-mip texture
            // also cannot be partially resident, so the runtime's texture manager has no way to demote
            // these under pressure -- its only option is to evict them whole, which is what showed up as
            // ground textures dropping out after walking for a while. Even with the compression this is
            // the half that makes the memory behaviour graceful rather than cliff-edged.
            //
            // Restricted to composites deliberately, and the restriction is not conservatism. The hash
            // computed below is the texture's public identity: it is what an rtx.conf texture tag is
            // stored under and what a replacement pack is authored against. Changing the bytes of anything
            // that came from a file on disk would silently invalidate both, with no diagnostic. A composite
            // has no file, cannot be tagged and cannot be replaced, so it is the one case where re-encoding
            // costs nothing downstream.
            static const bool compressComposites
                = envFlag("OPENMW_REMIX_COMPOSITE_COMPRESS", Settings::remix().mCompositeCompress);
            if (compressComposites
                && std::string_view(image.getFileName()) == Terrain::CompositeMap::sReadbackImageName)
            {
                // Bounded per frame, because the arrival rate is bursty and the cost is not small.
                //
                // Terrain composites are generated as chunks come into view, and measured bursts reach 82
                // in a single second. Encoding all of them the moment they appear put 678 ms into one
                // frame's scene submit, which the frame breakdown identified as the stutter. The work
                // itself is necessary and its result is cached, so the only thing wrong with it is doing
                // an unbounded amount of it between two presents.
                //
                // Over budget, this returns without caching anything. The caller falls back to the default
                // material for that chunk and the composite is retried next frame, so a burst fills in
                // over several frames instead of stopping the game. Retry is safe precisely because
                // nothing was cached: the identity is derived from the encoded bytes, so a deferred
                // composite has no half-built state to reconcile.
                // Off by default, because the visible cost is worse than the cost it saves.
                //
                // A budget of 4 ms did what it was meant to: composite encode per frame fell from 678 ms to
                // about 6, and the worst scene submit in a window went from 106-678 ms down to 43-142. But a
                // deferred composite leaves its chunk on the default material until a later frame, and in
                // motion that reads as ground squares flashing white for a second or two. For a showcase
                // that is a worse defect than the stutter it removes.
                //
                // The encoder rewrite is what actually mattered: one parallel region per composite instead
                // of one per mip level took the per-composite cost from about 7 ms to under 2, which is a
                // real reduction with no visual cost at all. That is kept; the deferral is not.
                //
                // Left in and tunable rather than deleted, because it is the right mechanism with the wrong
                // fallback. Given a way to keep the previous composite for a chunk, or to encode off the
                // frame thread, deferring becomes invisible and this becomes worth switching on.
                static const double budgetMs
                    = envMilliseconds("OPENMW_REMIX_COMPOSITE_BUDGET_MS", Settings::remix().mCompositeBudgetMs);

                if (budgetMs > 0.0 && mCompositeEncodeMs >= budgetMs)
                {
                    ++mCompositesDeferred;
                    return 0;
                }

                const auto encodeStart = std::chrono::steady_clock::now();

                std::vector<unsigned char> compressed;
                // The whole chain is four thirds of level zero plus the sub-block tail, and one allocation
                // beats eleven reallocations while a frame is waiting on this.
                compressed.reserve(
                    static_cast<std::size_t>(bc1LevelBytes(width, height) * 4ull / 3ull + 64ull));
                mipLevels = compressBc1WithMips(converted.data(), width, height, compressed);
                converted = std::move(compressed);
                uploadData = converted.data();
                uploadSize = converted.size();
                format = colour ? RemixRT::Runtime::Format_BC1_RGB
                                : RemixRT::Runtime::Format_BC1_RGB_Linear;
                cached.mFormat = "BC1_RGB(composite)";

                mCompositeEncodeMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - encodeStart)
                                          .count();
                // Whole BC1 chain, which is what a composite has always hashed. A composite has no file
                // behind it and cannot be tagged or replaced, so its identity is free to be whatever is
                // convenient -- but changing it for no reason would still churn the upload cache.
                identityBytes = converted.size();
            }
            else
            {
                // A mip chain for everything else that arrived uncompressed.
                //
                // Left until here so it sees the RGBA8 result rather than the source layout, and so the
                // composite path above -- which builds its own chain while block-compressing -- is not made
                // to do the work twice.
                //
                // The chain costs a third more memory than level 0 and is built with a box filter, which is
                // cheap next to what it buys: minification has something to sample, and the runtime can
                // demote these under pressure instead of evicting them outright.
                mipLevels = appendRgba8Mips(converted, width, height, colour);
                uploadData = converted.data();
                uploadSize = converted.size();
            }
        }
        else
        {
            Log(Debug::Verbose) << "Remix scene: texture data type 0x" << std::hex << dataType << std::dec
                                << " is not 8-bit; using the untextured material for it";
            mTextures.emplace(key, cached);
            return 0;
        }

        // Identity from content, not from the address the image happens to live at.
        //
        // This value is the texture's public identity: it is what a USD replacement is authored against,
        // what a texture tag is stored under in rtx.conf, and what Remix's texture list displays. Deriving
        // it from &image made all three useless across a restart -- tags went dead, replacements could
        // never bind, and the same texture showed a different hash every run.
        //
        // Hashed over exactly the bytes handed to the runtime, with XXH3-64, which is deliberately the same
        // algorithm over the same kind of input that Remix applies to a D3D9 texture (XXH3_64bits over the
        // staging buffer of subresource 0). For a compressed texture those bytes are the DDS blocks
        // unaltered, so this has a real chance of agreeing with a hash captured from another host driving
        // the same art -- which is the prerequisite for reusing replacement packs authored elsewhere.
        //
        // Dimensions and the colour interpretation fold in afterwards because identical bytes can be two
        // legitimately different textures: one image bound as sRGB albedo and as a linear normal map must
        // not collapse to a single entry, which is the distinction the old cache key spent its low bit on.
        unsigned long long hash = 0;
        if (mPackMatching && colour && mip0Size != 0)
        {
            // Exactly what Remix computes for a game texture: XXH3 over mip 0 alone, nothing folded in.
            // See D3D9CommonTexture::SetupForRtxFrom, which hashes the staging buffer of subresource 0.
            //
            // Verified rather than assumed, twice over. Every one of 399 textures sampled from this log
            // reproduced exactly by hashing tightly packed mip 0 of the source .dds; and 177 of the 244
            // material keys in an MGE-XE capture of the census office reproduce the same way, which is what
            // establishes that this is the formula Remix uses for a D3D9 texture. Folding in dimensions or a
            // colour flag, as an earlier revision did, moves every value off it.
            //
            // Be careful what this does and does not buy, because an earlier note here overstated it. 378 of
            // 1826 of these hashes appear among the 1282 mat_ keys in the NVIDIA demo pack, so the textures
            // are recognisable. The materials are not: createTexturedMaterial is handed `key | 1` as the
            // material hash, and a capture therefore records mat_<that>, which shares nothing with the pack.
            // Remix's own D3D9 path keys a material on its albedo texture hash, which is why an MGE-XE
            // capture binds and ours does not. Nothing in the pack can attach until the material hash is the
            // albedo texture hash -- with the consequence that surfaces differing only in roughness, blend
            // or alpha test would collapse onto one material, which is exactly what Remix does for D3D9 but
            // is not what this code currently assumes.
            hash = RemixRT::AssetHash::bytes(uploadData, mip0Size);
        }
        else
        {
            // Not a candidate for matching, so identity only has to be stable and distinct.
            //
            // The linear uploads are normal maps, which a mat_ key does not address anyway, and the colour
            // flag is folded in deliberately: without it the same file uploaded as both sRGB albedo and a
            // linear normal map would collapse onto one texture, and whichever arrived second would win.
            // Nothing in Morrowind's art does that today, but the guarantee is cheap.
            hash = RemixRT::AssetHash::bytes(uploadData, identityBytes);
            hash = RemixRT::AssetHash::combine(static_cast<unsigned long long>(width), hash);
            hash = RemixRT::AssetHash::combine(static_cast<unsigned long long>(height), hash);
            hash = RemixRT::AssetHash::combine(static_cast<unsigned long long>(format), hash);
            hash = RemixRT::AssetHash::combine(colour ? 1ull : 0ull, hash);
        }
        hash = RemixRT::AssetHash::avoidZero(hash);

        // Already uploaded under this identity, so there is nothing to upload.
        //
        // Two distinct osg::Image objects routinely hold the same file -- 127 of 1388 over a long walk --
        // because OpenMW's resource system reaches the same texture by more than one path. The identity is
        // the content, so both resolve here, and calling CreateTexture again would hand the runtime a
        // duplicate hash: at best a wasted staging copy and a second VkImage the fork never releases, at
        // worst it replaces the image a live material is already pointing at.
        //
        // A handful of these are genuinely different filenames holding identical bytes -- a mod shipping a
        // copy of another mod's texture, or one texture serving both genders of an outfit. Collapsing those
        // is correct rather than merely tolerable: Remix hashes pixels for D3D9 games too, so a replacement
        // authored against one of them is meant to apply to the other.
        if (auto shared = mUploadedTextures.find(hash); shared != mUploadedTextures.end())
        {
            // One more image naming this identity. Counted, because the count is what release is gated on:
            // the identity must outlive every image that resolves to it, not just the first.
            ++shared->second;
            // Re-claimed inside its grace window, so the pending destroy is cancelled outright. This is the
            // case the window exists for: a surface that goes away and comes straight back never pays for a
            // destroy and a re-upload, and never risks a create being refused for sharing a queued handle.
            mOrphanedSince.erase(hash);
            cached.mHash = hash;
            cached.mUsable = true;
            // Left at zero: this entry added no memory, so releasing it must subtract none.
            mTextures.emplace(key, cached);
            ++mTexturesShared;
            return hash;
        }

        if (mRuntime.createTexture(hash, static_cast<unsigned int>(width),
                static_cast<unsigned int>(height), mipLevels, format, uploadData, uploadSize)
            == 0)
        {
            mTextures.emplace(key, cached);
            return 0;
        }
        // First image to name this identity, so the count starts at one and the bytes are charged here --
        // once per upload rather than once per image, which is why the total is keyed by identity.
        mUploadedTextures.emplace(hash, 1u);
        mTextureBytes[hash] = static_cast<std::size_t>(uploadSize);
        mTextureBytesResident += static_cast<std::size_t>(uploadSize);
        mTextureBytesPeak = std::max(mTextureBytesPeak, mTextureBytesResident);

        cached.mHash = hash;
        cached.mUsable = true;
        cached.mBytes = static_cast<std::size_t>(uploadSize);
        mTextures.emplace(key, cached);
        ++mTexturesUploaded;

        // The hash is logged in upper-case hex specifically so it can be pasted from Remix's texture list
        // back into this log. Remix identifies every texture by the hash handed to it here and shows nothing
        // else about it, so without this there is no way to get from a thumbnail that looks wrong to the
        // file responsible -- which was previously a dead end for exactly the sort of "these normals are
        // off" report this exists to answer.
        // Generated terrain composites are excluded, and that is the whole point of the line rather than an
        // exception to it: a composite has no file on disk, so there is nothing for a pasted hash to lead
        // back to. They were also more than half the volume -- 511 of each in one two-minute run -- because
        // they are regenerated as chunks come into view, so they were both useless here and crowding out the
        // authored textures this exists to identify.
        //
        // Worth more than tidiness. Log::~Log ends in std::endl, which flushes, and the sink is a console as
        // well as a file; one measured frame spent 811 ms inside that flush. Logging on a per-composite basis
        // during play was itself a source of the stutter being investigated.
        const bool generatedComposite
            = std::string_view(image.getFileName()) == Terrain::CompositeMap::sReadbackImageName;

        if (!generatedComposite && mTexturesLogged < mTextureLogLimit)
        {
            ++mTexturesLogged;
            Log(Debug::Info) << "Remix texture " << mTexturesLogged << ": " << std::hex << std::uppercase
                             << hash << std::nouppercase << std::dec << " " << image.getFileName() << " "
                             << width << "x" << height << " " << cached.mFormat << " "
                             << (colour ? "sRGB" : "linear") << ", " << mipLevels
                             << " mip" << (mipLevels == 1 ? "" : "s")
                             << (mTexturesLogged == mTextureLogLimit ? " (last of these)" : "");
        }

        return hash;
    }

    const RemixScene::CachedSurfaceResponse& RemixScene::surfaceResponseFor(const osg::Image& image)
    {
        if (auto found = mSurfaceResponses.find(&image); found != mSurfaceResponses.end())
            return found->second;

        const SurfaceRule* rule = surfaceRuleFor(image.getFileName());

        CachedSurfaceResponse response;
        response.mRoughness = rule != nullptr ? rule->mRoughness : kTexturedRoughness;
        response.mMetallic = rule != nullptr ? rule->mMetallic : 0.0f;
        // Points into the static rule table, so there is nothing to own or outlive.
        response.mPattern = rule != nullptr ? rule->mPattern : nullptr;

        return mSurfaceResponses.emplace(&image, response).first->second;
    }

    unsigned long long RemixScene::materialFor(const SurfaceState& surface, float emissive)
    {
        // Water before any texture consideration. OpenMW's water is a shader effect -- its texture units
        // hold a normal map and render targets, none of which is an albedo -- so whatever is bound there
        // is not what the surface should look like. A path tracer wants the physical description
        // instead, and gets a much better result from it than the raster version manages.
        if (surface.mIsWater && mWaterMaterial != 0)
            return mWaterMaterial;

        // An explicitly supplied image wins. Composited terrain arrives this way because its albedo only
        // ever existed as a render target; mTexture for those chunks is that target and has no image.
        const osg::Image* image = surface.mExplicitImage.get();
        if (image == nullptr)
        {
            if (surface.mTexture == nullptr)
                return mDefaultMaterial;
            image = surface.mTexture->getImage();
        }
        if (image == nullptr)
            return mDefaultMaterial;

        // Draws whose texture is not an albedo at all -- see kNonSurfaceTexturePatterns.
        //
        // Returning 0 is the existing "drop this draw" signal: meshFor opens with `if (material == 0)
        // return 0;` and the particle path checks the same thing. Placed ahead of textureFor so these are
        // never uploaded either, rather than uploaded and then never sampled.
        if (isNonSurfaceTexture(image->getFileName()))
        {
            // Bounded, because there is one of these per animation frame per material and the point is to
            // confirm the classification fired, not to narrate every draw.
            static unsigned int s_nonSurfaceLogged = 0;
            constexpr unsigned int kNonSurfaceLogLimit = 8;
            if (s_nonSurfaceLogged < kNonSurfaceLogLimit)
            {
                ++s_nonSurfaceLogged;
                Log(Debug::Info) << "Remix scene: dropped a non-surface draw using " << image->getFileName()
                                 << " -- a post-process style pass, not an albedo"
                                 << (s_nonSurfaceLogged == kNonSurfaceLogLimit ? " (last of these)" : "");
            }

            return 0;
        }

        const unsigned long long textureHash = textureFor(*image, true);
        if (textureHash == 0)
            return mDefaultMaterial;

        // Alpha blending is turned into a cutout when no explicit test was asked for.
        //
        // This is the thing that makes foliage read as foliage. Morrowind's leaves, grates, ropes and
        // banners overwhelmingly use NiAlphaProperty's *blend* flag with no alpha test, so there is no
        // osg::AlphaFunc anywhere on the path and the threshold above comes out zero -- which the runtime
        // reads as "fully opaque" and draws the whole quad, leaves plus the transparent square around
        // them. A path tracer wants one decision per hit, so the transparency has to become a cutout;
        // Remix does the same thing to legacy draw calls through rtx.alphaBlendToCutout, but that path
        // cannot see API-submitted geometry, so the choice has to be made here.
        //
        // An explicit test always wins: a mesh that asked for a threshold gets the one it asked for.
        //
        // Unless the caller asked for real transparency instead, which is what mPreferBlend means. Then the
        // substitution is skipped and the blend mode is handed to the runtime as the material's own blend
        // state. Blending has to come from the material rather than from a per-instance
        // InstanceInfoBlendEXT, because that extension is only consulted when the material sets
        // useDrawCallAlphaState, and turning that on would route alpha testing through the legacy
        // draw-call path too -- where an API host has no legacy draw call to supply it.
        unsigned char alphaTestReference = surface.mAlphaTestReference;
        int blendType = kBlendTypeNone;
        if (surface.mPreferBlend && surface.mAlphaBlend)
        {
            // Additive gets the emissive blend type, which is the same translation the runtime applies to a
            // legacy SRC_ALPHA/ONE draw call. Note this stacks with the emissive term materialFor is handed
            // for additive particles: if flames come out blown, that term is the one to drop, since the
            // blend type now carries the "this glows" half of it.
            blendType = surface.mAdditive ? kBlendTypeAlphaEmissive : kBlendTypeAlpha;
        }
        else if (alphaTestReference == 0 && surface.mAlphaBlend)
        {
            alphaTestReference = mBlendCutout;
        }

        // Surface response, classified from the texture's own path. See kSurfaceRules for why the name is
        // the only source available and what that costs in confidence. Memoised per image -- see
        // surfaceResponseFor for why that matters here rather than being a micro-optimisation.
        const CachedSurfaceResponse& response = surfaceResponseFor(*image);
        const float roughness = response.mRoughness;
        const float metallic = response.mMetallic;

        // Normal map, uploaded linear. A missing one is not a failure: vanilla Morrowind ships none, and
        // these only appear when a texture pack provides them and OpenMW's "auto use object normal maps"
        // setting is on.
        unsigned long long normalHash = 0;
        if (surface.mNormalMap != nullptr && surface.mNormalMap->getImage() != nullptr)
            normalHash = textureFor(*surface.mNormalMap->getImage(), false);

        // Per-texel roughness, derived from the mesh's specular map when it has one.
        //
        // Without this every surface took the single constant above, classified from the texture's file
        // path by kSurfaceRules -- so a texture pack shipping real specular data had it discarded and got a
        // guess based on its name instead. The constant remains the fallback and still feeds
        // roughnessConstant, which the runtime uses wherever no map is bound.
        unsigned long long roughnessHash = 0;
        if (surface.mSpecularMap != nullptr && surface.mSpecularMap->getImage() != nullptr)
            roughnessHash = roughnessTextureFor(*surface.mSpecularMap->getImage());

        // Glow map, uploaded as colour because it names an emitted colour rather than a scalar.
        //
        // This is the one map vanilla Morrowind supplies in quantity. It comes from a NiTexturingProperty
        // glow slot instead of a filename convention, so no texture pack is needed for it to exist -- and
        // until now OpenMW read it off the mesh, tagged it, and this function dropped it. The runtime's
        // emissive slot was only ever fed the albedo, and only for additive particles.
        unsigned long long glowHash = 0;
        if (surface.mEmissiveMap != nullptr && surface.mEmissiveMap->getImage() != nullptr)
            glowHash = textureFor(*surface.mEmissiveMap->getImage(), true);

        // Recorded before the material-emissive fallback below can put the albedo into that slot, so "does
        // this asset ship a glow map" stays answerable afterwards. Both the [Remix PBR] tally and the
        // per-material log line report it, and letting the fallback answer it would make every emitting
        // surface look like it came with a map it does not have.
        const bool hasGlowMap = glowHash != 0;

        // A glow map does nothing without an intensity beside it: emissiveIntensity is what decides whether
        // the runtime treats the material as emitting at all, and every surface arriving here that is not an
        // additive particle brings zero. A figure the caller supplied still wins, so a flame that also
        // carries a glow map keeps the particle scale rather than being dimmed to this one.
        static const float glowIntensity = envFloat("OPENMW_REMIX_GLOW_INTENSITY", Settings::remix().mGlowIntensity);
        float emissiveIntensity = emissive;
        if (glowHash != 0 && emissiveIntensity <= 0.0f)
            emissiveIntensity = glowIntensity;

        // The same is true of an emissive colour, and for a long time only particles benefited from it.
        //
        // NiMaterialProperty is where Morrowind states "this surface emits" when the emission covers a whole
        // surface rather than a masked part of one, and mMaterialEmissive above already carries its
        // magnitude for every surface that arrives here. Until now submitParticles was the only caller that
        // did anything with it, so an opaque mesh asking to glow was read, believed and then dropped -- it
        // came through as its albedo alone, which for an asset whose albedo IS the glow art means the raw
        // texture rendered unlit. Glow in the Dahrk's windows are exactly that: the night state's pane and
        // the interior light shafts carry emissive (1, 1, 1) while the masonry around them carries zero.
        //
        // Scaled by its own setting rather than by glowIntensity, because the two are not the same claim. A
        // glow map is black wherever it must not emit, so a value above one only lifts texels that asked for
        // it; this covers the entire surface. 1.0 is what the fixed-function pipeline adds and therefore what
        // OpenMW's raster shows, which makes matching raster the default and brighter a deliberate choice.
        //
        // A caller-supplied figure still wins, so additive particles keep kParticleEmissive and alpha-blended
        // ones keep the particle scale instead of being re-derived here.
        static const float materialEmissiveScale
            = envFloat("OPENMW_REMIX_MATERIAL_EMISSIVE", Settings::remix().mMaterialEmissiveScale);
        if (emissiveIntensity <= 0.0f && surface.mMaterialEmissive > 0.0f)
        {
            emissiveIntensity = surface.mMaterialEmissive * materialEmissiveScale;

            // An intensity alone makes the surface emit a flat colour, which is not what the asset asked for.
            //
            // The runtime takes its emissive colour from the emissive texture when one is bound and from
            // emissiveColorConstant when none is, and the two REPLACE rather than multiply -- the albedo plays
            // no part in the emissive term. Since the constant here is a normalised unit hue, a pane whose NIF
            // says emissive (1, 1, 1) emits flat white over its whole area.
            //
            // The albedo is not the answer to that, for two reasons. The runtime already falls back to it on
            // its own when no emissive texture is bound, so naming it here changes nothing; and for these
            // assets the albedo is the wrong texture anyway. Glow in the Dahrk's night pane is a grey
            // luminance plate with the amber in its DARK map, which OpenMW's raster multiplies in and which
            // this host used to discard -- so an albedo-sourced emission is exactly the white being seen.
            //
            // Nothing more is needed here, because the albedo is now the composited product of the stages
            // and the runtime falls back to the albedo when no emissive texture is bound. A pane whose albedo
            // is base * dark therefore emits base * dark: the lattice and the amber together, which is what
            // the rasteriser draws and what the flattening exists to reproduce.
            //
            // This replaced a stopgap that bound the dark map alone into the emissive slot. That recovered the
            // colour and lost the lattice, because a dark map is half of a product and not a texture in its
            // own right. Flattening makes the whole question disappear rather than answering it twice.
        }

        // Terrain layer coverage, uploaded as a linear mask for the runtime's terrain baker to composite.
        //
        // Linear, not colour: this is data, and an sRGB decode applied to a coverage ramp would bend it.
        // The blend map is GL_ALPHA, which textureFor widens to RGBA8 with the source in the alpha channel
        // and white elsewhere, which is exactly where the bake shader reads it from.
        //
        // The mask is only half of what the baker needs. The other half is the map from the texcoords
        // submitted with this mesh to the mask's own UV space, which differ by the diffuse tiling and by
        // OpenMW's blend map inset -- see solveTexAffineBetween. A mask with no usable map is not sent at
        // all: sampling coverage through the wrong UVs would misplace every layer, where sending nothing
        // falls back to the vertex-alpha path that was already there.
        unsigned long long maskHash = 0;
        float maskTransform[6] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
        if (surface.mCoverageImage != nullptr)
        {
            const TexAffine diffuseAffine = surface.mHasTexMat
                ? TexAffine{ surface.mTexMat[0], surface.mTexMat[1], surface.mTexMat[2], surface.mTexMat[3],
                      surface.mTexMat[4], surface.mTexMat[5] }
                : kIdentityAffine;
            const TexAffine coverageAffine = surface.mHasCoverageTexMat
                ? TexAffine{ surface.mCoverageTexMat[0], surface.mCoverageTexMat[1],
                      surface.mCoverageTexMat[2], surface.mCoverageTexMat[3], surface.mCoverageTexMat[4],
                      surface.mCoverageTexMat[5] }
                : kIdentityAffine;

            float rowU[3];
            float rowV[3];
            if (solveTexAffineBetween(diffuseAffine, coverageAffine, rowU, rowV))
            {
                const unsigned long long uploaded = textureFor(*surface.mCoverageImage, false);
                if (uploaded != 0)
                {
                    maskHash = uploaded;
                    maskTransform[0] = rowU[0];
                    maskTransform[1] = rowU[1];
                    maskTransform[2] = rowU[2];
                    maskTransform[3] = rowV[0];
                    maskTransform[4] = rowV[1];
                    maskTransform[5] = rowV[2];
                }
            }
        }

        // Material constants from the NIF: tint, opacity, emissive colour, addressing.
        //
        // These three slots existed in the API and were pinned to white and 1.0 on the reasoning that the
        // texture should be what shows. That is the right default and the wrong constant: OpenMW has carried
        // NiMaterialProperty's mDiffuse, mAlpha and mEmissive all along.
        //
        // albedoConstant multiplies the albedo texture, so a tint below white darkens the surface. That is
        // intended where the asset asks for it, and the two things that keep it from darkening the world at
        // large are upstream: nifloader forces material diffuse to white when vertex colours are routed into
        // it, so the tint is not applied twice, and it only attaches a material at all when the values differ
        // from the defaults. OPENMW_REMIX_MATERIAL_COLOR=0 pins all three back if this reads wrong in game.
        static const bool materialConstants = envFlag("OPENMW_REMIX_MATERIAL_COLOR", Settings::remix().mMaterialConstants);

        float albedoConstant[3] = { 1.0f, 1.0f, 1.0f };
        float opacityConstant = 1.0f;
        // Split emission into a unit colour and a magnitude the same way the runtime does, so a green rune
        // stays green instead of reducing to a number and coming back white.
        float emissiveColour[3] = { 1.0f, 1.0f, 1.0f };

        if (materialConstants)
        {
            albedoConstant[0] = surface.mDiffuseColor[0];
            albedoConstant[1] = surface.mDiffuseColor[1];
            albedoConstant[2] = surface.mDiffuseColor[2];
            opacityConstant = surface.mMaterialAlpha;

            const float emissiveMagnitude = std::max(
                surface.mEmissiveColor[0], std::max(surface.mEmissiveColor[1], surface.mEmissiveColor[2]));
            if (emissiveMagnitude > 0.0f)
            {
                emissiveColour[0] = surface.mEmissiveColor[0] / emissiveMagnitude;
                emissiveColour[1] = surface.mEmissiveColor[1] / emissiveMagnitude;
                emissiveColour[2] = surface.mEmissiveColor[2] / emissiveMagnitude;
            }
        }

        // Addressing, read from the albedo texture where there is one. Terrain composites have no
        // osg::Texture2D behind them -- their albedo is a readback image -- so those keep the Repeat default,
        // which is what tiling terrain layers want anyway.
        unsigned char wrapU = 1;
        unsigned char wrapV = 1;
        if (surface.mTexture != nullptr)
        {
            // lss::Mdl::WrapMode: 0 Clamp, 1 Repeat. Only CLAMP* becomes Clamp; the mirrored modes have no
            // safe mapping here and Repeat is both the existing behaviour and the less damaging error.
            const auto wrapModeFor = [](osg::Texture::WrapMode mode) -> unsigned char {
                switch (mode)
                {
                    case osg::Texture::CLAMP:
                    case osg::Texture::CLAMP_TO_EDGE:
                    case osg::Texture::CLAMP_TO_BORDER:
                        return 0;
                    default:
                        return 1;
                }
            };
            wrapU = wrapModeFor(surface.mTexture->getWrap(osg::Texture::WRAP_S));
            wrapV = wrapModeFor(surface.mTexture->getWrap(osg::Texture::WRAP_T));
        }

        // Height for parallax occlusion mapping, split out of the normal map's alpha.
        //
        // Off by default. The depth is in world units and Morrowind is roughly seventy units to the metre,
        // so the value that reads as surface relief rather than as a swimming mess is a judgement against
        // the screen, not something derivable from the asset. OPENMW_REMIX_PARALLAX=1.0 is a reasonable
        // first try; Remix's own Displacement In Factor then scales it globally.
        //
        // Only offered when there is no terrain coverage mask. runtime.cpp routes that mask through the same
        // heightTexture slot, which is only safe while displacement stays zero -- the runtime gates the slot
        // on displacement being non-zero -- so setting displacement on a material carrying a mask would make
        // it read the coverage as a height field.
        static const float parallaxDepth = envFloat("OPENMW_REMIX_PARALLAX", Settings::remix().mParallaxDepth);
        unsigned long long heightHash = 0;
        float displaceIn = 0.0f;
        if (parallaxDepth > 0.0f && maskHash == 0 && surface.mHasNormalHeight
            && surface.mNormalMap != nullptr && surface.mNormalMap->getImage() != nullptr)
        {
            heightHash = heightTextureFor(*surface.mNormalMap->getImage());
            if (heightHash != 0)
                displaceIn = parallaxDepth;
        }

        // Everything baked into the material is part of its identity. The same texture can be a cutout on
        // one mesh and opaque on another, and can be paired with a normal map on one and not the other, so
        // any of these differing means a different material rather than a reused one.
        unsigned long long key = textureHash;
        const auto mix = [&key](unsigned long long value) {
            key = (key ^ value) * 0x100000001B3ull;
        };
        mix(alphaTestReference);
        // Offset so kBlendTypeNone does not mix in as the same value as kBlendTypeAlpha would after being
        // widened to unsigned. Without it a blended and a non-blended material sharing everything else
        // would collide on one cache entry, and whichever was built first would win for both.
        mix(static_cast<unsigned long long>(blendType + 1));
        mix(normalHash);
        // Folded in for the same reason as the normal: one albedo can appear with a specular map on one
        // mesh and without on another, and those are different materials.
        mix(roughnessHash);
        // The tint, opacity, emissive colour and addressing are identity too. One texture used tinted on one
        // mesh and plain on another is two materials; collapsing them means whichever was built first wins
        // for both, which is the same trap this block already avoids for cutouts and normal maps. Quantised
        // so that float noise does not manufacture a new material per draw.
        const auto mixUnit = [&mix](float value) {
            mix(static_cast<unsigned long long>(value * 255.0f + 0.5f));
        };
        mixUnit(albedoConstant[0]);
        mixUnit(albedoConstant[1]);
        mixUnit(albedoConstant[2]);
        mixUnit(opacityConstant);
        mixUnit(emissiveColour[0]);
        mixUnit(emissiveColour[1]);
        mixUnit(emissiveColour[2]);
        mix(static_cast<unsigned long long>(wrapU) | (static_cast<unsigned long long>(wrapV) << 8));
        // Height and its depth are identity too: the same albedo with and without displacement are two
        // materials, and collapsing them would give whichever was built first to both.
        mix(heightHash);
        mix(static_cast<unsigned long long>(displaceIn * 256.0f + 0.5f));
        // The sheet's frame count and rate are identity: the same atlas played at a different rate, or a
        // sheet against a single frame, are different materials.
        mix(static_cast<unsigned long long>(surface.mSpriteSheetCols)
            | (static_cast<unsigned long long>(surface.mSpriteSheetFps) << 8));
        mix(static_cast<unsigned long long>(roughness * 255.0f + 0.5f));
        mix(static_cast<unsigned long long>(metallic * 255.0f + 0.5f));
        mix(static_cast<unsigned long long>(emissiveIntensity * 16.0f + 0.5f));
        // Folded in for the same reason as the normal and the specular: one albedo appears with a glow map on
        // one mesh and without on another, and those are two materials. Only materials that carry a glow map
        // move -- everything else mixes a zero here, exactly as before.
        mix(glowHash);
        // The coverage mask and its map are part of the material's identity too, and this one is not
        // optional the way the others arguably are: a blend map belongs to a single chunk, so two chunks
        // sharing a land texture need two materials or the second would be drawn with the first's
        // coverage. This is what makes terrain materials per chunk per layer, which is why releasing them
        // with their chunk matters -- see MaterialTextures::mMask.
        mix(maskHash);
        if (maskHash != 0)
        {
            for (const float component : maskTransform)
            {
                unsigned int bits = 0;
                static_assert(sizeof(bits) == sizeof(component));
                std::memcpy(&bits, &component, sizeof(bits));
                mix(bits);
            }
        }

        if (auto found = mMaterials.find(key); found != mMaterials.end())
            return found->second;

        // One line per distinct material, capped. Each of these has cost real time to work out from the
        // screen alone: "alpha is not working" has three indistinguishable causes -- no cutout requested,
        // a cutout against a texture with no alpha channel, and a cutout at a threshold nothing can pass
        // -- and "everything looks like plastic" is either no rule matching or the rule being wrong.
        // Composites excluded for the same reason as in textureFor: no file behind them, regenerated
        // constantly, and the flush at the end of every log line is not free.
        if (std::string_view(image->getFileName()) != Terrain::CompositeMap::sReadbackImageName
            && mMaterialsLogged < mMaterialLogLimit)
        {
            ++mMaterialsLogged;
            const std::string transparencyDescription = blendType == kBlendTypeAlphaEmissive
                ? std::string("blend additive")
                : (blendType == kBlendTypeAlpha
                        ? std::string("blend alpha")
                        : "cutout " + std::to_string(static_cast<unsigned int>(alphaTestReference)));
            const unsigned long long cacheKey
                = (reinterpret_cast<unsigned long long>(image) << 1) | 1ull;
            const auto found = mTextures.find(cacheKey);
            Log(Debug::Info) << "Remix material " << mMaterialsLogged << ": " << image->getFileName()
                             << " " << image->s() << "x" << image->t() << " "
                             << (found != mTextures.end() ? found->second.mFormat : "?") << "; alphaTest "
                             << static_cast<unsigned int>(surface.mAlphaTestReference) << " blend "
                             << (surface.mAlphaBlend ? "yes" : "no") << " -> " << transparencyDescription
                             << "; rule "
                             << (response.mPattern != nullptr ? response.mPattern : "(none, default)")
                             << " roughness "
                             << roughness << " metallic " << metallic << "; normal map "
                             << (normalHash != 0 ? "yes" : "no")
                             // Printed so a surface that lit up unexpectedly can be traced back to its file,
                             // which is the whole point of these lines. Says where it came from too, because
                             // a glow map and a material emissive are different assets to go and look at.
                             << "; emissive " << emissiveIntensity
                             << (emissiveIntensity <= 0.0f
                                     ? ""
                                     : (hasGlowMap ? " (glow map)"
                                                      : (emissive > 0.0f ? " (caller)" : " (material)")))
                             << (mMaterialsLogged == mMaterialLogLimit ? " (last of these)" : "");
        }

        // `key | 1` is this material's identity, and deliberately not the albedo texture hash.
        //
        // Worth recording, because the opposite looks obviously right and was tried. Remix's D3D9 path
        // identifies a material by its albedo texture hash alone -- LegacyMaterialData::updateCachedHash is
        // literally `colorTextures[0].getImageHash()` -- so a capture from MGE-XE names every material after
        // its texture, and a pack is authored against those names. This host's key instead mixes the texture
        // hash with alpha test, blend, normal map, roughness, metallic and emissive, so a capture taken here
        // shares no material name with such a pack: 0 of 254 against the NVIDIA demo pack's 1282 keys, where
        // MGE-XE scores 131 of 244.
        //
        // None of which stops replacements binding, because binding does not go through this name.
        // fork_hooks::externalDrawMaterialReplacement tries the material hash first and then falls back to
        // the albedo texture hash, which is exactly the capture-authored key -- see the comment there, which
        // names this integration as the reason the fallback exists. Measured over two otherwise identical
        // sessions at 600k material lookups: 397,971 replacements bound with the identity below swapped for
        // the texture hash, and 396,753 with it as it is. A 0.3% difference, i.e. none.
        //
        // So switching would buy nothing and cost something: 12 of 762 textures in a session are used with
        // two different surface states -- foliage and fabric, blended in one place and alpha-cutout at a
        // specific threshold in another -- and keying on the texture alone collapses each pair onto whichever
        // was created first.
        const unsigned long long handle = mRuntime.createTexturedMaterial(key | 1ull, textureHash, roughness,
            metallic, alphaTestReference, normalHash, emissiveIntensity, blendType, maskHash,
            maskHash != 0 ? maskTransform : nullptr, roughnessHash, glowHash, albedoConstant,
            opacityConstant, emissiveColour, wrapU, wrapV, heightHash, displaceIn, 0.0f,
            surface.mSpriteSheetCols, surface.mSpriteSheetFps);
        if (handle == 0)
        {
            // Remembered as the fallback so a material the runtime refused is not retried every frame.
            mMaterials.emplace(key, mDefaultMaterial);
            return mDefaultMaterial;
        }

        mMaterials.emplace(key, handle);
        // Recorded for the mesh hash, which XORs the albedo texture hash the way Remix's D3D9 path does.
        // The normal hash is recorded alongside purely so a release can find this material from either
        // texture. An albedo-only index cannot enumerate the materials that name a normal map, so releasing
        // one would leave a live material pointing at a destroyed texture with nothing to detect it.
        mMaterialAlbedoHashes[handle] = MaterialTextures{ textureHash, normalHash, glowHash, maskHash };

        // Coverage summary, reported every 64 materials rather than per material.
        //
        // The per-material lines above stop at mMaterialLogLimit and describe individual surfaces, which
        // cannot answer the question that actually matters: how much of this scene's art is supplying PBR
        // maps at all. A count with no denominator cannot either -- "214 normals" means nothing without
        // knowing whether that is out of 300 materials or 3000 -- so both are reported together.
        //
        // Sixty-four because material creation is bursty: a cell load builds hundreds at once and then the
        // count sits still for minutes, so a per-material line would flood and a timer would mostly report
        // nothing new.
        ++mMaterialsBuilt;
        if (normalHash != 0)
            ++mMaterialsWithNormal;
        if (roughnessHash != 0)
            ++mMaterialsWithRoughness;
        if (emissiveIntensity > 0.0f)
            ++mMaterialsWithEmissive;
        if (hasGlowMap)
            ++mMaterialsWithGlow;

        if (mMaterialsBuilt - mMaterialSummaryAt >= 64)
        {
            mMaterialSummaryAt = mMaterialsBuilt;
            const auto percent = [total = mMaterialsBuilt](std::uint64_t n) {
                return total != 0 ? static_cast<int>((n * 100 + total / 2) / total) : 0;
            };
            Log(Debug::Info) << "[Remix PBR] " << mMaterialsBuilt << " materials built; normal map "
                             << mMaterialsWithNormal << " (" << percent(mMaterialsWithNormal)
                             << "%), roughness from specular " << mMaterialsWithRoughness << " ("
                             << percent(mMaterialsWithRoughness) << "%), glow map "
                             << mMaterialsWithGlow << " (" << percent(mMaterialsWithGlow)
                             << "%), emissive " << mMaterialsWithEmissive << " ("
                             << percent(mMaterialsWithEmissive) << "%)";
        }
        return handle;
    }

    void RemixScene::drawSubmitted(unsigned long long mesh, const float* transform,
        unsigned int categoryFlags, bool doubleSided, const SceneUtil::RigGeometry* rig,
        unsigned int pickingValue, float distanceSquared)
    {
        // OpenMW's own sky, dropped by default, because this host asks Remix to draw a sky of its own.
        //
        // Two skies were being lit at once. RemixSky drives rtx.atmosphere.* -- a physical sky model,
        // procedural clouds, stars, and real distant lights for the sun and each moon -- while this path
        // also submitted OpenMW's sky dome, its sun and sun-flash billboards, both moon billboards and its
        // cloud layer, tagged Category_Sky. Remix turns Category_Sky geometry into the environment, and the
        // note by mExempt below says what that means: it becomes the dominant area light in every cell.
        // So the scene was lit by a procedural atmosphere and by a textured dome carrying a starfield and two
        // moon quads, added together.
        //
        // That is what an unexplained pale light at midnight is. It also explains why none of Remix's
        // night-sky controls could touch it: nightSkyBrightness, starBrightness and the moon gains all tune
        // the procedural sky, and none of them is what the submitted dome contributes.
        //
        // Weather particles are unaffected. Rain and snow carry Category_Sky from Mask_WeatherParticles but
        // reach the runtime through submitParticles, not this queue, so they are still submitted.
        //
        // On by default only if someone wants the old behaviour back: with Remix's own sky off, or skyMode
        // set to something that does not draw one, the dome is the only sky there is and dropping it would
        // leave the world unlit.
        static const bool submitSkyGeometry
            = envFlag("OPENMW_REMIX_SKY_GEOMETRY", Settings::remix().mSkyGeometry);
        if (!submitSkyGeometry && (categoryFlags & RemixRT::Runtime::Category_Sky) != 0)
        {
            ++mSkyInstancesDropped;
            return;
        }

        PendingInstance pending;
        pending.mMesh = mesh;
        std::memcpy(pending.mTransform, transform, sizeof(pending.mTransform));
        pending.mCategoryFlags = categoryFlags;
        pending.mRig = rig;
        pending.mPickingValue = pickingValue;
        pending.mDistanceSquared = distanceSquared;
        pending.mDoubleSided = doubleSided;

        // Unknown identities are admitted unweighed: particle meshes and the probe quad are individually
        // tiny, and the point of the budget is the merged distant chunks, which are always in the map
        // because this traversal built them.
        const auto primitives = mMeshPrimitives.find(mesh);
        pending.mPrimitives = primitives != mMeshPrimitives.end() ? primitives->second : 0u;

        // The sky is never budgeted, whatever its distance.
        //
        // It has to be exempt precisely *because* the budget is now distance-ordered. OpenMW's sky dome is
        // submitted as ordinary geometry, and its bounding centre is further away than anything else in the
        // scene, so a distance sort puts it last and a binding budget would drop it first. Under path
        // tracing the sky is the dominant area light in every cell, interiors included -- losing it is not a
        // degraded far field, it is the world going dark.
        pending.mExempt = (categoryFlags & RemixRT::Runtime::Category_Sky) != 0;

        mPendingInstances.push_back(pending);
    }

    void RemixScene::flushSubmissions()
    {
        static const bool nearestFirst = envFlag("OPENMW_REMIX_BUDGET_NEAREST_FIRST", Settings::remix().mBudgetNearestFirst);

        // Sort keys, not instances. A PendingInstance is ~88 bytes, and at the 20,000-instance ceiling
        // sorting them directly would move well over a megabyte through a merge sort every frame -- a
        // measurable share of a submit stage that runs in 1.44 ms. The key is eight bytes and contiguous,
        // so the sort touches a 160 KB array instead and the instances are read once, in order.
        mSubmissionOrder.clear();
        mSubmissionOrder.reserve(mPendingInstances.size());
        for (unsigned int index = 0; index < static_cast<unsigned int>(mPendingInstances.size()); ++index)
        {
            const PendingInstance& pending = mPendingInstances[index];
            // Exemption folded into the key rather than branched on in the comparator: distances are never
            // negative, so -1 sorts every exempt instance ahead of the field for free.
            mSubmissionOrder.push_back(
                { pending.mExempt ? -1.0f : pending.mDistanceSquared, index });
        }

        if (nearestFirst)
        {
            // Ties broken by index, so equal distances keep traversal order and a frame is reproducible.
            // That makes an unstable sort safe, which is worth having: the comparator is total.
            std::sort(mSubmissionOrder.begin(), mSubmissionOrder.end(),
                [](const SubmissionKey& left, const SubmissionKey& right) {
                    if (left.mDistanceSquared != right.mDistanceSquared)
                        return left.mDistanceSquared < right.mDistanceSquared;
                    return left.mIndex < right.mIndex;
                });
        }

        const unsigned int budget = primitiveBudget();
        for (const SubmissionKey& key : mSubmissionOrder)
        {
            const PendingInstance& pending = mPendingInstances[key.mIndex];
            if (!pending.mExempt && pending.mPrimitives != 0)
            {
                if (mPrimitivesSubmitted + pending.mPrimitives > budget)
                {
                    // Not a break. An instance that does not fit is skipped rather than ending the pass,
                    // because one enormous merged chunk arriving mid-list must not shut out every smaller
                    // instance behind it -- and with the list sorted, everything behind it is further away
                    // and cheaper to keep.
                    ++mInstancesOverBudget;
                    continue;
                }
                mPrimitivesSubmitted += pending.mPrimitives;
            }
            submitInstanceNow(pending);
        }
        mPendingInstances.clear();
    }

    unsigned int RemixScene::primitiveBudget()
    {
        // No meaningful upper bound to impose here, so none is imposed. This counts triangles, and the
        // only real ceiling is the runtime's NEE prefix-sum index -- clamping to that silently would be
        // worse than letting a deliberately large value through, because the whole reason this is tunable
        // is to measure against that ceiling rather than assume it.
        static const unsigned int budget = envUInt("OPENMW_REMIX_PRIMITIVE_BUDGET", kPrimitiveBudget,
            std::numeric_limits<unsigned int>::max());
        return budget;
    }

    void RemixScene::submitInstanceNow(const PendingInstance& pending)
    {
        const unsigned long long mesh = pending.mMesh;
        const float* const transform = pending.mTransform;
        const unsigned int categoryFlags = pending.mCategoryFlags;
        const bool doubleSided = pending.mDoubleSided;
        const SceneUtil::RigGeometry* const rig = pending.mRig;
        const unsigned int pickingValue = pending.mPickingValue;

        if (rig == nullptr)
        {
            mRuntime.drawInstance(mesh, transform, categoryFlags, doubleSided, nullptr, 0, pickingValue);
            return;
        }

        // Bone matrices are read every frame rather than cached, because that is the entire per-frame
        // cost of a skinned instance and the whole reason this path exists. They are current at this
        // point in the frame: the Remix submit runs after osgViewer's update traversal, which is what
        // drives RigGeometry::updateBounds and through it Skeleton::updateBoneMatrices.
        if (!rig->getBoneMatrices(mBoneMatrixScratch))
        {
            // The skin has no resolved skeleton yet, which happens for a frame or two after a cell loads.
            // Submitting the bind pose instead would put a T-posed actor in the scene, which is a worse
            // answer than nothing.
            ++mSkinnedDropped;
            return;
        }

        // The identity bone buildSkinning reserved. It has to be here and it has to be last, because the
        // index buildSkinning wrote for unweighted vertices is the bone count it saw.
        mBoneMatrixScratch.emplace_back(osg::Matrixf::identity());

        const unsigned int boneCount = static_cast<unsigned int>(
            std::min<std::size_t>(mBoneMatrixScratch.size(), RemixRT::Runtime::kMaxBones));

        mBoneTransformScratch.resize(static_cast<std::size_t>(boneCount) * 12);
        for (unsigned int bone = 0; bone < boneCount; ++bone)
        {
            const osg::Matrixf& matrix = mBoneMatrixScratch[bone];
            float* out = mBoneTransformScratch.data() + static_cast<std::size_t>(bone) * 12;
            // Same row-vector to column-vector conversion as the instance transform: OSG applies p * M
            // with translation in the fourth row, Remix applies M * p with translation in the fourth
            // column, so the 3x3 is transposed and the translation moves.
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                    out[row * 4 + col] = matrix(col, row);
                out[row * 4 + 3] = matrix(3, row);
            }
        }

        if (mRuntime.drawInstance(mesh, transform, categoryFlags, doubleSided,
                mBoneTransformScratch.data(), boneCount, pickingValue))
            ++mSkinnedInstances;
    }

    void RemixScene::submitParticles(const osgParticle::ParticleSystem& particles,
        const SurfaceState& surface, const osg::Matrixd& localToWorld, const osg::Vec3f& cameraRight,
        const osg::Vec3f& cameraUp, unsigned int categoryFlags)
    {
        const int count = particles.numParticles();
        if (count <= 0)
            return;

        // Whether the particles are already in world space, by the same test OpenMW itself uses.
        //
        // osgParticle::ParticleSystem has no getReferenceFrame(), so NifOsg::Loader records the answer by
        // pushing the string "worldspace" onto the drawable's user data container when the NIF asked for
        // absolute placement, and Resource::SceneManager reads it back the same way. Duplicated here rather
        // than shared because that helper is file-local, but it has to agree: applying the node transform to
        // particles that are already in world coordinates transforms them twice, which throws every effect
        // across the cell and does it worse the further the emitter is from the origin.
        const osg::UserDataContainer* userData = particles.getUserDataContainer();
        const bool worldSpace = userData != nullptr && userData->getNumDescriptions() > 0
            && userData->getDescriptions()[0] == "worldspace";

        // Emissive only for additive particles. This is the distinction kParticleEmissive's own comment
        // said was worth making once particles were on screen and could be judged -- they are, and smoke
        // was coming through as a solid white puff, because a fixed emissive of 6.0 over a pale grey smoke
        // texture saturates every channel long before it reaches the tonemapper.
        //
        // Flames, glows, sparks and magic blend additively and genuinely emit, so they keep the term. Smoke,
        // fog and dust blend with ONE_MINUS_SRC_ALPHA: they occlude what is behind them and are lit by the
        // scene like anything else, which is what a path tracer is good at. Handing them an emissive term
        // does not merely brighten them, it removes them from the lighting solution entirely -- an emitter
        // takes no shadow and picks up no colour from its surroundings, so smoke stops being smoke.
        static const float alphaEmissiveScale
            = envFloat("OPENMW_REMIX_PARTICLE_EMISSIVE", Settings::remix().mParticleEmissive);
        const float particleEmissive
            = surface.mAdditive ? kParticleEmissive : surface.mMaterialEmissive * alphaEmissiveScale;
        const unsigned long long material = materialFor(surface, particleEmissive);
        if (material == 0)
            return;

        // Rebuilt whole rather than updated. The API has no mesh-update entry point, and unlike a skinned
        // mesh there is nothing static to keep: a particle's position, size and colour all change every
        // frame, and particles are born and die. The saving grace is scale -- a candle flame is a handful of
        // quads, where a character is thousands of vertices -- so the churn this would be unacceptable for
        // in submitGeometry is affordable here.
        const void* key = &particles;
        if (auto found = mParticleMeshes.find(key); found != mParticleMeshes.end())
        {
            mRuntime.destroyMesh(found->second.mHandle);
            mParticleMeshes.erase(found);
        }

        mVertexScratch.clear();
        mIndexScratch.clear();

        for (int i = 0; i < count; ++i)
        {
            const osgParticle::Particle* particle = particles.getParticle(i);
            if (particle == nullptr || !particle->isAlive())
                continue;

            osg::Vec3f centre = particle->getPosition();
            if (!worldSpace)
                centre = osg::Vec3f(osg::Vec3d(centre) * localToWorld);

            // Half extent, because the quad spans the size in each direction from the centre.
            const float half = particle->getCurrentSize() * 0.5f;
            if (!(half > 0.0f))
                continue;

            // Rotated in the billboard plane by the particle's own angle, which osgParticle integrates from
            // the NIF's rotation speed. Without it every puff in a plume shares the camera basis exactly, so
            // thirty quads read as one sprite stamped thirty times instead of a turning volume.
            osg::Vec3f across = cameraRight * half;
            osg::Vec3f down = cameraUp * half;
            if (const float angle = particle->getAngle().z(); angle != 0.0f)
            {
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const osg::Vec3f rotatedAcross = across * cosine + down * sine;
                down = down * cosine - across * sine;
                across = rotatedAcross;
            }
            // Facing the camera, which for a billboard is the direction the quad is built against.
            const osg::Vec3f normal = (cameraRight ^ cameraUp);

            // The particle's own colour, and more importantly its alpha.
            //
            // osgParticle interpolates both across a particle's lifetime, and that fade is how a puff appears
            // and dissipates. Left at the zero a default-constructed Vertex carries, every particle was
            // handed black at zero alpha and none of the lifetime shaping reached the runtime at all -- which
            // is not a subtle loss, since Remix binds this as color0 in B8G8R8A8_UNORM and the struct is
            // layout-compatible with remixapi_HardcodedVertex, so it is read verbatim.
            //
            // Colour and alpha are separate in osgParticle -- _current_color carries the colour range and
            // _current_alpha the alpha range -- and are combined by multiplying, which is what its own
            // rendering does. Multiplying also means whichever of the two a NIF actually animates comes
            // through, rather than depending on which one it chose.
            const osg::Vec4f particleColour = particle->getCurrentColor();
            const auto toByte = [](float value) {
                return static_cast<unsigned int>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
            };
            const unsigned int packedColour = (toByte(particleColour.w() * particle->getCurrentAlpha()) << 24)
                | (toByte(particleColour.x()) << 16) | (toByte(particleColour.y()) << 8)
                | toByte(particleColour.z());

            const unsigned int base = static_cast<unsigned int>(mVertexScratch.size());
            const osg::Vec3f corners[4] = { centre - across - down, centre + across - down,
                centre + across + down, centre - across + down };
            constexpr float texcoords[4][2] = { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f } };

            for (int corner = 0; corner < 4; ++corner)
            {
                RemixRT::Runtime::Vertex vertex = {};
                vertex.mPosition[0] = corners[corner].x();
                vertex.mPosition[1] = corners[corner].y();
                vertex.mPosition[2] = corners[corner].z();
                vertex.mNormal[0] = normal.x();
                vertex.mNormal[1] = normal.y();
                vertex.mNormal[2] = normal.z();
                vertex.mTexcoord[0] = texcoords[corner][0];
                vertex.mTexcoord[1] = texcoords[corner][1];
                vertex.mColor = packedColour;
                mVertexScratch.push_back(vertex);
            }

            mIndexScratch.insert(mIndexScratch.end(),
                { base, base + 1u, base + 2u, base, base + 2u, base + 3u });
        }

        if (mVertexScratch.empty() || mIndexScratch.empty())
            return;

        // Hashed from the address and the frame, because the handle *is* the hash: reusing one while the
        // previous mesh is still queued for destruction would alias the two.
        //
        // Holding this hash stable across frames was tried, on the theory that creating under a live hash is
        // an in-place update the way it is for lights. It is not, for meshes: the runtime keeps the original
        // contents and ignores the new ones, so every particle system froze on the geometry it had when it
        // was first seen -- fires and torch flames stopped moving. Whatever the per-frame cost of rebuilding
        // these, it cannot be avoided this way.
        const unsigned long long hash
            = ((reinterpret_cast<unsigned long long>(key) * 0x9E3779B97F4A7C15ull) ^ (mFrame << 1)) | 1ull;
        ++mMeshesCreated;
        const unsigned long long mesh = mRuntime.createMesh(hash, mVertexScratch.data(),
            static_cast<unsigned int>(mVertexScratch.size()), mIndexScratch.data(),
            static_cast<unsigned int>(mIndexScratch.size()), material);
        if (mesh == 0)
            return;

        ParticleMesh entry;
        entry.mHandle = mesh;
        entry.mMaterial = material;
        entry.mLastUsedFrame = mFrame;

        // Retire the previous frame's mesh for this system, then record the new one.
        //
        // emplace() was silently doing nothing here. The key is the system's address, which is the same
        // every frame, so from the second frame onward the insert failed and the map kept last frame's
        // handle with last frame's timestamp. releaseStaleParticleMeshes then found that timestamp stale,
        // destroyed the old handle and erased the key -- leaving the mesh created *this* frame untracked
        // and therefore never destroyed. The pattern alternated, so almost exactly half of every frame's
        // particle meshes leaked inside the runtime: around 45 a frame, a thousand a second, each holding
        // vertex and index buffers and an acceleration structure.
        //
        // It was invisible in the scene accounting because particle meshes are counted here, not in
        // mMeshes, so the cached-mesh figure looked healthy while GPU memory drained. The symptom was a
        // fifteen second stall and VK_ERROR_DEVICE_LOST after a few minutes or a few cell changes,
        // whichever came first.
        //
        // The handle now comes back identical every frame, because the hash is stable and the handle is the
        // hash, so the guard below finds nothing to retire and the entry is simply restamped. That is the
        // whole point: the destroy that used to run here every frame for every system is gone, and with it
        // the queued-destruction traffic that made a stable hash unusable in the first place.
        //
        // The destroy is kept for the case that still needs it -- a handle that genuinely changed, meaning
        // the runtime issued a different one rather than updating the mesh in place. Retiring the old one
        // then is correct and cannot tombstone the new one, since the two differ.
        if (auto existing = mParticleMeshes.find(key); existing != mParticleMeshes.end())
        {
            if (existing->second.mHandle != mesh)
                mRuntime.destroyMesh(existing->second.mHandle);
            existing->second = entry;
        }
        else
        {
            mParticleMeshes.emplace(key, entry);
        }

        // Identity transform: the quads were built in world space, both so that the camera basis could be
        // used directly and because world-space particle systems have no node transform to apply anyway.
        constexpr float identity[12]
            = { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f };
        // Picking values for particles start above the traversal's ceiling rather than continuing its
        // count, because the two paths advance independently and the runtime only warns once about a
        // collision before dropping it. Reserving a disjoint range is cheaper than coordinating a shared
        // counter across them, and it keeps flames and smoke selectable in the developer menu -- the
        // textures most likely to need tagging by hand.
        ++mParticleInstances;
        mRuntime.drawInstance(
            mesh, identity, categoryFlags, true, nullptr, 0, kMaxInstancesPerFrame + mParticleInstances);
        mLastParticleCount += static_cast<unsigned int>(mVertexScratch.size() / 4);
    }

    void RemixScene::releaseTextureClaim(const CachedTexture& cached)
    {
        if (!cached.mUsable || cached.mHash == 0)
            return;

        const auto claimed = mUploadedTextures.find(cached.mHash);
        if (claimed != mUploadedTextures.end() && claimed->second > 0)
            --claimed->second;
    }

    void RemixScene::releaseOrphanedTextures()
    {
        // Pass one: drop the claims of entries whose image OpenMW has freed. No destroys here, so a cell
        // that pages back in during this same frame can re-claim its identities and pay nothing.
        for (auto it = mTextures.begin(); it != mTextures.end();)
        {
            if (it->second.mImage.valid())
            {
                ++it;
                continue;
            }

            const unsigned long long orphaned = it->second.mUsable ? it->second.mHash : 0ull;
            releaseTextureClaim(it->second);
            it = mTextures.erase(it);

            // Start the clock the first time an identity is left unclaimed. Recorded here rather than in
            // pass two so the window is measured from when the last image actually went away.
            if (orphaned != 0)
            {
                if (const auto claimed = mUploadedTextures.find(orphaned);
                    claimed != mUploadedTextures.end() && claimed->second == 0)
                {
                    mOrphanedSince.emplace(orphaned, mFrame);
                }
            }
        }

        // Pass two: destroy the identities nothing claims any more, newest work first budgeted. Reaching a
        // count of zero is the whole safety argument -- it means no live image resolves to this identity, so
        // no live surface can be drawing with it and no lookup can produce it again without re-uploading.
        unsigned int destroyed = 0;
        for (auto it = mUploadedTextures.begin();
             it != mUploadedTextures.end() && destroyed < kTextureDestroysPerFrame;)
        {
            if (it->second > 0)
            {
                ++it;
                continue;
            }

            // Unclaimed, but not yet for long enough. Destroying now would risk the runtime refusing a later
            // create that shares the handle, and would throw away an identity a returning surface is about
            // to ask for again.
            const auto since = mOrphanedSince.find(it->first);
            if (since == mOrphanedSince.end() || mFrame - since->second < kTextureGraceFrames)
            {
                ++it;
                continue;
            }

            const unsigned long long hash = it->first;

            // Materials naming this texture go first, through any of its slots, or they would be left
            // pointing at a texture that no longer exists. mMaterials is keyed by surface state rather than by handle, so
            // it has to be searched by value; mDefaultMaterial is deliberately never matched here because it
            // is stored as the value for every material the runtime refused, and destroying it mid-session
            // would take every one of those surfaces with it.
            for (auto mat = mMaterialAlbedoHashes.begin(); mat != mMaterialAlbedoHashes.end();)
            {
                if (mat->second.mAlbedo != hash && mat->second.mNormal != hash
                    && mat->second.mGlow != hash && mat->second.mMask != hash)
                {
                    ++mat;
                    continue;
                }

                const unsigned long long handle = mat->first;
                if (handle == mDefaultMaterial)
                {
                    ++mat;
                    continue;
                }

                for (auto key = mMaterials.begin(); key != mMaterials.end();)
                    key = (key->second == handle) ? mMaterials.erase(key) : std::next(key);

                mRuntime.destroyMaterial(handle);
                mat = mMaterialAlbedoHashes.erase(mat);
                ++mMaterialsReleased;
            }

            mRuntime.destroyTexture(hash);

            if (const auto bytes = mTextureBytes.find(hash); bytes != mTextureBytes.end())
            {
                mTextureBytesResident -= std::min(mTextureBytesResident, bytes->second);
                mTextureBytes.erase(bytes);
            }

            // Erased in the same breath as the destroy. Leaving it would be worse than a leak: the upload
            // path would report this content as already present, hand back a hash the runtime no longer
            // holds, and never attempt the re-upload that would put it right.
            it = mUploadedTextures.erase(it);
            mOrphanedSince.erase(hash);
            ++mTexturesReleased;
            ++destroyed;
        }
    }

    void RemixScene::releaseStaleParticleMeshes()
    {
        for (auto it = mParticleMeshes.begin(); it != mParticleMeshes.end();)
        {
            if (it->second.mLastUsedFrame == mFrame)
            {
                ++it;
                continue;
            }
            mRuntime.destroyMesh(it->second.mHandle);
            it = mParticleMeshes.erase(it);
        }
    }

    unsigned int RemixScene::buildSkinning(const SceneUtil::RigGeometry& rig, unsigned int vertexCount)
    {
        const auto* influences = rig.getInfluences();
        if (influences == nullptr || influences->empty() || vertexCount == 0)
            return 0;

        // One slot past the real bones, holding an identity transform.
        //
        // RigGeometry::setInfluences drops the empty weight set, so a vertex with no influences at all is
        // absent from the grouping -- and the CPU path leaves those vertices at their bind-pose position,
        // because it writes only the vertices it finds in the groups. There is no way to express "not
        // skinned" on the GPU, where every vertex is the weighted sum of some bones, so the identity has
        // to be a bone. Without it those vertices would collapse to the origin and drag a triangle fan
        // across the model with them.
        const std::size_t realBones = rig.getBoneCount();
        if (realBones == 0 || realBones + 1 > RemixRT::Runtime::kMaxBones)
        {
            Log(Debug::Warning) << "Remix scene: a skin references " << realBones
                                << " bones, over the runtime's limit of " << RemixRT::Runtime::kMaxBones
                                << "; it will not be skinned";
            return 0;
        }
        const unsigned int identityBone = static_cast<unsigned int>(realBones);

        unsigned int bonesPerVertex = 1;
        for (const auto& [weights, vertices] : *influences)
            bonesPerVertex
                = std::max(bonesPerVertex, static_cast<unsigned int>(weights.size()));

        const std::size_t slots = static_cast<std::size_t>(bonesPerVertex) * vertexCount;
        // Every vertex starts fully weighted onto the identity bone, so a vertex the grouping never
        // mentions is already correct and needs no separate pass to find.
        mWeightScratch.assign(slots, 0.0f);
        mBoneIndexScratch.assign(slots, identityBone);
        for (unsigned int vertex = 0; vertex < vertexCount; ++vertex)
            mWeightScratch[static_cast<std::size_t>(vertex) * bonesPerVertex] = 1.0f;

        for (const auto& [weights, vertices] : *influences)
        {
            // Normalised, because the runtime derives the last weight of each tuple as one minus the
            // others rather than reading it. A tuple summing to 0.99 does not produce a slightly dimmer
            // vertex, it hands 0.01 to whichever bone sits in the last slot -- which for a vertex with
            // fewer influences than bonesPerVertex is a padding entry.
            float total = 0.0f;
            for (const auto& [bone, weight] : weights)
                total += weight;
            if (!(total > 0.0f))
                continue;

            for (unsigned short vertex : vertices)
            {
                if (vertex >= vertexCount)
                    continue;
                const std::size_t base = static_cast<std::size_t>(vertex) * bonesPerVertex;
                std::size_t slot = 0;
                for (const auto& [bone, weight] : weights)
                {
                    if (bone >= realBones)
                        continue;
                    mWeightScratch[base + slot] = weight / total;
                    mBoneIndexScratch[base + slot] = static_cast<unsigned int>(bone);
                    ++slot;
                }
                // Any remaining slots keep the identity bone at weight zero. Pointing them at a bone that
                // already influences this vertex would do as well; what matters is that the index is
                // valid, since the runtime multiplies by it before checking the weight.
                for (; slot < bonesPerVertex; ++slot)
                    mWeightScratch[base + slot] = 0.0f;
            }
        }

        return bonesPerVertex;
    }

    unsigned long long RemixScene::meshFor(osg::Geometry& geometry, unsigned long long material,
        const SurfaceState& surface, const SceneUtil::RigGeometry* rig)
    {
        if (material == 0)
            return 0;

        // Keyed on the bind pose for a skinned mesh, which is deliberate and is where the efficiency of
        // this whole approach comes from. RigGeometry's copy constructor shares mSourceGeometry, so every
        // actor wearing the same body part or armour piece resolves to one Remix mesh, submitted once and
        // instanced with different bone transforms.
        // Address alone; the terrain layer selects among the identities held under it. See
        // mGeometryIdentities in the header for why the layer is not part of the key.
        const std::uintptr_t key = reinterpret_cast<std::uintptr_t>(&geometry);

        const auto* positions = dynamic_cast<const osg::Vec3Array*>(geometry.getVertexArray());
        if (positions == nullptr || positions->empty())
            return 0;

        const unsigned int vertexCount = static_cast<unsigned int>(positions->size());

        // Fast path, and the reason the memo exists: hashing this geometry's content every frame would
        // cost far more than sharing saves. The checks are the ones the mesh cache itself used to make --
        // a changed vertex count means different geometry at a reused address, a changed modified counter
        // means the same geometry animated underneath us, and the material and texture matrix are baked in
        // at creation so neither can be swapped afterwards.
        if (auto memo = mGeometryIdentities.find(key); memo != mGeometryIdentities.end())
        {
            for (GeometryIdentity& identity : memo->second)
            {
                if (identity.mCoverageLayer != surface.mCoverageLayer)
                    continue;

                if (identity.mVertexCount == vertexCount && identity.mMaterial == material
                    && identity.mModifiedCount == positions->getModifiedCount()
                    && std::equal(std::begin(identity.mTexMat), std::end(identity.mTexMat),
                        std::begin(surface.mTexMat)))
                {
                    // The mesh can still be missing here, because eviction works on meshes and this memo
                    // is only a lookup cache. Falling through rebuilds it rather than returning a dead
                    // handle.
                    if (auto found = mMeshes.find(identity.mMeshHash); found != mMeshes.end())
                    {
                        identity.mLastUsedFrame = mFrame;
                        found->second.mLastUsedFrame = mFrame;
                        mLastMeshBonesPerVertex = found->second.mBonesPerVertex;
                        return found->second.mHandle;
                    }
                }

                // Only one identity per layer, so a failed revalidation means a rebuild rather than a
                // look at the next element.
                break;
            }
        }

        mIndexScratch.clear();
        osg::TriangleIndexFunctor<TriangleCollector> collector;
        collector.mIndices = &mIndexScratch;
        collector.mVertexCount = vertexCount;
        geometry.accept(collector);
        if (mIndexScratch.empty())
            return 0;

        const auto* normals = dynamic_cast<const osg::Vec3Array*>(geometry.getNormalArray());
        const auto* texcoords = dynamic_cast<const osg::Vec2Array*>(geometry.getTexCoordArray(0));

        mVertexScratch.clear();
        mVertexScratch.resize(vertexCount);
        for (unsigned int i = 0; i < vertexCount; ++i)
        {
            RemixRT::Runtime::Vertex& vertex = mVertexScratch[i];
            const osg::Vec3& position = (*positions)[i];
            vertex.mPosition[0] = position.x();
            vertex.mPosition[1] = position.y();
            vertex.mPosition[2] = position.z();

            // A per-vertex normal array is the common case; anything else (a single overall normal, or
            // none at all) gets an up vector rather than garbage. Wrong-but-consistent shading is far
            // easier to recognise than uninitialised normals.
            if (normals != nullptr && normals->size() == positions->size())
            {
                const osg::Vec3& normal = (*normals)[i];
                vertex.mNormal[0] = normal.x();
                vertex.mNormal[1] = normal.y();
                vertex.mNormal[2] = normal.z();
            }
            else
            {
                vertex.mNormal[0] = 0.0f;
                vertex.mNormal[1] = 0.0f;
                vertex.mNormal[2] = 1.0f;
            }

            if (texcoords != nullptr && texcoords->size() == positions->size())
            {
                const float s = (*texcoords)[i].x();
                const float t = (*texcoords)[i].y();
                if (surface.mHasTexMat)
                {
                    // Baked here because the vertex layout has nowhere else to put it: Remix vertices
                    // carry raw texcoords and the material carries no UV transform.
                    vertex.mTexcoord[0] = s * surface.mTexMat[0] + t * surface.mTexMat[2] + surface.mTexMat[4];
                    vertex.mTexcoord[1] = s * surface.mTexMat[1] + t * surface.mTexMat[3] + surface.mTexMat[5];
                }
                else
                {
                    vertex.mTexcoord[0] = s;
                    vertex.mTexcoord[1] = t;
                }
            }
            else
            {
                vertex.mTexcoord[0] = 0.0f;
                vertex.mTexcoord[1] = 0.0f;
            }

            // Opaque white, except for a terrain layer, whose coverage rides in the alpha.
            //
            // Sampled from the raw texcoord rather than the one written above: that one has the diffuse
            // tiling baked in and repeats many times across the chunk, while the blend map is stretched
            // once over it. Using the wrong one would tile the coverage along with the diffuse.
            unsigned int alpha = 0xFFu;
            if (surface.mCoverageImage != nullptr && texcoords != nullptr
                && texcoords->size() == positions->size())
            {
                const float s = (*texcoords)[i].x();
                const float t = (*texcoords)[i].y();
                float cs = s;
                float ct = t;
                if (surface.mHasCoverageTexMat)
                {
                    cs = s * surface.mCoverageTexMat[0] + t * surface.mCoverageTexMat[2]
                        + surface.mCoverageTexMat[4];
                    ct = s * surface.mCoverageTexMat[1] + t * surface.mCoverageTexMat[3]
                        + surface.mCoverageTexMat[5];
                }
                alpha = sampleImageAlpha(*surface.mCoverageImage, cs, ct);
            }

            vertex.mColor = 0x00FFFFFFu | (alpha << 24);
        }

        // Identity as Remix's own D3D9 path would compute it, so that a replacement pack authored against a
        // Morrowind capture binds to geometry submitted here. This value is what Remix knows the mesh as
        // and what a USD replacement is keyed on, so an address-derived one meant no replacement could ever
        // bind and every capture named its meshes differently.
        //
        // AssetHash::d3d9Geometry documents the formulation and how each part of it was verified. The
        // material is XORed in because the runtime does the same -- DrawCallState::getHash is the geometry
        // hash for the active rule XOR the material hash -- and it uses the albedo texture hash for the
        // latter, which is not this host's material handle. Hence the albedo lookup.
        //
        // An earlier attempt at this was abandoned, and the note explaining why claimed the difference was
        // structural: that Morrowind draws sub-ranges of shared vertex buffers, so its vertexCount could
        // never correspond to a standalone mesh's. That turned out to be wrong. The geometry descriptor,
        // which is where vertexCount lands, reproduces from a capture's own point count 40 times out of 40.
        // The real causes were a 16-bit index buffer and an opposite triangle winding, both handled in
        // d3d9Geometry and neither visible from the runtime source alone.
        unsigned long long hash = 0;
        // Half of what OPENMW_REMIX_PACK_MATCH=0 switches off; the texture identity in textureFor is the
        // other half. With it off this falls through to the content-hash branch below, which no pack entry
        // can match. Textures and materials are still uploaded and the scene still renders -- only the
        // identity a surface is offered under changes.
        const unsigned long long geometryHash = mPackMatching
            ? RemixRT::AssetHash::d3d9Geometry(mVertexScratch.data(), sizeof(RemixRT::Runtime::Vertex),
                vertexCount, mIndexScratch.data(), static_cast<unsigned int>(mIndexScratch.size()))
            : 0ull;
        if (geometryHash != 0)
        {
            const auto albedo = mMaterialAlbedoHashes.find(material);
            hash = RemixRT::AssetHash::avoidZero(
                geometryHash ^ (albedo != mMaterialAlbedoHashes.end() ? albedo->second.mAlbedo : 0ull));
        }
        else
        {
            // Geometry Remix's D3D9 path could not have described -- too many vertices for a 16-bit index,
            // or an index count that is not whole triangles. No pack entry can exist for it, so identity
            // only has to be stable and collision-free, and hashing everything submitted gives that.
            unsigned long long contentHash = RemixRT::AssetHash::bytes(
                mVertexScratch.data(), mVertexScratch.size() * sizeof(RemixRT::Runtime::Vertex));
            contentHash = RemixRT::AssetHash::bytesSeeded(
                mIndexScratch.data(), mIndexScratch.size() * sizeof(unsigned int), contentHash);
            hash = RemixRT::AssetHash::avoidZero(contentHash);
        }

        // Identical content already submitted? Then share it -- one Remix mesh, one BLAS, however many
        // instances reference it.
        //
        // The material and texture matrix have to match as well, and the hash no longer distinguishes
        // either on its own. It covers positions, indices and counts, plus the albedo texture via the XOR,
        // because that is what Remix computes -- so it says nothing about surface state or texcoords, and
        // the texture matrix is baked into the texcoords this builds. Two collisions therefore remain
        // reachable: one geometry drawn with two materials that share an albedo texture but differ in alpha
        // test or blend, which a session shows for 12 of 762 textures, and one geometry drawn with two
        // different texture matrices.
        //
        // Both are resolved by moving to a derived hash rather than by widening the real one, since
        // widening it would break the agreement with Remix that the whole change exists to establish. The
        // unperturbed value is always tried first, so the common case keeps the replacement-addressable
        // identity and only the colliding variant gives it up.
        bool reused = false;
        unsigned long long reusedHandle = 0;

        // Claims an identity: 0 free, 1 already ours and reusable, 2 held by different geometry.
        const auto tryClaim = [&](unsigned long long candidate) -> int {
            auto found = mMeshes.find(candidate);
            if (found == mMeshes.end())
                return 0;
            if (found->second.mMaterial == material
                && std::equal(std::begin(found->second.mTexMat), std::end(found->second.mTexMat),
                    std::begin(surface.mTexMat)))
            {
                found->second.mLastUsedFrame = mFrame;
                mLastMeshBonesPerVertex = found->second.mBonesPerVertex;
                reusedHandle = found->second.mHandle;
                reused = true;
                return 1;
            }
            return 2;
        };

        // A colliding variant takes a second identity derived from what it submits, never the next free
        // slot in the cache.
        //
        // Searching for a free slot was a leak, and a bad one. The result depended on what the cache
        // already held rather than on the geometry, so the same mesh landed on a different identity every
        // frame -- the slot it took last frame was occupied by its own previous entry -- and each frame
        // created another mesh. The cache grew without bound, cell loads slowed as it grew, and the process
        // eventually died. Widening the search from eight probes to seventy-two made it nine times worse
        // and produced the giveaway: "no free mesh identity after 72 attempts", which is impossible for
        // genuine collisions and only makes sense if the search was walking its own leaked entries.
        //
        // Hashing everything submitted, plus the material and texture matrix, is fully determined by this
        // draw. The same variant therefore resolves to the same identity on every frame and in every cell,
        // which is what makes it cacheable at all. It is not the value Remix would compute, so this variant
        // gives up being replacement-addressable -- the accepted trade, and the reason the parity hash is
        // always tried first so that the common case keeps it.
        if (tryClaim(hash) == 2)
        {
            unsigned long long variant = RemixRT::AssetHash::bytes(
                mVertexScratch.data(), mVertexScratch.size() * sizeof(RemixRT::Runtime::Vertex));
            variant = RemixRT::AssetHash::bytesSeeded(
                mIndexScratch.data(), mIndexScratch.size() * sizeof(unsigned int), variant);
            variant = RemixRT::AssetHash::combine(material, variant);
            variant = RemixRT::AssetHash::bytesSeeded(
                surface.mTexMat, sizeof(surface.mTexMat), variant);
            hash = RemixRT::AssetHash::avoidZero(variant);

            if (tryClaim(hash) == 2)
            {
                // Two different submissions agreeing on every vertex, index, the material and the texture
                // matrix, yet not equal. Dropping the mesh is right: there is no identity left to give it,
                // and aliasing it onto the other would corrupt the registry.
                static unsigned int sHardCollisions = 0;
                if (++sHardCollisions <= 4)
                    Log(Debug::Warning) << "Remix: full-content mesh identity collision on a mesh of "
                                        << vertexCount << " vertices; dropping it. This should be "
                                        << "unreachable.";
                return 0;
            }
        }

        if (reused)
        {
            GeometryIdentity& identity = identityFor(key, surface.mCoverageLayer);
            identity.mMeshHash = hash;
            identity.mLastUsedFrame = mFrame;
            identity.mVertexCount = vertexCount;
            identity.mModifiedCount = positions->getModifiedCount();
            identity.mMaterial = material;
            std::copy(std::begin(surface.mTexMat), std::end(surface.mTexMat), std::begin(identity.mTexMat));
            ++mMeshesShared;
            return reusedHandle;
        }

        // Never hand createMesh a hash that is already live for different geometry.
        //
        // The loop above gives up after eight perturbations, and before this guard existed it then fell
        // through and created a mesh under a hash another mesh was still using. createMesh takes the hash
        // as the handle verbatim, so those two geometries became one entry, and destroyMesh's contract --
        // release a hash before reusing it for different geometry -- was broken. That corrupts the mesh
        // registry, and it surfaces wherever geometry churns rather than at the point of the mistake:
        // cell changes, teleports, menus.
        //
        // Reaching this point at all is a consequence of matching Remix's formulation. Hashing everything
        // submitted, as this did before, made collisions unreachable because normals and texcoords
        // separated variants that positions and indices alone do not. Agreement with Remix is worth more
        // than that headroom, but it has to be paid for here rather than by corrupting state.
        //
        // Keeping the search going is the right response: a perturbed hash is as good as any other for a
        // variant that has already given up being replacement-addressable, and each round is a hash and a
        // map probe. Bailing out entirely is reserved for the case where even that fails, where dropping
        // one mesh from the frame is plainly better than aliasing it onto another.
        RemixRT::Runtime::Skinning skinning;
        if (rig != nullptr)
        {
            skinning.mBonesPerVertex = buildSkinning(*rig, vertexCount);
            skinning.mWeights = mWeightScratch.data();
            skinning.mBoneIndices = mBoneIndexScratch.data();
        }

        // Bounded per frame for a different reason than destroys are. This is not a startup cost that
        // settles once a cell is resident: animated geometry fails the identity memo's modified-count gate
        // every frame, re-hashes to a new identity every frame, and arrives here every frame. So this is a
        // standing workload of one acceleration structure build per animated drawable per frame -- measured
        // at a mean of 124 builds a frame against 152 skinned instances, roughly 3100 builds a second with
        // no end. A device that cannot retire that fails its fence sync, which is reported as a bare
        // device loss with no fault behind it, because nothing invalid happened; the work simply never
        // finished.
        //
        // Deferring costs the overflow one frame of freshness. Nothing is destroyed, no memo is written,
        // and the next frame retries, so the visible effect is that new geometry appears a frame or two
        // late and animation updates at a reduced rate while under load.
        //
        // This bounds a symptom. The standing cost exists because rigged geometry submits OpenMW's
        // CPU-deformed vertices while also submitting bone weights for the runtime to skin -- so the
        // content hash moves every frame even though the runtime is being handed everything it needs to
        // skin a static bind pose itself. Submitting the source pose makes the hash stable, the memo hit,
        // and this ceiling unnecessary.
        // The ceiling the paragraph above describes, which until now was only described.
        //
        // Checked here rather than earlier so that everything cheaper than a build still happens: a memo hit
        // and a shared mesh both return above this point, so a saturated frame still reuses every mesh it
        // already has and only turns away genuinely new ones. Nothing is written before returning -- no
        // primitive count, no cache entry, no identity -- so the retry next frame is an ordinary miss.
        static const unsigned int meshBuildBudget
            = envUInt("OPENMW_REMIX_MESH_BUILD_BUDGET", kMeshBuildsPerFrame, kMaxInstancesPerFrame);
        if (meshBuildBudget > 0 && mMeshesCreated >= meshBuildBudget)
        {
            ++mMeshBuildsDeferred;
            return 0;
        }

        ++mMeshesCreated;
        // Recorded so the submission budget can weigh an instance without re-deriving its geometry. Keyed by
        // identity, which is also the handle the runtime returns, so the submit path can look it up.
        mMeshPrimitives[hash] = static_cast<unsigned int>(mIndexScratch.size() / 3);
        const unsigned long long handle = mRuntime.createMesh(hash, mVertexScratch.data(), vertexCount,
            mIndexScratch.data(), static_cast<unsigned int>(mIndexScratch.size()), material,
            skinning.mBonesPerVertex > 0 ? &skinning : nullptr);
        if (handle == 0)
            return 0;

        CachedMesh cached;
        cached.mHandle = handle;
        cached.mLastUsedFrame = mFrame;
        cached.mVertexCount = vertexCount;
        cached.mIndexCount = static_cast<unsigned int>(mIndexScratch.size());
        cached.mMaterial = material;
        cached.mModifiedCount = positions->getModifiedCount();
        cached.mBonesPerVertex = skinning.mBonesPerVertex;
        std::copy(std::begin(surface.mTexMat), std::end(surface.mTexMat), std::begin(cached.mTexMat));
        mMeshes[hash] = cached;

        GeometryIdentity& identity = identityFor(key, surface.mCoverageLayer);
        identity.mMeshHash = hash;
        identity.mLastUsedFrame = mFrame;
        identity.mVertexCount = vertexCount;
        identity.mModifiedCount = positions->getModifiedCount();
        identity.mMaterial = material;
        std::copy(std::begin(surface.mTexMat), std::end(surface.mTexMat), std::begin(identity.mTexMat));

        mLastMeshBonesPerVertex = skinning.mBonesPerVertex;
        return handle;
    }

    void RemixScene::evictStaleMeshes()
    {
        if (mFrame < kMeshEvictionFrames)
            return;

        // Destroys are budgeted per frame because staleness arrives in a herd rather than a trickle. Every
        // mesh a cell contributed stops being submitted on the same frame that cell unloads, so they all
        // fall due together kMeshEvictionFrames later -- thousands of them, on one frame. Each destroyMesh
        // releases an acceleration structure, and releasing thousands without submitting a frame in between
        // stalls the device long enough for the watchdog to reset it, which arrives as a bare
        // VK_ERROR_DEVICE_LOST with nothing logged ahead of it. Spreading the same work across consecutive
        // frames retires the herd in comparable wall time while leaving every individual frame presentable.
        unsigned int destroyed = 0;
        for (auto it = mMeshes.begin(); it != mMeshes.end() && destroyed < kMeshDestroysPerFrame;)
        {
            if (mFrame - it->second.mLastUsedFrame > kMeshEvictionFrames)
            {
                mRuntime.destroyMesh(it->second.mHandle);
                mMeshPrimitives.erase(it->first);
                it = mMeshes.erase(it);
                ++destroyed;
            }
            else
            {
                ++it;
            }
        }

        // Logged only while the budget is saturated, which is the state worth knowing about: a backlog is
        // draining, and its size is the number this was silently doing in one frame before.
        if (destroyed == kMeshDestroysPerFrame && mFrame % 30 == 0)
            Log(Debug::Info) << "Remix scene: retiring stale meshes at the " << kMeshDestroysPerFrame
                             << "-per-frame budget; " << mMeshes.size() << " still cached.";

        // The identity memos are evicted on the same schedule rather than alongside their mesh, because
        // the mapping is many-to-one now: several geometries can name the same mesh, so a mesh going away
        // says nothing about whether any particular memo is still wanted. Leaving them would grow the map
        // for the whole session as the player moves and cells page out.
        for (auto it = mGeometryIdentities.begin(); it != mGeometryIdentities.end();)
        {
            std::vector<GeometryIdentity>& identities = it->second;
            const std::uint64_t frame = mFrame;
            identities.erase(std::remove_if(identities.begin(), identities.end(),
                                 [frame](const GeometryIdentity& identity) {
                                     return frame - identity.mLastUsedFrame > kMeshEvictionFrames;
                                 }),
                identities.end());

            // A geometry whose every layer has expired takes its map slot with it, so the map is bounded
            // by live content rather than by everything seen this session.
            if (identities.empty())
                it = mGeometryIdentities.erase(it);
            else
                ++it;
        }
    }

    RemixScene::GeometryIdentity& RemixScene::identityFor(std::uintptr_t address, unsigned int layer)
    {
        std::vector<GeometryIdentity>& identities = mGeometryIdentities[address];
        for (GeometryIdentity& identity : identities)
            if (identity.mCoverageLayer == layer)
                return identity;

        identities.emplace_back();
        identities.back().mCoverageLayer = layer;
        return identities.back();
    }

    void RemixScene::retainGeometry(osg::Geometry& geometry)
    {
        // Lookup only, never an insert: a geometry with no memo has no mesh to keep alive, and inserting
        // here would grow the map with empty slots for everything the cull rejects.
        auto memo = mGeometryIdentities.find(reinterpret_cast<std::uintptr_t>(&geometry));
        if (memo == mGeometryIdentities.end())
            return;

        ++mMeshesRetained;

        for (GeometryIdentity& identity : memo->second)
        {
            identity.mLastUsedFrame = mFrame;

            // The memo is stamped whether or not the mesh is still there. If it has already been
            // evicted the next submission rebuilds it, and the memo staying current is what stops the
            // slot itself expiring underneath a geometry that is still in the scene.
            if (auto found = mMeshes.find(identity.mMeshHash); found != mMeshes.end())
                found->second.mLastUsedFrame = mFrame;
        }
    }

    void RemixScene::retainCulledDrawable(osg::Drawable& drawable)
    {
        // Resolved exactly as the submission path resolves it, because a different answer here would
        // stamp the wrong slot and the retention would silently do nothing.
        //
        // The two dynamic_casts sit behind the null check rather than ahead of it, which the submission
        // path has no reason to do but this one does: it runs for every drawable the cull rejects, twelve
        // to fifteen thousand of them a frame outdoors. A RigGeometry and a MorphGeometry are both
        // Drawables that are not Geometries, so asGeometry() answers for the ordinary case -- terrain,
        // statics and merged paging chunks, which is nearly all of that number -- in one virtual call.
        osg::Geometry* geometry = drawable.asGeometry();
        if (geometry == nullptr)
        {
            if (const auto* rig = dynamic_cast<const SceneUtil::RigGeometry*>(&drawable))
            {
                geometry = rig->getSourceGeometry().get();
            }
            else if (const auto* morph = dynamic_cast<const SceneUtil::MorphGeometry*>(&drawable))
            {
                // Whichever frame-parity buffer holds the last completed blend, which is the one the
                // submission path would have hashed. The other parity's memo is left to expire: a morph
                // rebuilds its mesh whenever the pose moves regardless, so retaining both would hold a
                // mesh that the next submission cannot reuse anyway.
                geometry = const_cast<osg::Geometry*>(morph->getMorphedGeometry());
            }
        }

        if (geometry != nullptr)
            retainGeometry(*geometry);
    }
}

namespace MWRender
{
    bool RemixScene::resubmitCamera()
    {
        // The defaults the members carry are used when nothing has been submitted yet, which is the intro
        // video: it plays before a world exists, so there has never been a camera. Any valid camera will do
        // there -- an empty scene path-traces to black whichever way it points, and black is what belongs
        // behind a full-screen video. The point is only to give Remix a frame it will raytrace, so that the
        // composite at the end of that path runs and the video reaches the screen.
        const bool ok = mRuntime.setupCameraParameterized(mLastCameraEye, mLastCameraForward, mLastCameraUp,
            mLastCameraRight, mLastCameraFov, mLastCameraAspect, mLastCameraNear, mLastCameraFar);

        if (!ok && mHaveLastCamera)
        {
            // Worth one line: a camera the runtime accepted during the last submit and rejects now means
            // something changed about the runtime's state, not about the camera.
            static bool reported = false;
            if (!reported)
            {
                reported = true;
                Log(Debug::Warning) << "Remix: the runtime rejected a repeat of the last camera, so frames "
                                       "presented outside the main loop will not be raytraced and the "
                                       "interface drawn in them will not appear";
            }
        }

        return ok;
    }
}
