#include "remixscene.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

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
#include <osg/StateSet>
#include <osg/TexMat>
#include <osg/Texture2D>

#include <components/debug/debuglog.hpp>
#include <components/sceneutil/lightmanager.hpp>
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
    constexpr std::uint64_t kMeshEvictionFrames = 600;

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
    constexpr float kLightRadiusDefault = 0.6435f;

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
        return 600;
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

    /// Default for OPENMW_REMIX_LIGHT_INTENSITY, mirroring rtx.lightConversionIntensityFactor.
    ///
    /// Duplicated rather than read back from the runtime because the API still exposes no getter for an
    /// option, which means the two can drift. Kept at the same default so they agree until someone changes
    /// one. Note this is not the same knob as rtx.externalLight.intensityFactor below: that one tunes these
    /// lights and is live, this one only records what the runtime's own conversion path defaults to.
    constexpr float kLightIntensityFactorDefault = 0.65f;

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
    float lightEndDistance(const osg::Light& light, float range, float brightness)
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
        const float parsed = std::strtof(value, nullptr);
        return parsed > 0.0f ? parsed : fallback;
    }

    /// Reads a non-negative integer environment variable, keeping \a fallback when unset or unparseable.
    ///
    /// Zero is honoured rather than rejected: for a budget it is the meaningful "submit none of this"
    /// value, which is the same reason envByte exists alongside envFloat.
    unsigned int envUInt(const char* name, unsigned int fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        char* end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (end == value || parsed < 0)
            return fallback;
        return static_cast<unsigned int>(std::min<long long>(parsed, kMaxInstancesPerFrame));
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

    /// Samples an image's alpha at \a s, \a t and returns it as 0..255.
    ///
    /// Goes through osg::Image::getColor rather than indexing the data directly, because OpenMW's blend
    /// maps are not one format: they arrive as GL_ALPHA, GL_LUMINANCE_ALPHA or GL_RGBA depending on how the
    /// chunk was built, and getColor already knows how to read each. This runs once per vertex when a
    /// terrain layer's mesh is built, not per frame, so the indirection is not on a hot path.
    ///
    /// Clamped rather than wrapped: a blend map covers its chunk exactly once, so a texcoord landing
    /// outside it means the vertex is on the chunk's own edge, and wrapping there would fetch coverage from
    /// the opposite side of the chunk.
    unsigned int sampleImageAlpha(const osg::Image& image, float s, float t)
    {
        if (image.data() == nullptr || image.s() <= 0 || image.t() <= 0)
            return 0xFFu;

        const float cs = std::clamp(s, 0.0f, 1.0f);
        const float ct = std::clamp(t, 0.0f, 1.0f);
        const osg::Vec4 colour = image.getColor(osg::Vec2(cs, ct));
        return static_cast<unsigned int>(std::clamp(colour.a(), 0.0f, 1.0f) * 255.0f + 0.5f);
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
            if (envFlag("OPENMW_REMIX_TERRAIN", true))
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
                        ++mCulled;
                        return;
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
                    mergeState(*passes.front(), surface);

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
                mScene.lastMeshIsSkinned() ? rig : nullptr, mInstances + 1);
            mScene.noteInstancePosition(mMatrix(3, 0), mMatrix(3, 1), mMatrix(3, 2));
            ++mInstances;

            // The base layer is submitted; the rest of the ground goes over it.
            if (terrain != nullptr)
                submitTerrainLayers(*terrain, *geometry, categories, transform);
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
            unsigned int categories, const float (&baseTransform)[12])
        {
            static const bool enabled = envFlag("OPENMW_REMIX_TERRAIN_LAYERS", true);
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

                const unsigned long long layerMesh = mScene.submitGeometry(geometry, layer, nullptr);
                if (layerMesh == 0)
                    continue;

                mScene.drawSubmitted(layerMesh, baseTransform, categories, true, nullptr, mInstances + 1);
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

            static const bool enabled = envFlag("OPENMW_REMIX_GROUNDCOVER", true);
            static const unsigned int budget
                = envUInt("OPENMW_REMIX_GROUNDCOVER_BUDGET", kMaxGroundcoverCopies);
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

                if (fadeEnd > 0.0f && (at - mEye).length() > fadeEnd)
                    continue;

                float transform[12];
                writeTransform(world, transform);

                mScene.drawSubmitted(mesh, transform, categories, true, nullptr, mInstances + 1);
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
            if (composite.mReadback != nullptr)
            {
                surface.mExplicitImage = composite.mReadback.get();
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

        /// Folds one state set into \a surface. Later calls override earlier ones, which matches OSG's
        /// precedence for attributes that carry no override flag.
        static void mergeState(const osg::StateSet& stateSet, MWRender::RemixScene::SurfaceState& surface)
        {
            if (const auto* texture = dynamic_cast<const osg::Texture2D*>(
                    stateSet.getTextureAttribute(0, osg::StateAttribute::TEXTURE)))
            {
                surface.mTexture = texture;
            }

            // Every unit is examined for a role tag, and the unit index is deliberately not assumed.
            // Shader::ShaderVisitor binds an auto-detected normal map at `texAttributes.size()` -- the
            // next free unit, whatever that happens to be for this state set -- and records what it is by
            // attaching a SceneUtil::TextureType naming it. The tag is the only reliable identification.
            const unsigned int units = static_cast<unsigned int>(stateSet.getTextureAttributeList().size());
            for (unsigned int unit = 0; unit < units; ++unit)
            {
                const auto* type = dynamic_cast<const SceneUtil::TextureType*>(
                    stateSet.getTextureAttribute(unit, SceneUtil::TextureType::AttributeType));
                if (type == nullptr)
                    continue;

                // "normalHeightMap" is a normal map with height in alpha. The normal half is what Remix is
                // being given; the height half would need Remix's separate heightTexture slot and a
                // channel split, which is not done here.
                if (type->getName() != "normalMap" && type->getName() != "normalHeightMap")
                    continue;

                if (const auto* normal = dynamic_cast<const osg::Texture2D*>(
                        stateSet.getTextureAttribute(unit, osg::StateAttribute::TEXTURE)))
                {
                    surface.mNormalMap = normal;
                }
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
        }

        void pushState(osg::Node& node)
        {
            mCategoryStack.push_back(currentCategories() | categoriesFor(node.getNodeMask()));

            MWRender::RemixScene::SurfaceState surface = currentSurface();
            if (const osg::StateSet* stateSet = node.getStateSet())
                mergeState(*stateSet, surface);
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
        const bool emissive = envFlag("OPENMW_REMIX_EMISSIVE", false);
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
        const bool terrainComposite = envFlag("OPENMW_REMIX_TERRAIN_COMPOSITE", false);
        Terrain::CompositeMap::sReadbackEnabled = terrainComposite;
        Log(Debug::Info) << "Remix scene: terrain composite readback "
                         << (terrainComposite ? "ENABLED -- ground is correctly blended but softer, since a "
                                                "chunk's whole albedo is one 512x512 composite"
                                              : "off; ground uses the tiled base layer, which is sharp but "
                                                "shows hard edges where chunks pick different layers "
                                                "(OPENMW_REMIX_TERRAIN_COMPOSITE=1 to compare)");

        mLightRadius = envFloat("OPENMW_REMIX_LIGHT_RADIUS", kLightRadiusDefault);
        mLightIntensityFactor
            = envFloat("OPENMW_REMIX_LIGHT_INTENSITY", kLightIntensityFactorDefault);
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
        for (const auto& [key, cached] : mTextures)
        {
            if (cached.mUsable)
                mRuntime.destroyTexture(cached.mHash);
        }
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
        // Deliberately non-const: getLight takes a frame index because the osg::Light is double
        // buffered for the draw thread. Reading either buffer is fine here -- this traversal runs on the
        // update thread, before the draw -- and the frame counter is this class's own, so it only has to
        // be stable, not aligned with OSG's.
        auto& mutableSource = const_cast<SceneUtil::LightSource&>(source);
        const osg::Light* light = mutableSource.getLight(static_cast<size_t>(mFrame));
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

            // Recreated in place, and emphatically NOT destroyed first. Creating with an existing hash
            // *is* the runtime's update path. Destroying first cannot work: destroys are queued and
            // drained late in the frame, while creates go straight into the command stream, and the
            // drain builds a tombstone set from the queued destroys that suppresses any create sharing a
            // handle with one. So destroy-then-recreate reliably ends with the light gone -- and gone for
            // good, because this cache then believes it exists and never rebuilds it. That is what made
            // every light die a frame or two after it first moved.
            mLights.erase(found);
        }

        // The hash is the identity and the handle both, so it has to be stable for a given light and
        // distinct from every other live one. OpenMW's light id is already unique among live sources;
        // mixing it keeps ids that differ by one from producing adjacent hashes.
        const unsigned long long hash
            = (static_cast<unsigned long long>(static_cast<unsigned int>(key) + 1u)
                  * 0x9E3779B97F4A7C15ull)
            | 1ull;

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

    unsigned int RemixScene::submit(osg::Node* sceneRoot, const osg::Camera& camera)
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
        if (envFlag("OPENMW_REMIX_PROBE_QUAD", false))
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
        static const bool cullEnabled = envFlag("OPENMW_REMIX_CULL", true);

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
        sceneRoot->accept(visitor);
        mLastInstanceCount = visitor.instances();
        mCulled = visitor.culled();

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
                             << " geometries shared an existing mesh), " << mLastLightCount << " lights and "
                             << mTexturesUploaded << " textures (" << mTexturesShared
                             << " uploads avoided, content already present); " << mSkinnedInstances
                             << " instances were skinned, " << mCulled
                             << " drawables outside the frustum were culled, and " << mSkinnedDropped
                             << " skins were not ready; " << mLastParticleCount << " particles from "
                             << mParticleMeshes.size() << " systems"
                             << "; camera eye " << eye.x() << ", " << eye.y()
                             << ", " << eye.z() << " looking " << forward.x() << ", " << forward.y()
                             << ", " << forward.z() << " up " << up.x() << ", " << up.y() << ", "
                             << up.z() << " fov " << fov << " near " << nearClip << " far " << farClip
                             << (populated ? "" : " -- nothing found, so Remix has nothing to raytrace");
            if (mHaveExtent)
            {
                // The camera sits inside this box when geometry really is around the player. If it does
                // not, the instance transforms are wrong and no amount of lighting or material work will
                // put anything on screen.
                Log(Debug::Info) << "Remix scene: instance origins span " << mMin[0] << ".." << mMax[0]
                                 << ", " << mMin[1] << ".." << mMax[1] << ", " << mMin[2] << ".."
                                 << mMax[2];
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

    unsigned long long RemixScene::textureFor(const osg::Image& image, bool colour)
    {
        // Keyed on the image *and* the colour interpretation. The same bytes uploaded as sRGB and as
        // linear are two different textures to the runtime, and one image used both ways -- a diffuse map
        // that some other mesh binds as a normal map -- would otherwise silently get whichever
        // interpretation was asked for first.
        const unsigned long long key
            = (reinterpret_cast<unsigned long long>(&image) << 1) | (colour ? 1ull : 0ull);
        if (auto found = mTextures.find(key); found != mTextures.end())
            return found->second.mUsable ? found->second.mHash : 0ull;

        // Cached even on failure, so an unsupported format is diagnosed once rather than per drawable
        // per frame. Written before every early return below.
        CachedTexture cached;

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
        if (colour && mip0Size != 0)
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
            hash = RemixRT::AssetHash::bytes(uploadData, uploadSize);
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
        if (mUploadedTextures.find(hash) != mUploadedTextures.end())
        {
            cached.mHash = hash;
            cached.mUsable = true;
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
        mUploadedTextures.insert(hash);

        cached.mHash = hash;
        cached.mUsable = true;
        mTextures.emplace(key, cached);
        ++mTexturesUploaded;

        // The hash is logged in upper-case hex specifically so it can be pasted from Remix's texture list
        // back into this log. Remix identifies every texture by the hash handed to it here and shows nothing
        // else about it, so without this there is no way to get from a thumbnail that looks wrong to the
        // file responsible -- which was previously a dead end for exactly the sort of "these normals are
        // off" report this exists to answer.
        if (mTexturesLogged < kTextureLogLimit)
        {
            ++mTexturesLogged;
            Log(Debug::Info) << "Remix texture " << mTexturesLogged << ": " << std::hex << std::uppercase
                             << hash << std::nouppercase << std::dec << " " << image.getFileName() << " "
                             << width << "x" << height << " " << cached.mFormat << " "
                             << (colour ? "sRGB" : "linear") << ", " << mipLevels
                             << " mip" << (mipLevels == 1 ? "" : "s")
                             << (mTexturesLogged == kTextureLogLimit ? " (last of these)" : "");
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
        const osg::Image* image = surface.mExplicitImage;
        if (image == nullptr)
        {
            if (surface.mTexture == nullptr)
                return mDefaultMaterial;
            image = surface.mTexture->getImage();
        }
        if (image == nullptr)
            return mDefaultMaterial;

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
        mix(static_cast<unsigned long long>(roughness * 255.0f + 0.5f));
        mix(static_cast<unsigned long long>(metallic * 255.0f + 0.5f));
        mix(static_cast<unsigned long long>(emissive * 16.0f + 0.5f));

        if (auto found = mMaterials.find(key); found != mMaterials.end())
            return found->second;

        // One line per distinct material, capped. Each of these has cost real time to work out from the
        // screen alone: "alpha is not working" has three indistinguishable causes -- no cutout requested,
        // a cutout against a texture with no alpha channel, and a cutout at a threshold nothing can pass
        // -- and "everything looks like plastic" is either no rule matching or the rule being wrong.
        if (mMaterialsLogged < kMaterialLogLimit)
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
                             << (mMaterialsLogged == kMaterialLogLimit ? " (last of these)" : "");
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
            metallic, alphaTestReference, normalHash, emissive, blendType);
        if (handle == 0)
        {
            // Remembered as the fallback so a material the runtime refused is not retried every frame.
            mMaterials.emplace(key, mDefaultMaterial);
            return mDefaultMaterial;
        }

        mMaterials.emplace(key, handle);
        // Recorded for the mesh hash, which XORs the albedo texture hash the way Remix's D3D9 path does.
        mMaterialAlbedoHashes[handle] = textureHash;
        return handle;
    }

    void RemixScene::drawSubmitted(unsigned long long mesh, const float* transform,
        unsigned int categoryFlags, bool doubleSided, const SceneUtil::RigGeometry* rig,
        unsigned int pickingValue)
    {
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
        const float particleEmissive = surface.mAdditive ? kParticleEmissive : 0.0f;
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

            const osg::Vec3f across = cameraRight * half;
            const osg::Vec3f down = cameraUp * half;
            // Facing the camera, which for a billboard is the direction the quad is built against.
            const osg::Vec3f normal = (cameraRight ^ cameraUp);

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
                mVertexScratch.push_back(vertex);
            }

            mIndexScratch.insert(mIndexScratch.end(),
                { base, base + 1u, base + 2u, base, base + 2u, base + 3u });
        }

        if (mVertexScratch.empty() || mIndexScratch.empty())
            return;

        // Hashed from the address and the frame, because the handle *is* the hash: reusing one while the
        // previous mesh is still queued for destruction would alias the two.
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
        mParticleMeshes.emplace(key, entry);

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
        // Injective over (geometry, layer) -- see mGeometryIdentities in the header for why this is a
        // combination rather than a hash.
        const std::uint64_t key = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&geometry))
                * 0x100000001B3ull
            + surface.mCoverageLayer;

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
            GeometryIdentity& identity = memo->second;
            if (identity.mVertexCount == vertexCount && identity.mMaterial == material
                && identity.mModifiedCount == positions->getModifiedCount()
                && std::equal(std::begin(identity.mTexMat), std::end(identity.mTexMat),
                    std::begin(surface.mTexMat)))
            {
                // The mesh can still be missing here, because eviction works on meshes and this memo is
                // only a lookup cache. Falling through rebuilds it rather than returning a dead handle.
                if (auto found = mMeshes.find(identity.mMeshHash); found != mMeshes.end())
                {
                    identity.mLastUsedFrame = mFrame;
                    found->second.mLastUsedFrame = mFrame;
                    mLastMeshBonesPerVertex = found->second.mBonesPerVertex;
                    return found->second.mHandle;
                }
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
        const unsigned long long geometryHash = RemixRT::AssetHash::d3d9Geometry(mVertexScratch.data(),
            sizeof(RemixRT::Runtime::Vertex), vertexCount, mIndexScratch.data(),
            static_cast<unsigned int>(mIndexScratch.size()));
        if (geometryHash != 0)
        {
            const auto albedo = mMaterialAlbedoHashes.find(material);
            hash = RemixRT::AssetHash::avoidZero(
                geometryHash ^ (albedo != mMaterialAlbedoHashes.end() ? albedo->second : 0ull));
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
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            auto found = mMeshes.find(hash);
            if (found == mMeshes.end())
                break;

            if (found->second.mMaterial == material
                && std::equal(std::begin(found->second.mTexMat), std::end(found->second.mTexMat),
                    std::begin(surface.mTexMat)))
            {
                found->second.mLastUsedFrame = mFrame;
                mLastMeshBonesPerVertex = found->second.mBonesPerVertex;
                reusedHandle = found->second.mHandle;
                reused = true;
                break;
            }

            hash = RemixRT::AssetHash::avoidZero(RemixRT::AssetHash::combine(material, hash));
        }

        if (reused)
        {
            GeometryIdentity identity;
            identity.mMeshHash = hash;
            identity.mLastUsedFrame = mFrame;
            identity.mVertexCount = vertexCount;
            identity.mModifiedCount = positions->getModifiedCount();
            identity.mMaterial = material;
            std::copy(std::begin(surface.mTexMat), std::end(surface.mTexMat), std::begin(identity.mTexMat));
            mGeometryIdentities[key] = identity;
            ++mMeshesShared;
            return reusedHandle;
        }

        RemixRT::Runtime::Skinning skinning;
        if (rig != nullptr)
        {
            skinning.mBonesPerVertex = buildSkinning(*rig, vertexCount);
            skinning.mWeights = mWeightScratch.data();
            skinning.mBoneIndices = mBoneIndexScratch.data();
        }

        ++mMeshesCreated;
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

        GeometryIdentity identity;
        identity.mMeshHash = hash;
        identity.mLastUsedFrame = mFrame;
        identity.mVertexCount = vertexCount;
        identity.mModifiedCount = positions->getModifiedCount();
        identity.mMaterial = material;
        std::copy(std::begin(surface.mTexMat), std::end(surface.mTexMat), std::begin(identity.mTexMat));
        mGeometryIdentities[key] = identity;

        mLastMeshBonesPerVertex = skinning.mBonesPerVertex;
        return handle;
    }

    void RemixScene::evictStaleMeshes()
    {
        if (mFrame < kMeshEvictionFrames)
            return;

        for (auto it = mMeshes.begin(); it != mMeshes.end();)
        {
            if (mFrame - it->second.mLastUsedFrame > kMeshEvictionFrames)
            {
                mRuntime.destroyMesh(it->second.mHandle);
                it = mMeshes.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // The identity memos are evicted on the same schedule rather than alongside their mesh, because
        // the mapping is many-to-one now: several geometries can name the same mesh, so a mesh going away
        // says nothing about whether any particular memo is still wanted. Leaving them would grow the map
        // for the whole session as the player moves and cells page out.
        for (auto it = mGeometryIdentities.begin(); it != mGeometryIdentities.end();)
        {
            if (mFrame - it->second.mLastUsedFrame > kMeshEvictionFrames)
                it = mGeometryIdentities.erase(it);
            else
                ++it;
        }
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
