#include "remixscene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <osg/Camera>
#include <osg/FrameStamp>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osg/TriangleIndexFunctor>

#include <osg/AlphaFunc>
#include <osg/Image>
#include <osg/Light>
#include <osg/StateSet>
#include <osg/TexMat>
#include <osg/Texture2D>

#include <components/debug/debuglog.hpp>
#include <components/sceneutil/lightmanager.hpp>
#include <components/sceneutil/morphgeometry.hpp>
#include <components/sceneutil/riggeometry.hpp>
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
    constexpr float kLightPowerDefault = 160000.0f;

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
        RemixRT::Runtime::TextureFormat mFormat;
        unsigned int mBlockBytes;
        const char* mName;
    };

    constexpr CompressedFormat kCompressedFormats[] = {
        // S3TC / DXT, GL_EXT_texture_compression_s3tc plus its sRGB counterpart.
        { 0x83F0, RemixRT::Runtime::Format_BC1_RGB, 8, "BC1_RGB" }, // GL_COMPRESSED_RGB_S3TC_DXT1
        { 0x8C4C, RemixRT::Runtime::Format_BC1_RGB, 8, "BC1_RGB(srgb)" },
        { 0x83F1, RemixRT::Runtime::Format_BC1_RGBA, 8, "BC1_RGBA" }, // DXT1 with a one-bit alpha
        { 0x8C4D, RemixRT::Runtime::Format_BC1_RGBA, 8, "BC1_RGBA(srgb)" },
        { 0x83F2, RemixRT::Runtime::Format_BC2, 16, "BC2" }, // DXT3, explicit four-bit alpha
        { 0x8C4E, RemixRT::Runtime::Format_BC2, 16, "BC2(srgb)" },
        { 0x83F3, RemixRT::Runtime::Format_BC3, 16, "BC3" }, // DXT5, interpolated alpha
        { 0x8C4F, RemixRT::Runtime::Format_BC3, 16, "BC3(srgb)" },
        // RGTC, GL_ARB_texture_compression_rgtc. Two-channel and linear: normal maps, never colour, so
        // these take the UNORM Remix format. The signed spelling is listed because the loader can produce
        // it, not because a signed normal map would be interpreted correctly here.
        { 0x8DBD, RemixRT::Runtime::Format_BC5, 16, "BC5" }, // GL_COMPRESSED_RG_RGTC2
        { 0x8DBE, RemixRT::Runtime::Format_BC5, 16, "BC5(signed)" },
        // BPTC, GL_ARB_texture_compression_bptc. What modern high-resolution replacement packs use.
        { 0x8E8C, RemixRT::Runtime::Format_BC7, 16, "BC7" }, // GL_COMPRESSED_RGBA_BPTC_UNORM
        { 0x8E8D, RemixRT::Runtime::Format_BC7, 16, "BC7(srgb)" },
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

    /// Roughness for textured surfaces.
    ///
    /// A single value for everything, because Morrowind's materials carry no roughness information at
    /// all -- there is one diffuse texture and nothing else. Fairly rough, since most of the game's
    /// surfaces are stone, wood and cloth; specific materials will need per-texture overrides later.
    constexpr float kTexturedRoughness = 0.65f;

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
        SubmitVisitor(MWRender::RemixScene& scene, unsigned int skipMask, const osg::Vec3f& eye)
            : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ACTIVE_CHILDREN)
            , mScene(scene)
            , mEye(eye)
        {
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
            osg::Geometry* geometry = drawable.asGeometry();

            // Skinned geometry arrives as a RigGeometry, which is a Drawable and not a Geometry, so
            // asGeometry() returns null and the drawable was simply dropped -- that is why actors were
            // missing every part of themselves that moves. What it does hold is the bind pose plus the
            // influence data needed to deform it, which is exactly what a path tracer wants: the mesh
            // goes over once and only the bone transforms are resubmitted per frame.
            auto* rig = dynamic_cast<SceneUtil::RigGeometry*>(&drawable);
            if (rig != nullptr)
                geometry = rig->getSourceGeometry().get();

            // Morph geometry, the other Drawable that hides a Geometry, is submitted in its base pose.
            //
            // Unlike skinning, there is nothing to hand the runtime here: the Remix API has no concept of
            // morph targets, so the alternative to the base pose is either an invisible mesh or a mesh
            // rebuilt from CPU-blended vertices every frame -- which is the churn the skinning path exists
            // to avoid, and for a much smaller payoff. Morrowind uses vertex morphs sparingly, so the
            // visible cost is a handful of meshes that do not animate rather than a handful missing.
            if (const auto* morph = dynamic_cast<const SceneUtil::MorphGeometry*>(&drawable))
                geometry = morph->getSourceGeometry().get();

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
            if (const auto* terrain = dynamic_cast<const Terrain::TerrainDrawable*>(&drawable))
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

            // OSG is row-vector (p * M); the Remix transform is applied as p' = M * p with translation
            // in the last column. Hence the transpose of the 3x3 and the translation read from OSG's
            // fourth row. If instances come out rotated or mirrored, this is the line to question.
            float transform[12];
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                    transform[row * 4 + col] = static_cast<float>(mMatrix(col, row));
                transform[row * 4 + 3] = static_cast<float>(mMatrix(3, row));
            }

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
            mScene.drawSubmitted(
                mesh, transform, categories, true, mScene.lastMeshIsSkinned() ? rig : nullptr);
            mScene.noteInstancePosition(mMatrix(3, 0), mMatrix(3, 1), mMatrix(3, 2));
            ++mInstances;
        }

        unsigned int instances() const { return mInstances; }
        bool clamped() const { return mClamped; }

    private:
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
        osg::Matrix mMatrix;
        std::vector<unsigned int> mCategoryStack;
        std::vector<MWRender::RemixScene::SurfaceState> mSurfaceStack;
        unsigned int mInstances = 0;
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

        mLightRadius = envFloat("OPENMW_REMIX_LIGHT_RADIUS", kLightRadiusDefault);
        mLightPower = envFloat("OPENMW_REMIX_LIGHT_POWER", kLightPowerDefault);
        Log(Debug::Info) << "Remix scene: light emitter radius " << mLightRadius
                         << " units, radiance at that radius "
                         << (mLightPower / (mLightRadius * mLightRadius))
                         << " per unit of OpenMW light colour (OPENMW_REMIX_LIGHT_RADIUS and "
                            "OPENMW_REMIX_LIGHT_POWER override both)";

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

        // Actor fade is OpenMW's way of dimming a carried light as its owner fades out. Ignoring it
        // leaves lights at full strength on invisible actors.
        const float fade = source.getActorFade();
        // Power divided by radius squared, so the emitter size and the brightness stay independent.
        const float scale
            = (mLightPower / (mLightRadius * mLightRadius)) * (fade > 0.0f ? fade : 1.0f);

        const float radiance[3]
            = { diffuse.r() * scale, diffuse.g() * scale, diffuse.b() * scale };
        if (radiance[0] <= 0.0f && radiance[1] <= 0.0f && radiance[2] <= 0.0f)
            return;

        const float position[3]
            = { static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) };

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

            if (!moved && !recoloured)
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
        mSkinnedDropped = 0;
        mHaveExtent = false;
        if (sceneRoot == nullptr || mDefaultMaterial == 0)
            return 0;

        ++mFrame;

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
        const unsigned int skipMask
            = Mask_GUI | Mask_RenderToTexture | Mask_FirstPerson | Mask_Debug | Mask_SimpleWater;

        SubmitVisitor visitor(*this, skipMask,
            osg::Vec3f(static_cast<float>(eye.x()), static_cast<float>(eye.y()),
                static_cast<float>(eye.z())));
        sceneRoot->accept(visitor);
        mLastInstanceCount = visitor.instances();

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

        // Log the first submission, and then again whenever the scene goes from empty to populated or
        // back. A one-shot on frame 1 was actively misleading: the first frame happens before the world
        // is up, so it reported zero instances and then never spoke again, which read as "the traversal
        // never works" when it only meant "there was nothing there yet".
        const bool populated = mLastInstanceCount > 0;
        if (!mLoggedFirstSubmit || populated != mWasPopulated)
        {
            mLoggedFirstSubmit = true;
            mWasPopulated = populated;
            Log(Debug::Info) << "Remix scene: handed over " << mLastInstanceCount << " instances from "
                             << mMeshes.size() << " meshes, " << mLastLightCount << " lights and "
                             << mTexturesUploaded << " textures; " << mSkinnedInstances
                             << " instances were skinned and " << mSkinnedDropped
                             << " skins were not ready"
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

    unsigned long long RemixScene::textureFor(const osg::Image& image)
    {
        const void* key = &image;
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

        const unsigned long long hash
            = (reinterpret_cast<unsigned long long>(key) * 0xD6E8FEB86659FD93ull) | 1ull;

        std::vector<unsigned char> converted;
        const void* uploadData = nullptr;
        unsigned long long uploadSize = 0;
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
            format = compressed->mFormat;
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
            format = RemixRT::Runtime::Format_RGBA8;
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

        if (mRuntime.createTexture(hash, static_cast<unsigned int>(width),
                static_cast<unsigned int>(height), mipLevels, format, uploadData, uploadSize)
            == 0)
        {
            mTextures.emplace(key, cached);
            return 0;
        }

        cached.mHash = hash;
        cached.mUsable = true;
        mTextures.emplace(key, cached);
        ++mTexturesUploaded;
        return hash;
    }

    unsigned long long RemixScene::materialFor(const SurfaceState& surface)
    {
        // Water before any texture consideration. OpenMW's water is a shader effect -- its texture units
        // hold a normal map and render targets, none of which is an albedo -- so whatever is bound there
        // is not what the surface should look like. A path tracer wants the physical description
        // instead, and gets a much better result from it than the raster version manages.
        if (surface.mIsWater && mWaterMaterial != 0)
            return mWaterMaterial;

        if (surface.mTexture == nullptr)
            return mDefaultMaterial;

        const osg::Image* image = surface.mTexture->getImage();
        if (image == nullptr)
            return mDefaultMaterial;

        const unsigned long long textureHash = textureFor(*image);
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
        unsigned char alphaTestReference = surface.mAlphaTestReference;
        if (alphaTestReference == 0 && surface.mAlphaBlend)
            alphaTestReference = mBlendCutout;

        // The alpha threshold is part of the material identity: the same texture can be a cutout on one
        // mesh and opaque on another, and OpenMW decides that per state set, not per texture.
        const unsigned long long key
            = textureHash ^ (static_cast<unsigned long long>(alphaTestReference) << 56);
        if (auto found = mMaterials.find(key); found != mMaterials.end())
            return found->second;

        // One line per distinct material, capped. "Alpha is not working" has three separate causes --
        // no cutout requested, a cutout requested against a texture with no alpha channel, or a cutout
        // requested at a threshold nothing can pass -- and they are indistinguishable from the screen.
        if (mMaterialsLogged < kMaterialLogLimit)
        {
            ++mMaterialsLogged;
            const auto found = mTextures.find(image);
            Log(Debug::Info) << "Remix material " << mMaterialsLogged << ": texture 0x" << std::hex
                             << textureHash << std::dec << " " << image->s() << "x" << image->t()
                             << " format " << (found != mTextures.end() ? found->second.mFormat : "?")
                             << " alphaTest " << static_cast<unsigned int>(surface.mAlphaTestReference)
                             << " blend " << (surface.mAlphaBlend ? "yes" : "no") << " -> cutout "
                             << static_cast<unsigned int>(alphaTestReference)
                             << (mMaterialsLogged == kMaterialLogLimit ? " (last of these)" : "");
        }

        const unsigned long long handle = mRuntime.createTexturedMaterial(
            key | 1ull, textureHash, kTexturedRoughness, 0.0f, alphaTestReference);
        if (handle == 0)
        {
            // Remembered as the fallback so a material the runtime refused is not retried every frame.
            mMaterials.emplace(key, mDefaultMaterial);
            return mDefaultMaterial;
        }

        mMaterials.emplace(key, handle);
        return handle;
    }

    void RemixScene::drawSubmitted(unsigned long long mesh, const float* transform,
        unsigned int categoryFlags, bool doubleSided, const SceneUtil::RigGeometry* rig)
    {
        if (rig == nullptr)
        {
            mRuntime.drawInstance(mesh, transform, categoryFlags, doubleSided);
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
                mBoneTransformScratch.data(), boneCount))
            ++mSkinnedInstances;
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
        const void* key = &geometry;

        const auto* positions = dynamic_cast<const osg::Vec3Array*>(geometry.getVertexArray());
        if (positions == nullptr || positions->empty())
            return 0;

        const unsigned int vertexCount = static_cast<unsigned int>(positions->size());

        if (auto found = mMeshes.find(key); found != mMeshes.end())
        {
            // Cheap staleness check. A full content hash every frame would cost more than it saves, but
            // a changed vertex count definitely means different geometry at the same address, and
            // reusing the cached mesh then would draw the wrong thing. The material and the texture
            // matrix are checked too, because both are baked in at creation time -- the material into
            // the surface, the matrix into the texcoords -- and neither can be swapped afterwards.
            if (found->second.mVertexCount == vertexCount && found->second.mMaterial == material
                && std::equal(std::begin(found->second.mTexMat), std::end(found->second.mTexMat),
                    std::begin(surface.mTexMat)))
            {
                found->second.mLastUsedFrame = mFrame;
                mLastMeshBonesPerVertex = found->second.mBonesPerVertex;
                return found->second.mHandle;
            }
            mRuntime.destroyMesh(found->second.mHandle);
            mMeshes.erase(found);
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

            // Opaque white. Vertex colours are ignored until materials are real; feeding them in now
            // would tint the flat debug material and make it harder to read.
            vertex.mColor = 0xFFFFFFFFu;
        }

        // Derived from the address, which is unique for as long as the geometry lives, and mixed so that
        // adjacent allocations do not produce adjacent hashes. The cache destroys the mesh on eviction,
        // so a later allocation reusing this address cannot inherit it.
        const unsigned long long hash
            = (reinterpret_cast<unsigned long long>(key) * 0x9E3779B97F4A7C15ull) | 1ull;

        RemixRT::Runtime::Skinning skinning;
        if (rig != nullptr)
        {
            skinning.mBonesPerVertex = buildSkinning(*rig, vertexCount);
            skinning.mWeights = mWeightScratch.data();
            skinning.mBoneIndices = mBoneIndexScratch.data();
        }

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
        cached.mBonesPerVertex = skinning.mBonesPerVertex;
        std::copy(std::begin(surface.mTexMat), std::end(surface.mTexMat), std::begin(cached.mTexMat));
        mMeshes.emplace(key, cached);
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
    }
}
