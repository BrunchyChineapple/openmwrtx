#include "remixscene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <osg/Camera>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osg/TriangleIndexFunctor>

#include <osg/AlphaFunc>
#include <osg/Image>
#include <osg/Light>
#include <osg/StateSet>
#include <osg/Texture2D>

#include <components/debug/debuglog.hpp>
#include <components/sceneutil/lightmanager.hpp>
#include <components/settings/values.hpp>

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
    /// OpenMW's lights are points with an attenuation curve; they carry no physical size. A path tracer
    /// needs one, because a true point light gives razor-sharp shadows and infinite radiance at the
    /// source. About twenty units is a fist-sized emitter at Morrowind's scale, which reads as a torch
    /// or a candle rather than a studio softbox.
    constexpr float kLightRadius = 20.0f;

    /// Converts OpenMW's light intensity into radiance for a sphere of kLightRadius.
    ///
    /// Not physically derived, and it cannot be: OpenMW's lighting is an artist-tuned attenuation curve
    /// evaluated per vertex, with no defined relationship to radiometric units. This is the constant
    /// that makes torches read as torches at the current radius, and it has to be retuned if
    /// kLightRadius changes, because radiance for a fixed total power scales with the inverse square of
    /// the radius.
    constexpr float kLightRadianceScale = 400.0f;

    /// Smallest change in a light's position or radiance that justifies recreating it.
    ///
    /// The API has no light-update call, so a change means destroy and recreate. OpenMW's lights jitter
    /// continuously -- carried lights follow animated bones, and many flicker every frame -- so an exact
    /// comparison would recreate almost every light every frame. These thresholds are below the point
    /// where a difference is visible in a path-traced frame.
    constexpr float kLightMoveEpsilon = 1.0f;
    constexpr float kLightRadianceEpsilon = 0.01f;

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
    constexpr unsigned int kGlDxt1Rgb = 0x83F0;
    constexpr unsigned int kGlDxt1Rgba = 0x83F1;
    constexpr unsigned int kGlDxt3 = 0x83F2;
    constexpr unsigned int kGlDxt5 = 0x83F3;

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
    unsigned char alphaTestReferenceFor(const osg::StateSet& stateSet)
    {
        // Only meaningful when the mode is actually enabled. A state set can carry an AlphaFunc while
        // leaving GL_ALPHA_TEST off, and honouring it then punches holes in solid geometry.
        if ((stateSet.getMode(GL_ALPHA_TEST) & osg::StateAttribute::ON) == 0)
            return 0;

        const auto* alphaFunc
            = dynamic_cast<const osg::AlphaFunc*>(stateSet.getAttribute(osg::StateAttribute::ALPHAFUNC));
        if (alphaFunc == nullptr)
            return kDefaultAlphaTestReference;

        // Only the "keep what is more opaque than this" comparisons map onto a path tracer's cutout.
        // The others exist but do not describe a cutout, and guessing at them would be worse than
        // treating the surface as opaque.
        const auto function = alphaFunc->getFunction();
        if (function != osg::AlphaFunc::GREATER && function != osg::AlphaFunc::GEQUAL)
            return 0;

        const float reference = alphaFunc->getReferenceValue();
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
        SubmitVisitor(MWRender::RemixScene& scene, unsigned int skipMask)
            : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ACTIVE_CHILDREN)
            , mScene(scene)
        {
            // The traversal mask is the whole mask mechanism as far as this visitor is concerned. OSG
            // tests traversalMask & nodeMask in Node::accept, so a subgraph masked exclusively as GUI or
            // render-to-texture is excluded here, while ordinary geometry with the default all-bits mask
            // passes.
            setTraversalMask(~skipMask);
        }

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
            MWRender::RemixScene::SurfaceState surface = currentSurface();
            if (const osg::StateSet* stateSet = drawable.getStateSet())
                mergeState(*stateSet, surface);

            const unsigned long long mesh = mScene.submitGeometry(*geometry, surface);
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
            mScene.drawSubmitted(mesh, transform, currentCategories(), true);
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

        /// Folds one state set into \a surface. Later calls override earlier ones, which matches OSG's
        /// precedence for attributes that carry no override flag.
        static void mergeState(const osg::StateSet& stateSet, MWRender::RemixScene::SurfaceState& surface)
        {
            if (const auto* texture = dynamic_cast<const osg::Texture2D*>(
                    stateSet.getTextureAttribute(0, osg::StateAttribute::TEXTURE)))
            {
                surface.mTexture = texture;
            }

            // Recomputed rather than inherited when the state set says anything about alpha testing, so
            // that a subgraph turning the test off is respected as well as one turning it on.
            if (stateSet.getMode(GL_ALPHA_TEST) != osg::StateAttribute::INHERIT
                || stateSet.getAttribute(osg::StateAttribute::ALPHAFUNC) != nullptr)
            {
                surface.mAlphaTestReference = alphaTestReferenceFor(stateSet);
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
        // Self-lit by default, and that is not a stopgap for looks: no lights are submitted yet, so a
        // purely reflective surface indoors receives nothing and renders black. Black geometry and
        // absent geometry look identical, which is exactly the distinction that needs making right now.
        const bool emissive = envFlag("OPENMW_REMIX_EMISSIVE", true);
        constexpr unsigned long long kDefaultMaterialHash = 0x0B7A5E'0000'0001ull;
        mDefaultMaterial = mRuntime.createFlatMaterial(
            kDefaultMaterialHash, 0.6f, 0.6f, 0.6f, 0.7f, 0.0f, emissive ? kWorldEmissive : 0.0f);
        if (mDefaultMaterial == 0)
            Log(Debug::Error) << "Remix scene: could not create the default material; geometry submission "
                                 "will not work";
        else
            Log(Debug::Info) << "Remix scene: default material is " << (emissive ? "self-lit" : "unlit")
                             << " (OPENMW_REMIX_EMISSIVE=0 to disable once real lights are submitted)";

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
        const float scale = kLightRadianceScale * (fade > 0.0f ? fade : 1.0f);

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
            const bool recoloured = std::abs(cached.mRadiance[0] - radiance[0]) > kLightRadianceEpsilon
                || std::abs(cached.mRadiance[1] - radiance[1]) > kLightRadianceEpsilon
                || std::abs(cached.mRadiance[2] - radiance[2]) > kLightRadianceEpsilon;

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
            = mRuntime.createSphereLight(hash, position, radiance, kLightRadius);
        if (handle == 0)
            return;

        CachedLight cached;
        cached.mHandle = handle;
        cached.mLastUsedFrame = mFrame;
        std::copy(std::begin(position), std::end(position), std::begin(cached.mPosition));
        std::copy(std::begin(radiance), std::end(radiance), std::begin(cached.mRadiance));
        cached.mRadius = kLightRadius;
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
        const unsigned int skipMask = Mask_GUI | Mask_RenderToTexture | Mask_FirstPerson | Mask_Debug;

        SubmitVisitor visitor(*this, skipMask);
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
                             << mTexturesUploaded << " textures"
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

    unsigned long long RemixScene::submitGeometry(osg::Geometry& geometry, const SurfaceState& surface)
    {
        return meshFor(geometry, materialFor(surface));
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
            switch (internalFormat)
            {
                case kGlDxt1Rgb:
                    format = RemixRT::Runtime::Format_BC1_RGB;
                    break;
                case kGlDxt1Rgba:
                    format = RemixRT::Runtime::Format_BC1_RGBA;
                    break;
                case kGlDxt3:
                    format = RemixRT::Runtime::Format_BC2;
                    break;
                case kGlDxt5:
                    format = RemixRT::Runtime::Format_BC3;
                    break;
                default:
                    Log(Debug::Verbose) << "Remix scene: unsupported compressed texture format 0x"
                                        << std::hex << internalFormat << std::dec
                                        << "; using the untextured material for it";
                    mTextures.emplace(key, cached);
                    return 0;
            }

            uploadData = data;
            uploadSize = static_cast<unsigned long long>(image.getTotalSizeInBytesIncludingMipmaps());
            const unsigned int levels = static_cast<unsigned int>(image.getNumMipmapLevels());
            mipLevels = levels > 0 ? levels : 1u;
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
        if (surface.mTexture == nullptr)
            return mDefaultMaterial;

        const osg::Image* image = surface.mTexture->getImage();
        if (image == nullptr)
            return mDefaultMaterial;

        const unsigned long long textureHash = textureFor(*image);
        if (textureHash == 0)
            return mDefaultMaterial;

        // The alpha threshold is part of the material identity: the same texture can be a cutout on one
        // mesh and opaque on another, and OpenMW decides that per state set, not per texture.
        const unsigned long long key
            = textureHash ^ (static_cast<unsigned long long>(surface.mAlphaTestReference) << 56);
        if (auto found = mMaterials.find(key); found != mMaterials.end())
            return found->second;

        const unsigned long long handle = mRuntime.createTexturedMaterial(
            key | 1ull, textureHash, kTexturedRoughness, 0.0f, surface.mAlphaTestReference);
        if (handle == 0)
        {
            // Remembered as the fallback so a material the runtime refused is not retried every frame.
            mMaterials.emplace(key, mDefaultMaterial);
            return mDefaultMaterial;
        }

        mMaterials.emplace(key, handle);
        return handle;
    }

    void RemixScene::drawSubmitted(
        unsigned long long mesh, const float* transform, unsigned int categoryFlags, bool doubleSided)
    {
        mRuntime.drawInstance(mesh, transform, categoryFlags, doubleSided);
    }

    unsigned long long RemixScene::meshFor(osg::Geometry& geometry, unsigned long long material)
    {
        if (material == 0)
            return 0;

        const void* key = &geometry;

        const auto* positions = dynamic_cast<const osg::Vec3Array*>(geometry.getVertexArray());
        if (positions == nullptr || positions->empty())
            return 0;

        const unsigned int vertexCount = static_cast<unsigned int>(positions->size());

        if (auto found = mMeshes.find(key); found != mMeshes.end())
        {
            // Cheap staleness check. A full content hash every frame would cost more than it saves, but
            // a changed vertex count definitely means different geometry at the same address, and
            // reusing the cached mesh then would draw the wrong thing. The material is checked too
            // because it is baked into the surface at creation time and cannot be swapped afterwards.
            if (found->second.mVertexCount == vertexCount && found->second.mMaterial == material)
            {
                found->second.mLastUsedFrame = mFrame;
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
                vertex.mTexcoord[0] = (*texcoords)[i].x();
                vertex.mTexcoord[1] = (*texcoords)[i].y();
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

        const unsigned long long handle = mRuntime.createMesh(hash, mVertexScratch.data(), vertexCount,
            mIndexScratch.data(), static_cast<unsigned int>(mIndexScratch.size()), material);
        if (handle == 0)
            return 0;

        CachedMesh cached;
        cached.mHandle = handle;
        cached.mLastUsedFrame = mFrame;
        cached.mVertexCount = vertexCount;
        cached.mIndexCount = static_cast<unsigned int>(mIndexScratch.size());
        cached.mMaterial = material;
        mMeshes.emplace(key, cached);
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
