#ifndef OPENMW_MWRENDER_REMIXSCENE_H
#define OPENMW_MWRENDER_REMIXSCENE_H

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <osg/Matrixf>
#include <osg/Node>
#include <osg/observer_ptr>
#include <osg/ref_ptr>

#include <components/remixrt/runtime.hpp>

namespace osg
{
    class Camera;
    class Drawable;
    class Geometry;
    class Image;
    class Texture2D;
}

namespace SceneUtil
{
    class LightSource;
    class MorphGeometry;
    class RigGeometry;
}

namespace osgParticle
{
    class ParticleSystem;
}

namespace MWRender
{
    /// Feeds OpenMW's scene graph to the Remix runtime, one frame at a time.
    ///
    /// Remix reconstructs a scene from meshes, materials, lights and a camera rather than intercepting
    /// draw calls, so something has to walk OSG and hand it over. That is this. It lives in mwrender
    /// rather than in components/remixrt because deciding *what* a subgraph is -- sky, terrain, actor,
    /// GUI -- is OpenMW policy, and VisMask is the thing that already knows.
    ///
    /// Design notes worth having before changing anything here:
    ///
    /// - **Units are OpenMW's, not metres.** The camera is submitted in OpenMW units, so geometry has to
    ///   match or the two desynchronise. One metre is about 70 units, which means Remix's distance-based
    ///   effects (volumetrics, fog, light falloff) are all being fed numbers ~70x larger than they would
    ///   see from a normal game. Converting is a real decision to make later; doing it to geometry alone
    ///   would be worse than not doing it at all.
    ///
    /// - **Meshes are cached by osg::Geometry pointer.** OpenMW shares geometry heavily between
    ///   instances -- ObjectPaging and Groundcover exist to do exactly that -- so converting per instance
    ///   would multiply the work by the instance count. The API has no mesh-update entry point, so a
    ///   changed mesh means destroy and recreate.
    ///
    /// - **The mesh hash IS the identity.** remixapi_MeshHandle is a reinterpret_cast of the hash the
    ///   caller supplies, so two live meshes sharing a hash silently alias. The hash is derived from the
    ///   geometry pointer, which is unique while it lives; entries are destroyed on eviction so a reused
    ///   address cannot inherit a stale mesh.
    class RemixScene
    {
    public:
        explicit RemixScene(RemixRT::Runtime& runtime);
        ~RemixScene();

        RemixScene(const RemixScene&) = delete;
        RemixScene& operator=(const RemixScene&) = delete;

        /// Submits the camera and everything visible under \a sceneRoot for this frame.
        ///
        /// Call once per frame before Runtime::present, which is what actually renders what was
        /// submitted. Returns the number of instances handed over, which is the single most useful
        /// number when the frame comes out empty: zero means the traversal found nothing, and anything
        /// else moves the question downstream.
        unsigned int submit(osg::Node* sceneRoot, const osg::Camera& camera);

        /// Instances submitted on the last call to submit().
        unsigned int lastInstanceCount() const { return mLastInstanceCount; }

        /// Whether the mesh the last submitGeometry() returned needs bone transforms per instance.
        bool lastMeshIsSkinned() const { return mLastMeshBonesPerVertex > 0; }

        /// Lights submitted on the last call to submit().
        unsigned int lastLightCount() const { return mLastLightCount; }

        /// Meshes currently held in the cache.
        unsigned int cachedMeshCount() const { return static_cast<unsigned int>(mMeshes.size()); }

        /// Describes the surface appearance a drawable inherits from the OSG state above it.
        ///
        /// Assembled by the traversal, because texture and alpha state in OSG live on state sets
        /// anywhere up the path, not on the drawable.
        struct SurfaceState
        {
            /// Albedo texture, or null for the untextured fallback material. Const because it is read
            /// out of state sets reached through const traversal; nothing here modifies the scene.
            const osg::Texture2D* mTexture = nullptr;
            /// Alpha-test threshold, 0..255. Zero means no cutout, i.e. fully opaque.
            unsigned char mAlphaTestReference = 0;
            /// Whether the surface is alpha blended.
            ///
            /// Tracked separately from the threshold because most of Morrowind's foliage asks for
            /// blending and no test at all, and a path tracer has to be told to make a cutout out of it
            /// -- see materialFor. Without this, alpha-blended leaves come through as solid polygons.
            bool mAlphaBlend = false;
            /// Whether the surface blends ADDITIVELY, i.e. its blend function has a destination factor of
            /// GL_ONE and so can only ever brighten what is behind it.
            ///
            /// This is the difference between a flame and a smoke plume, and it cannot be inferred from
            /// mAlphaBlend -- both are blended, and Morrowind uses the same particle machinery for both. An
            /// additive surface is emissive by construction; an alpha-blended one occludes and must be lit.
            bool mAdditive = false;
            /// An albedo supplied directly as an image, bypassing mTexture.
            ///
            /// Exists for composited terrain, whose real albedo is a render target with no image behind it
            /// -- so it is read back to the CPU and handed over this way. Takes precedence over mTexture
            /// when set, because for those chunks mTexture is the render target and unusable.
            /// Held by reference, not by pointer. This is a terrain chunk's readback image, produced on the
            /// draw thread and released when its chunk goes away, and it is captured here well before the
            /// upload that reads its pixels -- so a bare pointer only survives while the timing happens to
            /// be kind.
            osg::ref_ptr<const osg::Image> mExplicitImage;

            /// Per-vertex coverage for one terrain layer, taken from OpenMW's blend map and written into
            /// the vertex alpha.
            ///
            /// This is how a terrain layer's coverage reaches a path tracer at all. OpenMW draws each layer
            /// with the layer's diffuse tiled many times across the chunk and a blend map stretched once
            /// over it, and multiplies the two in a shader -- two textures at different frequencies, which
            /// a Remix surface with a single albedo cannot express. Sampling the blend map per vertex loses
            /// very little, because Morrowind's blend maps are already at roughly the terrain vertex
            /// density, and it needs no second texture and no bake.
            ///
            /// The consequence worth knowing is that the same chunk geometry submitted with different
            /// coverage produces a different vertex buffer and therefore a different mesh -- one per layer
            /// rather than one per chunk. That is also what a D3D9 game doing multi-pass terrain hands the
            /// runtime, so it is the cost the runtime is built for rather than a surprise.
            const osg::Image* mCoverageImage = nullptr;
            /// Blend map texture matrix, applied to the raw texcoord before sampling mCoverageImage.
            /// Separate from mTexMat because that one carries the *diffuse* tiling; the two differ by
            /// exactly the frequency this whole mechanism exists to reconcile.
            bool mHasCoverageTexMat = false;
            float mCoverageTexMat[6] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
            /// Which layer of its chunk this is, so two layers sharing a geometry address do not collide
            /// in the mesh memo. Zero for everything that is not a terrain layer.
            unsigned int mCoverageLayer = 0;

            /// Whether this surface wants real order-independent transparency instead of the cutout that
            /// materialFor otherwise substitutes for alpha blending.
            ///
            /// The substitution is right for foliage and wrong for particles, and nothing about the state
            /// set distinguishes the two -- both arrive as "blend on, no alpha test". So the caller that
            /// knows which it is has to say so, which in practice means the particle path sets this.
            bool mPreferBlend = false;
            /// The 2D affine part of unit 0's texture matrix, as
            /// { m00, m01, m10, m11, m30, m31 }, applied to texcoords as a row vector.
            ///
            /// Has to be carried because Remix vertices have no UV transform: the only place a texture
            /// matrix can go is baked into the texcoords. Ignoring it is not cosmetic -- OpenMW scales
            /// terrain UVs this way to tile a layer across a chunk, so without it the base texture is
            /// stretched once over the whole chunk and reads as flat, featureless ground.
            float mTexMat[6] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
            /// Whether mTexMat differs from identity, so the common case costs one bool rather than six
            /// float comparisons per drawable per frame.
            bool mHasTexMat = false;
            /// Water gets a refractive material rather than whatever texture it happens to carry.
            bool mIsWater = false;
            /// Tangent-space normal map, when one is bound. Found by reading each texture unit's
            /// SceneUtil::TextureType tag rather than by assuming a unit index, because the ShaderVisitor
            /// picks the unit dynamically from however many the state set already had.
            const osg::Texture2D* mNormalMap = nullptr;
        };

        /// Converts and caches one drawable, returning its Remix mesh handle, or 0 if unusable.
        /// Public only because the traversal visitor lives in the .cpp and calls back into here.
        ///
        /// @param rig when non-null, \a geometry is that rig's bind pose and the mesh is created with
        ///        skinning data derived from the rig's influences.
        unsigned long long submitGeometry(osg::Geometry& geometry, const SurfaceState& surface,
            const SceneUtil::RigGeometry* rig = nullptr);

        /// Queues one instance. Companion to submitGeometry, same reason for being public.
        ///
        /// Genuinely queues, rather than handing over immediately: the instance is held until the
        /// traversal finishes so that the triangle budget can admit nearest-first. See
        /// flushSubmissions for why that matters.
        ///
        /// @param rig when non-null, its current bone matrices are submitted with the instance. Must be
        ///        the same rig the mesh was created from. Read at flush time, still within this frame.
        /// @param pickingValue identifies this draw so the developer menu can resolve a click in the
        ///        scene to it. Must be distinct per draw within a frame; zero leaves it unpickable.
        /// @param distanceSquared world-space distance from the eye to this instance's bounding-sphere
        ///        centre, squared. The budget's sort key. Must be the *bound* centre and not the
        ///        transform's translation: ObjectPaging flattens static transforms into the vertices of
        ///        a merged chunk, so the translation of the largest instances in the scene says nothing
        ///        about where their geometry actually is.
        void drawSubmitted(unsigned long long mesh, const float* transform, unsigned int categoryFlags,
            bool doubleSided, const SceneUtil::RigGeometry* rig = nullptr,
            unsigned int pickingValue = 0, float distanceSquared = 0.0f);

        /// Records a submitted instance's world-space origin, for the diagnostic extent log.
        void noteInstancePosition(double x, double y, double z);

        /// Marks \a drawable's mesh as still part of the loaded scene without submitting it, for a
        /// drawable the traversal reached and then rejected.
        ///
        /// Culling is a visibility decision taken every frame; eviction is a residency decision about
        /// whether content still exists. Those were the same signal -- mesh lifetime was driven by
        /// "submitted recently" -- so anything the frustum rejected stopped being touched, fell due
        /// kMeshEvictionFrames later and had its acceleration structure destroyed. Turning back rebuilt
        /// it. At half a second of eviction window that is a BLAS rebuild for most of the world every
        /// time the player spins round, and a smaller one every few steps as objects cross the
        /// apparent-size threshold.
        ///
        /// Being reached by the traversal at all is the residency signal wanted here: the drawable is in
        /// the scene graph, so its cell is loaded. Content that genuinely goes away -- an unloaded cell,
        /// or an animated mesh superseded by the next pose -- stops being traversed, is never retained,
        /// and still evicts on the same schedule as before.
        ///
        /// Public for the same reason as submitGeometry: the traversal lives in the .cpp.
        void retainCulledDrawable(osg::Drawable& drawable);

        /// retainCulledDrawable for a caller that has already resolved the geometry.
        void retainGeometry(osg::Geometry& geometry);

        /// Submits one OpenMW light source, at the world position \a x, \a y, \a z.
        ///
        /// Public for the same reason as submitGeometry: the traversal lives in the .cpp.
        void submitLight(const SceneUtil::LightSource& source, double x, double y, double z);

        /// Turns one particle system into camera-facing quads and submits them.
        ///
        /// Separate from submitGeometry because a particle system has no geometry to cache: osgParticle
        /// builds its quads during drawing and they differ every frame, so the mesh is rebuilt rather
        /// than reused.
        ///
        /// @param localToWorld the accumulated transform at the drawable, applied only when the system's
        ///        particles are in local coordinates. Some are already in world space.
        /// @param cameraRight world-space camera right, for orienting the quads.
        /// @param cameraUp world-space camera up.
        void submitParticles(const osgParticle::ParticleSystem& particles, const SurfaceState& surface,
            const osg::Matrixd& localToWorld, const osg::Vec3f& cameraRight, const osg::Vec3f& cameraUp,
            unsigned int categoryFlags);

        /// Re-sends the camera from the last submit, without submitting any geometry.
        ///
        /// For the loops that drive the viewer themselves -- loading screens, the modal message box, video
        /// playback. They present a frame but submit no scene, and a frame with no camera is one Remix does
        /// not raytrace: it never reaches injectRTX, and the screen-overlay composite lives at the end of
        /// that path, after tone mapping. So the interface those loops draw was composited by nothing. It
        /// showed as a frozen picture during cell loads and a black screen during the intro videos, which
        /// are the same fault seen with and without a previous frame to leave on screen.
        ///
        /// Only the camera. Submitting geometry here would mean walking a scene graph that a cell load is
        /// in the middle of rebuilding, to draw a world that is deliberately not being shown. An empty
        /// scene path-traces to black, which is the right backdrop for a loading screen and for a video.
        ///
        /// False when no camera has been submitted yet and none could be synthesised.
        bool resubmitCamera();

    private:
        /// Submits a self-lit quad a fixed distance in front of the camera.
        ///
        /// Bisects the two ways "instances submitted, nothing on screen" can happen. This quad shares the
        /// mesh, material and instance path with real geometry but takes its transform straight from the
        /// camera basis, so it cannot be in the wrong place. Visible means the path works and the world
        /// transforms are wrong; invisible means the path itself is, and the world transforms are not
        /// worth looking at yet.
        void submitProbeQuad(const double* eye, const double* forward, const double* up,
            const double* right);

        /// Builds the probe quad's mesh and emissive material. Called once, on first use.
        void ensureProbeAssets();
        struct CachedMesh
        {
            unsigned long long mHandle = 0;
            /// Frame this mesh was last submitted on, so unused ones can be released. Without this the
            /// cache would grow without bound as the player moves and terrain repages.
            std::uint64_t mLastUsedFrame = 0;
            unsigned int mVertexCount = 0;
            unsigned int mIndexCount = 0;
            /// Baked into the mesh's surface at creation time, so a drawable whose material changes
            /// needs the mesh rebuilt -- the API has no way to retarget an existing one.
            unsigned long long mMaterial = 0;
            /// The texture matrix the cached texcoords were built with. Baked in for the same reason as
            /// the material: it lives in the vertex data, so a change means a rebuild.
            float mTexMat[6] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
            /// The vertex array's modified count when this mesh was built.
            ///
            /// A vertex count comparison catches different geometry reusing an address, but not the same
            /// geometry whose contents changed underneath us -- which is exactly what morph animation is.
            /// OSG bumps this counter on every dirty(), and MorphGeometry::cull dirties the array on the
            /// frames it recomputes, so comparing it rebuilds precisely when the pose moved and never
            /// otherwise. Static geometry never touches it, and skinned meshes are keyed on their bind
            /// pose, so neither pays anything for this.
            unsigned int mModifiedCount = 0;
            /// Non-zero when the mesh was created with skinning, in which case instances of it have to
            /// supply bone transforms. Zero for static geometry, and also for a skin the runtime or this
            /// code refused, so it answers "does this instance need bones" rather than "is this a rig".
            unsigned int mBonesPerVertex = 0;
        };

        /// What a given osg::Geometry resolved to the last time it was submitted.
        ///
        /// Meshes are identified to Remix by a hash of their content, which is what makes the identity
        /// stable across runs and lets identical geometry share one mesh. Content hashing is too expensive
        /// to repeat for every drawable every frame, though, so the answer is memoised against the
        /// geometry's address and revalidated with the same cheap checks the mesh cache itself used to
        /// make: vertex count, OSG's modified counter, and the material and texture matrix that are baked
        /// into the mesh at creation.
        ///
        /// Separate from CachedMesh because the two are no longer one-to-one: several geometries can now
        /// resolve to the same mesh, which is the point.
        struct GeometryIdentity
        {
            unsigned long long mMeshHash = 0;
            /// Frame the geometry this memo describes was last known to be part of the loaded scene --
            /// which is not the same question as whether it was drawn. See retainGeometry.
            std::uint64_t mLastUsedFrame = 0;
            unsigned int mVertexCount = 0;
            unsigned int mModifiedCount = 0;
            unsigned long long mMaterial = 0;
            float mTexMat[6] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
            /// Which layer of its terrain chunk this identity describes; zero for everything that is not
            /// a terrain layer. Held in the value rather than folded into the map key so that a lookup by
            /// geometry address alone reaches every layer -- see mGeometryIdentities.
            unsigned int mCoverageLayer = 0;
        };

        struct CachedTexture
        {
            unsigned long long mHash = 0;
            /// Zero means the image was examined and found unusable -- an unsupported pixel format, or
            /// data already released. Cached as a negative result so the conversion is not retried for
            /// every drawable sharing it, every frame.
            bool mUsable = false;
            /// The uploaded format, as a literal, for the diagnostic in materialFor. Worth keeping
            /// because a cutout against a format with no alpha channel -- BC1_RGB, or an RGB source
            /// widened with alpha forced to 255 -- is silently a no-op, and that looks exactly like the
            /// cutout never having been requested.
            const char* mFormat = "none";
            /// Frame this texture was last resolved for a drawable. Drives eviction: without it the cache
            /// only ever grew, because destroyTexture was reachable solely from shutdown. Every other
            /// cache in this class already carries the same stamp.
            std::uint64_t mLastUsedFrame = 0;
            /// The image this entry was built from, observed rather than held.
            ///
            /// Observed, so caching an image here does not keep it alive -- the point is to learn when
            /// OpenMW frees it, which is the only trustworthy signal that this entry is garbage. An idle
            /// timer cannot answer that question: a texture can go hundreds of frames without being
            /// resolved and still belong to a cell the player is standing in.
            ///
            /// It also repairs a correctness hole. The map key is the image's address, and addresses are
            /// reused once OpenMW frees an image, so a later image landing on a freed one's address used to
            /// resolve to the *previous* image's texture. Comparing the observer against the image in hand
            /// detects that. Nothing noticed before only because nothing was ever released.
            osg::observer_ptr<const osg::Image> mImage;
            /// Bytes handed to createTexture, so a release can subtract what an upload added. Zero for a
            /// negative result and for an entry that shares another's identity.
            std::size_t mBytes = 0;
        };

        /// Whether identities are computed the way Remix's D3D9 path would, so that a replacement pack
        /// authored against a Morrowind capture binds to geometry submitted here.
        ///
        /// This is the whole emulation, not one piece of it: mesh identity reproduces Remix's D3D9 geometry
        /// hash XORed with the albedo texture hash, and texture identity reproduces Remix's D3D9 texture
        /// hash so that a pack's mat_ overrides attach. OPENMW_REMIX_PACK_MATCH=0 turns both off together
        /// and replaces them with identities that are merely stable and distinct, which nothing authored
        /// against a Morrowind capture can match.
        ///
        /// Everything else is untouched. Remix still renders, replacement assets are still loaded and still
        /// able to bind, and the mod layer is still consulted -- what changes is only which identity a
        /// surface is offered under. That distinction matters because it makes this the mode an offline
        /// converter targets: a pack re-keyed to the identities this produces binds with the emulation off,
        /// and the emulation can then be deleted rather than merely disabled.
        bool mPackMatching = true;

        /// Ceiling on texture destroys in one frame.
        ///
        /// Textures and their materials were never released during play, so a session accumulated every
        /// texture it had ever seen. That was survivable while little was replaced; once mesh replacements
        /// bound and the active grid stopped batching, each cell load pulled in far more distinct materials
        /// and their high-resolution replacement textures, and video memory ran out -- a long stall, then
        /// VK_ERROR_DEVICE_LOST. Terrain made it worse still: with composite readback on, every terrain
        /// chunk contributes one uncompressed RGBA8 composite, and a session measured 338 of them.
        ///
        /// A cell boundary orphans a whole cell's textures on one frame, and a texture destroy releases far
        /// more memory than a mesh destroy, so the herd-arrival argument that budgets evictStaleMeshes
        /// applies here with more force.
        static constexpr unsigned int kTextureDestroysPerFrame = 32;

        /// Frames an unclaimed identity is left alone before it is destroyed. See mOrphanedSince.
        ///
        /// Long enough that a surface which reappears shortly after vanishing -- a menu reopened, a cell
        /// stepped back into -- re-claims its identity instead of being destroyed and rebuilt, and far longer
        /// than the runtime can still be holding a queued destroy for the handle.
        static constexpr std::uint64_t kTextureGraceFrames = 120;

        /// Releases textures whose source image OpenMW has freed, and the materials naming them.
        ///
        /// Deliberately not an idle timer. An earlier attempt released textures unused for a fixed number
        /// of frames, which destroyed textures out from under materials still drawing with them and turned
        /// surfaces permanently white -- permanently, because the identity is the content hash, so the
        /// re-upload is suppressed by the runtime's tombstone set, and because the dedupe record was never
        /// cleared so no re-upload was even attempted. Liveness here is a fact taken from OpenMW's own
        /// resource lifetime rather than inferred from a clock.
        ///
        /// Order is enforced by counting. A content hash can be named by several images, so a texture is
        /// destroyed only once the last image naming it is gone, and the materials referencing that hash --
        /// through either their albedo or their normal slot -- are destroyed first, because a material
        /// outliving its texture refers to nothing.
        void releaseOrphanedTextures();

        /// Gives up one cache entry's claim on its uploaded identity, without destroying anything.
        ///
        /// Separate from the destroy on purpose. Dropping the claim is safe at any point in a frame;
        /// destroying is not, because re-creating a hash the runtime still has queued for destruction is
        /// suppressed rather than honoured. So claims are released as soon as they are known to be dead and
        /// the destroy is left to the end-of-frame sweep, which also gives a re-claim within the same frame
        /// the chance to make the destroy unnecessary.
        void releaseTextureClaim(const CachedTexture& cached);

        struct CachedLight
        {
            unsigned long long mHandle = 0;
            std::uint64_t mLastUsedFrame = 0;
            /// Position and radiance the handle was created with. Remix has no light-update entry point,
            /// so a light that moves or changes colour has to be destroyed and recreated -- and OpenMW's
            /// lights do both constantly, since they are carried by actors and many of them flicker.
            /// Recreating unconditionally every frame would churn hundreds of handles a frame, so the
            /// values are kept here and compared.
            float mPosition[3] = { 0.0f, 0.0f, 0.0f };
            float mRadiance[3] = { 0.0f, 0.0f, 0.0f };
            float mRadius = 0.0f;
        };

        /// Releases lights not submitted this frame.
        void releaseStaleLights();

        /// Picks up live edits to the light emitter radius and intensity factor from Remix's developer
        /// menu, so lighting can be tuned while looking at it instead of over a restart.
        ///
        /// A poll rather than a callback because the values cross a process-internal boundary as strings
        /// in Remix's game-state store; there is no notification channel coming back the other way.
        void pollLightTuning();

        /// Releases particle meshes whose system was not submitted this frame.
        void releaseStaleParticleMeshes();

        /// One particle system's mesh, rebuilt every frame it is visible.
        struct ParticleMesh
        {
            unsigned long long mHandle = 0;
            unsigned long long mMaterial = 0;
            std::uint64_t mLastUsedFrame = 0;
        };
        /// Keyed by particle system address. Unlike mMeshes this is never a cache hit -- the contents
        /// change every frame -- so the entry exists only to hold the handle that has to be destroyed
        /// before the replacement is created, and to notice when a system stops being submitted.
        std::unordered_map<const void*, ParticleMesh> mParticleMeshes;
        /// Particles submitted on the last call to submit(), for the report line.
        unsigned int mLastParticleCount = 0;
        /// Particle *instances* this frame, as opposed to the quad count above. Only used to hand each one
        /// a distinct object-picking value, offset past the traversal's range so the two cannot collide.
        unsigned int mParticleInstances = 0;

        /// Converts and caches \a geometry against \a material, returning its mesh handle, or 0.
        unsigned long long meshFor(osg::Geometry& geometry, unsigned long long material,
            const SurfaceState& surface, const SceneUtil::RigGeometry* rig);

        /// The memo slot for one \a layer of the geometry at \a address, appended empty if that pair has
        /// not been seen. The returned reference is invalidated by the next call.
        GeometryIdentity& identityFor(std::uintptr_t address, unsigned int layer);

        /// Expands \a rig's grouped influences into the flat per-vertex arrays the runtime wants.
        ///
        /// Fills mWeightScratch and mBoneIndexScratch and returns the bones-per-vertex count, or 0 if the
        /// rig cannot be skinned. The returned count is what both arrays are strided by.
        unsigned int buildSkinning(const SceneUtil::RigGeometry& rig, unsigned int vertexCount);

        /// Resolves \a surface to a material handle, uploading its texture on first use.
        /// Falls back to the untextured default material rather than dropping the drawable.
        ///
        /// @param emissive radiance scale for a self-lit surface. Non-zero only for particles.
        unsigned long long materialFor(const SurfaceState& surface, float emissive = 0.0f);

        /// Uploads \a image, returning the texture hash to reference it by, or 0 if unusable.
        ///
        /// @param colour true for anything whose values are a colour and so want sRGB decoding on sample;
        ///        false for data -- a normal map above all, where a gamma curve applied to what are
        ///        supposed to be vector components tilts every normal toward the surface.
        unsigned long long textureFor(const osg::Image& image, bool colour);

        /// Releases meshes not submitted for a while.
        void evictStaleMeshes();

        RemixRT::Runtime& mRuntime;

        /// The last camera handed to the runtime, kept so resubmitCamera can repeat it. Plain floats in the
        /// order the API takes them, rather than the OSG matrices they came from, so that repeating a frame
        /// cannot re-derive them differently from the frame it is repeating.
        float mLastCameraEye[3] = { 0.f, 0.f, 0.f };
        float mLastCameraForward[3] = { 0.f, -1.f, 0.f };
        float mLastCameraUp[3] = { 0.f, 0.f, 1.f };
        float mLastCameraRight[3] = { 1.f, 0.f, 0.f };
        float mLastCameraFov = 60.f;
        float mLastCameraAspect = 1.7777f;
        float mLastCameraNear = 1.f;
        float mLastCameraFar = 6666.f;
        bool mHaveLastCamera = false;
        unsigned long long mDefaultMaterial = 0;
        unsigned long long mWaterMaterial = 0;
        unsigned long long mProbeMaterial = 0;
        unsigned long long mProbeMesh = 0;
        /// Keyed by the mesh's content hash, which is also the identity Remix knows it by and the key a USD
        /// replacement is authored against. Previously keyed by geometry address, which is why the same
        /// mesh presented a different identity on every run.
        std::unordered_map<unsigned long long, CachedMesh> mMeshes;

        /// Keyed by osg::Geometry address, purely as a memo to keep content hashing off the per-frame path.
        /// Unlike mMeshes this is not an identity -- it is a cache of a lookup, and an address reused by a
        /// different geometry is caught by the revalidation checks in GeometryIdentity.
        ///
        /// One entry per terrain layer, because a chunk's layers all share one osg::Geometry and a
        /// single-entry map made them evict one another -- every layer rebuilt its mesh every frame.
        ///
        /// The layer index lives in the value and not in the key, which the key deliberately used to
        /// carry. retainGeometry has to reach every identity a geometry owns, and it runs at the cull
        /// sites, before any state set has been merged -- so it does not know the layer, and the layer
        /// count is data-driven per chunk rather than bounded, so it cannot enumerate them either. Keying
        /// on the address alone makes that one lookup. The vector holds a single element for everything
        /// that is not terrain, which is nearly all of it.
        std::unordered_map<std::uintptr_t, std::vector<GeometryIdentity>> mGeometryIdentities;
        /// Keyed by osg::Image address. Textures and materials are never evicted: OpenMW's resource
        /// system shares images aggressively and keeps them alive for the session, the set is bounded by
        /// how many distinct textures the game has, and re-uploading one costs a full staging copy.
        /// Keyed by the image address shifted left one, with the colour/linear flag in the low bit.
        std::unordered_map<unsigned long long, CachedTexture> mTextures;
        /// Keyed by texture hash combined with the alpha-test threshold, since those two are all that
        /// currently distinguish one material from another.
        std::unordered_map<unsigned long long, unsigned long long> mMaterials;
        /// Every texture a material references.
        struct MaterialTextures
        {
            /// Needed because a mesh's identity is its geometry hash XOR its material's hash, and Remix's
            /// D3D9 path takes the latter to be the albedo texture hash alone. This host's material handle
            /// deliberately is not that value -- it mixes in surface state so two materials sharing a
            /// texture stay distinct -- so the texture hash has to be carried alongside rather than
            /// recovered from the handle. Recomputing it at mesh build time is not an option: it hashes the
            /// whole mip 0.
            unsigned long long mAlbedo = 0;
            /// Recorded solely so a release can find this material from either of its textures. Nothing
            /// reads it for rendering. Without it the reverse lookup is incomplete in exactly the case a
            /// texture pack makes common -- every normal map is a texture that live materials name and that
            /// an albedo-only index cannot enumerate, so releasing one would strand a material on a
            /// destroyed texture with no way to notice.
            unsigned long long mNormal = 0;
            /// The terrain layer coverage mask, recorded for the same reason as mNormal and with more at
            /// stake. A blend map belongs to one chunk, so these come and go with the terrain rather than
            /// living as long as the session, and every one of them is named by a live material. Leaving
            /// the release scan unable to see them would strand a material per chunk per layer.
            unsigned long long mMask = 0;
        };
        /// Material handle to the textures it references.
        std::unordered_map<unsigned long long, MaterialTextures> mMaterialAlbedoHashes;

        /// One texture's classified surface response, memoised so the classification runs once per texture
        /// rather than once per drawable per frame.
        ///
        /// Stores what is consumed rather than the matched rule itself, because the rule type is private to
        /// the implementation file. The pattern is kept only for logging and points into the static rule
        /// table, so it needs no lifetime management; null means no rule matched.
        struct CachedSurfaceResponse
        {
            float mRoughness = 0.0f;
            float mMetallic = 0.0f;
            const char* mPattern = nullptr;
        };

        /// Keyed by osg::Image address, with the same never-evicted lifetime reasoning as mTextures above:
        /// the answer depends only on the image's own filename, which does not change while it is alive.
        std::unordered_map<const osg::Image*, CachedSurfaceResponse> mSurfaceResponses;

        /// Classifies a texture's path into a surface response, or returns the memoised answer.
        ///
        /// Worth memoising because it is pure string work -- a lowercased copy of the path, then a
        /// substring scan over fifty rules -- and it sits ahead of the material cache in materialFor, so
        /// every drawable paid for it every frame even when the material itself was already built.
        const CachedSurfaceResponse& surfaceResponseFor(const osg::Image& image);
        /// Keyed by LightSource::getId, which OpenMW already guarantees unique per live light source --
        /// unlike the node address, which is reused as cells page in and out.
        std::unordered_map<int, CachedLight> mLights;
        std::uint64_t mFrame = 0;

    public:
        /// The scene's own frame counter, exposed so the traversal can key per-frame work to the same clock
        /// the submission uses rather than inventing a second one.
        std::uint64_t frameNumber() const { return mFrame; }

    private:
        unsigned int mLastInstanceCount = 0;
        unsigned int mLastLightCount = 0;
        unsigned int mTexturesUploaded = 0;

        /// Identities already handed to CreateTexture, so the same content is never uploaded twice, mapped
        /// to how many mTextures entries currently name each one.
        ///
        /// Separate from mTextures because that map is keyed per osg::Image and several images can share one
        /// identity -- a mod shipping a copy of another's texture, or one file serving both genders of an
        /// outfit. The count is what makes releasing safe: reaching zero is the only proof that no live
        /// image still resolves to this identity, and therefore that destroying it cannot strand a surface.
        ///
        /// An entry must be erased when its texture is destroyed. Leaving it behind is not a leak but
        /// something worse: the lookup above would report the content as already present, hand back a hash
        /// the runtime no longer holds, and never attempt the upload that would fix it.
        std::unordered_map<unsigned long long, unsigned int> mUploadedTextures;
        /// Bytes resident per uploaded identity, and the running total.
        ///
        /// Counts exist for textures but said nothing about memory -- a 2048-square composite and a
        /// 64-square icon incremented the same counter -- so a cache that was visibly growing could not be
        /// weighed against video memory at all. Keyed by identity rather than by image so the dedupe path
        /// does not count the same upload twice.
        std::unordered_map<unsigned long long, std::size_t> mTextureBytes;
        /// Triangles per mesh identity, for the submission budget below.
        std::unordered_map<unsigned long long, unsigned int> mMeshPrimitives;
        /// Triangles admitted so far this frame, and how many instances the budget turned away.
        unsigned int mPrimitivesSubmitted = 0;
        unsigned int mInstancesOverBudget = 0;
        /// Milliseconds spent encoding terrain composites in the frame in progress, and how many were put
        /// off to a later frame once that budget was spent. Reset per frame with the counters above.
        ///
        /// Composites arrive in bursts as chunks come into view -- measured at up to 82 in one second --
        /// and encoding all of them immediately put 678 ms into a single frame's scene submit. Bounding the
        /// work per frame spreads a burst over several frames instead of stopping the game.
        double mCompositeEncodeMs = 0.0;
        unsigned int mCompositesDeferred = 0;

        /// The most expensive submit since the last scene log, with the state that could explain it.
        ///
        /// A mean cannot describe this and neither can the frame report, which measures submit as one
        /// number. Measured submits run a few milliseconds most frames and spike to 88-168 ms, and one
        /// frame of 97.5 ms was 98.6% accounted for by submit alone -- so the spike is real, it is ours,
        /// and it was the second largest remaining stutter with nothing saying which half of the function
        /// it was in.
        ///
        /// Split into the two halves that do the work, because they fail for unrelated reasons: the
        /// traversal walks the scene graph and its cost tracks how much geometry is loaded, while the flush
        /// hands instances to the runtime and builds acceleration structures for whatever is new. Steady
        /// state is roughly 9,000-11,000 instances with 85 meshes built, and a burst of new meshes is a
        /// very different problem from a traversal that got longer.
        struct WorstSubmit
        {
            double mTotalMs = 0.0;
            double mTraversalMs = 0.0;
            double mFlushMs = 0.0;
            double mCompositeEncodeMs = 0.0;
            unsigned int mInstances = 0;
            unsigned int mMeshesCreated = 0;
            unsigned int mPrimitives = 0;
            unsigned int mCulled = 0;
            std::uint64_t mFrame = 0;
        };
        WorstSubmit mWorstSubmit;

        /// One queued instance, held from drawSubmitted until flushSubmissions.
        ///
        /// The transform is copied by value rather than pointed at. The traversal writes it into a stack
        /// array per drawable, so a pointer would dangle the moment that scope ended.
        struct PendingInstance
        {
            unsigned long long mMesh;
            float mTransform[12];
            unsigned int mCategoryFlags;
            const SceneUtil::RigGeometry* mRig;
            unsigned int mPickingValue;
            float mDistanceSquared;
            unsigned int mPrimitives; ///< Zero for an identity with no recorded triangle count.
            bool mDoubleSided;
            bool mExempt; ///< Admitted regardless of the budget. See flushSubmissions.
        };

        /// This frame's queued instances, cleared at the start of every submit.
        std::vector<PendingInstance> mPendingInstances;

        /// Sort key for one queued instance: what to order by, and which instance it refers to.
        struct SubmissionKey
        {
            float mDistanceSquared; ///< Negative for a budget-exempt instance, so it sorts first.
            unsigned int mIndex; ///< Into mPendingInstances. Also the tie-break, to keep frames stable.
        };

        /// Scratch for flushSubmissions' sort. A member so its capacity survives between frames.
        std::vector<SubmissionKey> mSubmissionOrder;

        /// Sorts the queued instances nearest-first, admits them under the triangle budget, hands over.
        ///
        /// Nearest-first is the whole point, and it is what makes the budget usable rather than merely
        /// protective. Enforced greedily at drawSubmitted time, the budget dropped whatever happened to be
        /// traversed last -- and traversal order for paged chunks is quadtree order, which is spatially
        /// coherent but has nothing to do with distance. So a binding budget deleted an arbitrary wedge of
        /// the world, near geometry included, which is why it could only be set high enough never to bind.
        /// Sorted by distance it degrades as a draw-distance reduction instead, which is a visual trade
        /// worth making and can therefore be set to a number that actually protects the runtime.
        ///
        /// Deferring costs nothing else. Submission order is meaningless to a path tracer -- there is no
        /// depth test and no blend order, the runtime builds an acceleration structure from the whole set --
        /// so when the budget does not bind this reorders the handover and changes nothing observable.
        void flushSubmissions();

        /// Hands one admitted instance to the runtime, resolving bone matrices if it is skinned.
        ///
        /// Split out of drawSubmitted when that became a queue. Called only from flushSubmissions, which is
        /// still inside the frame that queued it -- which is what keeps the rig's bone matrices valid, since
        /// they are current from the update traversal onwards rather than only at the moment of traversal.
        void submitInstanceNow(const PendingInstance& pending);

        /// The frame triangle budget in force, from OPENMW_REMIX_PRIMITIVE_BUDGET or the default below.
        ///
        /// Env-tunable because the right value is an open question that needs measuring in-game rather than
        /// deciding here, and because rebuilding to try a number is a poor way to spend a test run.
        static unsigned int primitiveBudget();

        /// Default ceiling on triangles submitted in one frame.
        ///
        /// **The 67,108,863 figure the runtime logs is not the real limit.** `PRIMITIVE_INDEX_BIT_COUNT`
        /// is 26 in the runtime's `instance_definitions.h`, but grepping the runtime for it finds exactly
        /// one use: the log message that prints it. Nothing packs a primitive index into 26 bits. The field
        /// that actually constrains the count is the NEE cache's prefix-sum ID, which is **24** bits --
        /// `NEE_CACHE_INVALID_ID` is `0xffffff`, and `update_nee_cache.comp.slang` masks tasks with
        /// `& 0xffffff` and packs them as `(range << 24) | prefixSumID`. With the sentinel reserved the
        /// usable range is 16,777,214.
        ///
        /// So this default is **3.3x above the real ceiling**, not comfortably below a 67M one. It is left
        /// where it is only because lowering it was unsafe while the budget dropped arbitrary geometry;
        /// with flushSubmissions sorting nearest-first that objection is gone, and the number wants
        /// re-measuring downwards. Use OPENMW_REMIX_PRIMITIVE_BUDGET to try values without a rebuild.
        ///
        /// What exceeding it costs: truncated prefix-sum IDs resolve to a garbage surface and primitive,
        /// which the runtime's own source notes "can cause an out-of-range shared memory access in the
        /// update shader" (`integrator_indirect.slangh`). Measured at 143,278,765 triangles -- 8.5x the
        /// real ceiling -- arriving as an AppHangTransient with three nvlddmkm resets behind it.
        ///
        /// Set below whatever limit is chosen rather than at it, because this counts what this traversal
        /// hands over and the runtime adds its own replacement geometry on top.
        static constexpr unsigned int kPrimitiveBudget = 56000000u;
        /// Frame each identity's last claim was dropped, for identities nothing claims any more.
        ///
        /// A grace period, not bookkeeping. A material handle is derived deterministically from surface
        /// state, so destroying a material and later rebuilding the identical surface produces the *same*
        /// handle -- and a create sharing a handle with a destroy the runtime still has queued is suppressed
        /// rather than honoured. Destroying the moment a claim reaches zero therefore risks handing back a
        /// handle the runtime has quietly refused, which is most likely exactly where transient surfaces
        /// churn fastest: GUI textures being freed and rebuilt as a menu is opened and closed.
        ///
        /// Waiting also makes most of those destroys unnecessary rather than merely safe. A surface that
        /// comes straight back re-claims its identity within the window and is never destroyed at all.
        std::unordered_map<unsigned long long, std::uint64_t> mOrphanedSince;
        std::size_t mTextureBytesResident = 0;
        std::size_t mTextureBytesPeak = 0;
        /// Textures and materials released since the last report, for the per-frame log.
        unsigned int mTexturesReleased = 0;
        unsigned int mMaterialsReleased = 0;

        /// How many uploads were avoided because the content was already present.
        unsigned int mTexturesShared = 0;

        /// How many times a geometry resolved to a mesh that already existed with identical content.
        ///
        /// Reported because the saving is invisible otherwise: sharing shows up only as a mesh count that
        /// is lower than the instance count, which is also true for plenty of other reasons.
        unsigned int mMeshesShared = 0;
        /// Emitter radius and radiance-times-radius-squared for converted lights. Read from the
        /// environment once at construction so they can be tuned without a rebuild; Remix's own light
        /// options act on its legacy conversion path and cannot reach lights created through the API.
        float mLightRadius = 0.0f;
        /// Mirrors rtx.lightConversionIntensityFactor, which the runtime applies to the lights it converts
        /// itself. Applied here for the same reason and with the same default, so API lights and legacy
        /// ones respond to tuning the same way.
        float mLightIntensityFactor = 0.0f;

        /// Frame the light tuning last changed, and whether that change is still waiting to be logged.
        /// Only used to hold the log back until a drag stops, so the record is of the value settled on.
        std::uint64_t mLightTuningChangedFrame = 0;
        bool mLightTuningPendingLog = false;
        /// Threshold an alpha-blended surface is cut out at when it asks for no explicit test.
        /// Environment-tunable so foliage can be dialled in without a rebuild; 0 disables the
        /// substitution and leaves blended surfaces solid.
        unsigned char mBlendCutout = 0;
        /// How many distinct materials have been described in the log, and the ceiling on that. Bounded
        /// because Morrowind has thousands of textures and the interesting ones are all in the first few
        /// dozen the player walks into.
        unsigned int mMaterialsLogged = 0;
        // Raised from 48 to answer a specific question: how often is one texture used with more than one
        // surface state? That decides whether a material can be identified by its albedo texture hash, the
        // way Remix identifies a D3D9 material, without losing a distinction OpenMW currently makes. At 48
        // the answer was "never in 45 samples", which is not an answer.
        static constexpr unsigned int kMaterialLogLimit = 4096;

        /// Capped like the material log, but higher: the point of it is to be able to look up an arbitrary
        /// hash seen in Remix's texture list, so covering only the first few textures of a session would
        /// miss most of what anyone would want to ask about.
        unsigned int mTexturesLogged = 0;
        static constexpr unsigned int kTextureLogLimit = 4096;


        /// Same idea for lights, and a smaller sample: the interesting question is whether the derivation
        /// produces sane radiance for a real attenuation curve, and a dozen answers that in one glance.
        unsigned int mLightsLogged = 0;
        static constexpr unsigned int kLightLogLimit = 12;
        /// World-space extent of the instance origins submitted this frame. Logged next to the camera
        /// position, because "geometry is nowhere near the camera" and "geometry is right there but not
        /// being rendered" are different problems and the instance count alone cannot tell them apart.
        double mMin[3] = { 0.0, 0.0, 0.0 };
        double mMax[3] = { 0.0, 0.0, 0.0 };
        bool mHaveExtent = false;
        /// Scratch, reused across meshes so conversion does not allocate per geometry per frame.
        std::vector<RemixRT::Runtime::Vertex> mVertexScratch;
        std::vector<unsigned int> mIndexScratch;
        /// Skinning scratch. The weights and indices are only needed while a mesh is being created, but
        /// the bone matrices are rebuilt for every skinned instance every frame, which is the one place
        /// in this file where a per-frame allocation would actually show up in a profile.
        std::vector<float> mWeightScratch;
        std::vector<unsigned int> mBoneIndexScratch;
        std::vector<osg::Matrixf> mBoneMatrixScratch;
        std::vector<float> mBoneTransformScratch;
        /// Skinned instances submitted on the last call to submit(), and how many were dropped because
        /// their skin was not usable. Reported together: skinned actors going missing is otherwise
        /// indistinguishable from the traversal not reaching them.
        unsigned int mSkinnedInstances = 0;
        /// Meshes handed to the runtime this frame, which is one acceleration structure build each.
        ///
        /// Separate from mMeshes.size(), the cache population, because that number cannot distinguish a
        /// warm cache from rebuilding everything every frame -- and the two differ by the cost of hundreds
        /// of BLAS builds. Particle systems are expected to appear here every frame by construction; static
        /// geometry appearing here every frame is a fault.
        unsigned int mMeshesCreated = 0;

        /// Meshes the per-frame build ceiling turned away this frame, per frame.
        ///
        /// Reported because a deferral is otherwise invisible: the instance simply is not in the scene for a
        /// frame, which looks like a culling or content bug rather than a budget doing its job. A number here
        /// on a cell change is expected; a number here every frame while standing still would mean the
        /// ceiling sits below the standing load and animation is being throttled by it.
        unsigned int mMeshBuildsDeferred = 0;

        /// Drawables the traversal rejected this frame whose mesh was held resident anyway, per frame.
        ///
        /// The counter that says whether retainGeometry is doing anything: against mCulled it reads as
        /// "this many were rejected, this many of them would previously have expired and been rebuilt on
        /// the way back". A retained count near zero while mCulled is in the thousands means the memo
        /// lookup is missing, not that there was nothing to keep.
        unsigned int mMeshesRetained = 0;
        unsigned int mSkinnedDropped = 0;
        /// Bones per vertex of the mesh the last submitGeometry resolved to, so the caller can tell
        /// whether the instance it is about to queue needs bone transforms. Carried on the object rather
        /// than returned because a mesh that failed its skinning is still a usable static mesh, and the
        /// two answers -- handle, and whether it is skinned -- come from the same lookup.
        unsigned int mLastMeshBonesPerVertex = 0;
        /// Drawables the frustum test rejected this frame. Reported so the saving is visible: without it the
        /// only evidence culling is working is an instance count that went down for unstated reasons.
        unsigned int mCulled = 0;
        bool mLoggedFirstSubmit = false;
        bool mLoggedClamp = false;
        /// Whether the last submission found anything, so the transition can be reported rather than
        /// only the first frame -- which happens before the world exists and says nothing useful.
        bool mWasPopulated = false;
    };
}

#endif
