#ifndef OPENMW_MWRENDER_REMIXSCENE_H
#define OPENMW_MWRENDER_REMIXSCENE_H

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <osg/Matrixf>
#include <osg/Node>
#include <osg/ref_ptr>

#include <components/remixrt/runtime.hpp>

namespace osg
{
    class Camera;
    class Image;
    class Texture2D;
}

namespace SceneUtil
{
    class LightSource;
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
            const osg::Image* mExplicitImage = nullptr;

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
        /// @param rig when non-null, its current bone matrices are submitted with the instance. Must be
        ///        the same rig the mesh was created from.
        /// @param pickingValue identifies this draw so the developer menu can resolve a click in the
        ///        scene to it. Must be distinct per draw within a frame; zero leaves it unpickable.
        void drawSubmitted(unsigned long long mesh, const float* transform, unsigned int categoryFlags,
            bool doubleSided, const SceneUtil::RigGeometry* rig = nullptr,
            unsigned int pickingValue = 0);

        /// Records a submitted instance's world-space origin, for the diagnostic extent log.
        void noteInstancePosition(double x, double y, double z);

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
            std::uint64_t mLastUsedFrame = 0;
            unsigned int mVertexCount = 0;
            unsigned int mModifiedCount = 0;
            unsigned long long mMaterial = 0;
            float mTexMat[6] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
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
        };

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
        /// Keyed on geometry address combined with the terrain layer index, not on the address alone.
        ///
        /// A chunk's layers all share one osg::Geometry, so an address-only key made them evict one
        /// another and every layer rebuilt its mesh every frame. The combination is injective rather than
        /// a hash: multiplying by an odd constant is invertible modulo 2^64, so distinct addresses stay
        /// distinct, and adding a layer index far smaller than the multiplier cannot reach the next
        /// address's slot. No collisions to reason about.
        std::unordered_map<std::uint64_t, GeometryIdentity> mGeometryIdentities;
        /// Keyed by osg::Image address. Textures and materials are never evicted: OpenMW's resource
        /// system shares images aggressively and keeps them alive for the session, the set is bounded by
        /// how many distinct textures the game has, and re-uploading one costs a full staging copy.
        /// Keyed by the image address shifted left one, with the colour/linear flag in the low bit.
        std::unordered_map<unsigned long long, CachedTexture> mTextures;
        /// Keyed by texture hash combined with the alpha-test threshold, since those two are all that
        /// currently distinguish one material from another.
        std::unordered_map<unsigned long long, unsigned long long> mMaterials;

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

        /// Identities already handed to CreateTexture, so the same content is never uploaded twice.
        ///
        /// Separate from mTextures because that map is keyed per osg::Image and several images can share one
        /// identity. Never evicted, for the same reason mTextures is not: the set is bounded by how many
        /// distinct textures the game has.
        std::unordered_set<unsigned long long> mUploadedTextures;

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
        static constexpr unsigned int kMaterialLogLimit = 48;

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
