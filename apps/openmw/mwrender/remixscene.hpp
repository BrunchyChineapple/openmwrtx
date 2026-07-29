#ifndef OPENMW_MWRENDER_REMIXSCENE_H
#define OPENMW_MWRENDER_REMIXSCENE_H

#include <cstdint>
#include <unordered_map>
#include <vector>

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
        };

        /// Converts and caches one drawable, returning its Remix mesh handle, or 0 if unusable.
        /// Public only because the traversal visitor lives in the .cpp and calls back into here.
        unsigned long long submitGeometry(osg::Geometry& geometry, const SurfaceState& surface);

        /// Queues one instance. Companion to submitGeometry, same reason for being public.
        void drawSubmitted(
            unsigned long long mesh, const float* transform, unsigned int categoryFlags, bool doubleSided);

        /// Records a submitted instance's world-space origin, for the diagnostic extent log.
        void noteInstancePosition(double x, double y, double z);

        /// Submits one OpenMW light source, at the world position \a x, \a y, \a z.
        ///
        /// Public for the same reason as submitGeometry: the traversal lives in the .cpp.
        void submitLight(const SceneUtil::LightSource& source, double x, double y, double z);

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

        /// Converts and caches \a geometry against \a material, returning its mesh handle, or 0.
        unsigned long long meshFor(
            osg::Geometry& geometry, unsigned long long material, const SurfaceState& surface);

        /// Resolves \a surface to a material handle, uploading its texture on first use.
        /// Falls back to the untextured default material rather than dropping the drawable.
        unsigned long long materialFor(const SurfaceState& surface);

        /// Uploads \a image, returning the texture hash to reference it by, or 0 if unusable.
        unsigned long long textureFor(const osg::Image& image);

        /// Releases meshes not submitted for a while.
        void evictStaleMeshes();

        RemixRT::Runtime& mRuntime;
        unsigned long long mDefaultMaterial = 0;
        unsigned long long mWaterMaterial = 0;
        unsigned long long mProbeMaterial = 0;
        unsigned long long mProbeMesh = 0;
        std::unordered_map<const void*, CachedMesh> mMeshes;
        /// Keyed by osg::Image address. Textures and materials are never evicted: OpenMW's resource
        /// system shares images aggressively and keeps them alive for the session, the set is bounded by
        /// how many distinct textures the game has, and re-uploading one costs a full staging copy.
        std::unordered_map<const void*, CachedTexture> mTextures;
        /// Keyed by texture hash combined with the alpha-test threshold, since those two are all that
        /// currently distinguish one material from another.
        std::unordered_map<unsigned long long, unsigned long long> mMaterials;
        /// Keyed by LightSource::getId, which OpenMW already guarantees unique per live light source --
        /// unlike the node address, which is reused as cells page in and out.
        std::unordered_map<int, CachedLight> mLights;
        std::uint64_t mFrame = 0;
        unsigned int mLastInstanceCount = 0;
        unsigned int mLastLightCount = 0;
        unsigned int mTexturesUploaded = 0;
        /// Emitter radius and radiance-times-radius-squared for converted lights. Read from the
        /// environment once at construction so they can be tuned without a rebuild; Remix's own light
        /// options act on its legacy conversion path and cannot reach lights created through the API.
        float mLightRadius = 0.0f;
        float mLightPower = 0.0f;
        /// Threshold an alpha-blended surface is cut out at when it asks for no explicit test.
        /// Environment-tunable so foliage can be dialled in without a rebuild; 0 disables the
        /// substitution and leaves blended surfaces solid.
        unsigned char mBlendCutout = 0;
        /// How many distinct materials have been described in the log, and the ceiling on that. Bounded
        /// because Morrowind has thousands of textures and the interesting ones are all in the first few
        /// dozen the player walks into.
        unsigned int mMaterialsLogged = 0;
        static constexpr unsigned int kMaterialLogLimit = 48;
        /// World-space extent of the instance origins submitted this frame. Logged next to the camera
        /// position, because "geometry is nowhere near the camera" and "geometry is right there but not
        /// being rendered" are different problems and the instance count alone cannot tell them apart.
        double mMin[3] = { 0.0, 0.0, 0.0 };
        double mMax[3] = { 0.0, 0.0, 0.0 };
        bool mHaveExtent = false;
        /// Scratch, reused across meshes so conversion does not allocate per geometry per frame.
        std::vector<RemixRT::Runtime::Vertex> mVertexScratch;
        std::vector<unsigned int> mIndexScratch;
        bool mLoggedFirstSubmit = false;
        bool mLoggedClamp = false;
        /// Whether the last submission found anything, so the transition can be reported rather than
        /// only the first frame -- which happens before the world exists and says nothing useful.
        bool mWasPopulated = false;
    };
}

#endif
