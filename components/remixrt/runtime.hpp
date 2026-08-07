#ifndef OPENMW_COMPONENTS_REMIXRT_RUNTIME_H
#define OPENMW_COMPONENTS_REMIXRT_RUNTIME_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct SDL_Window;

namespace RemixRT
{
    /// Decodes one IEEE 754 binary16.
    ///
    /// Lives here because the readback surface is half-float, so anything that wants to inspect Remix's
    /// output as numbers rather than bytes needs it -- the brightness probe being the reason it exists.
    /// Handles subnormals and infinities rather than only the normal range, because a path tracer's output
    /// legitimately contains both very small values in shadow and very large ones in a highlight.
    float halfToFloat(unsigned short bits);

    /// Owns the lifetime of the RTX Remix runtime and the handle to its C API.
    ///
    /// Remix reconstructs a scene from meshes, materials, lights and a camera and path-traces it.
    /// Its published entry point is the D3D9 API, but it also exposes a C API that accepts scene data
    /// directly, which is the route this integration uses -- OpenMW submits its own geometry rather
    /// than Remix intercepting draw calls. OpenMW is 64-bit and the Remix runtime is x64-only, so the
    /// 32-bit bridge that legacy titles need is not involved here.
    ///
    /// Remix still requires a D3D9Ex device as its host object even when no D3D9 draw is ever issued.
    /// Startup() creates one headlessly; treat it as a device-creation formality, not a rendering path.
    ///
    /// This class is deliberately platform-neutral in its interface and does not leak <windows.h> or
    /// the Remix headers to its callers. On non-Windows builds every method is a no-op that reports
    /// failure, so the component compiles and links everywhere OpenMW does.
    class Runtime
    {
    public:
        Runtime();
        ~Runtime();

        Runtime(const Runtime&) = delete;
        Runtime& operator=(const Runtime&) = delete;

        /// True when the user asked for the Remix backend.
        ///
        /// Gated on the OPENMW_REMIX environment variable while this backend is experimental, rather
        /// than on a settings.cfg entry, so that a half-finished renderer cannot be reached from the
        /// launcher UI. engine.cpp already reads OPENMW_OSG_STATS_FILE the same way. Promote this to
        /// a real setting once the backend renders a complete frame.
        static bool requested();

        /// Loads the runtime and brings up its device. Safe to call once; returns false on any failure
        /// without throwing, because a missing or mismatched runtime should degrade to normal OpenMW
        /// rendering rather than prevent the game from starting.
        ///
        /// @param window the SDL window used to size Remix's render resolution. Remix does NOT
        ///        present into it; it presents into a window of its own -- see createPresentWindow in
        ///        runtime.cpp.
        ///
        /// Reads three further environment variables, all off by default:
        ///   OPENMW_REMIX_WINDOW       show Remix's own window, which is what makes the developer
        ///                             menu reachable.
        ///   OPENMW_REMIX_UI           menu to open with it: 0 none, 1 basic, 2 advanced. Default 2.
        ///   OPENMW_REMIX_WINDOW_SIZE  override Remix's render resolution, e.g. "1600x900".
        bool initialize(SDL_Window* window);

        /// Everything OpenGL needs to import Remix's output image as a texture.
        struct ExternalImage
        {
            unsigned long long mHandle = 0; ///< Win32 handle, owned by Remix; do not close it.
            unsigned long long mMemorySize = 0;
            unsigned long long mMemoryOffset = 0;
            unsigned int mHandleType = 0; ///< VkExternalMemoryHandleTypeFlagBits
            unsigned int mFormat = 0; ///< VkFormat
            unsigned int mWidth = 0;
            unsigned int mHeight = 0;
            bool mOptimalTiling = false;
        };

        /// The semaphore pair that orders Remix's copy against OpenGL's sampling of the shared image.
        ///
        /// Both are Win32 handles for *binary* Vulkan semaphores, and both stay owned by Remix.
        /// Note they are exported as NT handles, where the image is exported as a KMT handle, so the
        /// two imports need different GL handle types.
        struct ExternalSync
        {
            unsigned long long mCopyComplete = 0; ///< Remix signals after the copy; GL waits on it.
            unsigned long long mConsumerDone = 0; ///< GL signals after sampling; Remix waits on it.

            bool valid() const { return mCopyComplete != 0 && mConsumerDone != 0; }
        };

        /// Allocates the shared render target that Remix blits its final colour into, and queries the
        /// exportable memory behind it. Must be called after initialize().
        bool createOutputTarget(unsigned int width, unsigned int height);

        /// Valid only after a successful createOutputTarget().
        const ExternalImage& outputImage() const;

        /// Asks the runtime for the output synchronisation semaphores, creating them on first call.
        /// Not fatal if it fails; the consumer then has to fall back to sampling without ordering,
        /// which is undefined and in practice reads as black.
        bool createOutputSync();

        /// Valid only after a successful createOutputSync().
        const ExternalSync& outputSync() const;

        /// Asks Remix for an image to draw OpenMW's GUI into, which it then composites over the
        /// path-traced frame.
        ///
        /// The opposite direction to createOutputTarget: there Remix produces and OpenGL consumes, here
        /// OpenGL produces and Remix consumes. Same mechanism either way -- exportable Vulkan memory, one
        /// allocation, two APIs addressing it.
        ///
        /// Remix allocates rather than OpenMW because the alternative is DrawScreenOverlay, which takes
        /// CPU pixels: 33MB a frame at 4K, the same round trip that had to come out of the render loop to
        /// make this playable.
        ///
        /// Unsynchronised, like the output path, and for the same measured reason -- glWaitSemaphoreEXT
        /// fails on this driver under every condition tried. The consequence here is far milder: the worst
        /// case is Remix compositing a GUI frame while it is being drawn, and interface elements move
        /// rarely and slightly.
        bool createOverlayImage(unsigned int width, unsigned int height);

        /// Valid only after a successful createOverlayImage().
        const ExternalImage& overlayImage() const;

        /// Starts or stops Remix compositing the overlay image, and sets its opacity.
        ///
        /// Separate from allocation so compositing can stop without losing the image -- during a loading
        /// screen, where the GUI is not being drawn and the last frame would otherwise stay on screen.
        bool setOverlayEnabled(bool enabled, float opacity = 1.0f);

        /// Hands Remix the camera for this frame.
        ///
        /// Both matrices are 16 floats in OSG's layout, which is row-major with the row-vector
        /// convention (v * M) -- the same convention D3D9 uses, so they map straight across.
        ///
        /// Prefer setupCameraParameterized. This path makes handedness, matrix majorness and OpenMW's
        /// reversed-Z projection all load-bearing simultaneously, and gives no way to tell which of them
        /// is wrong when the result looks off.
        bool setupCamera(const float* view, const float* projection);

        /// Hands Remix the camera as explicit parameters rather than matrices.
        ///
        /// Preferred, because it removes three separate conventions from the list of things that can be
        /// silently wrong: row versus column vectors, handedness, and OpenMW's reversed-Z depth range.
        /// The runtime builds its own matrices from these, so only the values have to be right.
        ///
        /// @param eye     camera position in world space
        /// @param forward viewing direction, need not be normalised
        /// @param up      up vector
        /// @param right   right vector
        /// @param fovYDegrees vertical field of view
        /// @param aspect  width / height
        /// @param nearPlane,farPlane in the same units as eye -- OpenMW units, not metres
        bool setupCameraParameterized(const float* eye, const float* forward, const float* up,
            const float* right, float fovYDegrees, float aspect, float nearPlane, float farPlane);

        /// Blits Remix's final colour into the shared render target. Cheap, GPU-side.
        ///
        /// Unsynchronised: the consumer has no ordering guarantee against this copy. Kept for
        /// comparison against the synced path; prefer copyOutputSynced.
        bool copyOutput();

        /// As copyOutput, but ordered against the OpenGL consumer through the semaphore pair.
        ///
        /// @param consumerSignalledSinceLastCall must be true only when the GL side really did signal
        ///        "consumer done" since the previous call. These are binary semaphores, so claiming a
        ///        signal that never happened blocks Remix's render thread with no way out. Pass false
        ///        for the first call and for any frame where the composite did not run.
        ///
        /// Also arms the runtime's developer menu overlay: a host consuming output this way is not
        /// presenting, so Remix draws its menu into the copied image instead of into a swapchain
        /// image that is never produced.
        bool copyOutputSynced(bool consumerSignalledSinceLastCall);

        /// As copyOutputSynced, but Remix does not signal that the copy finished -- it only waits for us to
        /// finish sampling before it overwrites the image.
        ///
        /// This is the handshake that actually works from OpenGL. glSignalSemaphoreEXT on an imported
        /// semaphore succeeds; glWaitSemaphoreEXT on one returns GL_INVALID_OPERATION no matter how long ago
        /// the matching signal was submitted, whether a texture barrier is supplied, or whether exactly one
        /// signal is outstanding -- all of which were measured rather than assumed.
        ///
        /// One direction is enough for the hazard that matters. The damaging race is Remix overwriting the
        /// image while we are still reading it, and our own signal closes that. What is given up is knowing
        /// when the copy finished, so we may sample an image one copy behind -- which is already the case,
        /// because the copy is deferred past the draw for exactly this reason.
        ///
        /// @param consumerSignalledSinceLastCall carries the same requirement as copyOutputSynced: false
        ///        unless we really did signal since the previous call, or Remix's render thread waits on a
        ///        signal that never comes.
        bool copyOutputWaitOnly(bool consumerSignalledSinceLastCall);

        /// Drives a Remix frame. Required: the raytracing output that copyOutput() reads is only
        /// produced as part of presenting.
        bool present();

        /// Reads the shared render target into system memory as tightly packed B8G8R8A8.
        ///
        /// A GPU-to-CPU round trip, so it stalls the pipeline. It exists because it is a
        /// defined-behaviour way to get Remix's output on screen that depends on none of the
        /// cross-API synchronisation: the same mechanism probeOutputNonBlack already proves works.
        /// Treat it as a diagnostic and tuning path, not the shipping one -- at 4K it moves 33 MB a
        /// frame. One staging surface is reused across calls.
        ///
        /// Bytes per pixel of the readback, and of the shared output surface it comes from.
        ///
        /// Eight, not four: the surface is half-float RGBA to match the runtime's own final colour image,
        /// so the readback carries the range the renderer produced instead of an eight-bit clamp of it.
        static constexpr unsigned int kOutputBytesPerPixel = 8;

        /// @param out resized to width * height * kOutputBytesPerPixel on success.
        bool readOutputPixels(std::vector<unsigned char>& out, unsigned int& outWidth, unsigned int& outHeight);

        /// Nanoseconds the last readOutputPixels spent in each of its three stages.
        ///
        /// Split three ways because the first measurement of this lumped them and drew the wrong
        /// conclusion from it. \a queueNanoseconds is GetRenderTargetData, which merely queues the
        /// transfer and returns almost immediately. \a lockNanoseconds is LockRect, which blocks until the
        /// GPU has finished both the rendering being read and the transfer itself -- so it is a
        /// synchronisation stall, and it was measured at over sixty milliseconds a frame at 4K.
        /// \a copyNanoseconds is the row-by-row copy out of the mapped surface, pure host bandwidth, about
        /// three milliseconds for a 4K frame.
        ///
        /// Keeping them apart is what distinguishes "stop copying so much" from "stop waiting", and only
        /// the second one was ever worth real effort.
        void lastReadbackSplit(unsigned long long& queueNanoseconds, unsigned long long& lockNanoseconds,
            unsigned long long& copyNanoseconds) const;

        /// Reads the shared render target back to the CPU and logs whether anything in it is non-black.
        ///
        /// This is the only way to tell "Remix rendered nothing" apart from "Remix rendered something and
        /// OpenGL is not seeing it". It goes entirely through D3D9, touching none of the interop, so its
        /// answer is independent of the GL side. Slow -- it stalls on a GPU readback -- so call it once,
        /// not per frame.
        bool probeOutputNonBlack();

        /// One vertex in the only layout the Remix API accepts.
        ///
        /// Mirrors remixapi_HardcodedVertex exactly, including its 28 bytes of tail padding, because
        /// CreateMesh hard-binds the field offsets and a 64-byte stride. Redeclared here rather than
        /// exposing remix_c.h, which would drag <windows.h> into everything that submits geometry.
        /// The static_asserts in runtime.cpp keep the two definitions honest.
        ///
        /// One UV set, one colour set, no tangent channel. Normal mapping therefore has to come from
        /// the material's normal/tangent textures; tangents cannot ride in on the vertex buffer.
        struct Vertex
        {
            float mPosition[3];
            float mNormal[3];
            float mTexcoord[2];
            unsigned int mColor; ///< packed 8-bit BGRA
            unsigned int mPad[7];
        };

        /// Instance category bits, mirroring remixapi_InstanceCategoryBit.
        ///
        /// Duplicated here so callers can classify geometry without including remix_c.h, which would
        /// pull <windows.h> into the rest of OpenMW. The values must match the API's exactly; the
        /// static_asserts in runtime.cpp enforce that.
        ///
        /// Setting these is what makes Remix's usual texture-hash tagging workflow unnecessary: that
        /// exists so Remix can infer what a D3D9 draw *is*, and submitting through the API means we can
        /// simply say.
        enum InstanceCategory : unsigned int
        {
            Category_Sky = 1u << 2,
            /// Offsets an instance off the surface it sits on, so a coincident overlay is not ambiguous.
            ///
            /// Needed for per-layer terrain. A terrain layer is the same triangles as the ground beneath it
            /// at exactly the same depth, which a rasteriser resolves with a depth-equal test and a blend.
            /// A path tracer has no such test: a blended surface sitting exactly at the opaque hit distance
            /// is on the boundary of the transparency traversal's range, so whether it contributes at all is
            /// decided by floating-point luck. The visible result is one layer winning outright across a
            /// region with a hard edge where the winner changes -- which is what per-layer terrain looked
            /// like before this, and it is not a blending failure at all.
            ///
            /// The runtime's own D3D9 path solves this the same way, through rtx.terrainAsDecals. That
            /// global switch is unusable here because it can only see the Terrain category and so relabels
            /// the base layer too, leaving decals stacked on nothing -- see the note in
            /// rtx_fork_submit.cpp. Setting it per instance avoids that: the host knows which pass is the
            /// ground and which are overlays.
            Category_DecalStatic = 1u << 11,
            Category_Particle = 1u << 9,
            Category_Terrain = 1u << 16,
            Category_AnimatedWater = 1u << 17,
        };

        /// Creates a flat, untextured opaque material. Idempotent for a given hash.
        ///
        /// @param hash caller-owned and must be unique; the runtime uses it as the identity.
        /// @param emissive radiance added regardless of lighting, as a multiplier on \a emissiveColour.
        ///        Zero means a purely reflective surface. Non-zero is the only way to see geometry at
        ///        all before lights are submitted, so it doubles as the bring-up diagnostic: a surface
        ///        that is invisible with emission on is not in the scene, not merely unlit.
        /// @return an opaque handle, or 0 on failure.
        unsigned long long createFlatMaterial(unsigned long long hash, float red, float green, float blue,
            float roughness, float metallic, float emissive = 0.0f,
            const float* emissiveColour = nullptr);

        /// Per-vertex bone influences for a skinned mesh.
        ///
        /// The vertices of a skinned mesh are the bind pose and never change, so the mesh is created once
        /// and only the bone transforms are submitted per frame. That is the whole point of going through
        /// this rather than rebuilding a deformed mesh every frame: the runtime rebuilds acceleration
        /// structures from bone transforms far more cheaply than from new geometry, and a mesh whose
        /// handle changes every frame also defeats the temporal accumulation a path tracer depends on.
        struct Skinning
        {
            /// Influences per vertex. Every vertex has exactly this many slots; unused ones take a weight
            /// of zero. Any value is legal, though four is what NIF skins use.
            unsigned int mBonesPerVertex = 0;
            /// mBonesPerVertex * vertexCount weights, vertex-major.
            ///
            /// These must sum to one per vertex, and not approximately: the runtime reads only the first
            /// mBonesPerVertex-1 of each tuple and derives the last as the remainder. Weights that sum to
            /// less than one therefore do not dim the vertex, they hand the shortfall to whichever bone
            /// happens to occupy the last slot.
            const float* mWeights = nullptr;
            /// mBonesPerVertex * vertexCount bone indices, vertex-major, each below kMaxBones. Indices
            /// are packed one per byte downstream, so a larger value is silently truncated.
            const unsigned int* mBoneIndices = nullptr;
        };

        /// Creates a triangle mesh with a single surface.
        ///
        /// @param hash caller-owned and must be unique across live meshes: the runtime derives the mesh
        ///        handle from it directly, so a collision silently aliases two different meshes.
        /// @param skinning optional. When given, \a vertices are the bind pose and each instance has to
        ///        supply bone transforms; see drawInstance.
        /// @return an opaque handle, or 0 on failure.
        unsigned long long createMesh(unsigned long long hash, const Vertex* vertices,
            unsigned int vertexCount, const unsigned int* indices, unsigned int indexCount,
            unsigned long long material, const Skinning* skinning = nullptr);

        /// Releases a mesh. Must be called before its hash is reused for different geometry.
        void destroyMesh(unsigned long long mesh);

        /// Pixel formats accepted by createTexture, mirroring remixapi_Format.
        ///
        /// Duplicated here for the same reason as InstanceCategory: so callers need not include
        /// remix_c.h and drag <windows.h> in with it. The static_asserts in runtime.cpp keep them equal.
        /// The sRGB variants are the ones to use for colour, and the choice is not cosmetic: the runtime
        /// linearises on sample for an _SRGB format and does not for a _UNORM one, so naming the wrong
        /// variant leaves the albedo off by a gamma curve. Textures that are not colour -- normal maps
        /// above all -- have to use the UNORM variants for exactly the same reason.
        /// Every layout appears twice, once sRGB and once linear, because the same bytes mean different
        /// things depending on what the texture is for and only the caller knows which. A normal map
        /// uploaded through an sRGB format has a gamma curve applied to what are supposed to be vector
        /// components, which tilts every normal toward the surface and shows up as flat, weak bumps
        /// rather than as an obvious failure.
        enum TextureFormat : unsigned int
        {
            Format_RGBA8 = 43, ///< 8-bit RGBA, sRGB-encoded
            Format_RGBA8_Linear = 37,
            Format_BGRA8 = 50, ///< 8-bit BGRA, sRGB-encoded
            Format_BGRA8_Linear = 44,
            Format_BC1_RGB = 132, ///< DXT1 without alpha, sRGB-encoded
            Format_BC1_RGB_Linear = 131,
            Format_BC1_RGBA = 134, ///< DXT1 with a one-bit alpha, sRGB-encoded
            Format_BC1_RGBA_Linear = 133,
            Format_BC2 = 138, ///< DXT3, explicit four-bit alpha, sRGB-encoded
            Format_BC2_Linear = 137,
            Format_BC3 = 136, ///< DXT5, interpolated alpha, sRGB-encoded
            Format_BC3_Linear = 135,
            Format_BC5 = 139, ///< Two-channel. Normal maps only, so linear is the only variant.
            Format_BC7 = 146, ///< High-quality RGBA, sRGB-encoded
            Format_BC7_Linear = 145,
        };

        /// Uploads a texture the runtime can then be told to use by hash.
        ///
        /// This is the only workable route for a host whose assets are not loose files. Remix's material
        /// fields take file paths, and OpenMW reads its textures out of BSA archives through its own VFS,
        /// so there is no path to give. Uploading the decoded pixels and referencing them by hash avoids
        /// unpacking the archives to a scratch directory purely to satisfy a path-based API.
        ///
        /// @param hash caller-owned identity, and the handle: a collision aliases two textures.
        /// @param mipLevels number of mip levels present in \a data, packed tightly one after another
        ///        with no padding, largest first. One means no mips, which is legal but shimmers badly
        ///        at distance because minification then has nothing to fall back on.
        /// @return an opaque handle, or 0 on failure.
        unsigned long long createTexture(unsigned long long hash, unsigned int width,
            unsigned int height, unsigned int mipLevels, TextureFormat format, const void* data,
            unsigned long long dataSize);

        /// Releases an uploaded texture.
        void destroyTexture(unsigned long long texture);

        /// Per-category VRAM the runtime is holding, in bytes.
        ///
        /// Every field is a direct read of the runtime's own allocator accounting. Nothing here is
        /// estimated or derived, which is the point: VRAM questions on this integration had been argued
        /// from symptoms when the runtime was already keeping the answer and nobody was asking for it.
        ///
        /// Covers only what the runtime owns. OpenMW's own GL allocations are not in here, and neither is
        /// anything outside the runtime's allocator -- see mDriverAllocated for that gap.
        struct VramStats
        {
            unsigned long long mTotalAllocated = 0; ///< Everything the runtime's allocator holds.
            unsigned long long mTotalUsed = 0; ///< Of that, what is actually in use.

            /// Allocated minus used: chunks the allocator keeps rather than returning to the driver.
            ///
            /// The difference between holding and consuming. The allocator retains freed chunks in a
            /// high-water-mark pattern, so a large value here is retention that a compaction request would
            /// return, not memory anything needs. Reading a rising total without this field is how a
            /// retention pattern gets mistaken for a leak.
            unsigned long long mPoolRetained = 0;

            unsigned long long mReplacementGeometry = 0; ///< Vertex and index data for pack meshes.
            unsigned long long mBuffers = 0;

            /// Acceleration structures -- the BVH.
            ///
            /// The number that says whether the geometry handed over is affordable, which is not the same
            /// question as how many triangles it was. A replacement swaps a mesh for a heavier one without
            /// changing anything this host counts, so this is the only place that shows up.
            unsigned long long mAccelerationStructure = 0;

            unsigned long long mOpacityMicromap = 0;

            /// Replacement and material textures: the pool rtx.texturemanager.fixedBudgetMiB bounds.
            ///
            /// Textures this host uploads through createTexture are **not** counted here. Those are a
            /// separate pool with different rules -- they have no file behind them, so they cannot be
            /// demoted to a lower mip, only released when the host releases them. Conflating the two is
            /// what led to a texture budget being blamed for a single-mip upload problem it cannot reach.
            unsigned long long mMaterialTextures = 0;

            unsigned long long mRenderTargets = 0;

            /// What the driver reports for this process on device-local heaps, and the budget it is
            /// measured against. This is the Task Manager and nvidia-smi view.
            ///
            /// mDriverAllocated minus mTotalAllocated is everything outside the runtime's own allocator:
            /// DLSS and NGX working memory, raytracing pipeline state, bindless descriptor pools, NRC.
            /// That gap is invisible to every other field here and is not small.
            unsigned long long mDriverAllocated = 0;
            unsigned long long mDriverBudget = 0;

            /// Population of the runtime's own texture cache.
            ///
            /// Climbing while this host's identity count stays flat localises the growth to the runtime
            /// side rather than to what OpenMW uploads. Note it is a high-water mark rather than a live
            /// count -- freed slots become sentinels instead of being removed -- so read the trend.
            unsigned int mTextureCacheCount = 0;
        };

        /// Reads the runtime's VRAM accounting into \a out.
        ///
        /// @return false if the runtime is not running, or is older than this entry point; \a out is
        ///         zeroed in that case rather than left untouched.
        bool vramStats(VramStats& out) const;

        /// Creates a material that samples an uploaded texture for its albedo.
        ///
        /// @param textureHash the hash passed to createTexture. Referenced through the runtime's
        ///        "0x<hex>" pseudo-path convention, which exists precisely so an uploaded texture can be
        ///        named where a file path is expected.
        /// @param alphaTestReference 0 disables alpha testing and makes the surface fully opaque.
        ///        Non-zero rejects texels whose alpha is not greater than this, which is how cutout
        ///        foliage and lattices keep their holes.
        /// @return an opaque handle, or 0 on failure.
        /// @param normalTextureHash optional tangent-space normal map, uploaded through a *linear*
        ///        format. Zero for none, which leaves the geometric normal in place.
        /// @param emissive radiance multiplier for self-lit surfaces, taking the albedo texture as the
        ///        emissive colour. Zero for an ordinary surface. Needed for particles: a flame is a
        ///        light source, and a path tracer given a non-emissive flame quad renders grey cardboard.
        /// @param blendType Remix BlendType to give the surface real order-independent transparency, or a
        ///        negative value to leave it unblended. Zero is kAlpha and one is kAlphaEmissive; the full
        ///        list is BlendType in the runtime's surface_shared.h, which the API does not export, hence
        ///        an int. Blending has to be set on the material because the per-instance
        ///        InstanceInfoBlendEXT alternative is only read when the material sets useDrawCallAlphaState,
        ///        which also diverts alpha testing to the legacy draw call an API host does not have.
        ///        Needed for particles: a cutout leaves smoke as hard-edged blobs, and the runtime only
        ///        treats an instance as a particle at all when its blending is enabled.
        /// @param maskTextureHash optional terrain layer coverage mask, uploaded through a *linear*
        ///        format. Carried in the material's height slot, which nothing here otherwise writes and
        ///        which the runtime only reads when displacement is enabled -- see the implementation.
        ///        Only the terrain baker consumes it. It exists because a terrain layer needs its diffuse
        ///        tiled many times across the chunk and its coverage stretched once over it, and a single
        ///        albedo slot cannot express both frequencies.
        unsigned long long createTexturedMaterial(unsigned long long hash, unsigned long long textureHash,
            float roughness, float metallic, unsigned char alphaTestReference,
            unsigned long long normalTextureHash = 0, float emissive = 0.0f, int blendType = -1,
            unsigned long long maskTextureHash = 0);

        /// Creates a translucent, refractive material -- water, glass.
        ///
        /// @param refractiveIndex 1.33 for water, about 1.5 for glass.
        /// @param transmittance linear RGB tint acquired over \a measurementDistance of the medium.
        ///        This is absorption, not surface colour: a value of one in a channel means that
        ///        channel passes through unattenuated.
        /// @param measurementDistance in the same units as the geometry -- OpenMW units, so a value
        ///        that looks right for a metre-scaled game will be roughly seventy times too short.
        /// @return an opaque handle, or 0 on failure.
        unsigned long long createTranslucentMaterial(unsigned long long hash, float refractiveIndex,
            const float* transmittance, float measurementDistance);

        /// Releases a material.
        void destroyMaterial(unsigned long long material);

        /// Creates or replaces a spherical light.
        ///
        /// @param hash caller-owned identity, as with meshes: the handle is the hash, so a collision
        ///        aliases two lights. Recreating with the same hash replaces the previous one.
        /// @param position world space, in the same units as the camera and geometry -- OpenMW units.
        /// @param radiance linear RGB, and *not* a 0..1 colour: it is the radiance of the sphere's
        ///        surface, so it scales with how small the sphere is. A dim value here yields a light
        ///        that is present in the scene and invisible in the image.
        /// @param radius of the emitting sphere. Also the softness control -- a larger sphere gives
        ///        softer shadows for the same total power, which has to be compensated in \a radiance.
        /// @return an opaque handle, or 0 on failure.
        unsigned long long createSphereLight(unsigned long long hash, const float* position,
            const float* radiance, float radius);

        /// Releases a light.
        void destroyLight(unsigned long long light);

        /// Adds a previously created light to this frame.
        ///
        /// Separate from creation for the same reason instances are separate from meshes: creation is
        /// the expensive half and is cached, while presence in the frame is per-frame state.
        bool drawLight(unsigned long long light);

        /// Queues one instance of a mesh for this frame.
        ///
        /// @param transform 12 floats, three rows of four, rotation in the leading 3x3 and translation
        ///        in the last column, applied as p' = M * p.
        /// @param categoryFlags remixapi_InstanceCategoryBit values, taken from OpenMW's VisMask so
        ///        Remix is told what a thing *is* rather than inferring it from a texture hash.
        /// @param boneTransforms optional, 12 floats per bone in the same layout as \a transform, one
        ///        entry per bone the mesh's skinning indices refer to. Required for a mesh created with
        ///        skinning and meaningless without one. At most kMaxBones.
        /// @param boneCount number of entries in \a boneTransforms.
        /// @param objectPickingValue identifies this draw to the developer menu, so clicking the scene
        ///        selects what is under the cursor. Must be non-zero and distinct per draw within a
        ///        frame; zero opts out and leaves the instance unpickable.
        bool drawInstance(unsigned long long mesh, const float* transform, unsigned int categoryFlags,
            bool doubleSided, const float* boneTransforms = nullptr, unsigned int boneCount = 0,
            unsigned int objectPickingValue = 0);

        /// Most bones one skinned instance can have. The runtime packs bone indices one per byte, so
        /// this is a hard limit rather than a tuning value.
        static constexpr unsigned int kMaxBones = 256;

        /// True when OPENMW_REMIX_TESTSCENE asks for the built-in test scene instead of OpenMW's.
        static bool testSceneRequested();

        /// Submits a self-contained scene: one lit quad, viewed by a camera this code defines.
        ///
        /// Exists to separate "the Remix pipeline works" from "OpenMW is feeding it correctly". It
        /// depends on nothing from the engine -- not the camera, not the scene graph -- so if this
        /// renders then init, submission, rendering, the shared-image blit, the GL import and the
        /// composite are all proven, and any remaining problem is in what OpenMW hands over.
        ///
        /// Call instead of setupCamera, before present().
        bool submitTestScene();

        /// Releases the device and unloads the runtime. Idempotent, and called by the destructor.
        /// Must run before the SDL window it was given is destroyed.
        void shutdown();

        bool isReady() const;

        /// Sets an rtx.* configuration variable, e.g. "rtx.skyMode".
        ///
        /// This is one of the two channels that drive the fork's atmosphere and weather system; the
        /// other is setGameValue. Neither involves D3D9, which is why that whole feature set is
        /// portable to a host engine unchanged.
        bool setConfigVariable(const char* key, const char* value);

        /// Writes a key/value pair into Remix's game-state store. The fork's weather blender reads
        /// "__weather.target" and "__weather.blend_seconds" from here.
        bool setGameValue(const char* key, const char* value);

        /// Reads a float back out of Remix's game-state store.
        ///
        /// The store is the return path for values the developer menu owns. It is used instead of a
        /// dedicated config getter because the runtime has no API for reading an rtx.* option back,
        /// and adding one would mean a new vtable slot and an ABI bump for two floats.
        ///
        /// Leaves `out` untouched and returns false whenever the answer is not trustworthy -- runtime
        /// down, key never written, value unparseable -- so a caller can simply keep whatever it had.
        /// Note the runtime reports success for a *missing* key too, signalling absence through a zero
        /// size, so the return code alone is not enough to tell whether anything was read.
        bool getGameValueFloat(const char* key, float& out) const;

        /// Which developer-menu state the runtime is in: 0 none, 1 basic, 2 advanced.
        ///
        /// Worth reading rather than assuming. Setting the state is deferred to the end of the frame
        /// in which it was requested, so a read taken immediately after a set still reports the old
        /// value; and setConfigVariable cannot answer this question at all, because it reports whether
        /// the option *name* resolved and says nothing about whether the value parsed.
        int uiState() const;

        /// Requests a developer-menu state: 0 none, 1 basic, 2 advanced.
        ///
        /// Preferred over setConfigVariable("rtx.showUI", ...) because it hands the runtime a typed
        /// enum instead of a string, so there is no parse to get wrong or to fail silently. Takes
        /// effect at the end of the current Remix frame.
        bool setUiState(int state);

        /// Path the runtime was loaded from, for diagnostics. Empty when not loaded.
        const std::string& loadedFrom() const;

    private:
        /// Asks the runtime to draw its developer menu. Only meaningful when the present window is
        /// visible, because that menu is rasterised into the presented swapchain image and exists
        /// nowhere else -- see createPresentWindow in runtime.cpp.
        void enableDeveloperMenu();

        void releaseOutputTarget();
        void unloadModule();

        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };

    /// Presents Remix's frame from a loop that drives the viewer itself.
    ///
    /// OpenMW has four places that pump their own traversals instead of going through Engine::frame:
    /// loading screens, the modal message-box loop, video playback, and the screenshot manager. The first
    /// three put something on screen and so need a present; the fourth renders to its own target and must
    /// not have one.
    ///
    /// Without this they display nothing at all. The interface is drawn into Remix's overlay image
    /// correctly -- renderingTraversals runs the GUI camera -- but the frame is never presented, so the
    /// screen keeps whatever was last shown. It reads as the loading screen and the intro videos being
    /// missing, when in fact they were rendered and never shown.
    ///
    /// A free function with a settable implementation, rather than a method on something those call sites
    /// already hold, because none of them can reach the Runtime: two live in MyGUI's window manager and one
    /// in the loading screen, and widening MWBase::WindowManager for a Remix-specific concern would put it
    /// in a worse place than this. Does nothing until the engine installs a presenter, which it only does
    /// when Remix is what reaches the screen.
    void setNestedFramePresenter(std::function<void()> present);

    /// Runs whatever setNestedFramePresenter installed. Safe to call when nothing is installed, and safe
    /// to call when Remix is not in use at all -- both are no-ops.
    void presentNestedFrame();

    /// What the frames drawn by those loops cost, so the engine can both report and ration them.
    ///
    /// A measured cell load spent 10379.7 ms across seven of these presents while the load's own work was
    /// roughly 160 ms -- the progress bar cost sixty times what it was reporting on. Which fix that calls
    /// for depends on where the time is, and the mean alone cannot say: a per-frame path-traced cost and a
    /// one-off queue of asset compilation draining at the first synchronisation point produce the same
    /// mean over seven samples. Hence first, worst and least separately, and the traversal split out from
    /// the present -- OpenMW's incremental compile runs in the traversal, so if the cost is really asset
    /// compilation it belongs on that side of the line or in the first present alone.
    struct NestedFrameStats
    {
        unsigned int mFrames = 0;
        unsigned int mSkipped = 0;
        /// eventTraversal + updateTraversal + renderingTraversals, contributed by the loop itself.
        double mTraversalMs = 0.0;
        double mWorstTraversalMs = 0.0;
        /// resubmitCamera + present, contributed by the presenter.
        double mPresentMs = 0.0;
        double mFirstPresentMs = 0.0;
        double mWorstPresentMs = 0.0;
        double mLeastPresentMs = 0.0;
    };

    /// The process-wide nested frame statistics. Reset by whoever reports them.
    NestedFrameStats& nestedFrameStats();
}

#endif
