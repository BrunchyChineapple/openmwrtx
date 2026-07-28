#ifndef OPENMW_COMPONENTS_REMIXRT_RUNTIME_H
#define OPENMW_COMPONENTS_REMIXRT_RUNTIME_H

#include <memory>
#include <string>

struct SDL_Window;

namespace RemixRT
{
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

        /// Hands Remix the camera for this frame.
        ///
        /// Both matrices are 16 floats in OSG's layout, which is row-major with the row-vector
        /// convention (v * M) -- the same convention D3D9 uses, so they map straight across.
        bool setupCamera(const float* view, const float* projection);

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

        /// Drives a Remix frame. Required: the raytracing output that copyOutput() reads is only
        /// produced as part of presenting.
        bool present();

        /// Reads the shared render target back to the CPU and logs whether anything in it is non-black.
        ///
        /// This is the only way to tell "Remix rendered nothing" apart from "Remix rendered something and
        /// OpenGL is not seeing it". It goes entirely through D3D9, touching none of the interop, so its
        /// answer is independent of the GL side. Slow -- it stalls on a GPU readback -- so call it once,
        /// not per frame.
        bool probeOutputNonBlack();

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
}

#endif
