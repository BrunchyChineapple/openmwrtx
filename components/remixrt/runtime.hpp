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
        /// @param window the SDL window whose native handle Remix should attach to. May be null, in
        ///        which case Remix comes up without a window association.
        bool initialize(SDL_Window* window);

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

        /// Path the runtime was loaded from, for diagnostics. Empty when not loaded.
        const std::string& loadedFrom() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}

#endif
