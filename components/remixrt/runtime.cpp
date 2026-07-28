#include "runtime.hpp"

#include <cstdlib>

#include <components/debug/debuglog.hpp>

#ifdef _WIN32

#include <filesystem>

#include <SDL_syswm.h>
#include <SDL_video.h>

// remix_c.h includes <windows.h> for HWND/HMODULE and provides an inline loader helper that does the
// LoadLibrary + GetProcAddress + remixapi_InitializeLibrary dance. Keep this include confined to this
// translation unit so windows.h does not leak into the rest of OpenMW.
#include "remix_c.h"

namespace
{
    /// Resolve the runtime's location.
    ///
    /// OPENMW_REMIX_DLL wins if set. Otherwise look for a "remix" subdirectory next to the running
    /// executable. A bare "d3d9.dll" is deliberately never used as the search string: LoadLibrary
    /// would find the *system* d3d9.dll first and we would silently initialise the wrong library.
    std::filesystem::path resolveRuntimePath()
    {
        if (const char* explicitPath = std::getenv("OPENMW_REMIX_DLL");
            explicitPath != nullptr && *explicitPath != '\0')
            return std::filesystem::path(explicitPath);

        wchar_t exePath[MAX_PATH] = {};
        const DWORD written = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        if (written == 0 || written == MAX_PATH)
            return {};

        return std::filesystem::path(exePath).parent_path() / "remix" / "d3d9.dll";
    }

    const char* describe(remixapi_ErrorCode code)
    {
        switch (code)
        {
            case REMIXAPI_ERROR_CODE_SUCCESS:
                return "success";
            case REMIXAPI_ERROR_CODE_GENERAL_FAILURE:
                return "general failure";
            case REMIXAPI_ERROR_CODE_LOAD_LIBRARY_FAILURE:
                return "LoadLibrary failed";
            case REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS:
                return "invalid arguments";
            case REMIXAPI_ERROR_CODE_GET_PROC_ADDRESS_FAILURE:
                return "remixapi_InitializeLibrary not exported";
            case REMIXAPI_ERROR_CODE_ALREADY_EXISTS:
                return "already initialised";
            case REMIXAPI_ERROR_CODE_REGISTERING_NON_REMIX_D3D9_DEVICE:
                return "device was not created by Remix";
            case REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED:
                return "device was not registered";
            case REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION:
                return "incompatible API version";
            case REMIXAPI_ERROR_CODE_NOT_INITIALIZED:
                return "not initialised";
            case REMIXAPI_ERROR_CODE_SET_DLL_DIRECTORY_FAILURE:
                return "SetDllDirectory failed";
            case REMIXAPI_ERROR_CODE_GET_FULL_PATH_NAME_FAILURE:
                return "GetFullPathName failed";
            case REMIXAPI_ERROR_CODE_HRESULT_NO_REQUIRED_GPU_FEATURES:
                return "GPU lacks required features";
            case REMIXAPI_ERROR_CODE_HRESULT_DRIVER_VERSION_BELOW_MINIMUM:
                return "driver below minimum version";
            case REMIXAPI_ERROR_CODE_HRESULT_DXVK_INSTANCE_EXTENSION_FAIL:
                return "required Vulkan instance extension missing";
            case REMIXAPI_ERROR_CODE_HRESULT_VK_CREATE_INSTANCE_FAIL:
                return "vkCreateInstance failed";
            case REMIXAPI_ERROR_CODE_HRESULT_VK_CREATE_DEVICE_FAIL:
                return "vkCreateDevice failed";
            case REMIXAPI_ERROR_CODE_HRESULT_GRAPHICS_QUEUE_FAMILY_MISSING:
                return "no graphics queue family";
            default:
                return "unrecognised error code";
        }
    }

    HWND nativeHandle(SDL_Window* window)
    {
        if (window == nullptr)
            return nullptr;

        SDL_SysWMinfo info;
        SDL_VERSION(&info.version);
        if (SDL_GetWindowWMInfo(window, &info) != SDL_TRUE)
        {
            Log(Debug::Warning) << "Remix: could not obtain the native window handle: " << SDL_GetError();
            return nullptr;
        }
        if (info.subsystem != SDL_SYSWM_WINDOWS)
        {
            Log(Debug::Warning) << "Remix: SDL is not using the Windows video backend";
            return nullptr;
        }
        return info.info.win.window;
    }
}

namespace RemixRT
{
    struct Runtime::Impl
    {
        remixapi_Interface mApi = {};
        remixapi_HMODULE mModule = nullptr;
        std::string mLoadedFrom;
        bool mStarted = false;
    };

    bool Runtime::requested()
    {
        const char* value = std::getenv("OPENMW_REMIX");
        return value != nullptr && *value != '\0' && *value != '0';
    }

    Runtime::Runtime()
        : mImpl(std::make_unique<Impl>())
    {
    }

    Runtime::~Runtime()
    {
        shutdown();
    }

    bool Runtime::isReady() const
    {
        return mImpl->mStarted;
    }

    const std::string& Runtime::loadedFrom() const
    {
        return mImpl->mLoadedFrom;
    }

    bool Runtime::initialize(SDL_Window* window)
    {
        if (mImpl->mStarted)
            return true;

        const std::filesystem::path runtimePath = resolveRuntimePath();
        if (runtimePath.empty())
        {
            Log(Debug::Error) << "Remix: could not determine where to load the runtime from";
            return false;
        }
        if (!std::filesystem::exists(runtimePath))
        {
            Log(Debug::Error) << "Remix: runtime not found at " << runtimePath
                              << ". Set OPENMW_REMIX_DLL to the Remix d3d9.dll, or place it in a "
                                 "'remix' directory beside openmw.exe.";
            return false;
        }

        // The loader helper tries the plain path first, then retries with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
        // so the runtime's own sibling DLLs (DLSS, NRD, NRC, USD, ...) resolve out of its directory.
        remixapi_ErrorCode status = remixapi_lib_loadRemixDllAndInitialize(
            runtimePath.c_str(), &mImpl->mApi, &mImpl->mModule);
        if (status != REMIXAPI_ERROR_CODE_SUCCESS)
        {
            // An incompatible version here almost always means the vendored remix_c.h and the runtime
            // came from different branches. The fork reserves minor version 1000 and treats every
            // minor as breaking, so a stock Remix build will be rejected outright.
            Log(Debug::Error) << "Remix: failed to load " << runtimePath << ": " << describe(status) << " ("
                              << static_cast<unsigned>(status) << ")";
            mImpl->mModule = nullptr;
            return false;
        }

        remixapi_StartupInfo startup = {};
        startup.sType = REMIXAPI_STRUCT_TYPE_STARTUP_INFO;
        startup.hwnd = nativeHandle(window);
        startup.disableSrgbConversionForOutput = false;
        // OpenMW owns the window and presents through its own OpenGL context, so ultimately Remix's
        // output has to arrive as an image we composite, not as a second presenter on the same HWND.
        // Requesting no Vulkan swapchain is what makes dxvk_GetExternalSwapchain available, handing back
        // a raw VkImage for OpenGL interop.
        //
        // MEASURED 2026-07-28: this flag does NOT stop a swapchain being created. Startup() calls
        // CreateDeviceEx with hDeviceWindow set and Windowed true, and the log shows a real
        // 3840x2160 VK_FORMAT_R8G8B8A8_UNORM swapchain (FIFO, 3 images) on OpenMW's window. It is
        // harmless today only because nothing ever calls remixapi_Present, so that swapchain stays
        // idle. Do not assume this flag gives us a headless device -- resolving the presentation path
        // is its own piece of work, not something this flag already handled.
        startup.forceNoVkSwapchain = true;
        startup.editorModeEnabled = false;

        status = mImpl->mApi.Startup(&startup);
        if (status != REMIXAPI_ERROR_CODE_SUCCESS)
        {
            Log(Debug::Error) << "Remix: Startup failed: " << describe(status) << " ("
                              << static_cast<unsigned>(status) << ")";
            FreeLibrary(mImpl->mModule);
            mImpl->mModule = nullptr;
            mImpl->mApi = {};
            return false;
        }

        mImpl->mStarted = true;
        mImpl->mLoadedFrom = runtimePath.string();

        Log(Debug::Info) << "Remix: runtime initialised from " << runtimePath << " (API "
                         << REMIXAPI_VERSION_MAJOR << "." << REMIXAPI_VERSION_MINOR << "."
                         << REMIXAPI_VERSION_PATCH << ", hwnd " << (startup.hwnd ? "attached" : "none") << ")";
        return true;
    }

    void Runtime::shutdown()
    {
        if (mImpl->mStarted)
        {
            const remixapi_ErrorCode status = mImpl->mApi.Shutdown();
            if (status != REMIXAPI_ERROR_CODE_SUCCESS)
                Log(Debug::Warning) << "Remix: Shutdown reported " << describe(status);
            mImpl->mStarted = false;
        }

        if (mImpl->mModule != nullptr)
        {
            FreeLibrary(mImpl->mModule);
            mImpl->mModule = nullptr;
        }

        mImpl->mApi = {};
        mImpl->mLoadedFrom.clear();
    }

    bool Runtime::setConfigVariable(const char* key, const char* value)
    {
        if (!mImpl->mStarted || mImpl->mApi.SetConfigVariable == nullptr)
            return false;
        return mImpl->mApi.SetConfigVariable(key, value) == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    bool Runtime::setGameValue(const char* key, const char* value)
    {
        if (!mImpl->mStarted || mImpl->mApi.SetGameValue == nullptr)
            return false;
        return mImpl->mApi.SetGameValue(key, value) == REMIXAPI_ERROR_CODE_SUCCESS;
    }
}

#else // !_WIN32

// The Remix runtime is a Windows x64 DLL. Rather than make the whole component conditional in CMake
// and force every call site behind an #ifdef, the non-Windows build keeps the same symbols and reports
// that the backend is unavailable.

namespace RemixRT
{
    struct Runtime::Impl
    {
        std::string mLoadedFrom;
    };

    bool Runtime::requested()
    {
        return false;
    }

    Runtime::Runtime()
        : mImpl(std::make_unique<Impl>())
    {
    }

    Runtime::~Runtime() = default;

    bool Runtime::initialize(SDL_Window*)
    {
        Log(Debug::Warning) << "Remix: the runtime is Windows-only; the Remix backend is unavailable "
                               "on this platform";
        return false;
    }

    void Runtime::shutdown() {}

    bool Runtime::isReady() const
    {
        return false;
    }

    bool Runtime::setConfigVariable(const char*, const char*)
    {
        return false;
    }

    bool Runtime::setGameValue(const char*, const char*)
    {
        return false;
    }

    const std::string& Runtime::loadedFrom() const
    {
        return mImpl->mLoadedFrom;
    }
}

#endif
