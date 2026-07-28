// Must precede every include: see the note above the d3d9.h include below. windows.h has include
// guards, so if anything pulls it in while NOGDI is still defined, wingdi.h is skipped for good and
// d3d9.h will not compile. This file is also excluded from the precompiled header for the same reason
// (see components/CMakeLists.txt).
#undef NOGDI

#include "runtime.hpp"

#include <cstdlib>

#include <components/debug/debuglog.hpp>

#ifdef _WIN32

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include <SDL_syswm.h>
#include <SDL_video.h>

// remix_c.h only forward-declares the D3D9 interfaces as opaque types. We call methods on them --
// CreateDeviceEx, CreateRenderTarget, Release -- so the real declarations have to come first. d3d9.h
// depends on the Windows types and does not include windows.h itself.
//
// The project defines NOGDI globally (root CMakeLists.txt) because wingdi.h's RELATIVE macro collides
// with osgAnimation::MorphGeometry::RELATIVE. That also removes the GDI types d3d9.h needs -- RGNDATA
// in IDirect3DDevice9::Present, among others -- so it has to come back for this translation unit.
// Safe here specifically because nothing in this file touches osgAnimation. Do not move the D3D9
// includes into a header, or that collision becomes everyone's problem.
#undef NOGDI
#include <windows.h>

#include <d3d9.h>

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

    /// True when an environment variable is set to anything other than empty or "0".
    bool envFlag(const char* name)
    {
        const char* value = std::getenv(name);
        return value != nullptr && *value != '\0' && *value != '0';
    }

    /// Parses "1600x900" out of an environment variable. Leaves the outputs untouched and returns
    /// false when the variable is unset or malformed.
    bool envWindowSize(const char* name, uint32_t& width, uint32_t& height)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return false;

        unsigned parsedWidth = 0;
        unsigned parsedHeight = 0;
        if (std::sscanf(value, "%ux%u", &parsedWidth, &parsedHeight) != 2 || parsedWidth == 0
            || parsedHeight == 0)
        {
            Log(Debug::Warning) << "Remix: ignoring " << name << "='" << value
                                << "', expected a size like 1600x900";
            return false;
        }

        width = parsedWidth;
        height = parsedHeight;
        return true;
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

    constexpr const wchar_t* kHiddenWindowClass = L"OpenMWRemixPresentSink";

    LRESULT CALLBACK presentWindowProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
    {
        // Closing this window must not destroy it. Remix's swapchain is bound to this HWND and the
        // runtime keeps presenting into it every frame, so a destroyed window would leave the device
        // presenting to nothing. Hide instead, which lets the user dismiss the Remix view without
        // taking the runtime down.
        if (message == WM_CLOSE)
        {
            ShowWindow(hwnd, SW_HIDE);
            return 0;
        }
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }

    /// Creates the window Remix presents into.
    ///
    /// Remix only produces its raytracing output as part of presenting, so a frame has to be driven.
    /// Presenting onto OpenMW's own window would mean two presenters -- a Vulkan swapchain and
    /// OpenMW's OpenGL context -- fighting over one surface, so Remix gets a window of its own and we
    /// take the finished image via dxvk_CopyRenderingOutput.
    ///
    /// It is sized to the intended render resolution rather than 1x1, because the swapchain dimensions
    /// determine what resolution Remix renders at.
    ///
    /// @param visible show the window. Worth understanding what this does and does not change:
    ///        nothing about the render path, because the runtime already presents into this window's
    ///        swapchain every frame whether or not anyone can see it. What it buys is the **Remix
    ///        developer menu**, which the runtime rasterises into the presented swapchain image inside
    ///        D3D9SwapChainEx::PresentImage -- after the backbuffer has been blitted in, so it exists
    ///        only in the presented image and is not reachable through either the D3D9 backbuffer or
    ///        rtOutput.m_finalOutput. Showing the window is therefore the only way to reach that menu
    ///        that does not require a fork-side capture of the post-overlay image. DXVK subclasses
    ///        this HWND when it creates the swapchain, so once the window is visible and focused its
    ///        input reaches ImGui without any forwarding on our side.
    HWND createPresentWindow(uint32_t width, uint32_t height, bool visible)
    {
        static bool classRegistered = false;
        if (!classRegistered)
        {
            WNDCLASSEXW wc = {};
            wc.cbSize = sizeof(wc);
            // Our own proc rather than DefWindowProcW for the WM_CLOSE handling above. It also keeps
            // DXVK's subclassing on the ordinary path: it chains to whatever proc it replaced, and a
            // null previous proc is what makes it fall back to the 32-bit bridge message channel,
            // which has nothing to do with us.
            wc.lpfnWndProc = presentWindowProc;
            wc.hInstance = GetModuleHandleW(nullptr);
            wc.lpszClassName = kHiddenWindowClass;
            wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
            // A deliberately unmistakable background. With a null brush an unpainted window shows
            // whatever the compositor last had, which reads as plain white -- indistinguishable from
            // "Remix presented a blown-out frame". Anything still this colour has never been presented
            // into, which turns a guess into an observation.
            wc.hbrBackground = CreateSolidBrush(RGB(40, 0, 60));
            if (RegisterClassExW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
            {
                Log(Debug::Error) << "Remix: could not register the present window class, error "
                                  << GetLastError();
                return nullptr;
            }
            classRegistered = true;
        }

        // A caption when visible, so the window can be moved off OpenMW's; borderless when not, since
        // nothing will ever look at it. Sizing is the trap here: CreateWindowExW takes the *whole*
        // window size including non-client area, so a captioned window asked for 3840x2160 yields a
        // 3824x2121 client area, and Remix sizes its swapchain -- hence its render resolution -- to
        // the client area. AdjustWindowRectEx inflates the request by exactly the non-client size, so
        // the client area comes out at the resolution asked for whatever the style is. The previous
        // revision used WS_POPUP to dodge this by having no non-client area at all; doing it properly
        // means the style is now free to change.
        //
        // No thick frame or maximise box: a resize would change the client area, and the runtime
        // reacts to that by resetting the swapchain underneath us.
        const DWORD style = visible ? (WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX) : WS_POPUP;
        RECT rect = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };
        AdjustWindowRectEx(&rect, style, FALSE, 0);

        const HWND hwnd = CreateWindowExW(0, kHiddenWindowClass, L"RTX Remix (OpenMW)", style,
            CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top, nullptr,
            nullptr, GetModuleHandleW(nullptr), nullptr);
        if (hwnd == nullptr)
        {
            Log(Debug::Error) << "Remix: could not create the present window, error " << GetLastError();
            return nullptr;
        }

        if (visible)
        {
            // SHOWNOACTIVATE so OpenMW keeps keyboard focus at startup: its own menu is what starts a
            // game, and there is no 3D scene to look at until one is running. Click this window when
            // you want the Remix menu.
            ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        }

        return hwnd;
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

        // We create the device ourselves rather than calling remixapi_Startup, because Startup does not
        // hand back the IDirect3DDevice9Ex and we need it to allocate a shared render target.
        IDirect3D9Ex* mD3D9 = nullptr;
        IDirect3DDevice9Ex* mDevice = nullptr;
        HWND mPresentWindow = nullptr;

        IDirect3DSurface9* mOutputSurface = nullptr;
        ExternalImage mOutputImage;
        bool mHaveOutput = false;
        ExternalSync mOutputSync;

        // Built-in test scene, created once on first submit.
        remixapi_MaterialHandle mTestMaterial = nullptr;
        remixapi_MeshHandle mTestMesh = nullptr;
        remixapi_LightHandle mTestLight = nullptr;
        bool mTestSceneBuilt = false;
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

        // Size Remix's render resolution from OpenMW's window, but present into our own.
        uint32_t width = 0;
        uint32_t height = 0;
        if (const HWND gameWindow = nativeHandle(window))
        {
            RECT client = {};
            GetClientRect(gameWindow, &client);
            width = static_cast<uint32_t>(std::max(0l, client.right - client.left));
            height = static_cast<uint32_t>(std::max(0l, client.bottom - client.top));
        }
        if (width == 0 || height == 0)
        {
            Log(Debug::Error) << "Remix: could not determine the render resolution from the window";
            unloadModule();
            return false;
        }

        // Optional override of Remix's render resolution. Independent of OpenMW's window because it is
        // the present window's client area that sizes the swapchain. Useful for tuning: a 4K path
        // traced view is both slow and large enough to bury OpenMW's window.
        envWindowSize("OPENMW_REMIX_WINDOW_SIZE", width, height);

        const bool showPresentWindow = envFlag("OPENMW_REMIX_WINDOW");
        mImpl->mPresentWindow = createPresentWindow(width, height, showPresentWindow);
        if (mImpl->mPresentWindow == nullptr)
        {
            unloadModule();
            return false;
        }

        if (mImpl->mApi.dxvk_CreateD3D9 == nullptr || mImpl->mApi.dxvk_RegisterD3D9Device == nullptr)
        {
            Log(Debug::Error) << "Remix: the runtime does not expose the DXVK interop entry points";
            unloadModule();
            return false;
        }

        // editorModeEnabled is false. Note the legacy vtable slot reuses that argument to drive
        // forceNoVkSwapchain, so passing true here would silently switch the device into
        // external-swapchain mode as well; we want an ordinary swapchain on the hidden window.
        status = mImpl->mApi.dxvk_CreateD3D9(false, &mImpl->mD3D9);
        if (status != REMIXAPI_ERROR_CODE_SUCCESS || mImpl->mD3D9 == nullptr)
        {
            Log(Debug::Error) << "Remix: dxvk_CreateD3D9 failed: " << describe(status);
            unloadModule();
            return false;
        }

        D3DPRESENT_PARAMETERS present = {};
        present.BackBufferWidth = width;
        present.BackBufferHeight = height;
        present.BackBufferFormat = D3DFMT_UNKNOWN;
        present.BackBufferCount = 0;
        present.MultiSampleType = D3DMULTISAMPLE_NONE;
        present.SwapEffect = D3DSWAPEFFECT_DISCARD;
        present.hDeviceWindow = mImpl->mPresentWindow;
        present.Windowed = TRUE;
        present.EnableAutoDepthStencil = FALSE;
        present.AutoDepthStencilFormat = D3DFMT_UNKNOWN;

        HRESULT hr = mImpl->mD3D9->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL,
            mImpl->mPresentWindow, D3DCREATE_HARDWARE_VERTEXPROCESSING, &present, nullptr, &mImpl->mDevice);
        if (FAILED(hr) || mImpl->mDevice == nullptr)
        {
            // Remix aliases its GPU-capability failures into HRESULT space, so a raw HRESULT here is
            // often really one of the REMIXAPI_ERROR_CODE_HRESULT_* values.
            Log(Debug::Error) << "Remix: CreateDeviceEx failed, hr 0x" << std::hex << hr << std::dec << " ("
                              << describe(static_cast<remixapi_ErrorCode>(hr)) << ")";
            unloadModule();
            return false;
        }

        status = mImpl->mApi.dxvk_RegisterD3D9Device(mImpl->mDevice);
        if (status != REMIXAPI_ERROR_CODE_SUCCESS)
        {
            Log(Debug::Error) << "Remix: dxvk_RegisterD3D9Device failed: " << describe(status);
            unloadModule();
            return false;
        }

        mImpl->mStarted = true;
        mImpl->mLoadedFrom = runtimePath.string();

        Log(Debug::Info) << "Remix: runtime initialised from " << runtimePath << " (API "
                         << REMIXAPI_VERSION_MAJOR << "." << REMIXAPI_VERSION_MINOR << "."
                         << REMIXAPI_VERSION_PATCH << ", " << width << "x" << height << ", presenting to "
                         << (showPresentWindow ? "its own visible window" : "an offscreen window") << ")";

        // No longer conditional on the window being visible. The runtime now draws its menu into the
        // output image that copyOutputSynced copies, so the menu arrives through OpenMW's own frame
        // and does not need Remix's presenter -- which does not reach the screen in this process
        // anyway.
        enableDeveloperMenu();

        if (!createOutputTarget(width, height))
            return false;

        // Not fatal. Without the semaphores the GL consumer has no ordering against the copy, which is
        // undefined under GL_EXT_memory_object and reads as black; the composite logs that and carries
        // on so the failure is visible rather than silent.
        createOutputSync();
        return true;
    }

    int Runtime::uiState() const
    {
        if (!mImpl->mStarted || mImpl->mApi.GetUIState == nullptr)
            return -1;
        return static_cast<int>(mImpl->mApi.GetUIState());
    }

    bool Runtime::setUiState(int state)
    {
        if (!mImpl->mStarted || mImpl->mApi.SetUIState == nullptr)
            return false;

        remixapi_UIState mapped;
        switch (state)
        {
            case 0:
                mapped = REMIXAPI_UI_STATE_NONE;
                break;
            case 1:
                mapped = REMIXAPI_UI_STATE_BASIC;
                break;
            case 2:
                mapped = REMIXAPI_UI_STATE_ADVANCED;
                break;
            default:
                return false;
        }
        return mImpl->mApi.SetUIState(mapped) == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    void Runtime::enableDeveloperMenu()
    {
        // The runtime only draws the menu when its UI state is non-zero, and it defaults to none, so a
        // visible window on its own would show a bare render. 0 = none, 1 = basic, 2 = advanced;
        // advanced is the one carrying weather and the atmosphere presets.
        //
        // Set through SetUIState rather than setConfigVariable("rtx.showUI", "2"). The config route
        // takes a string, and its return value reports only that the option *name* resolved -- a value
        // that fails to parse into the runtime's enum is indistinguishable from success. SetUIState
        // takes the enum directly and routes to the same switchMenu the runtime's own hotkey uses.
        // Default off. The menu is drawn into the same image OpenMW composites, so leaving it on by
        // default would put it over every frame of an ordinary run and muddle any comparison against
        // the raster path. run-remix-menu.cmd asks for it explicitly.
        int requested = 0;
        if (const char* fromEnv = std::getenv("OPENMW_REMIX_UI"); fromEnv != nullptr && *fromEnv != '\0')
            requested = std::atoi(fromEnv);

        if (requested == 0)
            return;

        const bool uiOk = setUiState(requested);

        // Unlike the UI state this one genuinely is a config option with no typed entry point. It only
        // affects which menu the runtime's own Alt+X reopens after a close, so a silent parse failure
        // here costs a keystroke, not the feature.
        setConfigVariable("rtx.defaultToAdvancedUI", "True");

        // Deliberately not logging "accepted" here. The state change is deferred to the end of the
        // Remix frame, so nothing can be confirmed yet -- engine.cpp reads it back a few frames in and
        // logs what the runtime actually reports.
        Log(Debug::Info) << "Remix: requested developer menu state " << requested << " ("
                         << (uiOk ? "call succeeded, applies at end of frame" : "CALL FAILED")
                         << "). Alt+X toggles it, Alt+Delete its mouse cursor; the Remix window needs "
                            "keyboard focus first.";
    }

    bool Runtime::createOutputTarget(unsigned int width, unsigned int height)
    {
        if (!mImpl->mStarted)
            return false;

        releaseOutputTarget();

        // A non-null pSharedHandle is what makes DXVK allocate the image with
        // VkExportMemoryAllocateInfo plus a dedicated allocation, which is the whole point: without it
        // there is no Win32 handle to hand to OpenGL. It must start as null to request export mode
        // rather than import.
        HANDLE sharedHandle = nullptr;
        HRESULT hr = mImpl->mDevice->CreateRenderTarget(width, height, D3DFMT_A8R8G8B8,
            D3DMULTISAMPLE_NONE, 0, FALSE, &mImpl->mOutputSurface, &sharedHandle);
        if (FAILED(hr) || mImpl->mOutputSurface == nullptr)
        {
            Log(Debug::Error) << "Remix: could not create the shared output render target, hr 0x" << std::hex
                              << hr << std::dec;
            return false;
        }

        if (mImpl->mApi.dxvk_GetSurfaceExternalMemory == nullptr)
        {
            Log(Debug::Error) << "Remix: this runtime predates dxvk_GetSurfaceExternalMemory. Rebuild the "
                                 "Remix runtime from the matching branch.";
            releaseOutputTarget();
            return false;
        }

        remixapi_dxvk_ExternalMemoryInfo info = {};
        const remixapi_ErrorCode status
            = mImpl->mApi.dxvk_GetSurfaceExternalMemory(mImpl->mOutputSurface, &info);
        if (status != REMIXAPI_ERROR_CODE_SUCCESS)
        {
            Log(Debug::Error) << "Remix: dxvk_GetSurfaceExternalMemory failed: " << describe(status)
                              << " -- the surface was not created shareable";
            releaseOutputTarget();
            return false;
        }

        mImpl->mOutputImage.mHandle = info.handle;
        mImpl->mOutputImage.mMemorySize = info.memorySize;
        mImpl->mOutputImage.mMemoryOffset = info.memoryOffset;
        mImpl->mOutputImage.mHandleType = info.handleType;
        mImpl->mOutputImage.mFormat = info.format;
        mImpl->mOutputImage.mWidth = info.width;
        mImpl->mOutputImage.mHeight = info.height;
        mImpl->mOutputImage.mOptimalTiling = info.optimalTiling != 0;
        mImpl->mHaveOutput = true;

        Log(Debug::Info) << "Remix: shared output target " << info.width << "x" << info.height
                         << ", VkFormat " << info.format << ", " << info.memorySize << " bytes at offset "
                         << info.memoryOffset << ", handle type 0x" << std::hex << info.handleType << std::dec
                         << ", " << (info.optimalTiling ? "optimal" : "linear") << " tiling";
        return true;
    }

    const Runtime::ExternalImage& Runtime::outputImage() const
    {
        return mImpl->mOutputImage;
    }

    bool Runtime::setupCamera(const float* view, const float* projection)
    {
        if (!mImpl->mStarted || mImpl->mApi.SetupCamera == nullptr || view == nullptr
            || projection == nullptr)
            return false;

        remixapi_CameraInfo info = {};
        info.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO;
        info.type = REMIXAPI_CAMERA_TYPE_WORLD;
        std::memcpy(info.view, view, sizeof(info.view));
        std::memcpy(info.projection, projection, sizeof(info.projection));
        return mImpl->mApi.SetupCamera(&info) == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    bool Runtime::copyOutput()
    {
        if (!mImpl->mHaveOutput || mImpl->mApi.dxvk_CopyRenderingOutput == nullptr)
            return false;
        return mImpl->mApi.dxvk_CopyRenderingOutput(
                   mImpl->mOutputSurface, REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_FINAL_COLOR)
            == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    bool Runtime::createOutputSync()
    {
        mImpl->mOutputSync = {};

        if (!mImpl->mStarted)
            return false;
        if (mImpl->mApi.dxvk_GetOutputSyncSemaphores == nullptr)
        {
            Log(Debug::Error) << "Remix: this runtime predates dxvk_GetOutputSyncSemaphores. Rebuild "
                                 "the Remix runtime from the matching branch, or the composited frame "
                                 "will keep reading as black.";
            return false;
        }

        remixapi_dxvk_OutputSyncInfo info = {};
        const remixapi_ErrorCode status = mImpl->mApi.dxvk_GetOutputSyncSemaphores(&info);
        if (status != REMIXAPI_ERROR_CODE_SUCCESS)
        {
            Log(Debug::Error) << "Remix: dxvk_GetOutputSyncSemaphores failed: " << describe(status);
            return false;
        }

        mImpl->mOutputSync.mCopyComplete = info.copyComplete;
        mImpl->mOutputSync.mConsumerDone = info.consumerDone;

        Log(Debug::Info) << "Remix: output sync semaphores acquired (copyComplete 0x" << std::hex
                         << info.copyComplete << ", consumerDone 0x" << info.consumerDone << std::dec
                         << "); these are NT handles, unlike the image's KMT handle";
        return mImpl->mOutputSync.valid();
    }

    const Runtime::ExternalSync& Runtime::outputSync() const
    {
        return mImpl->mOutputSync;
    }

    bool Runtime::copyOutputSynced(bool consumerSignalledSinceLastCall)
    {
        if (!mImpl->mHaveOutput || mImpl->mApi.dxvk_CopyRenderingOutputSynced == nullptr)
            return false;

        return mImpl->mApi.dxvk_CopyRenderingOutputSynced(mImpl->mOutputSurface,
                   REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_FINAL_COLOR,
                   consumerSignalledSinceLastCall ? 1u : 0u)
            == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    bool Runtime::probeOutputNonBlack()
    {
        if (!mImpl->mHaveOutput || mImpl->mDevice == nullptr)
            return false;

        const UINT width = mImpl->mOutputImage.mWidth;
        const UINT height = mImpl->mOutputImage.mHeight;

        // A render target cannot be locked directly; GetRenderTargetData copies it into a system-memory
        // plain surface that can be.
        IDirect3DSurface9* staging = nullptr;
        HRESULT hr = mImpl->mDevice->CreateOffscreenPlainSurface(
            width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &staging, nullptr);
        if (FAILED(hr) || staging == nullptr)
        {
            Log(Debug::Error) << "Remix probe: CreateOffscreenPlainSurface failed, hr 0x" << std::hex << hr;
            return false;
        }

        hr = mImpl->mDevice->GetRenderTargetData(mImpl->mOutputSurface, staging);
        if (FAILED(hr))
        {
            Log(Debug::Error) << "Remix probe: GetRenderTargetData failed, hr 0x" << std::hex << hr;
            staging->Release();
            return false;
        }

        D3DLOCKED_RECT locked = {};
        hr = staging->LockRect(&locked, nullptr, D3DLOCK_READONLY);
        if (FAILED(hr))
        {
            Log(Debug::Error) << "Remix probe: LockRect failed, hr 0x" << std::hex << hr;
            staging->Release();
            return false;
        }

        // Sparse sample rather than every pixel: enough to answer "is anything lit" at 4K without
        // walking 33 MB.
        unsigned int maxChannel = 0;
        unsigned int nonBlack = 0;
        unsigned int sampled = 0;
        const auto* base = static_cast<const unsigned char*>(locked.pBits);
        for (UINT y = 0; y < height; y += 8)
        {
            const auto* row = reinterpret_cast<const unsigned int*>(base + locked.Pitch * y);
            for (UINT x = 0; x < width; x += 8)
            {
                const unsigned int pixel = row[x];
                ++sampled;
                for (int shift = 0; shift < 24; shift += 8)
                {
                    const unsigned int channel = (pixel >> shift) & 0xFFu;
                    maxChannel = std::max(maxChannel, channel);
                }
                if ((pixel & 0x00FFFFFFu) != 0)
                    ++nonBlack;
            }
        }

        staging->UnlockRect();
        staging->Release();

        Log(Debug::Info) << "Remix probe: sampled " << sampled << " pixels of the shared target, "
                         << nonBlack << " non-black, brightest channel " << maxChannel << "/255"
                         << (nonBlack == 0
                                    ? " -- Remix produced a black image, so the problem is upstream of the interop"
                                    : " -- Remix produced an image, so the problem is the GL side (sync or layout)");
        return nonBlack > 0;
    }

    bool Runtime::testSceneRequested()
    {
        const char* value = std::getenv("OPENMW_REMIX_TESTSCENE");
        return value != nullptr && *value != '\0' && *value != '0';
    }

    bool Runtime::submitTestScene()
    {
        if (!mImpl->mStarted)
            return false;

        const remixapi_Interface& api = mImpl->mApi;
        if (api.CreateMaterial == nullptr || api.CreateMesh == nullptr || api.DrawInstance == nullptr
            || api.CreateLight == nullptr || api.DrawLightInstance == nullptr
            || api.SetupCamera == nullptr)
            return false;

        // Handles are the caller's hashes, so any stable non-zero value works. Fixed constants keep
        // creation idempotent across frames.
        constexpr uint64_t kMaterialHash = 0x0BADC0DE01u;
        constexpr uint64_t kMeshHash = 0x0BADC0DE02u;
        constexpr uint64_t kLightHash = 0x0BADC0DE03u;

        if (!mImpl->mTestSceneBuilt)
        {
            remixapi_MaterialInfoOpaqueEXT opaque = {};
            opaque.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
            opaque.albedoConstant = { 0.8f, 0.8f, 0.8f };
            opaque.opacityConstant = 1.0f;
            opaque.roughnessConstant = 0.5f;
            opaque.metallicConstant = 0.0f;

            remixapi_MaterialInfo material = {};
            material.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
            material.pNext = &opaque;
            material.hash = kMaterialHash;
            // No texture paths: an untextured constant-albedo material keeps the test independent of
            // asset loading and the texture-hash lookup path.
            if (api.CreateMaterial(&material, &mImpl->mTestMaterial) != REMIXAPI_ERROR_CODE_SUCCESS)
            {
                Log(Debug::Error) << "Remix test scene: CreateMaterial failed";
                return false;
            }

            // A 2x2 quad on the XY plane facing +Z. remixapi_HardcodedVertex is the only accepted
            // layout: position, normal, one UV, one packed colour.
            remixapi_HardcodedVertex vertices[4] = {};
            const float positions[4][3]
                = { { -1.f, -1.f, 0.f }, { 1.f, -1.f, 0.f }, { -1.f, 1.f, 0.f }, { 1.f, 1.f, 0.f } };
            const float uvs[4][2] = { { 0.f, 1.f }, { 1.f, 1.f }, { 0.f, 0.f }, { 1.f, 0.f } };
            for (int i = 0; i < 4; ++i)
            {
                std::memcpy(vertices[i].position, positions[i], sizeof(positions[i]));
                vertices[i].normal[0] = 0.f;
                vertices[i].normal[1] = 0.f;
                vertices[i].normal[2] = 1.f;
                std::memcpy(vertices[i].texcoord, uvs[i], sizeof(uvs[i]));
                vertices[i].color = 0xFFFFFFFFu;
            }
            const uint32_t indices[6] = { 0, 1, 2, 2, 1, 3 };

            remixapi_MeshInfoSurfaceTriangles surface = {};
            surface.vertices_values = vertices;
            surface.vertices_count = 4;
            surface.indices_values = indices;
            surface.indices_count = 6;
            surface.skinning_hasvalue = 0;
            surface.material = mImpl->mTestMaterial;

            remixapi_MeshInfo mesh = {};
            mesh.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
            mesh.hash = kMeshHash;
            mesh.surfaces_values = &surface;
            mesh.surfaces_count = 1;
            if (api.CreateMesh(&mesh, &mImpl->mTestMesh) != REMIXAPI_ERROR_CODE_SUCCESS)
            {
                Log(Debug::Error) << "Remix test scene: CreateMesh failed";
                return false;
            }

            remixapi_LightInfoSphereEXT sphere = {};
            sphere.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT;
            sphere.position = { 2.f, 2.f, 3.f };
            sphere.radius = 0.2f;
            sphere.shaping_hasvalue = 0;
            sphere.volumetricRadianceScale = 1.0f;

            remixapi_LightInfo light = {};
            light.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO;
            light.pNext = &sphere;
            light.hash = kLightHash;
            // Bright: an under-lit test is indistinguishable from a broken one.
            light.radiance = { 800.f, 800.f, 800.f };
            light.isDynamic = 0;
            if (api.CreateLight(&light, &mImpl->mTestLight) != REMIXAPI_ERROR_CODE_SUCCESS)
            {
                Log(Debug::Error) << "Remix test scene: CreateLight failed";
                return false;
            }

            mImpl->mTestSceneBuilt = true;
            Log(Debug::Info) << "Remix test scene: built one quad, one sphere light";
        }

        // Parameterized camera rather than raw matrices, deliberately: it removes row-vs-column major,
        // handedness and reversed-Z from the list of things that could be wrong here. Those matter for
        // OpenMW's real camera, but this test exists to isolate the pipeline from them.
        remixapi_CameraInfoParameterizedEXT parameterized = {};
        parameterized.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT;
        parameterized.position = { 0.f, 0.f, 5.f };
        parameterized.forward = { 0.f, 0.f, -1.f };
        parameterized.up = { 0.f, 1.f, 0.f };
        parameterized.right = { 1.f, 0.f, 0.f };
        parameterized.fovYInDegrees = 60.f;
        parameterized.aspect = mImpl->mOutputImage.mHeight > 0
            ? static_cast<float>(mImpl->mOutputImage.mWidth) / static_cast<float>(mImpl->mOutputImage.mHeight)
            : 1.777f;
        parameterized.nearPlane = 0.1f;
        parameterized.farPlane = 1000.f;

        remixapi_CameraInfo camera = {};
        camera.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO;
        camera.pNext = &parameterized;
        camera.type = REMIXAPI_CAMERA_TYPE_WORLD;
        if (api.SetupCamera(&camera) != REMIXAPI_ERROR_CODE_SUCCESS)
            return false;

        remixapi_InstanceInfo instance = {};
        instance.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
        instance.mesh = mImpl->mTestMesh;
        instance.categoryFlags = 0;
        instance.doubleSided = 1;
        // Identity 3x4: the quad is already in world space.
        instance.transform.matrix[0][0] = 1.f;
        instance.transform.matrix[1][1] = 1.f;
        instance.transform.matrix[2][2] = 1.f;
        if (api.DrawInstance(&instance) != REMIXAPI_ERROR_CODE_SUCCESS)
            return false;

        return api.DrawLightInstance(mImpl->mTestLight) == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    bool Runtime::present()
    {
        if (!mImpl->mStarted || mImpl->mApi.Present == nullptr)
            return false;
        remixapi_PresentInfo info = {};
        info.sType = REMIXAPI_STRUCT_TYPE_PRESENT_INFO;
        info.hwndOverride = nullptr;
        return mImpl->mApi.Present(&info) == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    void Runtime::releaseOutputTarget()
    {
        mImpl->mHaveOutput = false;
        mImpl->mOutputImage = {};
        if (mImpl->mOutputSurface != nullptr)
        {
            mImpl->mOutputSurface->Release();
            mImpl->mOutputSurface = nullptr;
        }
    }

    void Runtime::unloadModule()
    {
        releaseOutputTarget();
        if (mImpl->mDevice != nullptr)
        {
            mImpl->mDevice->Release();
            mImpl->mDevice = nullptr;
        }
        if (mImpl->mD3D9 != nullptr)
        {
            mImpl->mD3D9->Release();
            mImpl->mD3D9 = nullptr;
        }
        if (mImpl->mPresentWindow != nullptr)
        {
            DestroyWindow(mImpl->mPresentWindow);
            mImpl->mPresentWindow = nullptr;
        }
        if (mImpl->mModule != nullptr)
        {
            FreeLibrary(mImpl->mModule);
            mImpl->mModule = nullptr;
        }
        mImpl->mApi = {};
        mImpl->mStarted = false;
    }

    void Runtime::shutdown()
    {
        // Shutdown() releases the device and D3D9 objects Remix knows about, so it has to run before we
        // drop our own references to them. Release the shared render target first: it was allocated from
        // that device.
        releaseOutputTarget();

        if (mImpl->mStarted && mImpl->mApi.Shutdown != nullptr)
        {
            const remixapi_ErrorCode status = mImpl->mApi.Shutdown();
            if (status != REMIXAPI_ERROR_CODE_SUCCESS)
                Log(Debug::Warning) << "Remix: Shutdown reported " << describe(status);
            mImpl->mStarted = false;
            // Shutdown() releases the registered device and factory itself, repeatedly, until their
            // refcounts hit zero. Ours are now dangling, so forget them rather than release again.
            mImpl->mDevice = nullptr;
            mImpl->mD3D9 = nullptr;
        }

        unloadModule();
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
        ExternalImage mOutputImage;
        ExternalSync mOutputSync;
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

    void Runtime::enableDeveloperMenu() {}

    int Runtime::uiState() const
    {
        return -1;
    }

    bool Runtime::setUiState(int)
    {
        return false;
    }

    bool Runtime::createOutputTarget(unsigned int, unsigned int)
    {
        return false;
    }

    bool Runtime::setupCamera(const float*, const float*)
    {
        return false;
    }

    bool Runtime::testSceneRequested()
    {
        return false;
    }

    bool Runtime::submitTestScene()
    {
        return false;
    }

    bool Runtime::probeOutputNonBlack()
    {
        return false;
    }

    const Runtime::ExternalImage& Runtime::outputImage() const
    {
        return mImpl->mOutputImage;
    }

    bool Runtime::copyOutput()
    {
        return false;
    }

    bool Runtime::createOutputSync()
    {
        return false;
    }

    const Runtime::ExternalSync& Runtime::outputSync() const
    {
        return mImpl->mOutputSync;
    }

    bool Runtime::copyOutputSynced(bool)
    {
        return false;
    }

    bool Runtime::present()
    {
        return false;
    }

    void Runtime::releaseOutputTarget() {}

    void Runtime::unloadModule() {}

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
