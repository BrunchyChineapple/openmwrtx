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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <iterator>

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

        /// Split cost of the last readback. Plain members rather than atomics: written and read on the
        /// same thread, since readOutputPixels is only ever called from the frame loop.
        unsigned long long mReadbackQueueNanoseconds = 0;
        unsigned long long mReadbackLockNanoseconds = 0;
        unsigned long long mReadbackCopyNanoseconds = 0;

        /// Staging surfaces for the pipelined readback, written one frame and read the next, plus whether
        /// each has ever been written -- reading one that has not would hand the composite a frame of
        /// uninitialised memory.
        IDirect3DSurface9* mReadbackSurfaces[2] = { nullptr, nullptr };
        bool mReadbackFilled[2] = { false, false };
        unsigned int mReadbackWriteIndex = 0;
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

        // The runtime's directory is put on the process DLL search path before loading, and reading the
        // loader helper is the only way to see why that is necessary.
        //
        // remixapi_lib_loadRemixDllAndInitialize tries three strategies in order and stops at the first
        // that works: a plain LoadLibraryW, then LoadLibraryExW with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        // then SetDllDirectoryW on the parent directory. Strategies two and three exist precisely to make
        // the runtime's sibling DLLs findable -- and neither ever runs here, because strategy one succeeds:
        // the path we pass is absolute and valid.
        //
        // That is enough for the runtime's *static* imports, since Windows always searches a module's own
        // directory for those. It is not enough for anything the runtime loads by name at run time, which
        // searches the executable's directory, System32 and PATH -- and this executable does not live next
        // to the runtime. NRC is the case that exposed it: enabling the Neural Radiance Cache made
        // nrc::vulkan::Initialize fail with a bare "unexpected condition" and no message of its own,
        // because NRC_Vulkan.dll and its CUDA runtime sit next to d3d9.dll where nothing was looking.
        //
        // SetDllDirectoryW rather than AddDllDirectory: AddDllDirectory requires
        // SetDefaultDllDirectories, which changes the search order for the whole process, and OpenMW loads
        // its own plugins by name. This adds one directory and leaves everything else alone. It has to stay
        // set for the life of the process rather than being restored after the load, because the
        // dependencies in question are loaded lazily -- NRC's arrive if and when the user switches it on.
        {
            const std::wstring runtimeDirectory = runtimePath.parent_path().wstring();
            if (!runtimeDirectory.empty() && SetDllDirectoryW(runtimeDirectory.c_str()) == 0)
            {
                // Not fatal. The runtime itself loads from an absolute path regardless; what is lost is the
                // lazily loaded extras, and each of those reports its own failure.
                Log(Debug::Warning) << "Remix: could not add " << runtimePath.parent_path()
                                    << " to the DLL search path (error " << GetLastError()
                                    << "); features that load their own libraries, NRC especially, will "
                                       "fail to initialise";
            }
        }

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
        // the present window's client area that sizes the swapchain.
        //
        // Overriding it has a consequence that is not obvious and is worth the warning below: the
        // developer menu is drawn by the runtime into the output image, but ImGui measures itself
        // against OpenMW's window. If the two disagree, ImGui's clip rectangles land outside the
        // smaller render target and the menu's contents are scissored away -- you get a window frame
        // and a title bar with nothing in them, which looks like a broken menu rather than a
        // resolution mismatch.
        const uint32_t windowWidth = width;
        const uint32_t windowHeight = height;
        if (envWindowSize("OPENMW_REMIX_WINDOW_SIZE", width, height)
            && (width != windowWidth || height != windowHeight))
        {
            Log(Debug::Warning) << "Remix: rendering at " << width << "x" << height
                                << " while OpenMW's window is " << windowWidth << "x" << windowHeight
                                << ". The developer menu will be clipped to an empty frame, because "
                                   "ImGui measures against the window and draws into the render "
                                   "target. Unset OPENMW_REMIX_WINDOW_SIZE to use the menu.";
        }

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

        // Point the developer menu's input at OpenMW's own window, not at the present window Remix's
        // swapchain sits on. The runtime derives ImGui's display size from this window's client rect and
        // positions its raw-input sink over it, hit-testing the cursor against that rectangle. Left
        // pointing at our present window -- which is hidden and somewhere else entirely -- keyboard
        // still works, because keystrokes carry no coordinates, but the mouse silently does nothing.
        if (const HWND gameWindow = nativeHandle(window))
        {
            if (mImpl->mApi.dxvk_SetDevMenuWindow != nullptr)
            {
                mImpl->mApi.dxvk_SetDevMenuWindow(gameWindow);
            }
            else
            {
                Log(Debug::Warning) << "Remix: this runtime predates dxvk_SetDevMenuWindow; the "
                                       "developer menu will draw but the mouse will not reach it.";
            }
        }

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
        // Half float, not A8R8G8B8, and this is the first link in the chain rather than a refinement.
        //
        // The runtime's final colour image is VK_FORMAT_R16G16B16A16_SFLOAT (rtx_resources.cpp, "final
        // output"). Asking for an eight-bit shared surface made dxvk_CopyRenderingOutput clamp and quantise
        // that on the way in, so everything downstream was working from an SDR copy of an HDR image and no
        // amount of care later could recover the range. Matching the source format is the only way the
        // question "how much range is there" can even be asked.
        //
        // Note this also changes the channel order: D3DFMT_A16B16G16R16F maps to the same Vulkan format as
        // the source, so the data arrives as RGBA rather than the BGRA an A8R8G8B8 surface produced.
        HRESULT hr = mImpl->mDevice->CreateRenderTarget(width, height, D3DFMT_A16B16G16R16F,
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

    // Runtime::Vertex must be layout-identical to the runtime's only accepted vertex format. CreateMesh
    // binds the field offsets and the 64-byte stride directly, so a mismatch here would not fail to
    // compile or to call -- it would read the wrong bytes as positions.
    static_assert(sizeof(Runtime::Vertex) == sizeof(remixapi_HardcodedVertex), "vertex size drift");
    static_assert(sizeof(Runtime::Vertex) == 64, "the API fixes the vertex stride at 64 bytes");
    static_assert(offsetof(Runtime::Vertex, mPosition) == offsetof(remixapi_HardcodedVertex, position),
        "position offset drift");
    static_assert(offsetof(Runtime::Vertex, mNormal) == offsetof(remixapi_HardcodedVertex, normal),
        "normal offset drift");
    static_assert(offsetof(Runtime::Vertex, mTexcoord) == offsetof(remixapi_HardcodedVertex, texcoord),
        "texcoord offset drift");
    static_assert(offsetof(Runtime::Vertex, mColor) == offsetof(remixapi_HardcodedVertex, color),
        "color offset drift");

    // The mirrored category bits must match the API's, or geometry gets classified as something else
    // entirely -- terrain submitted as sky, for instance, which changes how Remix lights it.
    static_assert(static_cast<unsigned int>(Runtime::Category_Sky)
            == static_cast<unsigned int>(REMIXAPI_INSTANCE_CATEGORY_BIT_SKY),
        "sky category drift");
    static_assert(static_cast<unsigned int>(Runtime::Category_Particle)
            == static_cast<unsigned int>(REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE),
        "particle category drift");
    static_assert(static_cast<unsigned int>(Runtime::Category_Terrain)
            == static_cast<unsigned int>(REMIXAPI_INSTANCE_CATEGORY_BIT_TERRAIN),
        "terrain category drift");
    static_assert(static_cast<unsigned int>(Runtime::Category_AnimatedWater)
            == static_cast<unsigned int>(REMIXAPI_INSTANCE_CATEGORY_BIT_ANIMATED_WATER),
        "animated water category drift");

    bool Runtime::setupCameraParameterized(const float* eye, const float* forward, const float* up,
        const float* right, float fovYDegrees, float aspect, float nearPlane, float farPlane)
    {
        if (!mImpl->mStarted || mImpl->mApi.SetupCamera == nullptr || eye == nullptr || forward == nullptr
            || up == nullptr || right == nullptr)
            return false;

        remixapi_CameraInfoParameterizedEXT parameterized = {};
        parameterized.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT;
        parameterized.position = { eye[0], eye[1], eye[2] };
        parameterized.forward = { forward[0], forward[1], forward[2] };
        parameterized.up = { up[0], up[1], up[2] };
        parameterized.right = { right[0], right[1], right[2] };
        parameterized.fovYInDegrees = fovYDegrees;
        parameterized.aspect = aspect;
        parameterized.nearPlane = nearPlane;
        parameterized.farPlane = farPlane;

        remixapi_CameraInfo camera = {};
        camera.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO;
        camera.pNext = &parameterized;
        camera.type = REMIXAPI_CAMERA_TYPE_WORLD;
        return mImpl->mApi.SetupCamera(&camera) == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    /// Fills in the sampler state every material has to carry.
    ///
    /// These three bytes are NOT safe to leave zeroed, and the failure mode is subtle enough to be worth
    /// spelling out. The runtime reads them as MDL enumerants (lss::Mdl::Filter, lss::Mdl::WrapMode) and
    /// zero is Nearest for the filter and Clamp for both wrap modes -- so a zero-initialised
    /// remixapi_MaterialInfo asks for point sampling with clamp-to-edge addressing.
    ///
    /// The wrap mode is the damaging half. SceneManager::processDrawCallState takes the sampler the API
    /// attached to the draw, overwrites its filter and address modes from these fields
    /// (POPULATE_SAMPLER_INFO in rtx_materials.h), and rebuilds it. Clamp-to-edge on geometry whose
    /// texcoords leave [0,1] -- which is most of Morrowind, and all of the terrain, whose layers tile by
    /// scaling UVs -- smears the last row and column of texels across everything past the edge. That
    /// reads as long streaks running down the surface, locked to it rather than to the camera, and it is
    /// indistinguishable at a glance from a UV bug or a denoiser artefact.
    ///
    /// The C++ wrapper in remix.h sets these in its constructor; the C struct used here does not, which
    /// is why it has to be done explicitly.
    static void applyDefaultSamplerState(remixapi_MaterialInfo& material)
    {
        material.filterMode = 1; // lss::Mdl::Filter::Linear
        material.wrapModeU = 1; // lss::Mdl::WrapMode::Repeat
        material.wrapModeV = 1; // lss::Mdl::WrapMode::Repeat
    }

    unsigned long long Runtime::createFlatMaterial(unsigned long long hash, float red, float green,
        float blue, float roughness, float metallic, float emissive, const float* emissiveColour)
    {
        if (!mImpl->mStarted || mImpl->mApi.CreateMaterial == nullptr || hash == 0)
            return 0;

        remixapi_MaterialInfoOpaqueEXT opaque = {};
        opaque.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
        opaque.albedoConstant = { red, green, blue };
        opaque.opacityConstant = 1.0f;
        opaque.roughnessConstant = roughness;
        opaque.metallicConstant = metallic;
        // Alpha test "always", i.e. never reject a hit. This is NOT a default that can be left alone:
        // the enum is Vulkan's compare-op ordering, so zero is kNever, and the runtime derives
        // isFullyOpaque as (blending disabled && alphaTestType == kAlways). Leave it zeroed and every
        // surface is non-opaque with a test that can never pass, so the path tracer rejects every hit
        // and rays pass straight through. The geometry is then present in the acceleration structure,
        // hit-tested, and completely invisible -- you see whatever is behind it, which for a scene with
        // Remix's own sky enabled means a sky-filled frame indoors.
        opaque.alphaTestType = 7; // AlphaTestType::kAlways, surface_shared.h
        opaque.alphaReferenceValue = 0;

        remixapi_MaterialInfo material = {};
        material.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
        material.pNext = &opaque;
        material.hash = hash;
        // The runtime keys emission off intensity being positive, so leaving it zero disables emission
        // whatever the colour says. Defaulting the colour to the albedo keeps a self-lit surface the
        // same hue as a lit one, which matters when emission is being used to make geometry visible.
        material.emissiveIntensity = emissive;
        material.emissiveColorConstant = emissiveColour != nullptr
            ? remixapi_Float3D{ emissiveColour[0], emissiveColour[1], emissiveColour[2] }
            : remixapi_Float3D{ red, green, blue };
        applyDefaultSamplerState(material);

        remixapi_MaterialHandle handle = nullptr;
        if (mImpl->mApi.CreateMaterial(&material, &handle) != REMIXAPI_ERROR_CODE_SUCCESS)
            return 0;
        return reinterpret_cast<unsigned long long>(handle);
    }

    unsigned long long Runtime::createMesh(unsigned long long hash, const Vertex* vertices,
        unsigned int vertexCount, const unsigned int* indices, unsigned int indexCount,
        unsigned long long material, const Skinning* skinning)
    {
        if (!mImpl->mStarted || mImpl->mApi.CreateMesh == nullptr || hash == 0 || vertices == nullptr
            || indices == nullptr || vertexCount == 0 || indexCount == 0)
            return 0;

        remixapi_MeshInfoSurfaceTriangles surface = {};
        // The reinterpret_cast is safe on the strength of the static_asserts above, and avoids copying
        // every vertex of every mesh into an identical struct purely to satisfy the type system.
        surface.vertices_values = reinterpret_cast<const remixapi_HardcodedVertex*>(vertices);
        surface.vertices_count = vertexCount;
        surface.indices_values = indices;
        surface.indices_count = indexCount;
        surface.material = reinterpret_cast<remixapi_MaterialHandle>(material);

        surface.skinning_hasvalue = 0;
        if (skinning != nullptr && skinning->mBonesPerVertex > 0 && skinning->mWeights != nullptr
            && skinning->mBoneIndices != nullptr)
        {
            // Rejected rather than clamped. The runtime sizes its buffers from these counts and strides
            // them by bonesPerVertex, so a count that disagrees with the vertex count does not produce a
            // partly-skinned mesh, it reads the wrong tuple for every vertex past the discrepancy.
            const unsigned long long expected
                = static_cast<unsigned long long>(skinning->mBonesPerVertex) * vertexCount;
            if (expected > std::numeric_limits<unsigned int>::max())
                return 0;

            surface.skinning_hasvalue = 1;
            surface.skinning_value.bonesPerVertex = skinning->mBonesPerVertex;
            surface.skinning_value.blendWeights_values = skinning->mWeights;
            surface.skinning_value.blendWeights_count = static_cast<unsigned int>(expected);
            surface.skinning_value.blendIndices_values = skinning->mBoneIndices;
            surface.skinning_value.blendIndices_count = static_cast<unsigned int>(expected);
        }

        remixapi_MeshInfo mesh = {};
        mesh.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
        mesh.hash = hash;
        mesh.surfaces_values = &surface;
        mesh.surfaces_count = 1;

        remixapi_MeshHandle handle = nullptr;
        if (mImpl->mApi.CreateMesh(&mesh, &handle) != REMIXAPI_ERROR_CODE_SUCCESS)
            return 0;
        return reinterpret_cast<unsigned long long>(handle);
    }

    void Runtime::destroyMesh(unsigned long long mesh)
    {
        if (!mImpl->mStarted || mImpl->mApi.DestroyMesh == nullptr || mesh == 0)
            return;
        mImpl->mApi.DestroyMesh(reinterpret_cast<remixapi_MeshHandle>(mesh));
    }

    static_assert(static_cast<unsigned int>(Runtime::Format_RGBA8) == REMIXAPI_FORMAT_R8G8B8A8_SRGB,
        "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_BGRA8) == REMIXAPI_FORMAT_B8G8R8A8_SRGB,
        "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_BC1_RGB) == REMIXAPI_FORMAT_BC1_RGB_SRGB,
        "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_BC1_RGBA) == REMIXAPI_FORMAT_BC1_RGBA_SRGB,
        "texture format drift");
    static_assert(
        static_cast<unsigned int>(Runtime::Format_BC2) == REMIXAPI_FORMAT_BC2_SRGB, "texture format drift");
    static_assert(
        static_cast<unsigned int>(Runtime::Format_BC3) == REMIXAPI_FORMAT_BC3_SRGB, "texture format drift");
    static_assert(
        static_cast<unsigned int>(Runtime::Format_BC5) == REMIXAPI_FORMAT_BC5_UNORM, "texture format drift");
    static_assert(
        static_cast<unsigned int>(Runtime::Format_BC7) == REMIXAPI_FORMAT_BC7_SRGB, "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_BC7_Linear) == REMIXAPI_FORMAT_BC7_UNORM,
        "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_RGBA8_Linear) == REMIXAPI_FORMAT_R8G8B8A8_UNORM,
        "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_BGRA8_Linear) == REMIXAPI_FORMAT_B8G8R8A8_UNORM,
        "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_BC1_RGB_Linear) == REMIXAPI_FORMAT_BC1_RGB_UNORM,
        "texture format drift");
    static_assert(static_cast<unsigned int>(Runtime::Format_BC1_RGBA_Linear) == REMIXAPI_FORMAT_BC1_RGBA_UNORM,
        "texture format drift");
    static_assert(
        static_cast<unsigned int>(Runtime::Format_BC2_Linear) == REMIXAPI_FORMAT_BC2_UNORM, "texture format drift");
    static_assert(
        static_cast<unsigned int>(Runtime::Format_BC3_Linear) == REMIXAPI_FORMAT_BC3_UNORM, "texture format drift");

    unsigned long long Runtime::createTexture(unsigned long long hash, unsigned int width,
        unsigned int height, unsigned int mipLevels, TextureFormat format, const void* data,
        unsigned long long dataSize)
    {
        if (!mImpl->mStarted || mImpl->mApi.CreateTexture == nullptr || hash == 0 || data == nullptr
            || dataSize == 0 || width == 0 || height == 0)
            return 0;

        remixapi_TextureInfo info = {};
        info.sType = REMIXAPI_STRUCT_TYPE_TEXTURE_INFO;
        info.hash = hash;
        info.width = width;
        info.height = height;
        info.depth = 1;
        info.mipLevels = mipLevels > 0 ? mipLevels : 1u;
        info.format = static_cast<remixapi_Format>(format);
        info.data = data;
        info.dataSize = dataSize;

        remixapi_TextureHandle handle = nullptr;
        if (mImpl->mApi.CreateTexture(&info, &handle) != REMIXAPI_ERROR_CODE_SUCCESS)
            return 0;
        return reinterpret_cast<unsigned long long>(handle);
    }

    void Runtime::destroyTexture(unsigned long long texture)
    {
        if (!mImpl->mStarted || mImpl->mApi.DestroyTexture == nullptr || texture == 0)
            return;
        mImpl->mApi.DestroyTexture(reinterpret_cast<remixapi_TextureHandle>(texture));
    }

    unsigned long long Runtime::createTexturedMaterial(unsigned long long hash,
        unsigned long long textureHash, float roughness, float metallic,
        unsigned char alphaTestReference, unsigned long long normalTextureHash, float emissive,
        int blendType)
    {
        if (!mImpl->mStarted || mImpl->mApi.CreateMaterial == nullptr || hash == 0 || textureHash == 0)
            return 0;

        // The runtime resolves a path of this shape against textures uploaded through CreateTexture
        // instead of against the filesystem. Lower case hex without leading zeros, because the lookup
        // parses it with strtoull and compares the parsed value, not the string.
        wchar_t pseudoPath[32] = {};
        std::swprintf(pseudoPath, std::size(pseudoPath), L"0x%llx", textureHash);

        wchar_t normalPath[32] = {};
        if (normalTextureHash != 0)
            std::swprintf(normalPath, std::size(normalPath), L"0x%llx", normalTextureHash);

        remixapi_MaterialInfoOpaqueEXT opaque = {};
        opaque.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
        // Left at white so the texture is what shows. The runtime multiplies the two, so a tinted
        // constant here would darken every textured surface for no reason.
        opaque.albedoConstant = { 1.0f, 1.0f, 1.0f };
        opaque.opacityConstant = 1.0f;
        opaque.roughnessConstant = roughness;
        opaque.metallicConstant = metallic;
        // See createFlatMaterial: zero is kNever, which rejects every hit and makes the surface
        // invisible. kAlways when no cutout is wanted, kGreater when one is.
        opaque.alphaTestType = alphaTestReference != 0 ? 4 /* kGreater */ : 7 /* kAlways */;
        opaque.alphaReferenceValue = alphaTestReference;
        if (blendType >= 0)
        {
            // blendType_hasvalue is what actually enables blending, not merely a presence flag: the runtime
            // feeds it straight into the opaque material's BlendEnabled and takes BlendType from the value
            // beside it. Left false, the value is ignored entirely.
            opaque.blendType_hasvalue = 1;
            opaque.blendType_value = blendType;
            // Not inverted. The inverted forms are for draw calls with the factors the other way round
            // (ONE_MINUS_SRC_ALPHA as the source), which nothing here produces.
            opaque.invertedBlend = 0;
        }

        remixapi_MaterialInfo material = {};
        material.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
        material.pNext = &opaque;
        material.hash = hash;
        material.albedoTexture = pseudoPath;
        if (normalTextureHash != 0)
            material.normalTexture = normalPath;
        if (emissive > 0.0f)
        {
            // The albedo doubles as the emissive colour, so a flame's own texture drives what it emits
            // rather than a flat tint. Radiance scale rather than colour, for the same reason the light
            // conversion separates the two.
            material.emissiveTexture = pseudoPath;
            material.emissiveIntensity = emissive;
            material.emissiveColorConstant = { 1.0f, 1.0f, 1.0f };
        }
        applyDefaultSamplerState(material);

        remixapi_MaterialHandle handle = nullptr;
        if (mImpl->mApi.CreateMaterial(&material, &handle) != REMIXAPI_ERROR_CODE_SUCCESS)
            return 0;
        return reinterpret_cast<unsigned long long>(handle);
    }

    unsigned long long Runtime::createTranslucentMaterial(unsigned long long hash,
        float refractiveIndex, const float* transmittance, float measurementDistance)
    {
        if (!mImpl->mStarted || mImpl->mApi.CreateMaterial == nullptr || hash == 0
            || transmittance == nullptr)
            return 0;

        remixapi_MaterialInfoTranslucentEXT translucent = {};
        translucent.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_TRANSLUCENT_EXT;
        translucent.refractiveIndex = refractiveIndex;
        translucent.transmittanceColor = { transmittance[0], transmittance[1], transmittance[2] };
        translucent.transmittanceMeasurementDistance = measurementDistance;
        // Not thin-walled: a thin wall models a surface with no interior, like a soap bubble, and would
        // discard the absorption that gives water its depth cue entirely.
        translucent.thinWallThickness_hasvalue = 0;
        translucent.useDiffuseLayer = 0;

        remixapi_MaterialInfo material = {};
        material.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
        material.pNext = &translucent;
        material.hash = hash;
        applyDefaultSamplerState(material);

        remixapi_MaterialHandle handle = nullptr;
        if (mImpl->mApi.CreateMaterial(&material, &handle) != REMIXAPI_ERROR_CODE_SUCCESS)
            return 0;
        return reinterpret_cast<unsigned long long>(handle);
    }

    void Runtime::destroyMaterial(unsigned long long material)
    {
        if (!mImpl->mStarted || mImpl->mApi.DestroyMaterial == nullptr || material == 0)
            return;
        mImpl->mApi.DestroyMaterial(reinterpret_cast<remixapi_MaterialHandle>(material));
    }

    unsigned long long Runtime::createSphereLight(
        unsigned long long hash, const float* position, const float* radiance, float radius)
    {
        if (!mImpl->mStarted || mImpl->mApi.CreateLight == nullptr || hash == 0 || position == nullptr
            || radiance == nullptr)
            return 0;

        remixapi_LightInfoSphereEXT sphere = {};
        sphere.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT;
        sphere.position = { position[0], position[1], position[2] };
        sphere.radius = radius;
        sphere.shaping_hasvalue = 0;
        // Volumetric contribution left at parity with the surface contribution. Zero here would leave
        // the light out of the fog entirely, which reads as the light not existing when looking through
        // any volumetric medium.
        sphere.volumetricRadianceScale = 1.0f;

        remixapi_LightInfo light = {};
        light.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO;
        light.pNext = &sphere;
        light.hash = hash;
        light.radiance = { radiance[0], radiance[1], radiance[2] };
        // Dynamic: OpenMW's lights move with the objects carrying them and flicker, so the runtime must
        // not assume the previous frame's position is still valid for reprojection.
        light.isDynamic = 1;

        remixapi_LightHandle handle = nullptr;
        if (mImpl->mApi.CreateLight(&light, &handle) != REMIXAPI_ERROR_CODE_SUCCESS)
            return 0;
        return reinterpret_cast<unsigned long long>(handle);
    }

    void Runtime::destroyLight(unsigned long long light)
    {
        if (!mImpl->mStarted || mImpl->mApi.DestroyLight == nullptr || light == 0)
            return;
        mImpl->mApi.DestroyLight(reinterpret_cast<remixapi_LightHandle>(light));
    }

    bool Runtime::drawLight(unsigned long long light)
    {
        if (!mImpl->mStarted || mImpl->mApi.DrawLightInstance == nullptr || light == 0)
            return false;
        return mImpl->mApi.DrawLightInstance(reinterpret_cast<remixapi_LightHandle>(light))
            == REMIXAPI_ERROR_CODE_SUCCESS;
    }

    float halfToFloat(unsigned short bits)
    {
        const unsigned int sign = static_cast<unsigned int>(bits & 0x8000u) << 16;
        const unsigned int exponent = (bits >> 10) & 0x1Fu;
        const unsigned int mantissa = bits & 0x3FFu;

        if (exponent == 0)
        {
            // Zero or subnormal. Scaling the mantissa by 2^-24 is exact and avoids a normalisation loop
            // that is easy to get subtly wrong for no benefit at this call rate.
            const float value = static_cast<float>(mantissa) * 5.9604644775390625e-8f;
            return (bits & 0x8000u) != 0 ? -value : value;
        }

        unsigned int result = 0;
        if (exponent == 0x1Fu)
        {
            // Infinity or NaN, preserved rather than clamped, so a probe reporting a peak cannot quietly
            // turn a broken frame into a plausible-looking number.
            result = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            // Rebias the exponent from 15 to 127 and shift the mantissa into place.
            result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }

        float out = 0.0f;
        std::memcpy(&out, &result, sizeof(out));
        return out;
    }

    static_assert(Runtime::kMaxBones == REMIXAPI_INSTANCE_INFO_MAX_BONES_COUNT, "bone limit drift");
    // Justifies pointing the API at the caller's flat float array instead of copying it per bone.
    static_assert(sizeof(remixapi_Transform) == 12 * sizeof(float), "transform layout drift");

    bool Runtime::drawInstance(unsigned long long mesh, const float* transform,
        unsigned int categoryFlags, bool doubleSided, const float* boneTransforms, unsigned int boneCount,
        unsigned int objectPickingValue)
    {
        if (!mImpl->mStarted || mImpl->mApi.DrawInstance == nullptr || mesh == 0 || transform == nullptr)
            return false;

        remixapi_InstanceInfo instance = {};
        instance.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
        instance.mesh = reinterpret_cast<remixapi_MeshHandle>(mesh);
        instance.categoryFlags = categoryFlags;
        instance.doubleSided = doubleSided ? 1 : 0;
        std::memcpy(instance.transform.matrix, transform, sizeof(instance.transform.matrix));

        // Chained rather than copied: remixapi_Transform is three rows of four floats, which is the
        // layout the caller already hands over, so the array can be pointed at directly. The struct only
        // has to outlive the call, and the runtime deep-copies what it keeps.
        //
        // Both extensions go on one pNext chain, newest first. Order carries no meaning -- the runtime
        // finds each by struct type with pnext::find -- but the links have to be built so neither is
        // orphaned, which is the easy mistake when a second one is added later.
        remixapi_InstanceInfoBoneTransformsEXT bones = {};
        if (boneTransforms != nullptr && boneCount > 0)
        {
            bones.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_BONE_TRANSFORMS_EXT;
            bones.boneTransforms_values = reinterpret_cast<const remixapi_Transform*>(boneTransforms);
            bones.boneTransforms_count = std::min(boneCount, kMaxBones);
            instance.pNext = &bones;
        }

        // What the developer menu resolves a click in the world to.
        //
        // Without this the menu cannot select anything by pointing at it, which is the primary way
        // textures get tagged and materials get picked for replacement. The runtime reads back a picking
        // image and maps the value found at that pixel to the draw's texture hash, but the fork only
        // records that mapping when the value is non-zero: externalDrawObjectPicking guards on
        // drawCall.drawCallID != 0 and stores nothing otherwise. A host that never attaches this struct
        // therefore submits every instance as zero, no metadata is kept for any of them, and the readback
        // has nothing to resolve -- clicking the scene silently does nothing, with the thumbnail grid
        // still working because that path keys on the texture hash directly.
        //
        // Values must be distinct per draw within a frame; the runtime warns once and drops the collision
        // otherwise. They need no stability across frames, because a request is answered from the frame
        // it was made against.
        remixapi_InstanceInfoObjectPickingEXT picking = {};
        if (objectPickingValue != 0)
        {
            picking.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_OBJECT_PICKING_EXT;
            picking.objectPickingValue = objectPickingValue;
            picking.pNext = instance.pNext;
            instance.pNext = &picking;
        }

        return mImpl->mApi.DrawInstance(&instance) == REMIXAPI_ERROR_CODE_SUCCESS;
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

    bool Runtime::readOutputPixels(
        std::vector<unsigned char>& out, unsigned int& outWidth, unsigned int& outHeight)
    {
        outWidth = 0;
        outHeight = 0;
        if (!mImpl->mHaveOutput || mImpl->mDevice == nullptr)
            return false;

        const UINT width = mImpl->mOutputImage.mWidth;
        const UINT height = mImpl->mOutputImage.mHeight;
        if (width == 0 || height == 0)
            return false;

        // Two staging surfaces, written one frame and read the next.
        //
        // This is the whole point of the rewrite, so it is worth stating what the single-surface version
        // cost. A render target cannot be mapped, so it has to be copied into a system-memory surface
        // first; GetRenderTargetData queues that copy and returns in under a microsecond, and LockRect
        // then blocks until the GPU has finished both the rendering being read *and* the transfer. Doing
        // both in one frame therefore makes the CPU wait for the GPU to go completely idle, every frame,
        // before it may touch the pixels. Measured at 4K that wait was 32 ms standing still and 70 ms
        // moving, against a 3 ms copy -- seventy to eighty percent of the whole frame spent waiting, with
        // the CPU and GPU taking turns instead of overlapping.
        //
        // Alternating two surfaces breaks the dependency: the copy queued this frame is not read until the
        // next one, by which time the GPU has had a full frame to complete it and the lock should find the
        // data already there. The cost is one extra frame of display latency, which is the standard price
        // for a pipelined readback and cheap next to a full sync.
        for (auto*& surface : mImpl->mReadbackSurfaces)
        {
            if (surface != nullptr)
                continue;
            const HRESULT createHr = mImpl->mDevice->CreateOffscreenPlainSurface(
                width, height, D3DFMT_A16B16G16R16F, D3DPOOL_SYSTEMMEM, &surface, nullptr);
            if (FAILED(createHr) || surface == nullptr)
            {
                Log(Debug::Error) << "Remix readback: CreateOffscreenPlainSurface failed, hr 0x"
                                  << std::hex << createHr << std::dec;
                return false;
            }
        }

        const unsigned int writeIndex = mImpl->mReadbackWriteIndex;
        const unsigned int readIndex = 1u - writeIndex;
        mImpl->mReadbackWriteIndex = readIndex;

        // Either of these can fail on every frame if something is wrong, so report once, not per frame.
        static bool loggedFailure = false;
        mImpl->mReadbackLockNanoseconds = 0;
        mImpl->mReadbackCopyNanoseconds = 0;

        const auto queueStart = std::chrono::steady_clock::now();
        HRESULT hr = mImpl->mDevice->GetRenderTargetData(
            mImpl->mOutputSurface, mImpl->mReadbackSurfaces[writeIndex]);
        mImpl->mReadbackQueueNanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - queueStart)
                                              .count();

        if (FAILED(hr))
        {
            if (!loggedFailure)
            {
                loggedFailure = true;
                Log(Debug::Error) << "Remix readback: GetRenderTargetData failed, hr 0x" << std::hex << hr
                                  << std::dec << " (reported once)";
            }
            return false;
        }
        mImpl->mReadbackFilled[writeIndex] = true;

        // Nothing to read on the very first frame, which is not a failure -- the composite already knows
        // to leave OpenMW's own frame alone when it is handed nothing.
        if (!mImpl->mReadbackFilled[readIndex])
            return false;

        const auto lockStart = std::chrono::steady_clock::now();
        D3DLOCKED_RECT locked = {};
        hr = mImpl->mReadbackSurfaces[readIndex]->LockRect(&locked, nullptr, D3DLOCK_READONLY);
        mImpl->mReadbackLockNanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - lockStart)
                                             .count();
        if (FAILED(hr))
        {
            if (!loggedFailure)
            {
                loggedFailure = true;
                Log(Debug::Error) << "Remix readback: LockRect failed, hr 0x" << std::hex << hr << std::dec
                                  << " (reported once)";
            }
            return false;
        }

        const auto copyStart = std::chrono::steady_clock::now();
        // Row by row: the locked pitch is not necessarily width * bytes per pixel.
        const size_t rowBytes = static_cast<size_t>(width) * kOutputBytesPerPixel;
        out.resize(rowBytes * height);
        const auto* src = static_cast<const unsigned char*>(locked.pBits);
        for (UINT y = 0; y < height; ++y)
        {
            std::memcpy(out.data() + rowBytes * y, src + static_cast<size_t>(locked.Pitch) * y, rowBytes);
        }
        mImpl->mReadbackSurfaces[readIndex]->UnlockRect();
        mImpl->mReadbackCopyNanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - copyStart)
                                             .count();

        outWidth = width;
        outHeight = height;
        return true;
    }

    void Runtime::lastReadbackSplit(unsigned long long& queueNanoseconds,
        unsigned long long& lockNanoseconds, unsigned long long& copyNanoseconds) const
    {
        queueNanoseconds = mImpl->mReadbackQueueNanoseconds;
        lockNanoseconds = mImpl->mReadbackLockNanoseconds;
        copyNanoseconds = mImpl->mReadbackCopyNanoseconds;
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
            Log(Debug::Error) << "Remix probe: CreateOffscreenPlainSurface failed, hr 0x" << std::hex << hr
                              << std::dec;
            return false;
        }

        hr = mImpl->mDevice->GetRenderTargetData(mImpl->mOutputSurface, staging);
        if (FAILED(hr))
        {
            Log(Debug::Error) << "Remix probe: GetRenderTargetData failed, hr 0x" << std::hex << hr
                              << std::dec;
            staging->Release();
            return false;
        }

        D3DLOCKED_RECT locked = {};
        hr = staging->LockRect(&locked, nullptr, D3DLOCK_READONLY);
        if (FAILED(hr))
        {
            Log(Debug::Error) << "Remix probe: LockRect failed, hr 0x" << std::hex << hr << std::dec;
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
            // See createFlatMaterial: zero is AlphaTestType::kNever, which makes the surface reject
            // every hit. The test scene got away with it because a single quad against a sky reads as
            // "dim" rather than "missing", which is precisely the sort of near-miss that makes a test
            // scene worse than useless.
            opaque.alphaTestType = 7; // AlphaTestType::kAlways

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
        for (auto*& surface : mImpl->mReadbackSurfaces)
        {
            if (surface != nullptr)
            {
                surface->Release();
                surface = nullptr;
            }
        }
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
            // Put our own window procedure back before destroying the window.
            //
            // DXVK subclasses the window it presents to, replacing GWLP_WNDPROC with
            // dxvk::D3D9WindowProc, and that hook reads per-window state which Shutdown() has already
            // freed by the time we get here. DestroyWindow then sends WM_DESTROY and WM_NCDESTROY
            // straight into the dead hook, which dereferences a stale pointer -- a crash on every clean
            // exit, in D3D9WindowProc beneath NtUserDestroyWindow, with the whole teardown chain from
            // ~Engine on the stack.
            //
            // This is a real restore rather than a workaround: presentWindowProc is the procedure this
            // window's class was registered with, so putting it back leaves the window exactly as we
            // created it. Doing it the other way round -- destroying the window before Shutdown -- would
            // instead hand the runtime a dead HWND to tear its swapchain down against.
            SetWindowLongPtrW(
                mImpl->mPresentWindow, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&presentWindowProc));
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

    bool Runtime::getGameValueFloat(const char* key, float& out) const
    {
        if (!mImpl->mStarted || mImpl->mApi.GetGameValue == nullptr)
            return false;

        char buffer[64] = {};
        std::uint32_t actualSize = 0;
        if (mImpl->mApi.GetGameValue(key, buffer, static_cast<std::uint32_t>(sizeof(buffer)), &actualSize)
            != REMIXAPI_ERROR_CODE_SUCCESS)
            return false;

        // A missing key is reported as success with a zero size, and an oversized value leaves the
        // buffer untouched rather than truncating it -- reading either would be reading uninitialised
        // memory or a stale value, so both count as "no answer".
        if (actualSize == 0 || actualSize > sizeof(buffer))
            return false;

        char* end = nullptr;
        const float parsed = std::strtof(buffer, &end);
        if (end == buffer || !std::isfinite(parsed))
            return false;

        out = parsed;
        return true;
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

    bool Runtime::setupCameraParameterized(
        const float*, const float*, const float*, const float*, float, float, float, float)
    {
        return false;
    }

    unsigned long long Runtime::createFlatMaterial(unsigned long long, float, float, float, float, float)
    {
        return 0;
    }

    unsigned long long Runtime::createMesh(
        unsigned long long, const Vertex*, unsigned int, const unsigned int*, unsigned int, unsigned long long)
    {
        return 0;
    }

    void Runtime::destroyMesh(unsigned long long) {}

    bool Runtime::drawInstance(unsigned long long, const float*, unsigned int, bool)
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

    bool Runtime::readOutputPixels(std::vector<unsigned char>&, unsigned int&, unsigned int&)
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

    bool Runtime::getGameValueFloat(const char*, float&) const
    {
        return false;
    }

    const std::string& Runtime::loadedFrom() const
    {
        return mImpl->mLoadedFrom;
    }
}

#endif
