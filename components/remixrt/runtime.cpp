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

    /// Creates the offscreen window Remix presents into.
    ///
    /// Remix only produces its raytracing output as part of presenting, so a frame has to be driven.
    /// Presenting onto OpenMW's window would mean two presenters -- a Vulkan swapchain and OpenMW's
    /// OpenGL context -- fighting over the same surface. Instead Remix gets a window of its own that is
    /// never shown, and we take the finished image via dxvk_CopyRenderingOutput and composite it into
    /// OpenMW's frame ourselves.
    ///
    /// It is sized to the intended render resolution rather than 1x1, because the swapchain dimensions
    /// determine what resolution Remix renders at.
    HWND createHiddenPresentWindow(uint32_t width, uint32_t height)
    {
        static bool classRegistered = false;
        if (!classRegistered)
        {
            WNDCLASSEXW wc = {};
            wc.cbSize = sizeof(wc);
            wc.lpfnWndProc = DefWindowProcW;
            wc.hInstance = GetModuleHandleW(nullptr);
            wc.lpszClassName = kHiddenWindowClass;
            if (RegisterClassExW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
            {
                Log(Debug::Error) << "Remix: could not register the present-sink window class, error "
                                  << GetLastError();
                return nullptr;
            }
            classRegistered = true;
        }

        // Never shown, so there is no ShowWindow call and the compositor never has to present it.
        // WS_EX_NOREDIRECTIONBITMAP would additionally suppress the redirection surface, but it needs
        // _WIN32_WINNT >= 0x0602 and OpenMW targets lower; not worth raising the whole project's
        // Windows version for a window that stays hidden.
        //
        // WS_POPUP rather than WS_OVERLAPPED, and this is not cosmetic: CreateWindowExW sizes the whole
        // window including borders and caption, so WS_OVERLAPPED at 3840x2160 yielded a 3824x2121 client
        // area and Remix sized its swapchain -- and therefore its render resolution -- to that instead.
        // WS_POPUP has no non-client area, so client size equals the size requested.
        const HWND hwnd = CreateWindowExW(0, kHiddenWindowClass, L"OpenMW Remix present sink", WS_POPUP,
            0, 0, static_cast<int>(width), static_cast<int>(height), nullptr, nullptr,
            GetModuleHandleW(nullptr), nullptr);
        if (hwnd == nullptr)
            Log(Debug::Error) << "Remix: could not create the present-sink window, error " << GetLastError();
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

        mImpl->mPresentWindow = createHiddenPresentWindow(width, height);
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
                         << REMIXAPI_VERSION_PATCH << ", " << width << "x" << height
                         << ", presenting to an offscreen window)";

        return createOutputTarget(width, height);
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
