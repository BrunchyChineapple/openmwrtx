#include "glinterop.hpp"

#include <chrono>
#include <cstdlib>
#include <string>

#include <components/debug/debuglog.hpp>

#include <osg/GLExtensions>
#include <osg/RenderInfo>
#include <osg/State>
#include <osg/Viewport>

namespace
{
    // From GL_EXT_memory_object and GL_EXT_memory_object_win32. Declared locally because the GL headers
    // OSG pulls in do not reliably define them.
    constexpr GLenum kTextureTilingExt = 0x9580;
    constexpr GLenum kDedicatedMemoryObjectExt = 0x9581;
    constexpr GLenum kOptimalTilingExt = 0x9584;
    constexpr GLenum kLinearTilingExt = 0x9585;
    // Values checked against the Khronos registry text for EXT_external_objects_win32, not inferred.
    // An earlier revision of this file had both of these one lower -- 0x9586 is
    // HANDLE_TYPE_OPAQUE_FD_EXT, from the *fd* extension, and 0x9587 is the NT-handle type -- so a KMT
    // handle was being imported while GL was told it was an NT handle. NVIDIA's driver accepted that
    // without raising a GL error, which is exactly why it went unnoticed.
    constexpr GLenum kHandleTypeOpaqueWin32Ext = 0x9587;
    constexpr GLenum kHandleTypeOpaqueWin32KmtExt = 0x9588;
    // From EXT_external_objects. GL_LAYOUT_GENERAL_EXT is the counterpart of VK_IMAGE_LAYOUT_GENERAL,
    // which is where DXVK leaves shared images: d3d9_common_texture.cpp skips its layout optimisation
    // whenever the image is shared.
    constexpr GLenum kLayoutGeneralExt = 0x958D;
    // Core GL since 1.2, but <GL/gl.h> on Windows stops at 1.1, so it is not necessarily declared.
    // Valid as a transfer format only, which is exactly how the readback path uses it.
    constexpr GLenum kBgra = 0x80E1;

    // VkExternalMemoryHandleTypeFlagBits values we know how to translate.
    constexpr unsigned int kVkHandleTypeOpaqueWin32 = 0x00000002;
    constexpr unsigned int kVkHandleTypeOpaqueWin32Kmt = 0x00000004;

    // VkFormat values Remix can hand us for a colour target.
    constexpr unsigned int kVkFormatR8G8B8A8Unorm = 37;
    constexpr unsigned int kVkFormatR8G8B8A8Srgb = 43;
    constexpr unsigned int kVkFormatB8G8R8A8Unorm = 44;
    constexpr unsigned int kVkFormatB8G8R8A8Srgb = 50;

    using PFN_glCreateMemoryObjectsEXT = void(GL_APIENTRY*)(GLsizei, GLuint*);
    using PFN_glDeleteMemoryObjectsEXT = void(GL_APIENTRY*)(GLsizei, const GLuint*);
    using PFN_glImportMemoryWin32HandleEXT = void(GL_APIENTRY*)(GLuint, GLuint64, GLenum, void*);
    using PFN_glMemoryObjectParameterivEXT = void(GL_APIENTRY*)(GLuint, GLenum, const GLint*);
    using PFN_glTexStorageMem2DEXT = void(GL_APIENTRY*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLuint, GLuint64);

    // Note glImportSemaphoreWin32HandleEXT takes no size, unlike the memory equivalent.
    using PFN_glGenSemaphoresEXT = void(GL_APIENTRY*)(GLsizei, GLuint*);
    using PFN_glDeleteSemaphoresEXT = void(GL_APIENTRY*)(GLsizei, const GLuint*);
    using PFN_glImportSemaphoreWin32HandleEXT = void(GL_APIENTRY*)(GLuint, GLenum, void*);
    // (semaphore, numBufferBarriers, buffers, numTextureBarriers, textures, layouts) -- the layouts
    // argument is srcLayouts on wait and dstLayouts on signal.
    using PFN_glWaitSemaphoreEXT
        = void(GL_APIENTRY*)(GLuint, GLuint, const GLuint*, GLuint, const GLuint*, const GLenum*);
    using PFN_glSignalSemaphoreEXT
        = void(GL_APIENTRY*)(GLuint, GLuint, const GLuint*, GLuint, const GLuint*, const GLenum*);

    struct Entrypoints
    {
        PFN_glCreateMemoryObjectsEXT createMemoryObjects = nullptr;
        PFN_glDeleteMemoryObjectsEXT deleteMemoryObjects = nullptr;
        PFN_glImportMemoryWin32HandleEXT importMemoryWin32Handle = nullptr;
        PFN_glMemoryObjectParameterivEXT memoryObjectParameteriv = nullptr;
        PFN_glTexStorageMem2DEXT texStorageMem2D = nullptr;
        PFN_glGenSemaphoresEXT genSemaphores = nullptr;
        PFN_glDeleteSemaphoresEXT deleteSemaphores = nullptr;
        PFN_glImportSemaphoreWin32HandleEXT importSemaphoreWin32Handle = nullptr;
        PFN_glWaitSemaphoreEXT waitSemaphore = nullptr;
        PFN_glSignalSemaphoreEXT signalSemaphore = nullptr;

        bool complete() const
        {
            return createMemoryObjects && deleteMemoryObjects && importMemoryWin32Handle && texStorageMem2D;
        }

        /// The semaphore half is reported separately: a driver can advertise the memory extensions
        /// without the semaphore ones, and the image import is worth attempting either way even though
        /// sampling it is not safe without the semaphores.
        bool semaphoresComplete() const
        {
            return genSemaphores && deleteSemaphores && importSemaphoreWin32Handle && waitSemaphore
                && signalSemaphore;
        }
    };

    Entrypoints resolveEntrypoints()
    {
        Entrypoints fns;
        osg::setGLExtensionFuncPtr(fns.createMemoryObjects, "glCreateMemoryObjectsEXT");
        osg::setGLExtensionFuncPtr(fns.deleteMemoryObjects, "glDeleteMemoryObjectsEXT");
        osg::setGLExtensionFuncPtr(fns.importMemoryWin32Handle, "glImportMemoryWin32HandleEXT");
        osg::setGLExtensionFuncPtr(fns.memoryObjectParameteriv, "glMemoryObjectParameterivEXT");
        osg::setGLExtensionFuncPtr(fns.texStorageMem2D, "glTexStorageMem2DEXT");
        osg::setGLExtensionFuncPtr(fns.genSemaphores, "glGenSemaphoresEXT");
        osg::setGLExtensionFuncPtr(fns.deleteSemaphores, "glDeleteSemaphoresEXT");
        osg::setGLExtensionFuncPtr(fns.importSemaphoreWin32Handle, "glImportSemaphoreWin32HandleEXT");
        osg::setGLExtensionFuncPtr(fns.waitSemaphore, "glWaitSemaphoreEXT");
        osg::setGLExtensionFuncPtr(fns.signalSemaphore, "glSignalSemaphoreEXT");
        return fns;
    }

    /// Maps the Vulkan format Remix reports to a GL sized internal format.
    ///
    /// The internal format has to describe the same bytes-per-pixel and component layout as the Vulkan
    /// image, because both APIs are reading one allocation. GL has no BGRA *internal* format -- BGRA
    /// exists only as a transfer format -- so a B8G8R8A8 image is imported as GL_RGBA8 and the red/blue
    /// swap is corrected when sampling. Returns 0 for formats we have not accounted for, rather than
    /// guessing.
    GLenum glInternalFormat(unsigned int vkFormat, bool& outNeedsSwizzle, bool& outIsSrgb)
    {
        outNeedsSwizzle = false;
        outIsSrgb = false;
        switch (vkFormat)
        {
            case kVkFormatR8G8B8A8Unorm:
                return GL_RGBA8;
            case kVkFormatR8G8B8A8Srgb:
                outIsSrgb = true;
                return GL_SRGB8_ALPHA8;
            case kVkFormatB8G8R8A8Unorm:
                outNeedsSwizzle = true;
                return GL_RGBA8;
            case kVkFormatB8G8R8A8Srgb:
                outNeedsSwizzle = true;
                outIsSrgb = true;
                return GL_SRGB8_ALPHA8;
            default:
                return 0;
        }
    }

    GLenum glHandleType(unsigned int vkHandleType)
    {
        switch (vkHandleType)
        {
            case kVkHandleTypeOpaqueWin32:
                return kHandleTypeOpaqueWin32Ext;
            case kVkHandleTypeOpaqueWin32Kmt:
                return kHandleTypeOpaqueWin32KmtExt;
            default:
                return 0;
        }
    }

    /// Drains and reports the GL error queue. Returns true if it was empty.
    bool checkGl(const char* what)
    {
        bool clean = true;
        for (GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError())
        {
            clean = false;
            Log(Debug::Error) << "Remix GL interop: " << what << " raised GL error 0x" << std::hex << error
                              << std::dec;
        }
        return clean;
    }

    /// Empties the GL error queue without attributing what it finds to us.
    ///
    /// glGetError returns errors accumulated since it was last called, from anywhere. The composite
    /// runs at the end of OSG's draw traversal, so anything OSG left pending would otherwise be
    /// reported against whichever of our calls happens to be checked first. Drain before measuring, so
    /// that a reported error is genuinely ours.
    void drainGl(const char* whose)
    {
        unsigned int count = 0;
        for (GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError())
        {
            ++count;
            if (count <= 4)
            {
                Log(Debug::Verbose) << "Remix GL interop: discarding a pre-existing GL error 0x"
                                    << std::hex << error << std::dec << " that arrived before " << whose
                                    << "; not ours";
            }
        }
        if (count > 0)
            Log(Debug::Verbose) << "Remix GL interop: drained " << count << " pre-existing GL error(s)";
    }

    /// Imports Remix's synchronisation semaphore pair.
    ///
    /// A free function rather than a member of ImportOperation because Entrypoints is file-local and
    /// so cannot appear in the class declaration.
    ///
    /// Failure is logged but left non-fatal to the image import: having the texture present with the
    /// reason recorded is more useful than unwinding everything. The consumer checks syncAvailable()
    /// and refuses to wait if the semaphores are missing, since waiting on a semaphore that has had no
    /// signal submitted is undefined behaviour in its own right.
    void importSyncSemaphores(unsigned int contextId, const Entrypoints& fns,
        const RemixRT::Runtime::ExternalSync& sync, unsigned int& outWait, unsigned int& outSignal)
    {
        if (!sync.valid())
        {
            Log(Debug::Warning) << "Remix GL interop: the runtime reported no synchronisation "
                                   "semaphores, so sampling the shared image would have no ordering "
                                   "against Remix's writes -- undefined, and in practice black.";
            return;
        }

        const bool haveSemaphore = osg::isGLExtensionSupported(contextId, "GL_EXT_semaphore");
        const bool haveSemaphoreWin32 = osg::isGLExtensionSupported(contextId, "GL_EXT_semaphore_win32");
        if (!haveSemaphore || !haveSemaphoreWin32 || !fns.semaphoresComplete())
        {
            Log(Debug::Error) << "Remix GL interop: driver lacks the semaphore extensions "
                              << "(GL_EXT_semaphore " << (haveSemaphore ? "yes" : "NO")
                              << ", GL_EXT_semaphore_win32 " << (haveSemaphoreWin32 ? "yes" : "NO")
                              << "); the shared image cannot be sampled safely.";
            return;
        }

        GLuint semaphores[2] = { 0, 0 };
        fns.genSemaphores(2, semaphores);
        if (semaphores[0] == 0 || semaphores[1] == 0 || !checkGl("glGenSemaphoresEXT"))
            return;

        // NT handles here, where the image used a KMT handle: Remix builds these through
        // RtxSemaphore::createBinary, which exports
        // VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT. Importing does not transfer ownership,
        // so Remix stays responsible for closing them.
        fns.importSemaphoreWin32Handle(semaphores[0], kHandleTypeOpaqueWin32Ext,
            reinterpret_cast<void*>(static_cast<uintptr_t>(sync.mCopyComplete)));
        fns.importSemaphoreWin32Handle(semaphores[1], kHandleTypeOpaqueWin32Ext,
            reinterpret_cast<void*>(static_cast<uintptr_t>(sync.mConsumerDone)));
        if (!checkGl("glImportSemaphoreWin32HandleEXT"))
        {
            fns.deleteSemaphores(2, semaphores);
            return;
        }

        outWait = semaphores[0];
        outSignal = semaphores[1];

        Log(Debug::Info) << "Remix GL interop: imported sync semaphores as GL " << outWait
                         << " (wait for Remix's copy) and " << outSignal
                         << " (signal when sampling is done)";
    }
}

namespace RemixRT
{
    ImportOperation::ImportOperation(
        const Runtime::ExternalImage& image, const Runtime::ExternalSync& sync)
        : osg::GraphicsOperation("RemixImportOperation", false)
        , mImage(image)
        , mSync(sync)
    {
    }

    void ImportOperation::operator()(osg::GraphicsContext* context)
    {
        mCompleted = true;

        if (context == nullptr || context->getState() == nullptr)
        {
            Log(Debug::Error) << "Remix GL interop: no GL state on the graphics context";
            return;
        }
        if (mImage.mHandle == 0 || mImage.mMemorySize == 0)
        {
            Log(Debug::Error) << "Remix GL interop: nothing to import; the runtime reported no shared image";
            return;
        }

        const unsigned int contextId = context->getState()->getContextID();

        // Report both extensions explicitly. memory_object alone is not enough -- the _win32 half is what
        // provides glImportMemoryWin32HandleEXT, and drivers can ship one without the other.
        const bool haveMemoryObject = osg::isGLExtensionSupported(contextId, "GL_EXT_memory_object");
        const bool haveMemoryObjectWin32 = osg::isGLExtensionSupported(contextId, "GL_EXT_memory_object_win32");
        if (!haveMemoryObject || !haveMemoryObjectWin32)
        {
            Log(Debug::Error) << "Remix GL interop: driver lacks the required extensions "
                              << "(GL_EXT_memory_object " << (haveMemoryObject ? "yes" : "NO")
                              << ", GL_EXT_memory_object_win32 " << (haveMemoryObjectWin32 ? "yes" : "NO")
                              << "). Single-window compositing is not possible on this driver.";
            return;
        }

        const Entrypoints fns = resolveEntrypoints();
        if (!fns.complete())
        {
            Log(Debug::Error) << "Remix GL interop: the extensions are advertised but their entry points "
                                 "could not be resolved";
            return;
        }

        const GLenum handleType = glHandleType(mImage.mHandleType);
        if (handleType == 0)
        {
            Log(Debug::Error) << "Remix GL interop: unsupported external memory handle type 0x" << std::hex
                              << mImage.mHandleType << std::dec;
            return;
        }

        bool needsSwizzle = false;
        bool isSrgb = false;
        const GLenum internalFormat = glInternalFormat(mImage.mFormat, needsSwizzle, isSrgb);
        if (internalFormat == 0)
        {
            Log(Debug::Error) << "Remix GL interop: unhandled VkFormat " << mImage.mFormat
                              << "; add it to glInternalFormat rather than guessing a layout";
            return;
        }

        checkGl("state before import");

        GLuint memoryObject = 0;
        fns.createMemoryObjects(1, &memoryObject);
        if (memoryObject == 0 || !checkGl("glCreateMemoryObjectsEXT"))
            return;

        // Remix's shared images are allocated with VkMemoryDedicatedAllocateInfo -- which is also why
        // the reported memory offset is always 0. The spec requires DEDICATED_MEMORY_OBJECT_EXT to be
        // set before importing such a handle, and the parameter becomes immutable once the import has
        // happened, so it has to be here rather than after. Omitting it was an earlier defect: the
        // import still succeeded without a GL error, which is exactly what made it easy to miss.
        if (fns.memoryObjectParameteriv != nullptr)
        {
            const GLint dedicated = GL_TRUE;
            fns.memoryObjectParameteriv(memoryObject, kDedicatedMemoryObjectExt, &dedicated);
            if (!checkGl("glMemoryObjectParameterivEXT(GL_DEDICATED_MEMORY_OBJECT_EXT)"))
            {
                fns.deleteMemoryObjects(1, &memoryObject);
                return;
            }
        }
        else
        {
            Log(Debug::Warning) << "Remix GL interop: glMemoryObjectParameterivEXT is unavailable, so "
                                   "the dedicated-allocation flag cannot be set; the import may be "
                                   "rejected or silently wrong.";
        }

        // The handle stays owned by Remix, so this must not be the *_KMT variant's ownership-transfer
        // form. glImportMemoryWin32HandleEXT does not take ownership of a KMT handle, which suits us:
        // Remix frees it with the surface.
        fns.importMemoryWin32Handle(
            memoryObject, static_cast<GLuint64>(mImage.mMemorySize), handleType,
            reinterpret_cast<void*>(static_cast<uintptr_t>(mImage.mHandle)));
        if (!checkGl("glImportMemoryWin32HandleEXT"))
        {
            fns.deleteMemoryObjects(1, &memoryObject);
            return;
        }

        GLuint texture = 0;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        // Tiling must match how Vulkan laid the image out. Remix reports it; getting this wrong yields
        // garbled output rather than an error, so it is not something to assume.
        glTexParameteri(GL_TEXTURE_2D, kTextureTilingExt,
            mImage.mOptimalTiling ? static_cast<GLint>(kOptimalTilingExt) : static_cast<GLint>(kLinearTilingExt));
        if (!checkGl("glTexParameteri(GL_TEXTURE_TILING_EXT)"))
        {
            glDeleteTextures(1, &texture);
            fns.deleteMemoryObjects(1, &memoryObject);
            return;
        }

        fns.texStorageMem2D(GL_TEXTURE_2D, 1, internalFormat, static_cast<GLsizei>(mImage.mWidth),
            static_cast<GLsizei>(mImage.mHeight), memoryObject,
            static_cast<GLuint64>(mImage.mMemoryOffset));
        if (!checkGl("glTexStorageMem2DEXT"))
        {
            glDeleteTextures(1, &texture);
            fns.deleteMemoryObjects(1, &memoryObject);
            return;
        }

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        checkGl("sampler state");

        mMemoryObject = memoryObject;
        mTextureName = texture;
        mNeedsChannelSwap = needsSwizzle;
        mSucceeded = true;

        Log(Debug::Info) << "Remix GL interop: imported " << mImage.mWidth << "x" << mImage.mHeight
                         << " VkFormat " << mImage.mFormat << " (" << mImage.mMemorySize
                         << " bytes) as GL texture " << texture << "; "
                         << (mImage.mOptimalTiling ? "optimal" : "linear") << " tiling"
                         << (needsSwizzle ? ", BGRA source needs a red/blue swap when sampling" : "")
                         << (isSrgb ? ", sRGB" : "");

        importSyncSemaphores(contextId, fns, mSync, mWaitSemaphore, mSignalSemaphore);
    }
}

namespace
{
    // Minimal pass-through. Positions come from gl_VertexID rather than a vertex buffer so there is no
    // attribute state to set up or restore.
    const char* const kVertexShader = R"(#version 330 core
uniform vec2 uFlip;
out vec2 vUv;
void main()
{
    // Two triangles covering the viewport, expressed as a 4-vertex strip.
    vec2 corner = vec2((gl_VertexID & 1) == 0 ? -1.0 : 1.0, (gl_VertexID & 2) == 0 ? -1.0 : 1.0);

    // uFlip corrects the orientation of Remix's image. Vertical only, confirmed on screen: both
    // sources put row 0 at the TOP -- a D3D9 surface on the readback path, a Vulkan image on the
    // imported one -- while OpenGL puts v = 0 at the BOTTOM.
    //
    // Worth knowing if this ever looks wrong again: a purely vertical flip reads as "upside down AND
    // mirrored", because reflecting glyphs about the horizontal axis looks like mirror writing. That
    // description prompted a horizontal flip as well, which then showed the image genuinely mirrored.
    // Do not add a horizontal flip on the strength of text looking backwards.
    //
    // Applied to the texture coordinate rather than to gl_Position, so the geometry still covers the
    // viewport the same way and only the sampling orientation changes.
    vUv = corner * uFlip * 0.5 + 0.5;
    gl_Position = vec4(corner, 0.0, 1.0);
}
)";

    // The imported texture is GL_RGBA8 over a Vulkan B8G8R8A8 allocation, so red and blue arrive
    // swapped and have to be corrected when sampling -- GL has no BGRA *internal* format.
    //
    // The readback path does not need that, because it uploads with GL_BGRA as the transfer format and
    // the driver reorders on the way in. Which path is active can change at runtime, so the choice is a
    // uniform rather than two programs: picking a shader once at creation would bake in whichever mode
    // happened to be active on the first frame.
    const char* const kFragmentShader = R"(#version 330 core
uniform sampler2D uImage;
uniform int uSwizzle;
in vec2 vUv;
out vec4 fColour;
void main()
{
    vec3 colour = texture(uImage, vUv).rgb;
    fColour = vec4(uSwizzle != 0 ? colour.bgr : colour, 1.0);
}
)";

    // Everything past GL 1.1 has to be reached through function pointers on Windows, because
    // <GL/gl.h> from the platform SDK stops there. osg::GLExtensions already resolves the ones we
    // need, so use it rather than resolving a second copy.
    GLuint compileShader(osg::GLExtensions* ext, GLenum stage, const char* source)
    {
        const GLuint shader = ext->glCreateShader(stage);
        ext->glShaderSource(shader, 1, &source, nullptr);
        ext->glCompileShader(shader);

        GLint compiled = GL_FALSE;
        ext->glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (compiled == GL_FALSE)
        {
            char log[1024] = {};
            GLsizei written = 0;
            ext->glGetShaderInfoLog(shader, sizeof(log) - 1, &written, log);
            Log(Debug::Error) << "Remix composite: shader compilation failed: " << log;
            ext->glDeleteShader(shader);
            return 0;
        }
        return shader;
    }
}

namespace RemixRT
{
    struct CompositeCallback::Resources
    {
        GLuint program = 0;
        GLuint vao = 0;
        GLint imageLocation = -1;
        GLint swizzleLocation = -1;
        GLint flipLocation = -1;
        Entrypoints fns;
        // Only used by the readback path: an ordinary texture we own and upload into, as opposed to the
        // imported one that aliases Remix's memory.
        GLuint readbackTexture = 0;
        unsigned int readbackWidth = 0;
        unsigned int readbackHeight = 0;
    };

    CompositeCallback::CompositeCallback(const ImportOperation* import)
        : mImport(import)
    {
        if (const char* value = std::getenv("OPENMW_REMIX_SYNC_NO_TEXBARRIER");
            value != nullptr && *value != '\0' && *value != '0')
        {
            mSkipTextureBarrier = true;
        }
        if (const char* value = std::getenv("OPENMW_REMIX_READBACK");
            value != nullptr && *value != '\0' && *value != '0')
        {
            mForceReadback = true;
            Log(Debug::Info) << "Remix composite: readback path forced by OPENMW_REMIX_READBACK";
        }

        // Orientation. The default is the confirmed-correct vertical flip; the override exists so a
        // different source convention costs a relaunch rather than a rebuild.
        if (const char* value = std::getenv("OPENMW_REMIX_FLIP"); value != nullptr && *value != '\0')
        {
            const std::string mode(value);
            if (mode == "none")
            {
                mFlipHorizontal = false;
                mFlipVertical = false;
            }
            else if (mode == "v")
            {
                mFlipHorizontal = false;
                mFlipVertical = true;
            }
            else if (mode == "h")
            {
                mFlipHorizontal = true;
                mFlipVertical = false;
            }
            else if (mode == "both")
            {
                mFlipHorizontal = true;
                mFlipVertical = true;
            }
            else
            {
                Log(Debug::Warning) << "Remix composite: ignoring OPENMW_REMIX_FLIP='" << mode
                                    << "'; expected none, v, h or both";
            }
        }

        Log(Debug::Info) << "Remix composite: orientation flip h=" << (mFlipHorizontal ? "yes" : "no")
                         << " v=" << (mFlipVertical ? "yes" : "no")
                         << " (override with OPENMW_REMIX_FLIP=none|v|h|both)";
    }

    void CompositeCallback::takeReadbackFrame(
        std::vector<unsigned char>& pixels, unsigned int width, unsigned int height)
    {
        if (width == 0 || height == 0)
            return;
        if (pixels.size() < static_cast<size_t>(width) * height * 4)
            return;

        const std::lock_guard<std::mutex> lock(mReadbackMutex);
        mReadbackPixels.swap(pixels);
        mReadbackWidth = width;
        mReadbackHeight = height;
        mReadbackFresh = true;
    }

    unsigned int CompositeCallback::uploadReadbackTexture(osg::GLExtensions*) const
    {
        unsigned int width = 0;
        unsigned int height = 0;
        bool fresh = false;
        {
            // Copied out under the lock rather than uploaded under it: the engine writes this from the
            // main thread while the draw thread is here, and holding the lock across a multi-megabyte
            // GL upload would serialise the two threads for no benefit.
            const std::lock_guard<std::mutex> lock(mReadbackMutex);
            fresh = mReadbackFresh;
            width = mReadbackWidth;
            height = mReadbackHeight;
            if (fresh)
            {
                // Swapped, not copied: the upload buffer's old contents go back to be refilled.
                mReadbackUpload.swap(mReadbackPixels);
                mReadbackFresh = false;
            }
        }

        if (mResources->readbackTexture == 0 || mResources->readbackWidth != width
            || mResources->readbackHeight != height)
        {
            if (width == 0 || height == 0 || mReadbackUpload.empty())
                return 0;

            if (mResources->readbackTexture == 0)
                glGenTextures(1, &mResources->readbackTexture);
            glBindTexture(GL_TEXTURE_2D, mResources->readbackTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(width),
                static_cast<GLsizei>(height), 0, kBgra, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            mResources->readbackWidth = width;
            mResources->readbackHeight = height;
            fresh = true;
            Log(Debug::Info) << "Remix composite: readback path active, " << width << "x" << height
                             << " uploaded per frame. This is a CPU round trip and costs real frame "
                             << "time; it exists so the image is visible without the semaphore "
                                "handshake.";
        }

        if (mResources->readbackTexture == 0)
            return 0;

        if (fresh && !mReadbackUpload.empty())
        {
            // Timed because this is the half of the readback cost nothing was accounting for. The
            // GPU-to-CPU side is on the frame loop where it is at least visible; this one is on the draw
            // thread, so it does not show up in the frame loop's own timings at all and can only be seen
            // by measuring it here.
            const auto start = std::chrono::steady_clock::now();
            glBindTexture(GL_TEXTURE_2D, mResources->readbackTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(mResources->readbackWidth),
                static_cast<GLsizei>(mResources->readbackHeight), kBgra, GL_UNSIGNED_BYTE,
                mReadbackUpload.data());
            checkGl("glTexSubImage2D(readback)");
            const auto end = std::chrono::steady_clock::now();

            // Note this measures the driver call returning, not the transfer completing: glTexSubImage2D
            // from client memory may copy into a staging buffer and return, leaving the upload to happen
            // later. So this is a lower bound on the cost, and a large value here is conclusive while a
            // small one is not.
            mUploadNanoseconds.fetch_add(
                static_cast<unsigned long long>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
                std::memory_order_relaxed);
            mUploadCount.fetch_add(1, std::memory_order_relaxed);
        }

        return mResources->readbackTexture;
    }

    void CompositeCallback::operator()(osg::RenderInfo& renderInfo) const
    {
        if (!mEnabled || mFailed || mImport == nullptr)
            return;

        // The readback path does not touch the imported image at all, so it stays available even when
        // the import failed outright -- which is the situation it exists for.
        const bool readback = readbackMode();
        if (!readback && !mImport->succeeded())
            return;

        const GLuint importedTexture = static_cast<GLuint>(mImport->textureName());
        if (!readback && importedTexture == 0)
            return;

        osg::State* state = renderInfo.getState();
        if (state == nullptr)
            return;
        osg::GLExtensions* ext = state->get<osg::GLExtensions>();
        if (ext == nullptr)
        {
            Log(Debug::Error) << "Remix composite: no GL extension table on the state";
            mFailed = true;
            return;
        }

        if (mResources == nullptr)
        {
            auto resources = std::make_unique<Resources>();

            const GLuint vs = compileShader(ext, GL_VERTEX_SHADER, kVertexShader);
            const GLuint fs = compileShader(ext, GL_FRAGMENT_SHADER, kFragmentShader);
            if (vs == 0 || fs == 0)
            {
                mFailed = true;
                return;
            }

            resources->program = ext->glCreateProgram();
            ext->glAttachShader(resources->program, vs);
            ext->glAttachShader(resources->program, fs);
            ext->glLinkProgram(resources->program);
            ext->glDeleteShader(vs);
            ext->glDeleteShader(fs);

            GLint linked = GL_FALSE;
            ext->glGetProgramiv(resources->program, GL_LINK_STATUS, &linked);
            if (linked == GL_FALSE)
            {
                char log[1024] = {};
                GLsizei written = 0;
                ext->glGetProgramInfoLog(resources->program, sizeof(log) - 1, &written, log);
                Log(Debug::Error) << "Remix composite: program link failed: " << log;
                ext->glDeleteProgram(resources->program);
                mFailed = true;
                return;
            }

            resources->imageLocation = ext->glGetUniformLocation(resources->program, "uImage");
            resources->swizzleLocation = ext->glGetUniformLocation(resources->program, "uSwizzle");
            resources->flipLocation = ext->glGetUniformLocation(resources->program, "uFlip");
            // A VAO is required in core profile even with no vertex attributes.
            ext->glGenVertexArrays(1, &resources->vao);
            resources->fns = resolveEntrypoints();

            mResources = std::move(resources);
            Log(Debug::Info) << "Remix composite: ready"
                             << (mImport->needsChannelSwap() ? " (BGRA source)" : "")
                             << (mImport->syncAvailable()
                                        ? ", synchronised against Remix"
                                        : ", WITHOUT synchronisation -- reads of the shared image are "
                                          "undefined; the readback path is the way to get a picture");
        }

        const osg::Viewport* viewport = renderInfo.getCurrentCamera()
            ? renderInfo.getCurrentCamera()->getViewport()
            : nullptr;

        // No glPushAttrib/glPopAttrib: they do not exist in a core profile. OpenMW's GUI draws after
        // this, so the state OSG believes is current has to be invalidated before returning -- see the
        // dirtyAll* calls at the end, which are load-bearing rather than tidiness.
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDepthMask(GL_FALSE);
        if (viewport != nullptr)
        {
            glViewport(static_cast<GLint>(viewport->x()), static_cast<GLint>(viewport->y()),
                static_cast<GLsizei>(viewport->width()), static_cast<GLsizei>(viewport->height()));
        }

        // Only order against Remix when a synchronised copy was actually issued for this frame.
        // Waiting on a semaphore with no signal submitted since the last wait is undefined behaviour
        // by the spec, not merely ineffective, so this must never be assumed.
        // The readback path carries its own pixels, so it neither needs nor may use the handshake.
        const bool sync = !readback && mImport->syncAvailable()
            && mSyncArmed.load(std::memory_order_relaxed) && !mSyncFailed.load(std::memory_order_relaxed);

        if (sync)
        {
            // Clear anything OSG left behind first, so the check after the wait measures only the wait.
            drainGl("the Remix semaphore wait");

            // The texture is named in the barrier so the driver makes Remix's write visible to it and
            // initialises its internal layout tracking. GL_LAYOUT_GENERAL_EXT is correct because DXVK
            // leaves shared images in VK_IMAGE_LAYOUT_GENERAL -- it skips its layout optimisation for
            // anything shared. A wrong layout here corrupts contents rather than raising an error.
            const GLuint barrierTexture = importedTexture;
            const GLenum barrierLayout = kLayoutGeneralExt;

            // Diagnostic bisect for GL_INVALID_OPERATION out of the wait: this call carries both a
            // semaphore and a texture barrier, and either can be at fault. Dropping the barrier
            // separates "the semaphore is unacceptable" from "the texture barrier is unacceptable".
            // Correctness needs the barrier, so this is for isolating a fault, not for shipping.
            const bool skipBarrier = mSkipTextureBarrier;
            mResources->fns.waitSemaphore(static_cast<GLuint>(mImport->waitSemaphore()), 0, nullptr,
                skipBarrier ? 0u : 1u, skipBarrier ? nullptr : &barrierTexture,
                skipBarrier ? nullptr : &barrierLayout);
            if (skipBarrier)
                Log(Debug::Warning) << "Remix composite: waited with NO texture barrier "
                                       "(OPENMW_REMIX_SYNC_NO_TEXBARRIER) -- diagnostic only, the "
                                       "sampled image is not guaranteed visible or correctly laid out";
            if (!checkGl("glWaitSemaphoreEXT"))
            {
                // Do not signal back after a failed wait, and do not try again on later frames: see
                // syncFailed(). The engine falls back to the unsynchronised copy from here on.
                mSyncFailed.store(true, std::memory_order_relaxed);
                Log(Debug::Error) << "Remix composite: the semaphore wait failed, so synchronisation is "
                                     "disabled from here on and the composited image is undefined. This "
                                     "is the black-frame case, not a cosmetic warning.";
            }
        }

        // Pick the source. The imported texture aliases Remix's memory directly; the readback texture is
        // ours, filled from a CPU copy the engine handed over this frame.
        GLuint texture = importedTexture;
        bool swizzle = mImport->needsChannelSwap();
        if (readback)
        {
            texture = uploadReadbackTexture(ext);
            // Uploaded with GL_BGRA as the transfer format, so the driver has already put the channels
            // in order and no shader-side swap is wanted.
            swizzle = false;
            if (texture == 0)
                return;
        }

        ext->glUseProgram(mResources->program);
        state->setActiveTextureUnit(0);
        glBindTexture(GL_TEXTURE_2D, texture);
        if (mResources->imageLocation >= 0)
            ext->glUniform1i(mResources->imageLocation, 0);
        if (mResources->swizzleLocation >= 0)
            ext->glUniform1i(mResources->swizzleLocation, swizzle ? 1 : 0);
        if (mResources->flipLocation >= 0)
            ext->glUniform2f(mResources->flipLocation, mFlipHorizontal ? -1.0f : 1.0f,
                mFlipVertical ? -1.0f : 1.0f);

        ext->glBindVertexArray(mResources->vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        ext->glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);
        ext->glUseProgram(0);
        glDepthMask(GL_TRUE);

        // Re-read the failure flag: the wait above may have set it, in which case the matching signal
        // must be skipped too.
        if (sync && !mSyncFailed.load(std::memory_order_relaxed))
        {
            const bool skipBarrier = mSkipTextureBarrier;
            // Hand the image back in the layout Remix expects to find it in, then record that the
            // signal happened. The engine reads that flag and only then lets Remix wait, which is what
            // keeps this binary pair balanced -- an unmatched wait would block Remix's render thread.
            const GLuint barrierTexture = texture;
            const GLenum barrierLayout = kLayoutGeneralExt;
            mResources->fns.signalSemaphore(static_cast<GLuint>(mImport->signalSemaphore()), 0, nullptr,
                skipBarrier ? 0u : 1u, skipBarrier ? nullptr : &barrierTexture,
                skipBarrier ? nullptr : &barrierLayout);
            if (checkGl("glSignalSemaphoreEXT"))
                mSignalled.store(true, std::memory_order_release);
            else
                mSyncFailed.store(true, std::memory_order_relaxed);
        }

        // Everything OSG caches about the GL context has to be marked stale, because OpenMW's GUI is
        // drawn after this and would otherwise inherit modes, attributes and array bindings that OSG
        // still believes it set. Vertex arrays are included for the VAO bound above: unbinding it is not
        // enough on its own, since OSG tracks its own idea of what is bound.
        state->dirtyAllModes();
        state->dirtyAllAttributes();
        state->dirtyAllVertexArrays();
    }

    CompositeDrawable::CompositeDrawable(CompositeCallback* composite)
        : mComposite(composite)
    {
        // A fullscreen overlay has no meaningful bounds, and letting OSG cull it against the view
        // frustum would drop it. Display lists and VBO management are OSG's, and this draws through raw
        // GL, so both are off.
        setCullingActive(false);
        setUseDisplayList(false);
        setUseVertexBufferObjects(false);
    }

    void CompositeDrawable::drawImplementation(osg::RenderInfo& renderInfo) const
    {
        if (mComposite != nullptr)
            (*mComposite)(renderInfo);
    }
}
