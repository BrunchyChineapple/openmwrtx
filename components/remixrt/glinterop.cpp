#include "glinterop.hpp"

#include <components/debug/debuglog.hpp>

#include <osg/GLExtensions>
#include <osg/State>

namespace
{
    // From GL_EXT_memory_object and GL_EXT_memory_object_win32. Declared locally because the GL headers
    // OSG pulls in do not reliably define them.
    constexpr GLenum kTextureTilingExt = 0x9580;
    constexpr GLenum kOptimalTilingExt = 0x9584;
    constexpr GLenum kLinearTilingExt = 0x9585;
    constexpr GLenum kHandleTypeOpaqueWin32Ext = 0x9586;
    constexpr GLenum kHandleTypeOpaqueWin32KmtExt = 0x9587;

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
    using PFN_glTexStorageMem2DEXT = void(GL_APIENTRY*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLuint, GLuint64);

    struct Entrypoints
    {
        PFN_glCreateMemoryObjectsEXT createMemoryObjects = nullptr;
        PFN_glDeleteMemoryObjectsEXT deleteMemoryObjects = nullptr;
        PFN_glImportMemoryWin32HandleEXT importMemoryWin32Handle = nullptr;
        PFN_glTexStorageMem2DEXT texStorageMem2D = nullptr;

        bool complete() const
        {
            return createMemoryObjects && deleteMemoryObjects && importMemoryWin32Handle && texStorageMem2D;
        }
    };

    Entrypoints resolveEntrypoints()
    {
        Entrypoints fns;
        osg::setGLExtensionFuncPtr(fns.createMemoryObjects, "glCreateMemoryObjectsEXT");
        osg::setGLExtensionFuncPtr(fns.deleteMemoryObjects, "glDeleteMemoryObjectsEXT");
        osg::setGLExtensionFuncPtr(fns.importMemoryWin32Handle, "glImportMemoryWin32HandleEXT");
        osg::setGLExtensionFuncPtr(fns.texStorageMem2D, "glTexStorageMem2DEXT");
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
}

namespace RemixRT
{
    ImportOperation::ImportOperation(const Runtime::ExternalImage& image)
        : osg::GraphicsOperation("RemixImportOperation", false)
        , mImage(image)
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
        mSucceeded = true;

        Log(Debug::Info) << "Remix GL interop: imported " << mImage.mWidth << "x" << mImage.mHeight
                         << " VkFormat " << mImage.mFormat << " (" << mImage.mMemorySize
                         << " bytes) as GL texture " << texture << "; "
                         << (mImage.mOptimalTiling ? "optimal" : "linear") << " tiling"
                         << (needsSwizzle ? ", BGRA source needs a red/blue swap when sampling" : "")
                         << (isSrgb ? ", sRGB" : "");
    }
}
