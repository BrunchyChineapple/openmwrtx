#include "glinterop.hpp"

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
        mNeedsChannelSwap = needsSwizzle;
        mSucceeded = true;

        Log(Debug::Info) << "Remix GL interop: imported " << mImage.mWidth << "x" << mImage.mHeight
                         << " VkFormat " << mImage.mFormat << " (" << mImage.mMemorySize
                         << " bytes) as GL texture " << texture << "; "
                         << (mImage.mOptimalTiling ? "optimal" : "linear") << " tiling"
                         << (needsSwizzle ? ", BGRA source needs a red/blue swap when sampling" : "")
                         << (isSrgb ? ", sRGB" : "");
    }
}

namespace
{
    // Minimal pass-through. Positions come from gl_VertexID rather than a vertex buffer so there is no
    // attribute state to set up or restore.
    const char* const kVertexShader = R"(#version 330 core
out vec2 vUv;
void main()
{
    // Two triangles covering the viewport, expressed as a 4-vertex strip.
    vec2 corner = vec2((gl_VertexID & 1) == 0 ? -1.0 : 1.0, (gl_VertexID & 2) == 0 ? -1.0 : 1.0);
    vUv = corner * 0.5 + 0.5;
    gl_Position = vec4(corner, 0.0, 1.0);
}
)";

    // The imported texture is GL_RGBA8 over a Vulkan B8G8R8A8 allocation, so red and blue arrive
    // swapped. GL has no BGRA internal format, so the swap is corrected here rather than at import.
    const char* const kFragmentShaderSwizzle = R"(#version 330 core
uniform sampler2D uImage;
in vec2 vUv;
out vec4 fColour;
void main()
{
    fColour = vec4(texture(uImage, vUv).bgr, 1.0);
}
)";

    const char* const kFragmentShaderDirect = R"(#version 330 core
uniform sampler2D uImage;
in vec2 vUv;
out vec4 fColour;
void main()
{
    fColour = vec4(texture(uImage, vUv).rgb, 1.0);
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
    };

    CompositeCallback::CompositeCallback(const ImportOperation* import)
        : mImport(import)
    {
    }

    void CompositeCallback::operator()(osg::RenderInfo& renderInfo) const
    {
        if (!mEnabled || mFailed || mImport == nullptr || !mImport->succeeded())
            return;

        const GLuint texture = static_cast<GLuint>(mImport->textureName());
        if (texture == 0)
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

            // Whether the swizzle is needed is a property of the imported image, so pick the shader
            // from what the runtime reported rather than hardcoding a channel order.
            const bool swizzle = mImport->needsChannelSwap();
            const GLuint vs = compileShader(ext, GL_VERTEX_SHADER, kVertexShader);
            const GLuint fs = compileShader(
                ext, GL_FRAGMENT_SHADER, swizzle ? kFragmentShaderSwizzle : kFragmentShaderDirect);
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
            // A VAO is required in core profile even with no vertex attributes.
            ext->glGenVertexArrays(1, &resources->vao);

            mResources = std::move(resources);
            Log(Debug::Info) << "Remix composite: ready" << (swizzle ? " (BGRA swizzle)" : "");
        }

        const osg::Viewport* viewport = renderInfo.getCurrentCamera()
            ? renderInfo.getCurrentCamera()->getViewport()
            : nullptr;

        // No glPushAttrib/glPopAttrib: they do not exist in a core profile, and this is the final draw
        // callback so nothing else draws afterwards this frame. Telling OSG its cached mode state is
        // stale is enough -- it re-applies what it needs at the start of the next frame.
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

        ext->glUseProgram(mResources->program);
        state->setActiveTextureUnit(0);
        glBindTexture(GL_TEXTURE_2D, texture);
        if (mResources->imageLocation >= 0)
            ext->glUniform1i(mResources->imageLocation, 0);

        ext->glBindVertexArray(mResources->vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        ext->glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);
        ext->glUseProgram(0);
        glDepthMask(GL_TRUE);

        state->dirtyAllModes();
        state->dirtyAllAttributes();
    }
}
