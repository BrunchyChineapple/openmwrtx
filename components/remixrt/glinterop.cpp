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
/// Spelled out for the same reason as kBgra: the GL headers reachable from here are the ones OSG exposes,
/// which predate these tokens. GL_RGBA16F and GL_HALF_FLOAT from the OpenGL 3.0 core additions.
constexpr GLenum kRgba16f = 0x881A;
constexpr GLenum kHalfFloat = 0x140B;

    // VkExternalMemoryHandleTypeFlagBits values we know how to translate.
    constexpr unsigned int kVkHandleTypeOpaqueWin32 = 0x00000002;
    constexpr unsigned int kVkHandleTypeOpaqueWin32Kmt = 0x00000004;

    // VkFormat values Remix can hand us for a colour target.
    constexpr unsigned int kVkFormatR8G8B8A8Unorm = 37;
    constexpr unsigned int kVkFormatR8G8B8A8Srgb = 43;
    constexpr unsigned int kVkFormatB8G8R8A8Unorm = 44;
    constexpr unsigned int kVkFormatB8G8R8A8Srgb = 50;
    /// VK_FORMAT_R16G16B16A16_SFLOAT. This is what Remix actually reports for its shared output target,
    /// so leaving it unmapped is what forced the CPU readback path and its per-frame round trip.
    constexpr unsigned int kVkFormatR16G16B16A16Sfloat = 97;

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

    /// Whether OPENMW_REMIX_READBACK asks for the CPU round trip.
    ///
    /// Read in two places -- here and in CompositeCallback -- because the import and the composite are
    /// separate objects with no shared state at construction, and the import has to know: holding a GL
    /// handle to Remix's memory is only safe if the semaphore handshake is going to guard it.
    bool readbackForced()
    {
        const char* value = std::getenv("OPENMW_REMIX_READBACK");
        return value != nullptr && *value != '\0' && *value != '0';
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
            case kVkFormatR16G16B16A16Sfloat:
                // Neither swizzled nor sRGB: the component order already matches, and a float format
                // carries linear values with no transfer function to undo. The readback path already
                // treats these bytes as four halves per pixel, which is the same interpretation.
                return kRgba16f;
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

        // Do not import at all when the readback path is forced.
        //
        // The import used to fail on its own for an unrelated reason -- VK_FORMAT_R16G16B16A16_SFLOAT had
        // no mapping -- which quietly meant the readback configuration never held a GL handle to Remix's
        // memory. Mapping the format made the import succeed everywhere, including here, and that turned
        // out to be actively dangerous rather than merely useless: the result is a GL texture aliasing
        // memory Remix writes every frame, in optimal tiling, with no semaphore handshake and no layout
        // transition. That is undefined, and it presented as a GPU fault (LiveKernelEvent 0x1a8) taking the
        // whole process down rather than anything catchable.
        //
        // Nothing samples it in this mode anyway: the composite uploads its own texture from the CPU copy.
        if (readbackForced())
        {
            Log(Debug::Info) << "Remix GL interop: import skipped, readback path is forced. Importing would "
                                "alias memory Remix writes with no synchronisation available to guard it.";
            return;
        }

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
    // uEncodeGamma raises the output through a power curve to compensate for brightness lost somewhere
    // past this framebuffer. It is a knob, not a correction with a known right answer -- see the long
    // note at the point it is uploaded for what has been measured and what has been ruled out.
    //
    // The short version, because it is easy to misread this as a colour-space conversion: the framebuffer
    // contents measure CORRECT without it. An external capture of the framebuffer looked right before
    // this existed and looks too bright with it. So this deliberately writes wrong data to make the panel
    // show the right picture, and the moment the real fault is found this should go away rather than be
    // retuned.
    //
    // uTestPattern is the tool that established that much, by substituting values chosen here instead of
    // sampling Remix at all.
    const char* const kFragmentShader = R"(#version 330 core
uniform sampler2D uImage;
uniform int uSwizzle;
uniform float uEncodeGamma;
uniform int uTestPattern;
in vec2 vUv;
out vec4 fColour;

vec3 encodeGamma(vec3 colour, float gamma)
{
    // A tunable power curve, not the piecewise IEC 61966-2-1 sRGB curve this replaced.
    //
    // The piecewise curve is the right choice when the job is "encode linear values as sRGB", and that
    // is what this started as. It is the wrong shape for what the uniform actually does now, which is
    // compensate for an unidentified loss between our framebuffer and the panel. A fixed curve cannot be
    // dialled in, and dialling it in is both what makes the image usable and the only measurement of the
    // loss currently available: the exponent that looks correct is the exponent the display is eating.
    //
    // pow(x, 1/2.2) tracks the sRGB curve closely except in deep shadow, where the linear toe near black
    // keeps sRGB from crushing. Worth remembering if the low end ever looks wrong at gamma 2.2 -- the
    // divergence is real, it is just small next to the effect being corrected.
    return pow(clamp(colour, 0.0, 1.0), vec3(1.0 / gamma));
}

void main()
{
    vec3 colour = texture(uImage, vUv).rgb;
    if (uSwizzle != 0)
        colour = colour.bgr;

    // Substitute a value this shader chose, so the screen can be compared against a number rather than
    // against another image. Every measurement so far says the framebuffer is correct and every reference
    // image has turned out to measure something other than the screen: OpenMW's own screenshot samples
    // one render stage too early and captures the raster frame, and Game Bar tone maps its own output on
    // an HDR desktop. Neither is ground truth, so this stops using images as evidence.
    //
    // The step wedge is the informative one. Uniform dimness and a compressed range look the same in a
    // photograph but not in a wedge: if every band is darker by about the same proportion the frame is
    // being scaled on the way to the display, whereas bands bunching toward black is a transfer function
    // being applied that should not be.
    if (uTestPattern == 1)
        colour = vec3(1.0);
    else if (uTestPattern == 2)
        colour = vec3(0.5);
    else if (uTestPattern == 3)
        colour = vec3(floor(clamp(vUv.x, 0.0, 0.999) * 5.0) * 0.25);

    if (uEncodeGamma > 1.001)
        colour = encodeGamma(colour, uEncodeGamma);
    fColour = vec4(colour, 1.0);
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
        GLint encodeGammaLocation = -1;
        GLint flipLocation = -1;
        GLint testPatternLocation = -1;
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
            // RGBA16F, matching the half-float shared surface the readback now comes from. An RGBA8
            // texture here would undo the point of widening the chain: the range would survive the copy
            // out of the runtime and then be clamped on upload instead.
            //
            // GL_RGBA rather than GL_BGRA for the transfer, because D3DFMT_A16B16G16R16F maps to the same
            // Vulkan format as the runtime's own output and so arrives in RGBA order, unlike the A8R8G8B8
            // surface this replaces.
            glTexImage2D(GL_TEXTURE_2D, 0, kRgba16f, static_cast<GLsizei>(width),
                static_cast<GLsizei>(height), 0, GL_RGBA, kHalfFloat, nullptr);
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
                static_cast<GLsizei>(mResources->readbackHeight), GL_RGBA, kHalfFloat,
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
            resources->encodeGammaLocation
                = ext->glGetUniformLocation(resources->program, "uEncodeGamma");
            resources->flipLocation = ext->glGetUniformLocation(resources->program, "uFlip");
            resources->testPatternLocation
                = ext->glGetUniformLocation(resources->program, "uTestPattern");
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
            // No shader-side swap: the half-float surface arrives in RGBA order already, since it shares a
            // Vulkan format with the runtime's own output rather than being the byte-reversed BGRA an
            // A8R8G8B8 surface produced.
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

        // Read once, not per frame: an environment variable cannot change while the process runs, and this
        // is on the draw thread.
        //
        // This is compensation for a fault we have not located, and it is labelled as such because
        // calling it a fix would hide the open question.
        //
        // What is established. Applying this makes the image match what the same runtime and the same
        // config produce under MGE-XE, and without it the frame is dim with collapsed midtones. But a
        // Game Bar capture -- which records the framebuffer contents, not the panel -- looked correct
        // BEFORE this was applied and looks too bright after. So the bits were already right, and this
        // makes the data wrong in order to make the display look right. The loss is downstream of our
        // framebuffer.
        //
        // The runtime source agrees the bits were right. dispatchSRGBDither runs inside the injectRTX
        // chain (rtx_context.cpp:769), operates in place on m_finalOutput with AccessType::ReadWrite, and
        // its shader applies linearToGamma when performSRGBConversion is set. That flag is
        // "!captureScreenImage && g_allowSrgbConversionForOutput": captureScreenImage is true only on a
        // single requested frame, and g_allowSrgbConversionForOutput is true here because this host calls
        // the legacy dxvk_CreateD3D9 slot with editorModeEnabled false. Nothing writes m_finalOutput
        // after that pass -- it is read into srcImage and blitted to the game target. The copy reads that
        // same image.
        //
        // Ruled out on our side: the framebuffer is not sRGB (OpenMW requests 8/8/8/0 with no
        // sRGB-capable attribute, engine.cpp), GL_FRAMEBUFFER_SRGB is disabled at the draw (probed), the
        // readback texture is GL_RGBA16F rather than an _SRGB format so sampling applies no decode, and
        // OpenMW's gamma ramp is identity at the configured gamma/contrast of 1.0.
        //
        // So this stays a knob rather than becoming a constant. The exponent that looks correct is the
        // only measurement of the loss we currently have, and a value near 2.2 would mean a full extra
        // sRGB decode is happening somewhere past the framebuffer, while something nearer 1.4 would mean
        // it is milder and probably not a transfer function at all.
        //
        //   OPENMW_REMIX_GAMMA=1    off, raw pass-through
        //   OPENMW_REMIX_GAMMA=2.2  a full sRGB-shaped encode
        static const float encodeGamma = []() {
            const char* value = std::getenv("OPENMW_REMIX_GAMMA");
            float gamma = 2.2f;
            if (value != nullptr && *value != '\0')
            {
                const double parsed = std::atof(value);
                // Reject nonsense rather than silently dividing by it: the shader divides by this.
                if (parsed >= 0.1 && parsed <= 10.0)
                    gamma = static_cast<float>(parsed);
                else
                    Log(Debug::Warning) << "Remix composite: ignoring OPENMW_REMIX_GAMMA='" << value
                                        << "', outside 0.1 to 10";
            }
            Log(Debug::Info)
                << "Remix composite: display compensation gamma " << gamma
                << (gamma > 1.001f ? "" : " (off, raw pass-through)")
                << ". This corrects for a loss between our framebuffer and the panel that has not been "
                   "located -- the framebuffer contents themselves measure correct without it. Tune with "
                   "OPENMW_REMIX_GAMMA.";
            return gamma;
        }();
        if (mResources->encodeGammaLocation >= 0)
            ext->glUniform1f(mResources->encodeGammaLocation, encodeGamma);
        if (mResources->flipLocation >= 0)
            ext->glUniform2f(mResources->flipLocation, mFlipHorizontal ? -1.0f : 1.0f,
                mFlipVertical ? -1.0f : 1.0f);

        // 1 = full white, 2 = mid grey, 3 = a five-step wedge from black to white. Logged once so a
        // screenshot of a flat white frame cannot be mistaken for a broken composite later.
        static const int testPattern = []() {
            const char* value = std::getenv("OPENMW_REMIX_TESTPATTERN");
            const int pattern = (value != nullptr && *value != '\0') ? std::atoi(value) : 0;
            if (pattern != 0)
                Log(Debug::Warning)
                    << "Remix composite: drawing test pattern " << pattern
                    << " INSTEAD of Remix's image (1=white 1.0, 2=grey 0.5, 3=step wedge). Compare "
                       "against a white window on the desktop: if this white is dimmer, the frame was "
                       "always correct and the display path is mapping it differently.";
            return pattern;
        }();
        if (mResources->testPatternLocation >= 0)
            ext->glUniform1i(mResources->testPatternLocation, testPattern);

        ext->glBindVertexArray(mResources->vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        ext->glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);
        ext->glUseProgram(0);
        // What actually landed in the framebuffer, read back from it.
        //
        // This exists because every explanation for "the screen is dimmer than a screenshot of it" has been
        // wrong so far, and they were all reasoned rather than measured. The readback probe says Remix hands
        // us a well-exposed image peaking near 1.0. This says what survives the composite draw into
        // OpenMW's default framebuffer. Between the two there is nowhere left for the brightness to hide:
        //
        //   both peaks near 1.0   -> the bits on screen are correct and the loss is past the framebuffer
        //                            entirely, which means the display path or the comparison itself
        //   this peak much lower  -> the composite draw is losing it, and that is a small amount of code
        //
        // A 32x32 block from the centre of the viewport rather than the whole frame: glReadPixels stalls
        // the pipeline, so this has to stay small, and it only runs every 600th composite.
        {
            static unsigned compositeFrames = 0;
            if (++compositeFrames % 600 == 0)
            {
                GLint viewport[4] = {};
                glGetIntegerv(GL_VIEWPORT, viewport);
                if (viewport[2] > 0 && viewport[3] > 0)
                {
                    // The whole framebuffer, sampled with the same prime stride the readback probe uses.
                    //
                    // The first version of this read a 32x32 block from the centre and compared its peak
                    // against the readback's whole-frame peak, which is not a comparison at all: one was the
                    // brightest pixel anywhere in a 4K frame, the other the middle of a dark wall. It
                    // reported 22/255 against 0.80 and that difference was unreadable. Reading the whole
                    // thing costs 33 MB once every six hundred frames, which is nothing for a diagnostic
                    // and is the only way the two numbers mean the same thing.
                    const std::size_t pixels
                        = static_cast<std::size_t>(viewport[2]) * static_cast<std::size_t>(viewport[3]);
                    std::vector<unsigned char> frame(pixels * 4, 0);
                    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGBA,
                        GL_UNSIGNED_BYTE, frame.data());

                    // A histogram, because peak is not exposure and reporting it as though it were sent
                    // this investigation down a blind alley for a long time.
                    //
                    // Peak said 0.667 and "8317 of 8320 non-black", and that was read as "Remix hands us a
                    // well-exposed image". It says nothing of the kind. A pitch-dark interior with one
                    // candle flame in it has exactly those numbers: the flame alone sets the peak, and
                    // non-black counts any pixel above zero, including 1/255. The frame can be almost
                    // entirely near-black and still report both.
                    //
                    // What settles it is where the bulk of the distribution sits, so keep the median and
                    // the tails. The test pattern proved the display reaches full brightness for a value
                    // of 1.0, so if the median here is very low then the image genuinely is dark and the
                    // remaining question is Remix's exposure, not this display path.
                    unsigned char peak = 0;
                    std::size_t nonBlack = 0;
                    std::size_t sampled = 0;
                    double sum = 0.0;
                    std::size_t histogram[256] = {};
                    for (std::size_t p = 0; p < pixels; p += 997)
                    {
                        ++sampled;
                        // Rec. 709 luma rather than the max channel. Max channel is what a peak probe
                        // wants; perceived brightness is what a dimness complaint is about, and a
                        // saturated blue reads far darker than its max channel suggests.
                        const double luma = 0.2126 * frame[p * 4] + 0.7152 * frame[p * 4 + 1]
                            + 0.0722 * frame[p * 4 + 2];
                        const unsigned char value = static_cast<unsigned char>(luma + 0.5);
                        ++histogram[value];
                        sum += luma;
                        peak = std::max(peak, std::max(
                            { frame[p * 4], frame[p * 4 + 1], frame[p * 4 + 2] }));
                        if (value != 0)
                            ++nonBlack;
                    }

                    const auto percentile = [&](double fraction) {
                        const std::size_t target
                            = static_cast<std::size_t>(double(sampled) * fraction);
                        std::size_t running = 0;
                        for (int bin = 0; bin < 256; ++bin)
                        {
                            running += histogram[bin];
                            if (running >= target)
                                return bin;
                        }
                        return 255;
                    };
                    const double mean = sampled > 0 ? sum / double(sampled) : 0.0;
                    const int median = percentile(0.50);
                    const int p90 = percentile(0.90);
                    const int p99 = percentile(0.99);
                    std::size_t belowTenPercent = 0;
                    for (int bin = 0; bin < 26; ++bin)
                        belowTenPercent += histogram[bin];

                    // One call, and it decides whether a transfer function is being applied twice. If the
                    // default framebuffer is sRGB and this is enabled, every value the composite writes is
                    // encoded again on the way in -- and the pass-through shader is handing it values that
                    // are already encoded.
                    const GLboolean framebufferSrgb = glIsEnabled(0x8DB9 /* GL_FRAMEBUFFER_SRGB */);

                    Log(Debug::Info)
                        << "Remix composite: framebuffer luma over " << sampled << " samples -- median "
                        << median << "/255 (" << (median / 255.0f) << "), mean "
                        << (mean / 255.0) << ", p90 " << p90 << ", p99 " << p99 << ", peak channel "
                        << int(peak) << " (" << (peak / 255.0f) << "), "
                        << (100.0 * double(belowTenPercent) / double(std::max<std::size_t>(sampled, 1)))
                        << "% below 0.1, " << nonBlack << " non-black, GL_FRAMEBUFFER_SRGB "
                        << (framebufferSrgb ? "ENABLED" : "disabled")
                        << ". The median is the exposure; the peak is one candle flame and was read as "
                           "exposure for far too long.";
                }
            }
        }

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
