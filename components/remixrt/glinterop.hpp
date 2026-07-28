#ifndef OPENMW_COMPONENTS_REMIXRT_GLINTEROP_H
#define OPENMW_COMPONENTS_REMIXRT_GLINTEROP_H

#include <osg/GraphicsContext>
#include <osg/ref_ptr>

#include "runtime.hpp"

namespace RemixRT
{
    /// Imports a Remix-owned image into OpenGL so its output can be composited into OpenMW's frame.
    ///
    /// Remix renders with Vulkan and OpenMW presents with OpenGL, so the image has to cross an API
    /// boundary. The route is GL_EXT_memory_object plus GL_EXT_memory_object_win32: Remix allocates its
    /// render target from exportable, dedicated Vulkan memory and hands us the Win32 handle and the
    /// allocation size, and OpenGL imports that memory and wraps a texture around it. No copy, no
    /// readback -- both APIs address the same pixels.
    ///
    /// Two rejected alternatives, recorded so they are not retried:
    ///   - dxvk_GetExternalSwapchain returns a raw VkImage handle. OpenGL cannot import a VkImage; the
    ///     EXT_memory_object extensions import *memory*, and a VkImage handle is a driver-internal
    ///     object with no cross-API meaning.
    ///   - WGL_NV_DX_interop2 is the usual answer for D3D-to-GL, but it is implemented by the OpenGL
    ///     driver and expects a device from the native D3D driver. Remix's IDirect3DDevice9Ex is DXVK
    ///     over Vulkan, so the driver has no way to recognise it.
    ///
    /// This must run on the thread that owns the GL context, which is why it is a GraphicsOperation
    /// rather than a plain function. Queue it with GraphicsContext::add.
    class ImportOperation : public osg::GraphicsOperation
    {
    public:
        explicit ImportOperation(const Runtime::ExternalImage& image);

        void operator()(osg::GraphicsContext* context) override;

        /// Valid once the operation has run. Zero means the import failed.
        unsigned int textureName() const { return mTextureName; }

        bool succeeded() const { return mSucceeded; }

        /// True once the operation has run, whether or not it worked.
        bool completed() const { return mCompleted; }

    private:
        Runtime::ExternalImage mImage;
        unsigned int mMemoryObject = 0;
        unsigned int mTextureName = 0;
        bool mSucceeded = false;
        bool mCompleted = false;
    };
}

#endif
