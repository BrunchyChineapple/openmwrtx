#ifndef OPENMW_COMPONENTS_REMIXRT_GLINTEROP_H
#define OPENMW_COMPONENTS_REMIXRT_GLINTEROP_H

#include <memory>

#include <osg/Camera>
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

        /// True when the source is BGRA and consumers must swap red and blue. GL has no BGRA internal
        /// format, so the image is imported as RGBA8 and the swap is left to whoever samples it.
        bool needsChannelSwap() const { return mNeedsChannelSwap; }

    private:
        Runtime::ExternalImage mImage;
        unsigned int mMemoryObject = 0;
        unsigned int mTextureName = 0;
        bool mSucceeded = false;
        bool mCompleted = false;
        bool mNeedsChannelSwap = false;
    };

    /// Draws the imported Remix image over OpenMW's frame.
    ///
    /// Installed as the main camera's final draw callback, so it runs on the draw thread after OpenMW
    /// has finished rendering and before the buffer swap -- exactly where a full-frame replacement
    /// belongs. Raw GL rather than an OSG subgraph because the source is an externally-owned texture
    /// name, and going through osg::Texture2D would mean convincing OSG it already owns a GL object.
    ///
    /// Deliberately a hard overwrite, not a blend: while the renderer is being brought up, seeing
    /// exactly what Remix produced -- including black -- is more useful than seeing it mixed with
    /// OpenMW's raster output and having to guess which pixels came from where.
    class CompositeCallback : public osg::Camera::DrawCallback
    {
    public:
        explicit CompositeCallback(const ImportOperation* import);

        void operator()(osg::RenderInfo& renderInfo) const override;

        /// Turns compositing on and off at runtime so the raster frame can be compared against the
        /// path-traced one without relaunching.
        void setEnabled(bool enabled) { mEnabled = enabled; }
        bool enabled() const { return mEnabled; }

    private:
        struct Resources;

        const ImportOperation* mImport;
        mutable std::unique_ptr<Resources> mResources;
        mutable bool mFailed = false;
        bool mEnabled = true;
    };
}

#endif
