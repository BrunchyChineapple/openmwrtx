#ifndef OPENMW_COMPONENTS_REMIXRT_GLINTEROP_H
#define OPENMW_COMPONENTS_REMIXRT_GLINTEROP_H

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <osg/Camera>
#include <osg/Drawable>
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
        ImportOperation(const Runtime::ExternalImage& image, const Runtime::ExternalSync& sync);

        void operator()(osg::GraphicsContext* context) override;

        /// Valid once the operation has run. Zero means the import failed.
        unsigned int textureName() const { return mTextureName; }

        bool succeeded() const { return mSucceeded; }

        /// True once the operation has run, whether or not it worked.
        bool completed() const { return mCompleted; }

        /// True when the source is BGRA and consumers must swap red and blue. GL has no BGRA internal
        /// format, so the image is imported as RGBA8 and the swap is left to whoever samples it.
        bool needsChannelSwap() const { return mNeedsChannelSwap; }

        /// GL semaphore to wait on before sampling the image. Zero when unavailable.
        unsigned int waitSemaphore() const { return mWaitSemaphore; }

        /// GL semaphore to signal once sampling is finished. Zero when unavailable.
        unsigned int signalSemaphore() const { return mSignalSemaphore; }

        /// True when both semaphores imported, so the consumer can order itself against Remix.
        /// When false, sampling the image is undefined -- the import is not usable as it stands.
        bool syncAvailable() const { return mWaitSemaphore != 0 && mSignalSemaphore != 0; }

        /// Tears the import down: texture, memory object and both semaphores.
        ///
        /// Exists because an import that cannot be synchronised must not merely go unused -- it must go
        /// away. A GL texture aliasing memory Remix writes every frame, with no handshake ordering the two,
        /// faulted the device rather than producing a wrong picture (LiveKernelEvent 0x1a8), and it did so
        /// while nothing was sampling it. So the handshake failing has to release the import, not just stop
        /// reading from it.
        ///
        /// Must be called on the thread holding the GL context. Safe to call more than once; every handle
        /// is zeroed, so textureName() and syncAvailable() report the import as gone afterwards.
        void release();

    private:
        Runtime::ExternalImage mImage;
        Runtime::ExternalSync mSync;
        unsigned int mMemoryObject = 0;
        unsigned int mTextureName = 0;
        unsigned int mWaitSemaphore = 0;
        unsigned int mSignalSemaphore = 0;
        bool mSucceeded = false;
        bool mCompleted = false;
        bool mNeedsChannelSwap = false;
    };

    /// Draws the imported Remix image over OpenMW's frame.
    ///
    /// Runs on the draw thread, after OpenMW's world rendering and before the GUI. Raw GL rather than an
    /// OSG subgraph because the source is an externally-owned texture name, and going through
    /// osg::Texture2D would mean convincing OSG it already owns a GL object.
    ///
    /// Driven through CompositeDrawable rather than as a camera draw callback. It began as the main
    /// camera's *final* draw callback, which is wrong for a reason worth recording: OSG runs that after
    /// the camera's entire render-stage tree, and OpenMW's GUI is a nested POST_RENDER camera inside that
    /// tree. So a full-frame overwrite there erases every UI element -- and there is no callback hook
    /// between "world drawn" and "nested post-render stages drawn" to move it to.
    ///
    /// Deliberately a hard overwrite, not a blend: while the renderer is being brought up, seeing
    /// exactly what Remix produced -- including black -- is more useful than seeing it mixed with
    /// OpenMW's raster output and having to guess which pixels came from where.
    class CompositeCallback : public osg::Camera::DrawCallback
    {
    public:
        explicit CompositeCallback(ImportOperation* import);

        void operator()(osg::RenderInfo& renderInfo) const override;

        /// Turns compositing on and off at runtime so the raster frame can be compared against the
        /// path-traced one without relaunching.
        void setEnabled(bool enabled) { mEnabled = enabled; }
        bool enabled() const { return mEnabled; }

        /// Tells the callback whether a synchronised copy was issued for this frame.
        ///
        /// Only then may it wait on Remix's semaphore: the spec makes waiting on a semaphore that has
        /// had no signal submitted undefined behaviour, so the wait has to be gated on the producer
        /// actually having run. Called from the engine's frame loop before the draw traversal.
        void setSyncArmed(bool armed) { mSyncArmed.store(armed, std::memory_order_relaxed); }

        /// Reports, and clears, whether the callback signalled Remix since this was last asked.
        ///
        /// The engine feeds the answer straight back to copyOutputSynced. This is what keeps the
        /// binary semaphore pairing honest: Remix only waits when a signal genuinely happened.
        /// Read from the main thread, written from the draw thread, hence the atomic.
        bool takeConsumerSignalled() const { return mSignalled.exchange(false, std::memory_order_acq_rel); }

        /// True once a semaphore operation has failed.
        ///
        /// Latching rather than retrying, because a failed wait leaves Remix's signal unconsumed, and
        /// these are binary semaphores: carrying on would signal again with no intervening wait and
        /// corrupt the pairing for every frame after. The engine drops to the unsynchronised copy so
        /// the degradation is visible and bounded instead.
        bool syncFailed() const { return mSyncFailed.load(std::memory_order_relaxed); }

        /// True when the callback wants CPU-read pixels rather than the imported image.
        ///
        /// Forced by OPENMW_REMIX_READBACK, and entered automatically once the semaphore handshake has
        /// failed, so a broken handshake degrades to a correct-but-slower picture instead of a black
        /// one. The engine polls this to decide whether to pay for the read.
        bool readbackMode() const { return mForceReadback || mSyncFailed.load(std::memory_order_relaxed); }
        /// True when the handshake is switched off outright rather than having failed. Deliberately not
        /// folded into readbackMode(): the point is to reach the state where neither the readback nor the
        /// semaphores are in play, which is exactly what those two flags being wired together prevented.
        bool syncDisabled() const { return mSyncDisabled; }

        /// True when the handshake runs one way: we signal that sampling is finished, and never wait for
        /// the copy.
        ///
        /// On by default because it is the only form of it this driver accepts. glSignalSemaphoreEXT on an
        /// imported semaphore succeeds; glWaitSemaphoreEXT on one returns GL_INVALID_OPERATION with a
        /// texture barrier and without, on the importing thread and context, with the signal submitted a
        /// full frame earlier, and with exactly one signal outstanding. Every one of those was measured.
        ///
        /// The engine must pair this with the wait-only copy so Remix stops signalling copyComplete: that
        /// semaphore is binary, and one left signalled with no consumer makes the next signal invalid.
        bool syncOneWay() const { return mSyncOneWay; }

        /// Hands over one frame of CPU-read pixels, tightly packed B8G8R8A8.
        ///
        /// Takes the buffer by swap rather than by copy: at 4K a frame is 33 MB, and copying it here
        /// and again before upload was two thirds of the path's memory traffic for no benefit. The
        /// caller gets back a buffer of arbitrary contents to refill, which is fine because the reader
        /// resizes and overwrites it completely.
        ///
        /// Called from the engine's frame loop on the main thread; consumed on the draw thread.
        void takeReadbackFrame(std::vector<unsigned char>& pixels, unsigned int width, unsigned int height);

        /// Reads and clears the accumulated cost of uploading readback frames.
        ///
        /// Exists because this work happens on the draw thread and so is invisible to any timing the frame
        /// loop does of itself. Written on the draw thread, read from the main thread.
        void takeUploadCost(unsigned long long& nanoseconds, unsigned int& uploads) const
        {
            nanoseconds = mUploadNanoseconds.exchange(0, std::memory_order_acq_rel);
            uploads = mUploadCount.exchange(0, std::memory_order_acq_rel);
        }

    private:
        struct Resources;

        /// Uploads the most recent readback frame and returns the texture to sample. 0 if unavailable.
        unsigned int uploadReadbackTexture(osg::GLExtensions* ext) const;

        /// Non-const because a failed handshake has to release the import rather than leave it aliasing
        /// Remix's memory unguarded.
        ImportOperation* mImport;
        mutable std::unique_ptr<Resources> mResources;
        mutable bool mFailed = false;
        bool mEnabled = true;
        bool mForceReadback = false;
        bool mSyncDisabled = false;

        /// Signal but never wait. See syncOneWay().
        bool mSyncOneWay = true;
        /// Mutable for the same reason the other once-only log flags are: the compositing path is const.
        mutable bool mLoggedOneWay = false;
        /// Vertical flip only: Remix's output has row 0 at the top, OpenGL samples v = 0 at the bottom.
        /// Confirmed on screen, not just derived. Overridable with OPENMW_REMIX_FLIP=none|v|h|both so a
        /// future source with a different convention does not need a rebuild to diagnose.
        bool mFlipHorizontal = false;
        bool mFlipVertical = true;
        mutable std::mutex mReadbackMutex;
        mutable std::vector<unsigned char> mReadbackPixels;
        /// Separate from mReadbackPixels so the GL upload happens outside the lock.
        mutable std::vector<unsigned char> mReadbackUpload;
        mutable unsigned int mReadbackWidth = 0;
        mutable unsigned int mReadbackHeight = 0;
        mutable bool mReadbackFresh = false;
        std::atomic<bool> mSyncArmed{ false };
        mutable std::atomic<bool> mSignalled{ false };
        mutable std::atomic<bool> mSyncFailed{ false };
        /// Consecutive failed semaphore waits, counted so a startup race can be told from a permanent
        /// ordering fault. Mutable for the same reason mSyncFailed is: the compositing path is const.
        mutable unsigned int mSyncWaitFailures = 0;
        /// One-shot guard for the synchronisation state report.
        mutable bool mLoggedSyncState = false;
        /// Nanoseconds accumulated in glTexSubImage2D, and how many uploads that covers. Integer
        /// nanoseconds rather than a floating-point millisecond count because atomic<double> arithmetic is
        /// a C++20 addition and this has to build wherever the rest of the engine does.
        mutable std::atomic<unsigned long long> mUploadNanoseconds{ 0 };
        mutable std::atomic<unsigned int> mUploadCount{ 0 };
        /// Diagnostic: drop the texture barrier from the semaphore operations, to tell a rejected
        /// semaphore apart from a rejected texture barrier. Read once at construction.
        bool mSkipTextureBarrier = false;
    };

    /// Scene-graph node that draws a CompositeCallback where it is placed in the render order.
    ///
    /// Exists so the composite can sit between OpenMW's world and OpenMW's GUI. OSG offers camera
    /// callbacks before the camera's own drawing and after its whole stage tree, but nothing in between,
    /// and the GUI lives inside that tree as a nested POST_RENDER camera. Content, unlike a callback, can
    /// be ordered: put this under its own POST_RENDER camera with a negative order and it draws after the
    /// world and before the GUI.
    ///
    /// Making it real content also avoids relying on OSG keeping a render stage for a camera whose
    /// subgraph produced no drawables.
    class CompositeDrawable : public osg::Drawable
    {
    public:
        explicit CompositeDrawable(CompositeCallback* composite);

        void drawImplementation(osg::RenderInfo& renderInfo) const override;

    private:
        osg::ref_ptr<CompositeCallback> mComposite;
    };
}

#endif
