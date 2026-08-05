#ifndef OPENMW_COMPONENTS_TERRAIN_COMPOSITEMAPRENDERER_H
#define OPENMW_COMPONENTS_TERRAIN_COMPOSITEMAPRENDERER_H

#include <osg/Drawable>

#include <mutex>
#include <set>
#include <string_view>

namespace osg
{
    class FrameBufferObject;
    class RenderInfo;
    class Texture2D;
}

namespace Terrain
{

    class CompositeMap : public osg::Referenced
    {
    public:
        CompositeMap();
        ~CompositeMap();
        std::vector<osg::ref_ptr<osg::Drawable>> mDrawables;
        osg::ref_ptr<osg::Texture2D> mTexture;
        size_t mCompiled;

        /// The state set of the first sub-quad's base layer, and the factor its texture matrix has to be
        /// scaled by to tile once per sub-quad across the whole chunk.
        ///
        /// Kept for consumers that need the layer texture itself rather than the composited result, which
        /// only ever exists on the GPU: mTexture is a render target with no osg::Image behind it. The
        /// path tracer needs a real albedo, and there is nowhere else to get one -- mDrawables cannot be
        /// used for it, because CompositeMapRenderer::compile releases each entry as it renders it.
        ///
        /// Null for a chunk composited from nothing, which should not happen but is not worth asserting.
        osg::ref_ptr<osg::StateSet> mBaseLayerPass;
        float mBaseLayerTiling = 1.f;

        /// CPU copy of the composited result, filled once when the map finishes compositing.
        ///
        /// This is what mBaseLayerPass above was a stand-in for. Taking the base layer alone gives each
        /// chunk one flat texture, so the ground comes out as hard-edged rectangles wherever neighbouring
        /// chunks pick different base layers -- the blend that makes terrain read as terrain is exactly the
        /// part being discarded. OpenMW already computes it correctly here, into a render target, so reading
        /// that back is far better than reproducing the blend by hand: it inherits the layer tiling, the
        /// blend maps, and the half-texel nudge material.cpp applies to match vanilla.
        ///
        /// Null until composited, and null entirely unless sReadbackEnabled -- see there for the cost.
        ///
        /// Published from the draw thread and consumed from the thread that submits the scene, so both ends
        /// go through readback() rather than touching this directly. OpenMW never calls setThreadingModel,
        /// which leaves osgViewer on AutomaticSelection and gives a separate draw thread on any machine with
        /// cores to spare -- so this is a genuine cross-thread handoff, not a formality. A ref_ptr assignment
        /// is neither atomic nor ordered, so a reader could see a torn pointer, or a pointer whose pixels are
        /// not yet visible to it.
        osg::ref_ptr<osg::Image> mReadback;

        /// Guards mReadback for the handoff described above.
        mutable std::mutex mReadbackMutex;

        /// A strong reference to the readback image, or null.
        ///
        /// Returns a reference rather than a raw pointer deliberately. The consumer holds the result across a
        /// texture upload, and this image belongs to a chunk that can be released as the player moves, so a
        /// bare pointer is a use-after-free waiting for the timing to line up.
        osg::ref_ptr<osg::Image> readback() const
        {
            std::lock_guard<std::mutex> lock(mReadbackMutex);
            return mReadback;
        }

        /// Publishes the readback image. Draw thread only.
        void setReadback(osg::ref_ptr<osg::Image> image)
        {
            std::lock_guard<std::mutex> lock(mReadbackMutex);
            mReadback = std::move(image);
        }

        /// Whether composited maps are copied back to the CPU.
        ///
        /// Off by default because it is pure cost for anyone who only rasterises: one glReadPixels per
        /// chunk, and the result kept for as long as the chunk lives. At the default 512x512 RGBA8 that is
        /// 1MB a chunk. Enabled by the Remix backend, which has no other way to obtain a real albedo.
        static bool sReadbackEnabled;

        /// The osg::Image::getFileName() every readback image carries.
        ///
        /// A composite has no file behind it, so this stands in as its origin in logs. It is also the only
        /// signal a consumer has that an image is a composite rather than loaded art, which the Remix
        /// backend routes on -- composites are the one class of texture it block-compresses on upload.
        /// Shared from here so the producer and that consumer cannot drift apart silently.
        static constexpr std::string_view sReadbackImageName = "terrain_composite";
    };

    /**
     * @brief The CompositeMapRenderer is responsible for updating composite map textures in a blocking or non-blocking
     * way.
     */
    class CompositeMapRenderer : public osg::Drawable
    {
    public:
        CompositeMapRenderer();
        ~CompositeMapRenderer();

        void drawImplementation(osg::RenderInfo& renderInfo) const override;

        void compile(CompositeMap& compositeMap, osg::RenderInfo& renderInfo) const;

        /// Set the available time in seconds for compiling (non-immediate) composite maps each frame
        void setMinimumTimeAvailableForCompile(double time);

        /// If current frame rate is higher than this, the extra time will be set aside to do more compiling
        void setTargetFrameRate(float framerate);

        /// Add a composite map to be rendered
        void addCompositeMap(CompositeMap* map, bool immediate = false);

        /// Mark this composite map to be required for the current frame
        void setImmediate(CompositeMap* map);

        size_t getCompileSetSize() const;

    private:
        float mTargetFrameRate;
        double mMinimumTimeAvailable;
        mutable osg::Timer mTimer;

        typedef std::set<osg::ref_ptr<CompositeMap>> CompileSet;

        mutable CompileSet mCompileSet;
        mutable CompileSet mImmediateCompileSet;

        mutable std::mutex mMutex;

        osg::ref_ptr<osg::FrameBufferObject> mFBO;
    };

}

#endif
