#ifndef OPENMW_COMPONENTS_TERRAIN_COMPOSITEMAPRENDERER_H
#define OPENMW_COMPONENTS_TERRAIN_COMPOSITEMAPRENDERER_H

#include <osg/Drawable>

#include <mutex>
#include <set>

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
