#include "compositemaprenderer.hpp"

#include <osg/FrameBufferObject>
#include <osg/Image>
#include <osg/RenderInfo>
#include <osg/Texture2D>

#include <algorithm>
#include <string>

namespace Terrain
{

    CompositeMapRenderer::CompositeMapRenderer()
        : mTargetFrameRate(120)
        , mMinimumTimeAvailable(0.0025)
    {
        setSupportsDisplayList(false);
        setCullingActive(false);

        mFBO = new osg::FrameBufferObject;
    }

    CompositeMapRenderer::~CompositeMapRenderer() = default;

    void CompositeMapRenderer::drawImplementation(osg::RenderInfo& renderInfo) const
    {
        double dt = mTimer.time_s();
        dt = std::min(dt, 0.2);
        mTimer.setStartTick();
        double targetFrameTime = 1.0 / static_cast<double>(mTargetFrameRate);
        double conservativeTimeRatio(0.75);
        double availableTime = std::max((targetFrameTime - dt) * conservativeTimeRatio, mMinimumTimeAvailable);

        std::lock_guard<std::mutex> lock(mMutex);

        if (mImmediateCompileSet.empty() && mCompileSet.empty())
            return;

        while (!mImmediateCompileSet.empty())
        {
            osg::ref_ptr<CompositeMap> node = *mImmediateCompileSet.begin();
            mImmediateCompileSet.erase(node);

            mMutex.unlock();
            compile(*node, renderInfo);
            mMutex.lock();
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(availableTime);
        while (!mCompileSet.empty() && std::chrono::steady_clock::now() < deadline)
        {
            osg::ref_ptr<CompositeMap> node = *mCompileSet.begin();
            mCompileSet.erase(node);

            mMutex.unlock();
            compile(*node, renderInfo);
            mMutex.lock();

            if (node->mCompiled < node->mDrawables.size())
            {
                // We did not compile the map fully.
                // Place it back to queue to continue work in the next time.
                mCompileSet.insert(node);
            }
        }
        mTimer.setStartTick();
    }

    void CompositeMapRenderer::compile(CompositeMap& compositeMap, osg::RenderInfo& renderInfo) const
    {
        // if there are no more external references we can assume the texture is no longer required
        if (compositeMap.mTexture->referenceCount() <= 1)
        {
            compositeMap.mCompiled = compositeMap.mDrawables.size();
            return;
        }

        osg::Timer timer;
        osg::State& state = *renderInfo.getState();
        osg::GLExtensions* ext = state.get<osg::GLExtensions>();

        if (!mFBO)
            return;

        if (!ext->isFrameBufferObjectSupported)
            return;

        osg::FrameBufferAttachment attach(compositeMap.mTexture);
        mFBO->setAttachment(osg::Camera::COLOR_BUFFER, attach);
        mFBO->apply(state, osg::FrameBufferObject::DRAW_FRAMEBUFFER);

        GLenum status = ext->glCheckFramebufferStatus(GL_FRAMEBUFFER_EXT);

        if (status != GL_FRAMEBUFFER_COMPLETE_EXT)
        {
            GLuint fboId = state.getGraphicsContext() ? state.getGraphicsContext()->getDefaultFboId() : 0;
            ext->glBindFramebuffer(GL_FRAMEBUFFER_EXT, fboId);
            OSG_ALWAYS << "Error attaching FBO" << std::endl;
            return;
        }

        // inform State that Texture attribute has changed due to compiling of FBO texture
        // should OSG be doing this on its own?
        state.haveAppliedTextureAttribute(state.getActiveTextureUnit(), osg::StateAttribute::TEXTURE);

        for (size_t i = compositeMap.mCompiled; i < compositeMap.mDrawables.size(); ++i)
        {
            osg::Drawable* drw = compositeMap.mDrawables[i];
            osg::StateSet* stateset = drw->getStateSet();

            if (stateset)
                renderInfo.getState()->pushStateSet(stateset);

            renderInfo.getState()->apply();

            glViewport(0, 0, compositeMap.mTexture->getTextureWidth(), compositeMap.mTexture->getTextureHeight());
            drw->drawImplementation(renderInfo);

            if (stateset)
                renderInfo.getState()->popStateSet();

            ++compositeMap.mCompiled;

            compositeMap.mDrawables[i] = nullptr;
        }
        if (compositeMap.mCompiled == compositeMap.mDrawables.size())
        {
            // Read the finished composite back before the FBO goes away. This is the only moment it is
            // available: the attachment is bound right now, and mDrawables is about to be released.
            //
            // Bound again as READ_FRAMEBUFFER because apply() above bound it for drawing only, and
            // glReadPixels reads from the read binding. Reading a colour attachment that was just rendered
            // into needs no explicit barrier in GL -- the pipeline orders it.
            if (CompositeMap::sReadbackEnabled && compositeMap.readback() == nullptr)
            {
                const int width = compositeMap.mTexture->getTextureWidth();
                const int height = compositeMap.mTexture->getTextureHeight();
                if (width > 0 && height > 0)
                {
                    mFBO->apply(state, osg::FrameBufferObject::READ_FRAMEBUFFER);

                    osg::ref_ptr<osg::Image> image = new osg::Image;
                    image->allocateImage(width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE);

                    // Read from the colour attachment explicitly. The read buffer is not implied by binding
                    // the FBO, and leaving it unset can read from whatever was current instead.
                    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
                    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image->data());

                    // Force alpha opaque. The composite target is GL_RGB -- it has no alpha channel at all
                    // (see ChunkManager::createCompositeMapRTT) -- so what GL returns for alpha when asked
                    // for RGBA is driver-dependent: 1.0 by the spec's reading, but 0 or uninitialised in
                    // practice. Terrain has no use for alpha, and a terrain albedo that arrives partially
                    // transparent renders as washed-out lighter patches with straight chunk edges, which is
                    // exactly what this produced before. Overwriting it removes the dependency rather than
                    // hoping the driver is generous.
                    unsigned char* pixels = image->data();
                    const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
                    for (std::size_t i = 0; i < count; ++i)
                        pixels[i * 4 + 3] = 255;

                    // Named so a consumer can tell chunks apart in a log, and so the texture identity
                    // derived from it is stable for a given chunk rather than depending on load order.
                    image->setFileName(std::string(CompositeMap::sReadbackImageName));
                    // Published last, under the lock, after the pixels are in. Ordering matters as much as
                    // the mutual exclusion: the consumer must not be able to see this pointer before it can
                    // see the data glReadPixels just wrote.
                    compositeMap.setReadback(std::move(image));
                }
            }

            compositeMap.mDrawables = std::vector<osg::ref_ptr<osg::Drawable>>();
        }

        state.haveAppliedAttribute(osg::StateAttribute::VIEWPORT);

        GLuint fboId = state.getGraphicsContext() ? state.getGraphicsContext()->getDefaultFboId() : 0;
        ext->glBindFramebuffer(GL_FRAMEBUFFER_EXT, fboId);
    }

    void CompositeMapRenderer::setMinimumTimeAvailableForCompile(double time)
    {
        mMinimumTimeAvailable = time;
    }

    void CompositeMapRenderer::setTargetFrameRate(float framerate)
    {
        mTargetFrameRate = framerate;
    }

    void CompositeMapRenderer::addCompositeMap(CompositeMap* compositeMap, bool immediate)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (immediate)
            mImmediateCompileSet.insert(compositeMap);
        else
            mCompileSet.insert(compositeMap);
    }

    void CompositeMapRenderer::setImmediate(CompositeMap* compositeMap)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        CompileSet::iterator found = mCompileSet.find(compositeMap);
        if (found == mCompileSet.end())
            return;
        else
        {
            mImmediateCompileSet.insert(compositeMap);
            mCompileSet.erase(found);
        }
    }

    size_t CompositeMapRenderer::getCompileSetSize() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mCompileSet.size();
    }

    bool CompositeMap::sReadbackEnabled = false;

CompositeMap::CompositeMap()
        : mCompiled(0)
    {
    }

    CompositeMap::~CompositeMap() {}

}
