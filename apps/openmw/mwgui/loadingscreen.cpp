#include "loadingscreen.hpp"

#include <algorithm>
#include <array>
#include <chrono>

#include <osgViewer/Viewer>

#include <osg/Texture2D>

#include <MyGUI_Gui.h>
#include <MyGUI_ScrollBar.h>
#include <MyGUI_TextBox.h>
#include <MyGUI_UString.h>

#include <components/debug/debuglog.hpp>
#include <components/misc/pathhelpers.hpp>
#include <components/misc/rng.hpp>
#include <components/myguiplatform/myguitexture.hpp>
#include <components/resource/resourcesystem.hpp>
#include <components/settings/values.hpp>
#include <components/vfs/manager.hpp>
#include <components/remixrt/runtime.hpp>
#include <components/vfs/recursivedirectoryiterator.hpp>

#include "../mwbase/environment.hpp"
#include "../mwbase/inputmanager.hpp"
#include "../mwbase/statemanager.hpp"
#include "../mwbase/windowmanager.hpp"

#include "backgroundimage.hpp"

namespace MWGui
{

    LoadingScreen::LoadingScreen(Resource::ResourceSystem* resourceSystem, osgViewer::Viewer* viewer)
        : WindowBase("openmw_loading_screen.layout")
        , mResourceSystem(resourceSystem)
        , mViewer(viewer)
        , mTargetFrameRate(120.0)
        , mLastWallpaperChangeTime(0.0)
        , mLastRenderTime(0.0)
        , mLoadingOnTime(0.0)
        , mImportantLabel(false)
        , mNestedLoadingCount(0)
        , mProgress(0)
        , mShowWallpaper(true)
    {
        getWidget(mLoadingText, "LoadingText");
        getWidget(mProgressBar, "ProgressBar");
        getWidget(mLoadingBox, "LoadingBox");
        getWidget(mSceneImage, "Scene");
        getWidget(mSplashImage, "Splash");

        mProgressBar->setScrollViewPage(1);

        findSplashScreens();
    }

    LoadingScreen::~LoadingScreen() {}

    void LoadingScreen::findSplashScreens()
    {
        auto isSupportedExtension = [](const std::string_view& ext) {
            static const std::array<std::string, 7> supportedExtensions{ { "tga", "dds", "ktx", "png", "bmp", "jpeg",
                "jpg" } };
            return !ext.empty()
                && std::find(supportedExtensions.begin(), supportedExtensions.end(), ext) != supportedExtensions.end();
        };

        constexpr VFS::Path::NormalizedView splash("splash/");
        for (const auto& name : mResourceSystem->getVFS()->getRecursiveDirectoryIterator(splash))
        {
            if (isSupportedExtension(Misc::getFileExtension(name)))
                mSplashScreens.push_back(name);
        }
        if (mSplashScreens.empty())
            Log(Debug::Warning) << "Warning: no splash screens found!";
    }

    void LoadingScreen::setLabel(const std::string& label, bool important)
    {
        mImportantLabel = important;

        mLoadingText->setCaptionWithReplacing(label);
        int padding = mLoadingBox->getWidth() - mLoadingText->getWidth();
        MyGUI::IntSize size(mLoadingText->getTextSize().width + padding, mLoadingBox->getHeight());
        size.width = std::max(300, size.width);
        mLoadingBox->setSize(size);

        if (MWBase::Environment::get().getWindowManager()->getMessagesCount() > 0)
            mLoadingBox->setPosition(mMainWidget->getWidth() / 2 - mLoadingBox->getWidth() / 2,
                mMainWidget->getHeight() / 2 - mLoadingBox->getHeight() / 2);
        else
            mLoadingBox->setPosition(mMainWidget->getWidth() / 2 - mLoadingBox->getWidth() / 2,
                mMainWidget->getHeight() - mLoadingBox->getHeight() - 8);
    }

    void LoadingScreen::setVisible(bool visible)
    {
        WindowBase::setVisible(visible);
        mSplashImage->setVisible(visible);
        mSceneImage->setVisible(visible);
    }

    double LoadingScreen::getTargetFrameRate() const
    {
        double frameRateLimit = MWBase::Environment::get().getFrameRateLimit();
        if (frameRateLimit > 0)
            return std::min(frameRateLimit, mTargetFrameRate);
        else
            return mTargetFrameRate;
    }

    // GL 3.0 token, and <GL/gl.h> on Windows stops at 1.1. Spelled out for the same reason
    // components/remixrt/glinterop.cpp spells out its framebuffer tokens.
    constexpr GLenum kReadFramebufferBinding = 0x8CAA;

    class CopyFramebufferToTextureCallback : public osg::Camera::DrawCallback
    {
    public:
        CopyFramebufferToTextureCallback(osg::Texture2D* texture)
            : mOneshot(true)
            , mTexture(texture)
        {
        }

        void operator()(osg::RenderInfo& renderInfo) const override
        {
            const osg::Viewport* viewPort = renderInfo.getCurrentCamera()->getViewport();
            int w = static_cast<int>(viewPort->width());
            int h = static_cast<int>(viewPort->height());

            // What this copy actually reads, reported once per loading screen.
            //
            // This is the background of every in-game loading screen: a snapshot of the frame that was on
            // screen when the load began. When Remix presents to its own window OpenMW's swap is disabled,
            // so the buffer being sampled here is one nothing else consumes -- which is a specific reason
            // it could come back empty, and an empty background is exactly what a black loading screen is.
            // Sampling it is the only way to tell that apart from the copy working and something later
            // discarding it.
            //
            // Four texels at the centre rather than the whole surface: this is a GPU-to-CPU read inside a
            // draw callback, so it stalls the pipeline, and mOneshot bounds it to one frame per screen.
            // The centre because the corners of a Morrowind frame are often legitimately black.
            if (mOneshot && RemixRT::diagEnabled(RemixRT::Diag::LoadingScreen))
            {
                GLint readBinding = 0;
                glGetIntegerv(kReadFramebufferBinding, &readBinding);

                std::array<unsigned char, 4 * 4 * 4> texels{};
                if (w >= 2 && h >= 2)
                    glReadPixels(w / 2 - 2, h / 2 - 2, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, texels.data());

                unsigned int brightest = 0;
                for (std::size_t i = 0; i < texels.size(); i += 4)
                    for (std::size_t c = 0; c < 3; ++c)
                        brightest = std::max(brightest, static_cast<unsigned int>(texels[i + c]));

                Log(Debug::Info) << "Loading screen background: copying " << w << "x" << h
                                 << " from read framebuffer " << readBinding
                                 << "; brightest of 16 sampled centre texels " << brightest << " of 255"
                                 << (brightest == 0
                                            ? " -- the source is black, so the background is empty before"
                                              " the copy rather than after it"
                                            : " -- the source has content");
            }

            mTexture->copyTexImage2D(*renderInfo.getState(), 0, 0, w, h);

            mOneshot = false;
        }

        void reset() { mOneshot = true; }

    private:
        mutable bool mOneshot;
        osg::ref_ptr<osg::Texture2D> mTexture;
    };

    class DontComputeBoundCallback : public osg::Node::ComputeBoundingSphereCallback
    {
    public:
        osg::BoundingSphere computeBound(const osg::Node&) const override { return osg::BoundingSphere(); }
    };

    void LoadingScreen::loadingOn()
    {
        // Early-out if already on
        if (mNestedLoadingCount++ > 0 && mMainWidget->getVisible())
            return;

        mLoadingOnTime = mTimer.time_m();

        // Assign dummy bounding sphere callback to avoid the bounding sphere of the entire scene being recomputed after
        // each frame of loading We are already using node masks to avoid the scene from being updated/rendered, but
        // node masks don't work for computeBound()
        mViewer->getSceneData()->setComputeBoundingSphereCallback(new DontComputeBoundCallback);

        if (const osgUtil::IncrementalCompileOperation* ico = mViewer->getIncrementalCompileOperation())
        {
            mOldIcoMin = ico->getMinimumTimeAvailableForGLCompileAndDeletePerFrame();
            mOldIcoMax = ico->getMaximumNumOfObjectsToCompilePerFrame();
        }

        setVisible(true);

        mShowWallpaper = MWBase::Environment::get().getStateManager()->getState() == MWBase::StateManager::State_NoGame;

        if (mShowWallpaper)
        {
            changeWallpaper();
        }

        // Which of the two backgrounds this screen will use, and the layout it will use it at.
        //
        // The two are entirely different mechanisms and only one of them was ever in question: wallpaper
        // mode puts a splash image from splash/ into a widget, while scene mode shows a texture copied
        // from the framebuffer. The selector is the game state, not the kind of load, so a save loaded
        // from the main menu takes wallpaper mode and every cell load afterwards takes scene mode --
        // which is why one loading screen per session can look right while the rest do not.
        //
        // The canvas size and the box rect are here because a widget laid out against a canvas of one size
        // and composited into an image of another lands in the wrong place at the wrong scale, and telling
        // that apart from a missing background needs both numbers rather than a screenshot.
        if (RemixRT::diagEnabled(RemixRT::Diag::LoadingScreen))
        {
            Log(Debug::Info) << "Loading screen on: " << (mShowWallpaper ? "wallpaper" : "scene") << " mode"
                             << " (state " << static_cast<int>(MWBase::Environment::get()
                                                                   .getStateManager()
                                                                   ->getState())
                             << ", " << mSplashScreens.size() << " splash images known)"
                             << "; canvas " << mMainWidget->getWidth() << "x" << mMainWidget->getHeight()
                             << ", loading box " << mLoadingBox->getWidth() << "x"
                             << mLoadingBox->getHeight() << " at " << mLoadingBox->getLeft() << ","
                             << mLoadingBox->getTop() << "; splash widget "
                             << (mSplashImage->getVisible() ? "visible" : "hidden") << ", scene widget "
                             << (mSceneImage->getVisible() ? "visible" : "hidden");
        }

        MWBase::Environment::get().getWindowManager()->pushGuiMode(mShowWallpaper ? GM_LoadingWallpaper : GM_Loading);
    }

    void LoadingScreen::loadingOff()
    {
        if (--mNestedLoadingCount > 0)
            return;

        if (mLastRenderTime < mLoadingOnTime)
        {
            // the loading was so fast that we didn't show loading screen at all
            // we may still want to show the label if the caller requested it
            if (mImportantLabel)
            {
                MWBase::Environment::get().getWindowManager()->messageBox(mLoadingText->getCaption());
                mImportantLabel = false;
            }
        }
        else
            mImportantLabel = false; // label was already shown on loading screen

        mViewer->getSceneData()->setComputeBoundingSphereCallback(nullptr);
        mViewer->getSceneData()->dirtyBound();

        setVisible(false);

        if (osgUtil::IncrementalCompileOperation* ico = mViewer->getIncrementalCompileOperation())
        {
            ico->setMinimumTimeAvailableForGLCompileAndDeletePerFrame(mOldIcoMin);
            ico->setMaximumNumOfObjectsToCompilePerFrame(mOldIcoMax);
        }

        MWBase::Environment::get().getWindowManager()->removeGuiMode(GM_Loading);
        MWBase::Environment::get().getWindowManager()->removeGuiMode(GM_LoadingWallpaper);
    }

    void LoadingScreen::changeWallpaper()
    {
        if (!mSplashScreens.empty())
        {
            std::string const& randomSplash = mSplashScreens.at(Misc::Rng::rollDice(mSplashScreens.size()));

            // TODO: add option (filename pattern?) to use image aspect ratio instead of 4:3
            // we can't do this by default, because the Morrowind splash screens are 1024x1024, but should be displayed
            // as 4:3
            mSplashImage->setVisible(true);
            mSplashImage->setBackgroundImage(randomSplash, true, Settings::gui().mStretchMenuBackground);
        }
        mSceneImage->setBackgroundImage({});
        mSceneImage->setVisible(false);
    }

    void LoadingScreen::setProgressRange(size_t range)
    {
        mProgressBar->setScrollRange(range + 1);
        mProgressBar->setScrollPosition(0);
        mProgressBar->setTrackSize(0);
        mProgress = 0;
    }

    void LoadingScreen::setProgress(size_t value)
    {
        // skip expensive update if there isn't enough visible progress
        if (mProgressBar->getWidth() <= 0
            || value - mProgress < mProgressBar->getScrollRange() / mProgressBar->getWidth())
            return;
        value = std::min(value, mProgressBar->getScrollRange() - 1);
        mProgress = value;
        mProgressBar->setScrollPosition(0);
        mProgressBar->setTrackSize(
            static_cast<int>(value / (float)(mProgressBar->getScrollRange()) * mProgressBar->getLineSize()));
        draw();
    }

    void LoadingScreen::increaseProgress(size_t increase)
    {
        mProgressBar->setScrollPosition(0);
        size_t value = mProgress + increase;
        value = std::min(value, mProgressBar->getScrollRange() - 1);
        mProgress = value;
        mProgressBar->setTrackSize(
            static_cast<int>(value / (float)(mProgressBar->getScrollRange()) * mProgressBar->getLineSize()));
        draw();
    }

    bool LoadingScreen::needToDrawLoadingScreen()
    {
        if (mTimer.time_m() <= mLastRenderTime + (1.0 / getTargetFrameRate()) * 1000.0)
            return false;

        // the minimal delay before a loading screen shows
        constexpr float initialDelay = 0.05f;

        bool alreadyShown = (mLastRenderTime > mLoadingOnTime);
        double diff = (mTimer.time_m() - mLoadingOnTime);

        if (!alreadyShown)
        {
            // bump the delay by the current progress - i.e. if during the initial delay the loading
            // has almost finished, no point showing the loading screen now
            diff -= mProgress / static_cast<float>(mProgressBar->getScrollRange()) * 100.f;
        }

        if (!mShowWallpaper && diff < initialDelay * 1000)
            return false;
        return true;
    }

    void LoadingScreen::setupCopyFramebufferToTextureCallback()
    {
        // Copy the current framebuffer onto a texture and display that texture as the background image
        // Note, we could also set the camera to disable clearing and have the background image transparent,
        // but then we get shaking effects on buffer swaps.

        if (!mTexture)
        {
            mTexture = new osg::Texture2D;
            mTexture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
            mTexture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
            mTexture->setInternalFormat(GL_RGB);
            mTexture->setResizeNonPowerOfTwoHint(false);
        }

        if (!mGuiTexture.get())
        {
            mGuiTexture = std::make_unique<MyGUIPlatform::OSGTexture>(mTexture);
        }

        if (!mCopyFramebufferToTextureCallback)
        {
            mCopyFramebufferToTextureCallback = new CopyFramebufferToTextureCallback(mTexture);
        }

        mViewer->getCamera()->removeInitialDrawCallback(mCopyFramebufferToTextureCallback);
        mViewer->getCamera()->addInitialDrawCallback(mCopyFramebufferToTextureCallback);
        mCopyFramebufferToTextureCallback->reset();

        mSplashImage->setBackgroundImage({});
        mSplashImage->setVisible(false);

        mSceneImage->setRenderItemTexture(mGuiTexture.get());
        // The widget is Y-down, the RTT image is Y-up, so this UV is inverted
        mSceneImage->getSubWidgetMain()->_setUVSet(MyGUI::FloatRect(0.f, 1.f, 1.f, 0.f));
        mSceneImage->setVisible(true);
    }

    void LoadingScreen::draw()
    {
        if (!needToDrawLoadingScreen())
            return;

        if (mShowWallpaper && mTimer.time_m() > mLastWallpaperChangeTime + 5000 * 1)
        {
            mLastWallpaperChangeTime = mTimer.time_m();
            changeWallpaper();
        }

        if (!mShowWallpaper && mLastRenderTime < mLoadingOnTime)
        {
            // With Remix presenting, the background this would build is already on screen and better.
            //
            // Scene mode exists to keep the world visible behind the progress bar, and it does that by
            // snapshotting the framebuffer. That is the right mechanism when OpenMW's own frame is what
            // reaches the screen. It is the wrong one here: Remix presents its retained scene for every
            // nested present, so the path-traced world is already behind the overlay -- and the buffer
            // being snapshotted is OpenMW's raster frame, which nothing presents in this mode and which
            // measured 12 of 255 at its brightest. So the snapshot was not failing to capture the world.
            // It was capturing an unlit one and painting it over a lit one.
            //
            // Hiding both background widgets leaves the overlay transparent everywhere the interface has
            // not drawn, which is exactly what lets Remix's frame through. The result is the look scene
            // mode was after in the first place, arrived at by not doing the work.
            //
            // Explicitly rather than by omission: setVisible(true) in loadingOn shows the scene widget,
            // and skipping the setup below would leave it showing whichever stale texture the last copy
            // left behind. That is how most of a session's loading screens came to show one frozen dark
            // frame -- the copy only runs on a screen's first draw, so seven of eight cell loads in a
            // measured run reused a single snapshot taken minutes earlier.
            if (RemixRT::nestedFramePresenterInstalled())
            {
                mSceneImage->setBackgroundImage({});
                mSceneImage->setVisible(false);
                mSplashImage->setBackgroundImage({});
                mSplashImage->setVisible(false);
            }
            else
            {
                setupCopyFramebufferToTextureCallback();
            }
        }

        MWBase::Environment::get().getInputManager()->update(0, true, true);

        osg::Stats* const stats = mViewer->getViewerStats();
        const unsigned frameNumber = mViewer->getFrameStamp()->getFrameNumber();

        stats->setAttribute(frameNumber, "Loading", 1);

        mResourceSystem->reportStats(frameNumber, stats);
        if (osgUtil::IncrementalCompileOperation* ico = mViewer->getIncrementalCompileOperation())
        {
            ico->setMinimumTimeAvailableForGLCompileAndDeletePerFrame(1.f / getTargetFrameRate());
            ico->setMaximumNumOfObjectsToCompilePerFrame(1000);
        }

        // at the time this function is called we are in the middle of a frame,
        // so out of order calls are necessary to get a correct frameNumber for the next frame.
        // refer to the advance() and frame() order in Engine::go()
        //
        // Timed separately from the present that follows, because the two answer different questions and
        // the answer decides what to do about a ten-second cell load. The incremental compile operation
        // runs here -- setMaximumNumOfObjectsToCompilePerFrame(1000) above is what makes a load compile its
        // assets rather than hitch afterwards -- so this side of the line is work that has to happen
        // whether or not anything is shown. The present is the part that only exists to show it.
        {
            const auto beforeTraversal = std::chrono::steady_clock::now();
            mViewer->eventTraversal();
            mViewer->updateTraversal();
            mViewer->renderingTraversals();

            RemixRT::NestedFrameStats& nestedStats = RemixRT::nestedFrameStats();
            const double traversalMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - beforeTraversal)
                                           .count();
            nestedStats.mTraversalMs += traversalMs;
            nestedStats.mWorstTraversalMs = std::max(nestedStats.mWorstTraversalMs, traversalMs);
        }

        // This loop drives the viewer itself, so nothing else will present the frame it just drew. Without
        // this the loading screen renders correctly into Remix's overlay image and is never shown.
        RemixRT::presentNestedFrame();
        mViewer->advance(mViewer->getFrameStamp()->getSimulationTime());

        mLastRenderTime = mTimer.time_m();
    }

}
