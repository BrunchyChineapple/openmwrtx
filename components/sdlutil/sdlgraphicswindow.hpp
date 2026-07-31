#ifndef OPENMW_COMPONENTS_SDLUTIL_SDLGRAPHICSWINDOW_H
#define OPENMW_COMPONENTS_SDLUTIL_SDLGRAPHICSWINDOW_H

#include <SDL_video.h>

#include <osgViewer/GraphicsWindow>

#include "vsyncmode.hpp"

namespace SDLUtil
{

    class GraphicsWindowSDL2 : public osgViewer::GraphicsWindow
    {
        SDL_Window* mWindow;
        SDL_GLContext mContext;

        bool mValid;
        bool mRealized;
        bool mOwnsWindow;
        VSyncMode mVSyncMode;
        bool mSwapEnabled = true;

        void init();

        virtual ~GraphicsWindowSDL2();

    public:
        GraphicsWindowSDL2(osg::GraphicsContext::Traits* traits, VSyncMode vsyncMode);

        bool isSameKindAs(const Object* object) const override
        {
            return dynamic_cast<const GraphicsWindowSDL2*>(object) != nullptr;
        }
        const char* libraryName() const override { return "osgViewer"; }
        const char* className() const override { return "GraphicsWindowSDL2"; }

        bool valid() const override { return mValid; }

        /// Stop presenting this window's rendering, while continuing to render it.
        ///
        /// For when something other than this context owns what the window shows. RTX Remix path traces the
        /// scene into a child window covering the whole client area, and OpenMW's swap is a blit across that
        /// same area, so the two alternate and the child is overwritten as fast as it is presented -- what
        /// you see is OpenMW, and the raytraced image never survives a frame. WS_CLIPCHILDREN on the parent
        /// does not help: OpenGL wants that style in place when the pixel format is set, and the driver's
        /// presentation path does not consult it afterwards.
        ///
        /// Rendering deliberately continues. The scene submission to Remix rides on OpenMW's cull, and its
        /// cull is also what keeps distant NPCs animating, so skipping the draw would cost both. Only the
        /// present is dropped, which is the one part nothing needs.
        void setSwapEnabled(bool enabled);

        /** Realise the GraphicsContext.*/
        bool realizeImplementation() override;

        /** Return true if the graphics context has been realised and is ready to use.*/
        bool isRealizedImplementation() const override { return mRealized; }

        /** Close the graphics context.*/
        void closeImplementation() override;

        /** Make this graphics context current.*/
        bool makeCurrentImplementation() override;

        /** Release the graphics context.*/
        bool releaseContextImplementation() override;

        /** Swap the front and back buffers.*/
        void swapBuffersImplementation() override;

        /** Set sync-to-vblank. */
        void setSyncToVBlank(bool on) override;
        void setSyncToVBlank(VSyncMode mode);

        /** Set Window decoration.*/
        bool setWindowDecorationImplementation(bool flag) override;

        /** Raise specified window */
        void raiseWindow() override;

        /** Set the window's position and size.*/
        bool setWindowRectangleImplementation(int x, int y, int width, int height) override;

        /** Set the name of the window */
        void setWindowName(const std::string& name) override;

        /** Set mouse cursor to a specific shape.*/
        void setCursor(MouseCursor cursor) override;

        /** Get focus.*/
        void grabFocus() override {}

        /** Get focus on if the pointer is in this window.*/
        void grabFocusIfPointerInWindow() override {}

        /** WindowData is used to pass in the SDL2 window handle attached to the GraphicsContext::Traits structure. */
        struct WindowData : public osg::Referenced
        {
            WindowData(SDL_Window* window)
                : mWindow(window)
            {
            }

            SDL_Window* mWindow;
        };

    private:
        void setSwapInterval(VSyncMode mode);
    };

}

#endif /* OSGGRAPHICSWINDOW_H */
