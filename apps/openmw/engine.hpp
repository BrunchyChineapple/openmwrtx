#ifndef ENGINE_H
#define ENGINE_H

#include <filesystem>

#include <components/compiler/extensions.hpp>
#include <components/debug/debuglog.hpp>
#include <components/esm/refid.hpp>
#include <components/files/collections.hpp>
#include <components/settings/settings.hpp>
#include <components/translation/translation.hpp>

#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include "mwbase/environment.hpp"

namespace Resource
{
    class ResourceSystem;
}

namespace SceneUtil
{
    class WorkQueue;
    class AsyncScreenCaptureOperation;
    class UnrefQueue;
}

namespace VFS
{
    class Manager;
}

namespace Compiler
{
    class Context;
}

namespace MWLua
{
    class LuaManager;
    class Worker;
}

namespace Stereo
{
    class Manager;
}

namespace Files
{
    struct ConfigurationManager;
}

namespace osgViewer
{
    class ScreenCaptureHandler;
}

namespace SceneUtil
{
    class SelectDepthFormatOperation;

    namespace Color
    {
        class SelectColorFormatOperation;
    }
}

namespace MWState
{
    class StateManager;
}

namespace MWGui
{
    class WindowManager;
}

namespace MWInput
{
    class InputManager;
}

namespace MWSound
{
    class SoundManager;
}

namespace MWWorld
{
    class World;
}

namespace MWScript
{
    class ScriptManager;
}

namespace MWMechanics
{
    class MechanicsManager;
}

namespace MWDialogue
{
    class DialogueManager;
}

namespace MWDialogue
{
    class Journal;
}

namespace L10n
{
    class Manager;
}

namespace RemixRT
{
    class Runtime;
    class ImportOperation;
    class CompositeCallback;
}

namespace MWRender
{
    class RemixScene;
    class RemixSky;
}

struct SDL_Window;

namespace OMW
{
    /// \brief Main engine class, that brings together all the components of OpenMW
    class Engine
    {
        SDL_Window* mWindow;
        // Declared right after mWindow so destruction order releases Remix before the window it is
        // attached to. Only constructed when the backend is requested; null otherwise.
        std::unique_ptr<RemixRT::Runtime> mRemix;
        // Runs once on the GL thread to import Remix's shared image. Held so the result can be read
        // after it has run.
        osg::ref_ptr<RemixRT::ImportOperation> mRemixImport;
        // Draws the imported image over OpenMW's frame. Held so compositing can be toggled.
        osg::ref_ptr<RemixRT::CompositeCallback> mRemixComposite;
        // Walks OpenMW's scene graph and hands the geometry to Remix. Declared after mRemix so it is
        // destroyed first: its destructor releases meshes through the runtime.
        std::unique_ptr<MWRender::RemixScene> mRemixScene;
        // Pushes OpenMW's weather into the runtime's atmosphere options. Holds no runtime resources of its
        // own -- only the last values it wrote, so it can avoid resending them -- but declared here beside
        // mRemixScene so the ordering rule above is obvious rather than accidental.
        std::unique_ptr<MWRender::RemixSky> mRemixSky;
        // Scratch for the readback display path. A member so the allocation is reused across frames
        // rather than churning a multi-megabyte buffer every frame.
        std::vector<unsigned char> mRemixReadback;
        /// Accumulated cost of each stage of the Remix frame, in milliseconds, over the current reporting
        /// window. Reset whenever the window is reported.
        struct RemixTiming
        {
            unsigned int mFrames = 0;
            double mFrameMs = 0.0;
            double mSubmitMs = 0.0;
            double mPresentMs = 0.0;
            double mCopyMs = 0.0;
            double mReadbackQueueMs = 0.0;
            double mReadbackLockMs = 0.0;
            double mReadbackCopyMs = 0.0;
            double mWorstReadbackMs = 0.0;
            double mOsgUpdateMs = 0.0;
            double mOsgRenderMs = 0.0;
        };
        RemixTiming mRemixTiming;
        /// Last frame's cost of OpenMW's own update and rendering traversals. Both are measured after the
        /// Remix pump has already reported, so the accumulator picks them up one frame late -- irrelevant
        /// over a 300-frame window, and the alternative is moving the report to the end of the frame where
        /// it is further from the thing it describes.
        double mRemixOsgUpdateMs = 0.0;
        double mRemixRenderMs = 0.0;
        // Whether the Remix developer menu currently owns the mouse. Tracked so the pointer is released
        // on transitions only; doing it every frame re-warps the cursor and pins it again.
        bool mRemixMenuHasMouse = false;
        // Pointer state to put back when the menu closes, captured when it opens.
        bool mRemixSavedMouseRelative = false;
        bool mRemixSavedMouseGrab = false;
        std::unique_ptr<VFS::Manager> mVFS;
        std::unique_ptr<Resource::ResourceSystem> mResourceSystem;
        osg::ref_ptr<SceneUtil::WorkQueue> mWorkQueue;
        std::unique_ptr<SceneUtil::UnrefQueue> mUnrefQueue;
        std::unique_ptr<MWWorld::World> mWorld;
        std::unique_ptr<MWSound::SoundManager> mSoundManager;
        std::unique_ptr<MWScript::ScriptManager> mScriptManager;
        std::unique_ptr<MWGui::WindowManager> mWindowManager;
        std::unique_ptr<MWMechanics::MechanicsManager> mMechanicsManager;
        std::unique_ptr<MWDialogue::DialogueManager> mDialogueManager;
        std::unique_ptr<MWDialogue::Journal> mJournal;
        std::unique_ptr<MWInput::InputManager> mInputManager;
        std::unique_ptr<MWState::StateManager> mStateManager;
        std::unique_ptr<MWLua::LuaManager> mLuaManager;
        std::unique_ptr<MWLua::Worker> mLuaWorker;
        std::unique_ptr<L10n::Manager> mL10nManager;
        MWBase::Environment mEnvironment;
        ToUTF8::FromType mEncoding;
        std::unique_ptr<ToUTF8::Utf8Encoder> mEncoder;
        Files::PathContainer mDataDirs;
        std::vector<std::string> mArchives;
        std::filesystem::path mResDir;
        osg::ref_ptr<osgViewer::Viewer> mViewer;
        osg::ref_ptr<osgViewer::ScreenCaptureHandler> mScreenCaptureHandler;
        osg::ref_ptr<SceneUtil::AsyncScreenCaptureOperation> mScreenCaptureOperation;
        osg::ref_ptr<SceneUtil::SelectDepthFormatOperation> mSelectDepthFormatOperation;
        osg::ref_ptr<SceneUtil::Color::SelectColorFormatOperation> mSelectColorFormatOperation;
        std::string mCellName;
        std::vector<std::string> mContentFiles;
        std::vector<std::string> mGroundcoverFiles;

        std::unique_ptr<Stereo::Manager> mStereoManager;

        bool mSkipMenu;
        bool mUseSound;
        bool mCompileAll;
        bool mCompileAllDialogue;
        int mWarningsMode;
        std::string mFocusName;
        bool mScriptConsoleMode;
        std::filesystem::path mStartupScript;
        int mActivationDistanceOverride;
        std::filesystem::path mSaveGameFile;
        // Grab mouse?
        bool mGrab;

        bool mExportFonts;
        unsigned int mRandomSeed;
        Debug::Level mMaxRecastLogLevel = Debug::Error;

        Compiler::Extensions mExtensions;
        std::unique_ptr<Compiler::Context> mScriptContext;

        Files::Collections mFileCollections;
        Translation::Storage mTranslationDataStorage;
        bool mNewGame;

        Files::ConfigurationManager& mCfgMgr;
        int mGlMaxTextureImageUnits;

        // not implemented
        Engine(const Engine&);
        Engine& operator=(const Engine&);

        void executeLocalScripts();

        bool frame(unsigned frameNumber, float dt);

        /// Prepare engine for game play
        void prepareEngine();

        void createWindow();
        void setWindowIcon();

    public:
        Engine(Files::ConfigurationManager& configurationManager);
        virtual ~Engine();

        /// Set data dirs
        void setDataDirs(const Files::PathContainer& dataDirs);

        /// Add BSA archive
        void addArchive(const std::string& archive);

        /// Set resource dir
        void setResourceDir(const std::filesystem::path& parResDir);

        /// Set start cell name
        void setCell(const std::string& cellName);

        /**
         * @brief addContentFile - Adds content file (ie. esm/esp, or omwgame/omwaddon) to the content files container.
         * @param file - filename (extension is required)
         */
        void addContentFile(const std::string& file);
        void addGroundcoverFile(const std::string& file);

        /// Disable or enable all sounds
        void setSoundUsage(bool soundUsage);

        /// Skip main menu and go directly into the game
        ///
        /// \param newGame Start a new game instead off dumping the player into the game
        /// (ignored if !skipMenu).
        void setSkipMenu(bool skipMenu, bool newGame);

        void setGrabMouse(bool grab) { mGrab = grab; }

        /// Initialise and enter main loop.
        void go();

        /// Compile all scripts (excludign dialogue scripts) at startup?
        void setCompileAll(bool all);

        /// Compile all dialogue scripts at startup?
        void setCompileAllDialogue(bool all);

        /// Font encoding
        void setEncoding(const ToUTF8::FromType& encoding);

        /// Enable console-only script functionality
        void setScriptConsoleMode(bool enabled);

        /// Set path for a script that is run on startup in the console.
        void setStartupScript(const std::filesystem::path& path);

        /// Override the game setting specified activation distance.
        void setActivationDistanceOverride(int distance);

        void setWarningsMode(int mode);

        void enableFontExport(bool exportFonts);

        /// Set the save game file to load after initialising the engine.
        void setSaveGameFile(const std::filesystem::path& savegame);

        void setRandomSeed(unsigned int seed);

        void setRecastMaxLogLevel(Debug::Level value) { mMaxRecastLogLevel = value; }
    };
}

#endif /* ENGINE_H */
