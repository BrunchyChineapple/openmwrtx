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
    class GuiOverlayTarget;
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
        // Imports the image OpenMW draws its interface into, for Remix to composite. Separate from
        // mRemixImport because it runs the other way round: that one imports what Remix produced, this one
        // imports what OpenMW is about to produce.
        osg::ref_ptr<RemixRT::ImportOperation> mRemixOverlayImport;
        // Binds that image around the interface camera's drawing. Held so the engine can see whether the
        // framebuffer ever came up and stop asking Remix to composite an image nothing is drawing into.
        osg::ref_ptr<RemixRT::GuiOverlayTarget> mRemixOverlayTarget;
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
        /// Wall-clock cost of OpenMW's own simulation phases, in milliseconds.
        ///
        /// All six run at the top of frame(), ahead of every Remix stamp, and together they were the entire
        /// unmeasured remainder. An eleven-second frame reported 0.0098% accounted with every instrumented
        /// span sub-millisecond, which is not a small error in the accounting -- it means the freeze was
        /// somewhere nothing was looking, and this is what was left.
        ///
        /// mWorld is the one to watch: cell loading runs synchronously inside it, which is what the
        /// multi-second loading bars are. mMechanics is actor AI and combat, the other candidate that
        /// scales with how much of the world is live rather than with how much of it is drawn.
        ///
        /// OpenMW already times these into its own ScopedProfile counters, but those only reach the F3
        /// overlay -- they are averaged for display and never logged, so they cannot answer "what did the
        /// one frame that took eleven seconds spend it on".
        struct SimulationPhases
        {
            double mInput = 0.0;
            double mSound = 0.0;
            /// Lua's queued changes from the previous frame being applied to the world, on this thread.
            /// Distinct from the Lua finish already measured, which is only the wait for the worker: this is
            /// the main thread doing the resulting work, and 206 loaded scripts is a lot of resulting work.
            double mLuaSync = 0.0;
            double mState = 0.0;
            double mMechanics = 0.0;
            double mPhysics = 0.0;
            /// Includes cell loading, which is synchronous. This is the prime suspect for the loading bars.
            double mWorld = 0.0;
            double mWindowManager = 0.0;

            double total() const
            {
                return mInput + mSound + mLuaSync + mState + mMechanics + mPhysics + mWorld
                    + mWindowManager;
            }
        };

        /// Accumulated cost of each stage of the Remix frame, in milliseconds, over the current reporting
        /// window. Reset whenever the window is reported.
        struct RemixTiming
        {
            unsigned int mFrames = 0;
            double mFrameMs = 0.0;
            /// Summed SimulationPhases::total(), so the mean accounting covers them too.
            double mPhasesTotalMs = 0.0;
            /// Frames presented from a loop that drives the viewer itself -- loading screens, modal message
            /// boxes, video playback -- and their total cost.
            ///
            /// Counted because each one is a full path-traced present. The loading screen redraws on every
            /// progress tick, gated only by 1/mTargetFrameRate, and mTargetFrameRate is 120 unless a
            /// framerate limit is set -- so the gate is 8.3 ms, which is shorter than one present at this
            /// resolution. A gate that never blocks means every tick of the progress bar buys a whole
            /// path-traced frame, and these do not appear in any per-frame measurement because
            /// Engine::frame is not the thing running.
            ///
            /// If a multi-second cell load turns out to be mostly this, the load is not slow: the time is
            /// going on drawing the progress bar that is reporting it.
            unsigned int mNestedPresents = 0;
            double mNestedPresentMs = 0.0;
            double mSubmitMs = 0.0;
            double mPresentMs = 0.0;
            double mCopyMs = 0.0;
            double mReadbackQueueMs = 0.0;
            double mReadbackLockMs = 0.0;
            double mReadbackCopyMs = 0.0;
            double mWorstReadbackMs = 0.0;
            double mOsgUpdateMs = 0.0;
            double mOsgRenderMs = 0.0;
            /// The single slowest frame in the window, broken down the same way as the mean.
            ///
            /// A mean cannot describe a freeze. Measured frames run 27-47 ms on average while individual
            /// frames reach four seconds and beyond, and averaging one of those over 300 frames adds about
            /// 13 ms -- invisible. Every component of the worst frame is kept so the freeze can be
            /// attributed to a component rather than to whatever happened to be logged next to it, which is
            /// how an opacity micromap message got mistaken for the cause.
            double mWorstFrameMs = 0.0;
            double mWorstFrameSubmitMs = 0.0;
            double mWorstFramePresentMs = 0.0;
            double mWorstFrameCopyMs = 0.0;
            double mWorstFrameReadbackMs = 0.0;
            double mWorstFrameOsgUpdateMs = 0.0;
            double mWorstFrameOsgRenderMs = 0.0;
            unsigned int mWorstFrameIndex = 0;
            /// Worst submit and worst present of the window, tracked independently of the worst frame.
            ///
            /// A spike in one does not have to land on the same frame as a spike in the other, and taking
            /// only the worst frame's breakdown would hide whichever did not coincide with it.
            double mWorstSubmitMs = 0.0;
            double mWorstPresentMs = 0.0;
            /// Worst OSG update and render traversal of the window.
            ///
            /// renderingTraversals runs after every timing point this frame takes, so its cost is only
            /// visible one frame later and can never appear on the same sample as the frame it belongs to.
            /// Measured freezes reported under 1% accounted for exactly that reason. A window maximum does
            /// not care which sample it is attributed to, so it reveals a four-second traversal either way.
            /// renderingTraversals is also where the terrain composite maps are rendered and read back and
            /// where OSG's pager compiles newly loaded assets, which makes it the largest unmeasured span
            /// in the frame.
            double mWorstOsgUpdateMs = 0.0;
            double mWorstOsgRenderMs = 0.0;
            /// Worst of the two spans nothing was watching.
            ///
            /// postCopy covers everything between the Remix copy and the draw traversal: input, the GUI,
            /// uiState, readOutputPixels and probeOutputNonBlack, the last two of which read GPU memory back
            /// and therefore block. luaFinish covers joining the Lua worker.
            ///
            /// These two were once described as closing the accounting, on the reasoning that every line of
            /// frame() was then inside some measured span. That was wrong, and believing it cost several
            /// runs: the six simulation phases at the top of frame() -- input, sound, state, mechanics,
            /// world and the window manager -- sit ahead of the first stamp and were in no span at all. See
            /// SimulationPhases.
            double mWorstPostCopyMs = 0.0;
            double mWorstLuaFinishMs = 0.0;
            /// The worst frame's own simulation breakdown, and the window maxima tracked independently of
            /// it for the same reason submit and present are.
            SimulationPhases mWorstFramePhases;
            SimulationPhases mWorstPhases;

            /// Frames over the stall threshold. A maximum alone cannot tell one four-second frame from
            /// forty hundred-millisecond ones, and every stall measured so far has been a single
            /// occurrence -- which is what points away from the phase it landed in.
            unsigned int mStalls = 0;

            /// What the process was doing across the worst frame: counters differenced over that frame
            /// alone, plus the queue depths that say whether a worker thread was mid-load.
            unsigned long long mWorstFramePageFaults = 0;
            unsigned long long mWorstFrameReadOps = 0;
            unsigned long long mWorstFrameReadBytes = 0;
            unsigned long long mWorstFrameWorkingSetMiB = 0;
            unsigned long long mWorstFrameAvailableRamMiB = 0;
            unsigned int mWorstFrameWorkQueue = 0;
            unsigned int mWorstFrameWorkThreads = 0;
            unsigned int mWorstFrameUnrefQueue = 0;
        };
        /// Last frame's cost of the two spans above. Recorded after the timing report runs, so like the OSG
        /// traversals they are picked up one frame late -- which the window maxima do not care about.
        double mRemixPostCopyMs = 0.0;
        double mRemixLuaFinishMs = 0.0;
        /// When the Remix copy finished. Held on the object because the span from there to the draw
        /// traversal crosses out of the scope the local timestamp lives in.
        std::chrono::steady_clock::time_point mRemixAfterCopyStamp = std::chrono::steady_clock::now();
        /// Wall clock at the previous frame's timing point.
        ///
        /// OSG's frametime is clamped -- OpenMW limits it so a long frame cannot destabilise physics -- and
        /// the clamp is 200 ms. Measured against it every freeze reported as exactly 200 ms with its
        /// components accounting for under 10%, which says the ceiling was hit, not where the time went. A
        /// steady_clock delta across the same point each frame is the only honest total.
        std::chrono::steady_clock::time_point mRemixFrameStamp = std::chrono::steady_clock::now();
        RemixTiming mRemixTiming;
        /// Last frame's cost of OpenMW's own update and rendering traversals. Both are measured after the
        /// Remix pump has already reported, so the accumulator picks them up one frame late -- irrelevant
        /// over a 300-frame window, and the alternative is moving the report to the end of the frame where
        /// it is further from the thing it describes.
        double mRemixOsgUpdateMs = 0.0;
        double mRemixRenderMs = 0.0;
        /// This frame's simulation phase costs. Filled at the top of frame(), read by the timing report
        /// further down the same frame, so unlike the traversals above these are never a frame late.
        SimulationPhases mRemixPhases;

        /// State of the current burst of nested frames -- one cell load, message box or video.
        ///
        /// Held on the object rather than as statics in the presenter because the ration has to be readable
        /// and resettable from outside it, and because a burst is a property of the engine's run rather
        /// than of the lambda.
        bool mRemixNestedSequenceActive = false;
        std::chrono::steady_clock::time_point mRemixNestedSequenceStart
            = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point mRemixNestedLastCall = std::chrono::steady_clock::now();
        /// Wall time already spent presenting this burst, which is what the share is measured against.
        double mRemixNestedSequenceSpentMs = 0.0;

        /// Previous frame's process counters, so the worst frame can be differenced rather than reported
        /// as a session total that says nothing about one frame.
        unsigned long long mRemixPrevPageFaults = 0;
        unsigned long long mRemixPrevReadOps = 0;
        unsigned long long mRemixPrevReadBytes = 0;
        /// False until the first sample has been taken, so the first frame is not charged with everything
        /// the process did before it.
        bool mRemixCountersPrimed = false;

        /// Whether the most recent copy into the shared image succeeded.
        ///
        /// A member rather than a local because on the synchronised path the copy is deferred past the draw
        /// traversal, so within a frame the composite samples what the *previous* copy left. That makes this
        /// the honest answer to "is there anything to composite", and it starts false so the first frame
        /// shows OpenMW's own image rather than an empty one.
        bool mRemixCopyOk = false;

        /// Whether a copyComplete signal has been submitted that the composite has not yet consumed.
        ///
        /// The pair is binary, so signalling one that is already signalled is invalid and unrecoverable. This
        /// enforces at most one in flight: while it is set, the copy is issued through the unsynchronised
        /// entry point so the image still updates without adding a second signal.
        bool mRemixSignalOutstanding = false;
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
