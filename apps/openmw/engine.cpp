#include "engine.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <future>
#include <system_error>
#include <thread>


#include <osgDB/ReaderWriter>
#include <osgDB/Registry>
#include <osgViewer/ViewerEventHandlers>

#include <SDL.h>

#include <components/debug/debuglog.hpp>
#include <components/debug/gldebug.hpp>

#include <components/misc/rng.hpp>
#include <components/misc/strings/format.hpp>

#include <components/vfs/manager.hpp>
#include <components/vfs/registerarchives.hpp>

#include <components/sdlutil/imagetosurface.hpp>
#include <components/myguiplatform/myguirendermanager.hpp>
#include <components/sdlutil/sdlgraphicswindow.hpp>

#include <components/resource/resourcesystem.hpp>
#include <components/resource/scenemanager.hpp>
#include <components/resource/stats.hpp>

#include <components/compiler/extensions0.hpp>

#include <components/stereo/stereomanager.hpp>

#include <components/sceneutil/glextensions.hpp>
#include <components/sceneutil/workqueue.hpp>

#include <components/files/configurationmanager.hpp>

#include <components/version/version.hpp>

#include <components/l10n/manager.hpp>

#include <components/loadinglistener/asynclistener.hpp>
#include <components/loadinglistener/loadinglistener.hpp>

#include <components/misc/frameratelimiter.hpp>

#include <components/sceneutil/color.hpp>
#include <components/remixrt/glinterop.hpp>
#include <components/remixrt/runtime.hpp>

#include "mwrender/remixscene.hpp"
#include "mwrender/remixsky.hpp"
#include "mwrender/renderingmanager.hpp"
#include "mwworld/weather.hpp"
#include <components/sceneutil/depth.hpp>
#include <components/sceneutil/screencapture.hpp>
#include <components/sceneutil/unrefqueue.hpp>
#include <components/sceneutil/util.hpp>

#include <components/settings/shadermanager.hpp>
#include <components/settings/values.hpp>

#include "mwinput/inputmanagerimp.hpp"

#include "mwgui/windowmanagerimp.hpp"

#include "mwlua/luamanagerimp.hpp"
#include "mwlua/worker.hpp"

#include "mwscript/interpretercontext.hpp"
#include "mwscript/scriptmanagerimp.hpp"

#include "mwsound/constants.hpp"
#include "mwsound/soundmanagerimp.hpp"

#include "mwworld/class.hpp"
#include "mwworld/datetimemanager.hpp"
#include "mwworld/worldimp.hpp"

#ifdef _WIN32
// Last, and deliberately so. windows.h defines near and far as empty macros left over from the segmented
// memory model, and OpenMW has functions taking parameters with those names -- Stereo::Manager::updateSettings
// among them -- so including it any earlier turns unrelated headers into syntax errors. Everything above is
// already parsed by the time it arrives, and the two macros are dropped immediately for the code below.
//
// Psapi is pulled in with a pragma rather than a CMake change because this is the only thing in the
// executable that wants it, which keeps the dependency next to its single use.
// WIN32_LEAN_AND_MEAN and NOMINMAX are already set project-wide, so they are not repeated here.
#include <windows.h>

#include <dbghelp.h>
#include <psapi.h>

#undef near
#undef far

#ifdef _MSC_VER
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "dbghelp.lib")
#endif
#endif

namespace
{
    /// Records the wall-clock cost of the enclosing scope into \a slot, in milliseconds.
    ///
    /// Scoped rather than a stamp pair because the phases this measures are already wrapped in
    /// ScopedProfile blocks with early returns and continues inside them -- a hand-placed end stamp would
    /// be skipped on exactly the frames worth measuring.
    class PhaseTimer
    {
    public:
        explicit PhaseTimer(double& slot)
            : mSlot(slot)
            , mStart(std::chrono::steady_clock::now())
        {
        }

        ~PhaseTimer()
        {
            mSlot = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mStart)
                        .count();
        }

        PhaseTimer(const PhaseTimer&) = delete;
        PhaseTimer& operator=(const PhaseTimer&) = delete;

    private:
        double& mSlot;
        std::chrono::steady_clock::time_point mStart;
    };

    /// Process-wide counters that a per-phase timer cannot see.
    ///
    /// The multi-second stalls land on a different phase every time -- input 1146 ms, mechanics 4007 ms,
    /// lua sync 3406 ms, world 2703 ms -- and each happened exactly once in its window. One of them was
    /// SDL event pumping taking over a second, which is not work that subsystem is capable of doing. A
    /// stall that can surface anywhere is not owned by the code it lands in, so the useful question is what
    /// the process as a whole was doing while it happened.
    ///
    /// These separate the remaining candidates. A page-fault burst means the working set is being paged and
    /// the stall is the memory manager. Disk reads mean I/O reached the frame thread. Neither, with the
    /// runtime already reporting zero acceleration structures and zero shader compiles for the same spike,
    /// leaves a lock held by a worker thread -- and that is worth knowing before writing any more timers.
    struct ProcessCounters
    {
        unsigned long long mPageFaults = 0;
        unsigned long long mReadOps = 0;
        unsigned long long mReadBytes = 0;
        unsigned long long mWorkingSetMiB = 0;
        unsigned long long mAvailableRamMiB = 0;
    };

    ProcessCounters sampleProcessCounters()
    {
        ProcessCounters counters;

#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS memory{};
        memory.cb = sizeof(memory);
        if (::GetProcessMemoryInfo(::GetCurrentProcess(), &memory, sizeof(memory)))
        {
            counters.mPageFaults = memory.PageFaultCount;
            counters.mWorkingSetMiB = memory.WorkingSetSize >> 20;
        }

        IO_COUNTERS io{};
        if (::GetProcessIoCounters(::GetCurrentProcess(), &io))
        {
            counters.mReadOps = io.ReadOperationCount;
            counters.mReadBytes = io.ReadTransferCount;
        }

        MEMORYSTATUSEX system{};
        system.dwLength = sizeof(system);
        if (::GlobalMemoryStatusEx(&system))
            counters.mAvailableRamMiB = system.ullAvailPhys >> 20;
#endif

        return counters;
    }

#ifdef _WIN32
    /// Captures the frame thread's call stack when a frame overruns, so a stall names the function it is
    /// stuck in instead of the phase it happened to land on.
    ///
    /// Every stall measured so far has been a single occurrence, and each landed on a different phase --
    /// input 1146 ms, mechanics 4222 ms, lua sync 3406 ms, world 2703 ms. One of them was SDL event
    /// pumping, which that subsystem cannot do. So the phase is not the culprit. The worst read 786 MiB
    /// across 249,690 operations inside one frame with the background work queue idle and 41 GiB of RAM
    /// free, which says something on this thread is loading synchronously. A phase timer cannot say what.
    /// A stack can, and guessing from four attributions that contradict each other is how several runs
    /// have already been spent.
    ///
    /// Deliberately does no symbol resolution and no allocation while the frame thread is suspended.
    /// DbgHelp is lock-protected, so resolving names against a thread that might itself hold that lock is a
    /// deadlock. Addresses are collected, the thread is resumed, and only then are they turned into names.
    class StallSampler
    {
    public:
        explicit StallSampler(double thresholdMs)
            : mThresholdMs(thresholdMs)
        {
            if (mThresholdMs <= 0.0)
                return;

            // A real handle to this thread, not the pseudo-handle GetCurrentThread returns: that one
            // always means "the calling thread", so the watcher would suspend itself.
            if (!::DuplicateHandle(::GetCurrentProcess(), ::GetCurrentThread(), ::GetCurrentProcess(),
                    &mThread, THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                    FALSE, 0))
            {
                mThread = nullptr;
                Log(Debug::Warning) << "Remix: could not duplicate the frame thread handle; stall stacks "
                                       "are unavailable";
                return;
            }

            mFrames.reserve(kMaxFrames);
            mHeartbeat.store(nowMs(), std::memory_order_relaxed);
            mRunning.store(true, std::memory_order_relaxed);
            mWorker = std::thread([this] { run(); });

            Log(Debug::Info) << "Remix: capturing a frame-thread stack for any frame over " << mThresholdMs
                             << " ms (OPENMW_REMIX_STALL_STACK_MS=0 to disable)";
        }

        ~StallSampler()
        {
            stop();
            if (mThread != nullptr)
                ::CloseHandle(mThread);
        }

        /// Disarms the watcher and waits for it. Idempotent, and it must be called before the frame loop is
        /// left rather than left to the destructor.
        ///
        /// Learned by hanging a shutdown. Once the frame loop stops, so does the heartbeat, so the watcher
        /// concludes the frame thread has stalled and suspends it -- during shutdown, when that thread is
        /// unloading libraries and holding the loader lock. StackWalk64 then wants the same lock, from a
        /// thread that cannot make progress until the one holding it resumes, and the process deadlocks with
        /// no output at all. The user had to kill it. A static destructor is too late for this, because it
        /// runs after the shutdown the watcher would be sampling.
        void stop()
        {
            mRunning.store(false, std::memory_order_relaxed);
            if (mWorker.joinable())
                mWorker.join();
        }

        StallSampler(const StallSampler&) = delete;
        StallSampler& operator=(const StallSampler&) = delete;

        /// Called once per frame from the frame thread. A stall is simply this going quiet.
        void heartbeat() { mHeartbeat.store(nowMs(), std::memory_order_relaxed); }

    private:
        /// Deep enough to get past a recursive teardown to whoever started it.
        ///
        /// Was 48, which was exactly the length of the one stack that mattered: a recursive OSG node
        /// destruction about twelve levels deep at four frames a level, truncated one frame short of its
        /// caller. A cap that swallows the answer is worse than no cap, and these are captured once per
        /// stall so the extra depth costs nothing worth counting.
        static constexpr std::size_t kMaxFrames = 192;

        static double nowMs()
        {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

        void run()
        {
            const HANDLE process = ::GetCurrentProcess();
            ::SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
            if (!::SymInitialize(process, nullptr, TRUE))
            {
                Log(Debug::Warning) << "Remix: SymInitialize failed; stall stacks will be addresses only";
            }

            // One capture per stall, rearmed only once the frame thread has moved on. Without this a
            // four-second stall would be sampled forty times and say the same thing forty times.
            double reported = 0.0;

            while (mRunning.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                const double beat = mHeartbeat.load(std::memory_order_relaxed);
                if (beat == reported || nowMs() - beat < mThresholdMs)
                    continue;

                mFrames.clear();
                if (capture())
                {
                    reported = beat;
                    report(nowMs() - beat, process);
                }
            }

            ::SymCleanup(process);
        }

        bool capture()
        {
            if (::SuspendThread(mThread) == static_cast<DWORD>(-1))
                return false;

            CONTEXT context{};
            context.ContextFlags = CONTEXT_FULL;
            const bool haveContext = ::GetThreadContext(mThread, &context) != FALSE;

            if (haveContext)
            {
                STACKFRAME64 frame{};
                frame.AddrPC.Offset = context.Rip;
                frame.AddrPC.Mode = AddrModeFlat;
                frame.AddrFrame.Offset = context.Rbp;
                frame.AddrFrame.Mode = AddrModeFlat;
                frame.AddrStack.Offset = context.Rsp;
                frame.AddrStack.Mode = AddrModeFlat;

                // The two DbgHelp callbacks are required on x64 -- unwinding reads the image's exception
                // directory, which is what SymFunctionTableAccess64 provides. They touch module tables
                // rather than PDBs, so this stays off the symbol-loading path; name resolution waits until
                // after the resume below.
                while (mFrames.size() < kMaxFrames
                    && ::StackWalk64(IMAGE_FILE_MACHINE_AMD64, ::GetCurrentProcess(), mThread, &frame,
                           &context, nullptr, ::SymFunctionTableAccess64, ::SymGetModuleBase64, nullptr))
                {
                    if (frame.AddrPC.Offset == 0)
                        break;
                    mFrames.push_back(frame.AddrPC.Offset);
                }
            }

            ::ResumeThread(mThread);
            return haveContext && !mFrames.empty();
        }

        void report(double stalledMs, HANDLE process)
        {
            // One line per frame rather than one long line, because these are read by eye and a
            // forty-deep stack on one line is unreadable.
            Log(Debug::Warning) << "Remix stall stack: frame thread has been busy for " << stalledMs
                                << " ms; innermost first:";

            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {};
            auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = MAX_SYM_NAME;

            for (std::size_t i = 0; i < mFrames.size(); ++i)
            {
                const DWORD64 address = mFrames[i];
                std::string name = "??";
                DWORD64 displacement = 0;
                if (::SymFromAddr(process, address, &displacement, symbol))
                    name = symbol->Name;

                std::string location;
                IMAGEHLP_LINE64 line{};
                line.SizeOfStruct = sizeof(line);
                DWORD lineDisplacement = 0;
                if (::SymGetLineFromAddr64(process, address, &lineDisplacement, &line)
                    && line.FileName != nullptr)
                {
                    location = std::string(" at ") + line.FileName + ":" + std::to_string(line.LineNumber);
                }

                Log(Debug::Warning) << "  #" << i << " " << name << location;
            }
        }

        double mThresholdMs = 0.0;
        HANDLE mThread = nullptr;
        std::atomic<double> mHeartbeat{ 0.0 };
        std::atomic<bool> mRunning{ false };
        std::thread mWorker;
        /// Reserved up front so the capture allocates nothing while the frame thread is suspended.
        std::vector<DWORD64> mFrames;
    };
#endif

    /// A frame longer than this is a stall rather than a slow frame, and is counted separately.
    ///
    /// Counted at all because a window maximum cannot distinguish one four-second frame from forty
    /// hundred-millisecond ones, and those want completely different fixes. Every multi-second stall
    /// measured so far has been a single occurrence in its window, which is itself the finding.
    constexpr double kStallMs = 500.0;

    /// A numeric environment override, or \a fallback when it is unset or does not parse.
    double envDouble(const char* name, double fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;

        try
        {
            return std::stod(value);
        }
        catch (const std::exception&)
        {
            Log(Debug::Warning) << "Remix: " << name << " is not a number, using " << fallback;
            return fallback;
        }
    }

#ifdef _WIN32
    /// The one stall sampler, or null when it is switched off.
    ///
    /// A function-local static rather than an Engine member because it is Windows-only and lives in this
    /// file's anonymous namespace; declaring it in the header would drag windows.h along with it. Built on
    /// the first call, which happens on the frame thread -- precisely the thread it needs a handle to.
    ///
    /// Off unless OPENMW_REMIX_STALL_STACK_MS is set, and deliberately so. It found every stall worth
    /// finding, but suspending the frame thread to walk its stack can deadlock against any lock that thread
    /// holds, which is not a hazard to leave armed by default in a build meant to be shown to people. It is
    /// a tool to reach for when something needs locating, not a permanent fixture.
    StallSampler* stallSampler()
    {
        static const double thresholdMs = envDouble("OPENMW_REMIX_STALL_STACK_MS", 0.0);
        if (thresholdMs <= 0.0)
            return nullptr;

        static StallSampler sampler(thresholdMs);
        return &sampler;
    }
#endif

    /// Tells the stall sampler the frame thread is still alive.
    void beatStallSampler()
    {
#ifdef _WIN32
        if (StallSampler* sampler = stallSampler())
            sampler->heartbeat();
#else
        // Nothing portable to do here: capturing another thread's stack has no standard spelling, and the
        // stalls being chased are on Windows.
#endif
    }

    /// Disarms the stall sampler. Called when the frame loop is left, because a stopped heartbeat is
    /// indistinguishable from a stall and shutdown is the worst possible moment to suspend the frame thread.
    void stopStallSampler()
    {
#ifdef _WIN32
        if (StallSampler* sampler = stallSampler())
            sampler->stop();
#endif
    }

    /// True when Remix presents to the screen itself and OpenMW must not move its image.
    ///
    /// The configuration frame generation requires. Interpolated frames are produced during Remix's present
    /// and live only in its swapchain, so a host displaying rtOutput.m_finalOutput -- the path tracer's
    /// target from before presentation -- throws all of them away. DLFG is not even brought up in a host
    /// that never presents; rtx_fork_upscaler_ui.cpp records that, having crashed the developer menu by
    /// reaching for DLFG's device pointer in exactly that situation.
    ///
    /// Read once. It cannot change within a run: it decides how the window and the swapchain are set up.
    bool remixPresentsToScreen()
    {
        static const bool value = []() -> bool {
            const char* env = std::getenv("OPENMW_REMIX_PRESENT");
            return env != nullptr && *env != '\0' && *env != '0';
        }();
        return value;
    }

    /// True when Remix presents into a window of its own rather than into OpenMW's.
    ///
    /// The distinction matters for what should happen when OpenMW's window is not visible. With a separate
    /// window there is still something on screen to keep rendering for, and OpenMW's window is expected to
    /// be inactive -- clicking the other one is how you reach the Remix menu. With Remix's surface parented
    /// to OpenMW's window there is only one window, so it not being visible means nothing is: the game
    /// should pause like any other, and presenting has to stop, because a minimised window has a zero-sized
    /// surface and presenting into one hangs.
    bool remixPresentsToSeparateWindow()
    {
        static const bool value = []() -> bool {
            const char* env = std::getenv("OPENMW_REMIX_WINDOW");
            return env != nullptr && *env != '\0' && *env != '0';
        }();
        return remixPresentsToScreen() && value;
    }
}

#include "mwrender/vismask.hpp"

#include "mwclass/classes.hpp"

#include "mwdialogue/dialoguemanagerimp.hpp"
#include "mwdialogue/journalimp.hpp"
#include "mwdialogue/scripttest.hpp"

#include "mwmechanics/mechanicsmanagerimp.hpp"

#include "mwstate/statemanagerimp.hpp"

#include "profile.hpp"

namespace
{
    void checkSDLError(int ret)
    {
        if (ret != 0)
            Log(Debug::Error) << "SDL error: " << SDL_GetError();
    }

    void initStatsHandler(Resource::Profiler& profiler)
    {
        const osg::Vec4f textColor(1.f, 1.f, 1.f, 1.f);
        const osg::Vec4f barColor(1.f, 1.f, 1.f, 1.f);
        const float multiplier = 1000;
        const bool average = true;
        const bool averageInInverseSpace = false;
        const float maxValue = 10000;

        OMW::forEachUserStatsValue([&](const OMW::UserStats& v) {
            profiler.addUserStatsLine(v.mLabel, textColor, barColor, v.mTaken, multiplier, average,
                averageInInverseSpace, v.mBegin, v.mEnd, maxValue);
        });
        // the forEachUserStatsValue loop is "run" at compile time, hence the settings manager is not available.
        // Unconditionnally add the async physics stats, and then remove it at runtime if necessary
        if (Settings::physics().mAsyncNumThreads == 0)
            profiler.removeUserStatsLine(" -Async");
    }

    struct ScreenCaptureMessageBox
    {
        void operator()(std::string filePath) const
        {
            if (filePath.empty())
            {
                MWBase::Environment::get().getWindowManager()->scheduleMessageBox(
                    "#{OMWEngine:ScreenshotFailed}", MWGui::ShowInDialogueMode_Never);

                return;
            }

            auto l10n = MWBase::Environment::get().getL10nManager()->getContext("OMWEngine");
            std::string message = l10n->formatMessage("ScreenshotMade", { "file" }, { L10n::toUnicode(filePath) });

            MWBase::Environment::get().getWindowManager()->scheduleMessageBox(
                std::move(message), MWGui::ShowInDialogueMode_Never);
        }
    };

    struct IgnoreString
    {
        void operator()(std::string) const {}
    };

    class IdentifyOpenGLOperation : public osg::GraphicsOperation
    {
    public:
        IdentifyOpenGLOperation()
            : GraphicsOperation("IdentifyOpenGLOperation", false)
        {
        }

        void operator()(osg::GraphicsContext* graphicsContext) override
        {
            Log(Debug::Info) << "OpenGL Vendor: " << glGetString(GL_VENDOR);
            Log(Debug::Info) << "OpenGL Renderer: " << glGetString(GL_RENDERER);
            Log(Debug::Info) << "OpenGL Version: " << glGetString(GL_VERSION);
            glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &mMaxTextureImageUnits);
        }

        int getMaxTextureImageUnits() const
        {
            if (mMaxTextureImageUnits == 0)
                throw std::logic_error("mMaxTextureImageUnits is not initialized");
            return mMaxTextureImageUnits;
        }

    private:
        int mMaxTextureImageUnits = 0;
    };

    void reportStats(unsigned frameNumber, osgViewer::Viewer& viewer, std::ostream& stream)
    {
        viewer.getViewerStats()->report(stream, frameNumber);
        osgViewer::Viewer::Cameras cameras;
        viewer.getCameras(cameras);
        for (osg::Camera* camera : cameras)
            camera->getStats()->report(stream, frameNumber);
    }
}

void OMW::Engine::executeLocalScripts()
{
    MWWorld::LocalScripts& localScripts = mWorld->getLocalScripts();

    localScripts.startIteration();
    std::pair<ESM::RefId, MWWorld::Ptr> script;
    while (localScripts.getNext(script))
    {
        MWScript::InterpreterContext interpreterContext(&script.second.getRefData().getLocals(), script.second);
        mScriptManager->run(script.first, interpreterContext);
    }
}

bool OMW::Engine::frame(unsigned frameNumber, float frametime)
{
    const osg::Timer_t frameStart = mViewer->getStartTick();
    const osg::Timer* const timer = osg::Timer::instance();
    osg::Stats* const stats = mViewer->getViewerStats();

    mEnvironment.setFrameDuration(frametime);

    try
    {
        // update input
        {
            ScopedProfile<UserStatsType::Input> profile(frameStart, frameNumber, *timer, *stats);
            // While Remix's developer menu is up it owns the pointer and the keyboard, so OpenMW has to
            // stop acting on them: otherwise the camera keeps turning under the menu and keystrokes are
            // read as bindings as well as by the menu. The existing disableControls/disableEvents pair is
            // exactly the right mechanism -- SDL is still pumped, so the window stays responsive and
            // window events are handled, but button presses and mouse motion are dropped.
            //
            // The menu's own hotkeys are unaffected because Remix reads them from raw input on its
            // overlay window, not from SDL, so Alt+X still closes it.
            PhaseTimer phase(mRemixPhases.mInput);
            mInputManager->update(frametime, mRemixMenuHasMouse, mRemixMenuHasMouse);
        }

        // When the window is minimized, pause the game. Currently this *has* to be here to work around a MyGUI bug.
        // If we are not currently rendering, then RenderItems will not be reused resulting in a memory leak upon
        // changing widget textures (fixed in MyGUI 3.3.2), and destroyed widgets will not be deleted (not fixed yet,
        // https://github.com/MyGUI/mygui/issues/21)
        {
            ScopedProfile<UserStatsType::Sound> profile(frameStart, frameNumber, *timer, *stats);

            // Stood down only when Remix presents into a window of its own.
            //
            // That case needs it gone: a fullscreen window is minimised the moment it loses focus, so
            // clicking Remix's window to reach the developer menu made this fire and stop the frame -- the
            // scene submission with it -- freezing the image in the window being watched.
            //
            // With Remix's surface parented to OpenMW's window the check has to stay, and removing it was a
            // mistake that traded a pause for a hang. There is then only one window, so its not being
            // visible means nothing is on screen, and continuing costs more than a wasted frame: a minimised
            // window's surface has zero extent, and presenting into that blocks. Under DLFG the present runs
            // on its own thread, so the game thread waits on a present that can never complete and the
            // process stops responding rather than pausing.
            //
            // Keeping it also keeps the MyGUI RenderItem leak it was guarding against, which is why it
            // exists at all.
            if (!mWindowManager->isWindowVisible() && !remixPresentsToSeparateWindow())
            {
                mSoundManager->pausePlayback();
                return false;
            }
            else
                mSoundManager->resumePlayback();

            // sound
            PhaseTimer phase(mRemixPhases.mSound);
            if (mUseSound)
                mSoundManager->update(frametime);
        }

        {
            ScopedProfile<UserStatsType::LuaSyncUpdate> profile(frameStart, frameNumber, *timer, *stats);
            PhaseTimer phase(mRemixPhases.mLuaSync);
            // Should be called after input manager update and before any change to the game world.
            // It applies to the game world queued changes from the previous frame.
            mLuaManager->synchronizedUpdate();
        }

        // update game state
        {
            ScopedProfile<UserStatsType::State> profile(frameStart, frameNumber, *timer, *stats);
            PhaseTimer phase(mRemixPhases.mState);
            mStateManager->update(frametime);
        }

        bool paused = mWorld->getTimeManager()->isPaused();

        {
            ScopedProfile<UserStatsType::Script> profile(frameStart, frameNumber, *timer, *stats);

            if (mStateManager->getState() != MWBase::StateManager::State_NoGame)
            {
                if (!mWindowManager->containsMode(MWGui::GM_MainMenu) || !paused)
                {
                    if (mWorld->getScriptsEnabled())
                    {
                        // local scripts
                        executeLocalScripts();

                        // global scripts
                        mScriptManager->getGlobalScripts().run();
                    }

                    mWorld->getWorldScene().markCellAsUnchanged();
                }

                if (!paused)
                {
                    double hours = (frametime * mWorld->getTimeManager()->getGameTimeScale()) / 3600.0;
                    mWorld->advanceTime(hours, true);
                    mWorld->rechargeItems(frametime, true);
                }
            }
        }

        // update mechanics
        {
            ScopedProfile<UserStatsType::Mechanics> profile(frameStart, frameNumber, *timer, *stats);
            PhaseTimer phase(mRemixPhases.mMechanics);

            if (mStateManager->getState() != MWBase::StateManager::State_NoGame)
            {
                mMechanicsManager->update(frametime, paused);
            }

            if (mStateManager->getState() == MWBase::StateManager::State_Running)
            {
                MWWorld::Ptr player = mWorld->getPlayerPtr();
                if (!paused && player.getClass().getCreatureStats(player).isDead())
                    mStateManager->endGame();
            }
        }

        // update physics
        {
            ScopedProfile<UserStatsType::Physics> profile(frameStart, frameNumber, *timer, *stats);
            PhaseTimer phase(mRemixPhases.mPhysics);

            if (mStateManager->getState() != MWBase::StateManager::State_NoGame)
            {
                mWorld->updatePhysics(frametime, paused, frameStart, frameNumber, *stats);
            }
        }

        // update world
        {
            ScopedProfile<UserStatsType::World> profile(frameStart, frameNumber, *timer, *stats);
            PhaseTimer phase(mRemixPhases.mWorld);

            if (mStateManager->getState() != MWBase::StateManager::State_NoGame)
            {
                mWorld->update(frametime, paused);
            }
        }

        // update GUI
        {
            ScopedProfile<UserStatsType::Gui> profile(frameStart, frameNumber, *timer, *stats);
            PhaseTimer phase(mRemixPhases.mWindowManager);
            mWindowManager->update(frametime);
        }
    }
    catch (const std::exception& e)
    {
        Log(Debug::Error) << "Error in frame: " << e.what();
    }

    const bool reportResource = stats->collectStats("resource");

    if (reportResource)
        stats->setAttribute(frameNumber, "UnrefQueue", static_cast<double>(mUnrefQueue->getSize()));

    mUnrefQueue->flush(*mWorkQueue);

    if (reportResource)
    {
        stats->setAttribute(frameNumber, "FrameNumber", frameNumber);

        mResourceSystem->reportStats(frameNumber, stats);

        stats->setAttribute(frameNumber, "WorkQueue", static_cast<double>(mWorkQueue->getNumItems()));
        stats->setAttribute(frameNumber, "WorkThread", static_cast<double>(mWorkQueue->getNumActiveThreads()));

        mMechanicsManager->reportStats(frameNumber, *stats);
        mWorld->reportStats(frameNumber, *stats);
        mLuaManager->reportStats(frameNumber, *stats);

        stats->setAttribute(frameNumber, "StringRefId Count", static_cast<double>(ESM::StringRefId::totalCount()));
    }

    mStereoManager->updateSettings(Settings::camera().mNearClip, Settings::camera().mViewingDistance);

    const auto beforeOsgUpdate = std::chrono::steady_clock::now();
    mViewer->eventTraversal();
    mViewer->updateTraversal();
    mRemixOsgUpdateMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - beforeOsgUpdate)
                            .count();

    // update focus object for GUI
    {
        ScopedProfile<UserStatsType::Focus> profile(frameStart, frameNumber, *timer, *stats);
        mWorld->updateFocusObject();
    }

    // if there is a separate Lua thread, it starts the update now
    mLuaWorker->allowUpdate(frameStart, frameNumber, *stats);

    // Drive a Remix frame before OpenMW draws.
    //
    // Ordering note: the composite happens in the main camera's final draw callback during
    // renderingTraversals below, so it samples the image produced by *this* call. Present has to come
    // first because the raytracing output that copyOutput reads only exists as a result of presenting.
    if (mRemix != nullptr && mRemix->isReady())
    {
        const auto beforeSubmit = std::chrono::steady_clock::now();
        const osg::Camera* camera = mViewer->getCamera();
        // OSG stores these as doubles; the Remix API takes float[4][4]. Named locals so the pointers
        // outlive the call.
        const osg::Matrixf view(camera->getViewMatrix());
        const osg::Matrixf projection(camera->getProjectionMatrix());

        // The test scene defines its own camera and geometry, so it deliberately bypasses OpenMW's.
        const bool useTestScene = RemixRT::Runtime::testSceneRequested();
        bool cameraOk = false;
        if (useTestScene)
        {
            cameraOk = mRemix->submitTestScene();
        }
        else if (mRemixScene != nullptr)
        {
            // Submits the camera and the visible scene together, because they have to agree: the camera
            // goes over as parameters rather than matrices, and geometry in the same units.
            cameraOk = mRemixScene->submit(mViewer->getSceneData(), *camera) > 0;

            // The sky is configuration rather than geometry, so it goes separately and its result does not
            // bear on whether there is a scene to raytrace. Driven from the values OpenMW's own sky was
            // handed this frame, which WeatherManager has already blended across any weather transition.
            if (mRemixSky != nullptr)
            {
                MWBase::World* world = MWBase::Environment::get().getWorld();
                MWRender::RenderingManager* rendering = world != nullptr ? world->getRenderingManager()
                                                                        : nullptr;
                MWRender::SkyManager* sky = rendering != nullptr ? rendering->getSkyManager() : nullptr;
                if (sky != nullptr)
                {
                    const MWWorld::Weather* next = world->getNextWeather();
                    mRemixSky->update(sky->getState(), world->isCellExterior(),
                        world->getCurrentWeatherScriptId(),
                        next != nullptr ? next->mScriptId : -1, world->getWeatherTransition(),
                        Settings::camera().mViewingDistance);
                }
            }
        }
        const auto afterSubmit = std::chrono::steady_clock::now();
        const bool presentOk = mRemix->present();
        const auto afterPresent = std::chrono::steady_clock::now();

        // Skip the synchronised copy entirely when nothing is going to consume the semaphores: either
        // the GL side already reported a failure, or the readback path is in use and reads the surface
        // through D3D9 instead. Signalling a binary semaphore nobody waits on is not free -- it leaves
        // the pair unbalanced -- and the synced entry point is the more complex path of the two.
        // syncDisabled() belongs in this list for the reason the comment above gives: it is the case where
        // nothing will consume the semaphores because the handshake was switched off on purpose. Leaving it
        // out would submit a signal with no waiter and unbalance the pair.
        const bool syncPointless = mRemixComposite == nullptr || mRemixComposite->syncFailed()
            || mRemixComposite->readbackMode() || mRemixComposite->syncDisabled();
        // Remix presents to the screen itself, and OpenMW stops moving its image at all.
        //
        // This is the configuration DLSS Frame Generation requires. Interpolated frames are produced during
        // Remix's present and exist only in its swapchain, so a host that displays rtOutput.m_finalOutput --
        // the path tracer's target from before presentation -- discards every one of them. Worse, DLFG is
        // never even brought up: rtx_fork_upscaler_ui.cpp says so outright, having crashed the developer
        // menu by reaching for DLFG's device pointer in a host where Remix never presents.
        //
        // So no copy and no composite here. Not merely unnecessary: leaving them on is what made the first
        // attempt at this unusable. Remix presenting for real while we copied from its render target
        // unsynchronised gave a static image with scanline tearing and crashed on cell load, because two
        // configurations were fighting over the same surface.
        //
        // The scene submission, the update traversal and OpenMW's own draw all continue. The submission is
        // what Remix renders, and OpenMW's cull is what keeps SemiActive skeletons animating.
        const bool remixPresents = remixPresentsToScreen();

        if (remixPresents)
        {
            if (mRemixComposite != nullptr && mRemixComposite->enabled())
            {
                mRemixComposite->setEnabled(false);
                Log(Debug::Info) << "Remix: compositing disabled -- Remix presents to its own window, so "
                                    "its image is not copied into OpenMW's frame. This is the only "
                                    "arrangement in which frame generation reaches the screen.";
            }
        }
        // Where the copy happens depends on whether the handshake is live, and this is the whole fix for the
        // wait that kept failing.
        //
        // The unsynchronised copy stays here, ahead of the readback below, because that readback reads the
        // image this call fills and would otherwise be reading a frame behind.
        //
        // The synchronised copy is deferred until after renderingTraversals -- see the deferred block near
        // the end of this function. It has to be. The composite's wait happens inside that traversal, so
        // issuing the copy first means the wait races a signal that EmitCs has queued but the CS thread may
        // not have run yet, which is exactly the GL_INVALID_OPERATION this path hit on its first armed
        // frame. Waiting before signalling removes the race, and because these are binary semaphores it is
        // also the only ordering that keeps exactly one signal outstanding at a time.
        //
        // Deferring costs one frame of latency: the composite samples the image the previous copy left. At
        // the framerates involved that is invisible, and it buys the readback's entire cost.
        if (syncPointless && !remixPresents)
        {
            mRemixCopyOk = mRemix->copyOutput();

            if (mRemixComposite != nullptr)
                mRemixComposite->setSyncArmed(false);
        }
        const auto afterCopy = std::chrono::steady_clock::now();
        // Also kept on the object, because the span from here to the draw traversal is measured outside
        // this block's scope.
        mRemixAfterCopyStamp = afterCopy;

        // Only overwrite OpenMW's frame when Remix actually has something in it.
        //
        // The composite is a full-frame overwrite, so with nothing submitted it replaces the entire game
        // -- picture, UI and all -- with black. That is indistinguishable from a hang: the game is
        // running and audible, no key does anything visible, and even the Remix menu is gone, because
        // that is drawn into the same output which is only produced when Remix raytraces. A failure
        // upstream of the compositor must not take the whole game down with it.
        if (mRemixComposite != nullptr && !remixPresents)
        {
            // The most recent copy rather than this frame's, because on the synchronised path the copy has
            // not happened yet at this point and the composite is about to sample what the previous one
            // left. On the first frame there is no copy yet, so compositing stays off and OpenMW's own
            // frame is shown, which is the intended behaviour rather than a black one.
            const bool haveContent = mRemixCopyOk && (useTestScene || cameraOk);
            if (haveContent != mRemixComposite->enabled())
            {
                mRemixComposite->setEnabled(haveContent);
                Log(Debug::Info) << "Remix: compositing " << (haveContent ? "enabled" : "disabled")
                                 << (haveContent ? "" : " -- nothing was submitted, so OpenMW's own frame "
                                                        "is being shown instead of an empty one");
            }
        }

        // Hand the mouse over while Remix's developer menu is open, and take it back afterwards.
        //
        // The menu gets its cursor position from GetCursorPos, and in gameplay OpenMW holds the pointer
        // in relative mode with the cursor grabbed, which pins the OS cursor in place. The menu then
        // draws a cursor that does not track the mouse and receives clicks at a stale position -- it
        // looks like the menu is ignoring input when in fact it is being told the mouse never moves.
        //
        // Remix normally solves this with rtx.blockInputToGameInUI, which signals the game over the
        // 32-bit bridge. There is no bridge in a native host, so the host has to do it.
        //
        // Done through SDL directly, and not through MWBase::InputManager::changeInputMode, which was
        // tried first and cannot work: MouseManager::updateCursorMode ignores the flag it is passed and
        // derives both relative mode and grab from WindowManager::isGuiMode(). OpenMW's input API has
        // no way to express "an external overlay wants the pointer", which is why the menu only becomes
        // clickable when OpenMW happens to enter its own GUI mode -- on death, for instance.
        //
        // Safe to go behind SDLUtil::InputWrapper's back specifically because this is symmetric: the
        // prior state is captured on open and restored on close, so the wrapper's cached values still
        // describe reality either side of the menu being up. Transitions only, since
        // updateCursorMode warps the cursor whenever it leaves relative mode.
        {
            // Only when the menu can actually be seen. It is drawn into Remix's output image, so if the
            // composite is not on screen the menu is invisible -- and handing it the pointer then, while
            // OpenMW also stops acting on input, leaves the game apparently frozen with nothing to
            // interact with and no visible way back. That is precisely what happened when the composite
            // was mis-ordered behind OpenMW's scene resolve: pressing Alt+X looked like a hard lock-up.
            // An overlay that cannot be shown does not get the pointer.
            // When Remix presents, the menu is on screen in Remix's own window and the composite is off by
            // design, so the composite's state no longer says anything about whether the menu is visible.
            const bool menuIsOnScreen
                = remixPresents || (mRemixComposite != nullptr && mRemixComposite->enabled());
            const bool menuWantsMouse = mRemix->uiState() > 0 && menuIsOnScreen;
            if (menuWantsMouse != mRemixMenuHasMouse)
            {
                mRemixMenuHasMouse = menuWantsMouse;
                if (menuWantsMouse)
                {
                    mRemixSavedMouseRelative = SDL_GetRelativeMouseMode() == SDL_TRUE;
                    mRemixSavedMouseGrab = mWindow != nullptr && SDL_GetWindowGrab(mWindow) == SDL_TRUE;
                    SDL_SetRelativeMouseMode(SDL_FALSE);
                    if (mWindow != nullptr)
                        SDL_SetWindowGrab(mWindow, SDL_FALSE);
                }
                else
                {
                    SDL_SetRelativeMouseMode(mRemixSavedMouseRelative ? SDL_TRUE : SDL_FALSE);
                    if (mWindow != nullptr)
                        SDL_SetWindowGrab(mWindow, mRemixSavedMouseGrab ? SDL_TRUE : SDL_FALSE);

                    // Then have OpenMW re-assert its own input state, rather than trusting that
                    // restoring the two SDL flags was enough.
                    //
                    // While the menu is up, both sides have been touching cursor state: this code moved
                    // the pointer out of relative mode, and the runtime drives the Win32 cursor display
                    // counter to force its own cursor to be the only one visible. Putting back what we
                    // personally changed leaves anything else out of step, and the symptom is OpenMW's
                    // cursor never coming back after the menu has been used once.
                    //
                    // changeInputMode derives everything -- cursor visibility, mouse look, the GUI
                    // cursor -- from the window manager's own idea of whether a GUI is open, so it is the
                    // authoritative answer rather than a guess at what the state was.
                    mInputManager->changeInputMode(mWindowManager->isGuiMode());
                }
                Log(Debug::Info) << "Remix: developer menu " << (menuWantsMouse ? "opened" : "closed")
                                 << "; pointer " << (menuWantsMouse ? "released to it" : "returned")
                                 << " (was relative=" << mRemixSavedMouseRelative
                                 << " grab=" << mRemixSavedMouseGrab << ")";
            }
        }

        // When the composite is on the readback path -- either forced, or because the handshake failed
        // -- it has no way to see Remix's pixels on its own, so read them here and hand them over. This
        // is a GPU-to-CPU round trip on the main thread and it costs real frame time; it buys a
        // correct, visible image without depending on the cross-API synchronisation.
        //
        // Skipped when Remix presents, whatever the composite was configured for. Nothing consumes the
        // pixels then, and this is the most expensive thing in the frame to do for no reason.
        if (mRemixComposite != nullptr && mRemixComposite->readbackMode() && !remixPresents)
        {
            unsigned int readWidth = 0;
            unsigned int readHeight = 0;
            if (mRemix->readOutputPixels(mRemixReadback, readWidth, readHeight))
            {
                // Report how bright Remix's output actually is, periodically.
                //
                // "The screen went black" has come up repeatedly and is ambiguous every time: it can mean
                // Remix rendered nothing, or that it rendered and the image is not reaching the screen,
                // or that the game itself is showing black -- a loading screen, a fade, the death
                // screen. Those need completely different investigations and nothing in the log
                // distinguished them.
                //
                // Free to measure here: these pixels are already in host memory for the upload, so this
                // costs a sparse walk of a buffer that was going to be touched anyway. Sampled rather
                // than scanned because at 4K a full pass is 33 MB. Must happen before takeReadbackFrame,
                // which swaps the buffer away.
                static unsigned readbackFrames = 0;
                if (++readbackFrames % 600 == 0 && !mRemixReadback.empty())
                {
                    // Half floats now, and the peak value is the point. The chain carries the runtime's
                    // own R16G16B16A16_SFLOAT range rather than an eight-bit clamp of it, so this answers
                    // the question that decides whether an HDR presentation path is worth building: a peak
                    // above 1.0 means there is real range being thrown away at the display, and a peak that
                    // sits at or under 1.0 means the tonemapper has already flattened everything and the
                    // fix belongs there instead.
                    float brightest = 0.0f;
                    std::size_t nonBlack = 0;
                    std::size_t sampled = 0;
                    std::size_t overOne = 0;
                    const auto* halfs = reinterpret_cast<const std::uint16_t*>(mRemixReadback.data());
                    const std::size_t pixels
                        = mRemixReadback.size() / RemixRT::Runtime::kOutputBytesPerPixel;
                    // Every 997th pixel: prime, so the stride cannot land in step with the row length
                    // and sample only one column of the image.
                    for (std::size_t p = 0; p < pixels; p += 997)
                    {
                        ++sampled;
                        const float value = std::max({ RemixRT::halfToFloat(halfs[p * 4 + 0]),
                            RemixRT::halfToFloat(halfs[p * 4 + 1]),
                            RemixRT::halfToFloat(halfs[p * 4 + 2]) });
                        brightest = std::max(brightest, value);
                        if (value > 0.0f)
                            ++nonBlack;
                        if (value > 1.0f)
                            ++overOne;
                    }
                    Log(Debug::Info) << "Remix readback: " << nonBlack << " of " << sampled
                                     << " sampled pixels non-black, " << overOne
                                     << " above 1.0, brightest " << brightest
                                     << ". A peak above 1.0 is range an SDR present is discarding; a peak "
                                        "at or below 1.0 means the tonemapper already flattened it and no "
                                        "amount of display plumbing will bring it back.";
                }

                mRemixComposite->takeReadbackFrame(mRemixReadback, readWidth, readHeight);
            }
        }

        // Where the frame actually goes.
        //
        // Averaged over a window and reported periodically rather than per frame, because a single frame
        // says nothing: shader compilation, cell loading and the first few frames of any path-traced scene
        // are all outliers, and one sample cannot be told apart from them. The maximum is carried
        // alongside the mean for the same reason in reverse -- a mean that looks fine with a maximum ten
        // times larger is a stutter, not a steady cost, and the two want different fixes.
        //
        // Everything here is wall time on the frame loop except the upload, which is charged to the draw
        // thread. That distinction matters: work on the draw thread overlaps the next frame's simulation
        // and only costs frame time once the draw thread is the critical path, so it cannot simply be
        // added to the rest.
        {
            const auto ms = [](auto duration) {
                return std::chrono::duration<double, std::milli>(duration).count();
            };

            mRemixTiming.mFrames += 1;
            mRemixTiming.mSubmitMs += ms(afterSubmit - beforeSubmit);
            mRemixTiming.mPresentMs += ms(afterPresent - afterSubmit);
            mRemixTiming.mCopyMs += ms(afterCopy - afterPresent);
            mRemixTiming.mOsgUpdateMs += mRemixOsgUpdateMs;
            mRemixTiming.mOsgRenderMs += mRemixRenderMs;

            unsigned long long queueNs = 0;
            unsigned long long lockNs = 0;
            unsigned long long copyNs = 0;
            mRemix->lastReadbackSplit(queueNs, lockNs, copyNs);
            const double readbackMs = static_cast<double>(queueNs + lockNs + copyNs) / 1.0e6;
            mRemixTiming.mReadbackQueueMs += static_cast<double>(queueNs) / 1.0e6;
            mRemixTiming.mReadbackLockMs += static_cast<double>(lockNs) / 1.0e6;
            mRemixTiming.mReadbackCopyMs += static_cast<double>(copyNs) / 1.0e6;
            mRemixTiming.mWorstReadbackMs = std::max(mRemixTiming.mWorstReadbackMs, readbackMs);

            // Keep the slowest frame of the window with its components intact, so a freeze can be
            // attributed instead of averaged away. Measured on the wall clock rather than frametime, which
            // is clamped at 200 ms and therefore reports every freeze as exactly 200 ms.
            const auto frameStamp = std::chrono::steady_clock::now();
            const double thisFrameMs = ms(frameStamp - mRemixFrameStamp);
            mRemixFrameStamp = frameStamp;

            // The mean total is the same wall-clock measure as the worst frame. It used to accumulate
            // frametime, which is clamped, so a window containing a ten-second freeze reported a mean built
            // from a 200 ms ceiling -- and then the accounted percentage was computed against that, which
            // made the accounting look far better than it was.
            mRemixTiming.mFrameMs += thisFrameMs;
            mRemixTiming.mPhasesTotalMs += mRemixPhases.total();

            mRemixTiming.mWorstSubmitMs
                = std::max(mRemixTiming.mWorstSubmitMs, ms(afterSubmit - beforeSubmit));
            mRemixTiming.mWorstPresentMs
                = std::max(mRemixTiming.mWorstPresentMs, ms(afterPresent - afterSubmit));
            mRemixTiming.mWorstOsgUpdateMs
                = std::max(mRemixTiming.mWorstOsgUpdateMs, mRemixOsgUpdateMs);
            mRemixTiming.mWorstOsgRenderMs = std::max(mRemixTiming.mWorstOsgRenderMs, mRemixRenderMs);
            mRemixTiming.mWorstPostCopyMs = std::max(mRemixTiming.mWorstPostCopyMs, mRemixPostCopyMs);
            mRemixTiming.mWorstLuaFinishMs
                = std::max(mRemixTiming.mWorstLuaFinishMs, mRemixLuaFinishMs);

            SimulationPhases& worstPhases = mRemixTiming.mWorstPhases;
            worstPhases.mInput = std::max(worstPhases.mInput, mRemixPhases.mInput);
            worstPhases.mSound = std::max(worstPhases.mSound, mRemixPhases.mSound);
            worstPhases.mLuaSync = std::max(worstPhases.mLuaSync, mRemixPhases.mLuaSync);
            worstPhases.mState = std::max(worstPhases.mState, mRemixPhases.mState);
            worstPhases.mMechanics = std::max(worstPhases.mMechanics, mRemixPhases.mMechanics);
            worstPhases.mPhysics = std::max(worstPhases.mPhysics, mRemixPhases.mPhysics);
            worstPhases.mWorld = std::max(worstPhases.mWorld, mRemixPhases.mWorld);
            worstPhases.mWindowManager
                = std::max(worstPhases.mWindowManager, mRemixPhases.mWindowManager);

            if (thisFrameMs > kStallMs)
                mRemixTiming.mStalls += 1;

            // Tell the watcher the frame thread is alive. Anything that stops this for longer than the
            // threshold gets its stack captured.
            beatStallSampler();

            // Sampled every frame and differenced, so what lands on the worst frame describes that frame
            // rather than the session. Cheap: three syscalls reading counters the kernel already keeps.
            //
            // Primed on the first sample instead of differencing against zero. Without that the first frame
            // reports every page fault and every byte the process has read since it started -- which is how
            // a startup frame came to claim 5,656,900 page faults and 2 GiB of reads, numbers that describe
            // the whole of startup and not the frame they were printed against.
            const ProcessCounters counters = sampleProcessCounters();
            if (!mRemixCountersPrimed)
            {
                mRemixCountersPrimed = true;
                mRemixPrevPageFaults = counters.mPageFaults;
                mRemixPrevReadOps = counters.mReadOps;
                mRemixPrevReadBytes = counters.mReadBytes;
            }
            const unsigned long long pageFaultDelta = counters.mPageFaults - mRemixPrevPageFaults;
            const unsigned long long readOpDelta = counters.mReadOps - mRemixPrevReadOps;
            const unsigned long long readByteDelta = counters.mReadBytes - mRemixPrevReadBytes;
            mRemixPrevPageFaults = counters.mPageFaults;
            mRemixPrevReadOps = counters.mReadOps;
            mRemixPrevReadBytes = counters.mReadBytes;

            if (thisFrameMs > mRemixTiming.mWorstFrameMs)
            {
                mRemixTiming.mWorstFrameMs = thisFrameMs;
                mRemixTiming.mWorstFramePhases = mRemixPhases;
                mRemixTiming.mWorstFramePageFaults = pageFaultDelta;
                mRemixTiming.mWorstFrameReadOps = readOpDelta;
                mRemixTiming.mWorstFrameReadBytes = readByteDelta;
                mRemixTiming.mWorstFrameWorkingSetMiB = counters.mWorkingSetMiB;
                mRemixTiming.mWorstFrameAvailableRamMiB = counters.mAvailableRamMiB;
                mRemixTiming.mWorstFrameWorkQueue
                    = mWorkQueue != nullptr ? static_cast<unsigned int>(mWorkQueue->getNumItems()) : 0u;
                mRemixTiming.mWorstFrameWorkThreads
                    = mWorkQueue != nullptr ? static_cast<unsigned int>(mWorkQueue->getNumActiveThreads()) : 0u;
                mRemixTiming.mWorstFrameUnrefQueue
                    = mUnrefQueue != nullptr ? static_cast<unsigned int>(mUnrefQueue->getSize()) : 0u;
                mRemixTiming.mWorstFrameSubmitMs = ms(afterSubmit - beforeSubmit);
                mRemixTiming.mWorstFramePresentMs = ms(afterPresent - afterSubmit);
                mRemixTiming.mWorstFrameCopyMs = ms(afterCopy - afterPresent);
                mRemixTiming.mWorstFrameReadbackMs = readbackMs;
                mRemixTiming.mWorstFrameOsgUpdateMs = mRemixOsgUpdateMs;
                mRemixTiming.mWorstFrameOsgRenderMs = mRemixRenderMs;
                mRemixTiming.mWorstFrameIndex = mRemixTiming.mFrames;
            }

            constexpr unsigned kTimingWindowFrames = 300;
            if (mRemixTiming.mFrames >= kTimingWindowFrames)
            {
                const double frames = static_cast<double>(mRemixTiming.mFrames);
                unsigned long long uploadNs = 0;
                unsigned int uploads = 0;
                if (mRemixComposite != nullptr)
                    mRemixComposite->takeUploadCost(uploadNs, uploads);
                const double uploadMs
                    = uploads > 0 ? static_cast<double>(uploadNs) / 1.0e6 / uploads : 0.0;

                const double submit = mRemixTiming.mSubmitMs / frames;
                const double present = mRemixTiming.mPresentMs / frames;
                const double readbackQueue = mRemixTiming.mReadbackQueueMs / frames;
                const double readbackLock = mRemixTiming.mReadbackLockMs / frames;
                const double readbackCopy = mRemixTiming.mReadbackCopyMs / frames;
                const double osgUpdate = mRemixTiming.mOsgUpdateMs / frames;
                const double osgRender = mRemixTiming.mOsgRenderMs / frames;
                const double frame = mRemixTiming.mFrameMs / frames;
                // What is left after everything measured. A large remainder means the budget is wrong and
                // the next thing to do is measure, not optimise -- which is exactly the mistake the first
                // pass at this made by assuming the readback dominated.
                const double phases = mRemixTiming.mPhasesTotalMs / frames;
                const double accounted = submit + present + (mRemixTiming.mCopyMs / frames) + readbackQueue
                    + readbackLock + readbackCopy + osgUpdate + osgRender + phases;

                Log(Debug::Info) << "Remix frame cost over " << mRemixTiming.mFrames
                                 << " frames, mean ms: frame " << frame << " | OpenMW update " << osgUpdate
                                 << " | OpenMW render " << osgRender << " | scene submit " << submit
                                 << " | present " << present << " | copy to shared "
                                 << (mRemixTiming.mCopyMs / frames) << " | readback queue " << readbackQueue
                                 << " | readback lock (GPU wait) " << readbackLock << " | readback copy "
                                 << readbackCopy << " | OpenMW simulation " << phases
                                 << " | upload CPU->GPU " << uploadMs << " (draw thread, "
                                 << uploads << " uploads); worst single readback "
                                 << mRemixTiming.mWorstReadbackMs
                                 << "; accounted " << accounted << " of " << frame << " ("
                                 << (frame > 0.0 ? 100.0 * accounted / frame : 0.0) << "%)";

                const double worstAccounted = mRemixTiming.mWorstFrameSubmitMs
                    + mRemixTiming.mWorstFramePresentMs + mRemixTiming.mWorstFrameCopyMs
                    + mRemixTiming.mWorstFrameReadbackMs + mRemixTiming.mWorstFrameOsgUpdateMs
                    + mRemixTiming.mWorstFrameOsgRenderMs + mRemixTiming.mWorstFramePhases.total();

                const SimulationPhases& wfp = mRemixTiming.mWorstFramePhases;
                const SimulationPhases& wp = mRemixTiming.mWorstPhases;
                Log(Debug::Info) << "Remix simulation phases, worst frame ms: world " << wfp.mWorld
                                 << " | physics " << wfp.mPhysics << " | mechanics " << wfp.mMechanics
                                 << " | lua sync " << wfp.mLuaSync << " | GUI " << wfp.mWindowManager
                                 << " | state " << wfp.mState << " | input " << wfp.mInput << " | sound "
                                 << wfp.mSound << " (total " << wfp.total()
                                 << "); worst anywhere in window: world " << wp.mWorld << " | physics "
                                 << wp.mPhysics << " | mechanics " << wp.mMechanics << " | lua sync "
                                 << wp.mLuaSync << " | GUI " << wp.mWindowManager << " | state "
                                 << wp.mState << " | input " << wp.mInput << " | sound " << wp.mSound
                                 << "; nested presents (loading screens, message boxes, video) "
                                 << mRemixTiming.mNestedPresents << " costing "
                                 << mRemixTiming.mNestedPresentMs << " ms total, mean "
                                 << (mRemixTiming.mNestedPresents > 0
                                            ? mRemixTiming.mNestedPresentMs / mRemixTiming.mNestedPresents
                                            : 0.0)
                                 << " ms";

                // Reported separately from the window figures above, and not reset with them, because a
                // burst does not line up with a 300-frame window: a cell load is one long frame, so the
                // whole burst lands inside a single sample. These accumulate over the session so the split
                // is readable even when the load that produced it was several windows ago.
                //
                // The split is the point. If the cost is asset compilation then it sits in the traversal
                // or in the first present alone, and rationing the presents would move the work rather
                // than remove it. If first, worst and least are all alike, each present is paying full
                // price for a path-traced frame and the ration is the fix.
                // Only when something actually stalled, so an ordinary window stays quiet.
                if (mRemixTiming.mStalls > 0)
                {
                    Log(Debug::Info)
                        << "Remix stall context: " << mRemixTiming.mStalls << " frame(s) over " << kStallMs
                        << " ms in this window; across the worst one, page faults "
                        << mRemixTiming.mWorstFramePageFaults << ", disk reads "
                        << mRemixTiming.mWorstFrameReadOps << " ops / "
                        << (mRemixTiming.mWorstFrameReadBytes >> 10) << " KiB; work queue "
                        << mRemixTiming.mWorstFrameWorkQueue << " items on "
                        << mRemixTiming.mWorstFrameWorkThreads << " active threads, unref queue "
                        << mRemixTiming.mWorstFrameUnrefQueue << "; working set "
                        << mRemixTiming.mWorstFrameWorkingSetMiB << " MiB, system RAM free "
                        << mRemixTiming.mWorstFrameAvailableRamMiB << " MiB";
                }

                const RemixRT::NestedFrameStats& nested = RemixRT::nestedFrameStats();
                if (nested.mFrames > 0 || nested.mSkipped > 0)
                {
                    Log(Debug::Info)
                        << "Remix nested frames this session: presented " << nested.mFrames << ", rationed "
                        << nested.mSkipped << "; present " << nested.mPresentMs << " ms total (first "
                        << nested.mFirstPresentMs << ", worst " << nested.mWorstPresentMs << ", least "
                        << nested.mLeastPresentMs << ", mean "
                        << (nested.mFrames > 0 ? nested.mPresentMs / nested.mFrames : 0.0)
                        << "); traversal " << nested.mTraversalMs << " ms total (worst "
                        << nested.mWorstTraversalMs << ")";
                }
                Log(Debug::Info) << "Remix worst frame in window: frame " << mRemixTiming.mWorstFrameMs
                                 << " ms (sample " << mRemixTiming.mWorstFrameIndex << ") | OpenMW update "
                                 << mRemixTiming.mWorstFrameOsgUpdateMs << " | OpenMW render "
                                 << mRemixTiming.mWorstFrameOsgRenderMs << " | scene submit "
                                 << mRemixTiming.mWorstFrameSubmitMs << " | present "
                                 << mRemixTiming.mWorstFramePresentMs << " | copy to shared "
                                 << mRemixTiming.mWorstFrameCopyMs << " | readback "
                                 << mRemixTiming.mWorstFrameReadbackMs << "; accounted " << worstAccounted
                                 << " of " << mRemixTiming.mWorstFrameMs << " ("
                                 << (mRemixTiming.mWorstFrameMs > 0.0
                                            ? 100.0 * worstAccounted / mRemixTiming.mWorstFrameMs
                                            : 0.0)
                                 << "%); worst anywhere in window: submit " << mRemixTiming.mWorstSubmitMs
                                 << ", present " << mRemixTiming.mWorstPresentMs << ", OpenMW update "
                                 << mRemixTiming.mWorstOsgUpdateMs << ", OpenMW render (traversal) "
                                 << mRemixTiming.mWorstOsgRenderMs << ", post-copy (input/GUI/readback) "
                                 << mRemixTiming.mWorstPostCopyMs << ", Lua finish "
                                 << mRemixTiming.mWorstLuaFinishMs;
                mRemixTiming = RemixTiming{};
            }
        }

        // Report the outcome of the pump once. Each of these can fail quietly -- a bad return here is
        // the difference between "Remix rendered black" and "Remix was never asked to render", and
        // guessing between those wastes far more time than one log line costs.
        static unsigned remixFrames = 0;
        ++remixFrames;
        if (remixFrames == 1)
        {
            Log(Debug::Info) << "Remix pump: " << (useTestScene ? "test scene" : "OpenMW scene")
                             << " submit=" << cameraOk << " present=" << presentOk
                             << " copyOutput=" << mRemixCopyOk
                             << (mRemixScene != nullptr
                                        ? " instances=" + std::to_string(mRemixScene->lastInstanceCount())
                                            + " meshes=" + std::to_string(mRemixScene->cachedMeshCount())
                                        : std::string());
            if (!useTestScene)
            {
                // Only meaningful when OpenMW's camera is the one being submitted.
                const osg::Vec3d eye = osg::Matrixd::inverse(camera->getViewMatrix()).getTrans();
                Log(Debug::Info) << "Remix pump: camera eye " << eye.x() << ", " << eye.y() << ", "
                                 << eye.z() << "; proj[10]=" << projection(2, 2)
                                 << " proj[14]=" << projection(3, 2)
                                 << " (reversed-Z shows as a positive proj[10] with an inverted near/far)";
            }
        }
        // Report the developer-menu state the runtime actually holds. Deliberately not on frame 1: the
        // request made during initialisation is applied at the end of a Remix frame, so an earlier read
        // would report the pre-request value and look like a failure.
        if (remixFrames == 3)
        {
            const int ui = mRemix->uiState();
            Log(Debug::Info) << "Remix: runtime reports developer menu state " << ui
                             << " (0 none, 1 basic, 2 advanced, -1 unavailable)"
                             << (ui > 0 ? " -- the menu is being drawn into Remix's own window"
                                        : " -- no menu is being drawn");
        }
        // Probe late enough that shader compilation has finished and Remix has had many frames to
        // converge, but only once -- it stalls on a GPU readback.
        if (remixFrames == 600)
            mRemix->probeOutputNonBlack();
    }

    // The suspected bulk of the frame, and the reason it is worth measuring separately: OpenMW still
    // renders the whole world in OpenGL here, at the window resolution, with its own shadow maps and
    // view distance -- and the composite then overwrites the result with Remix's image. If that is where
    // the time is going, the scene is being paid for twice and one copy is discarded.
    //
    // Note this includes the draw dispatch but not necessarily the GPU finishing: with a separate draw
    // thread, work can still be in flight when this returns. So a large number here is conclusive and a
    // small one is not, same caveat as the texture upload.
    const auto beforeRender = std::chrono::steady_clock::now();

    // Everything between the Remix copy and the draw traversal, as one span.
    //
    // Measured freezes of four seconds accounted for 0.35% across every timer in the frame, so the cost is
    // in a span nobody was watching. This is the largest of them: input, the GUI, uiState, readOutputPixels
    // and probeOutputNonBlack all sit here, and the last two read GPU memory back to the CPU, which blocks.
    mRemixPostCopyMs
        = std::chrono::duration<double, std::milli>(beforeRender - mRemixAfterCopyStamp).count();

    mViewer->renderingTraversals();
    mRemixRenderMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - beforeRender)
              .count();

    // The deferred synchronised copy. See the ordering note where syncPointless is computed above.
    //
    // The composite has now run: it waited on the previous copy's signal, sampled the image, and signalled
    // consumerDone back. Remix waits on that before it overwrites the image, so the signal submitted here
    // cannot land until the consumer has finished reading -- which is what keeps a binary pair balanced
    // across two APIs without any CPU synchronisation between them. A CPU sync here is not merely
    // unnecessary but unsafe: it was tried, it did not fix the wait, and it deadlocked against the overlay
    // thread's window management.
    if (mRemix != nullptr && mRemix->isReady() && mRemixComposite != nullptr
        && !remixPresentsToScreen() && !mRemixComposite->syncFailed()
        && !mRemixComposite->readbackMode() && !mRemixComposite->syncDisabled())
    {
        const auto beforeDeferredCopy = std::chrono::steady_clock::now();

        // Read after the traversal deliberately. It reports whether the composite signalled during the draw
        // that just finished, so it describes the sampling this copy must not overwrite.
        const bool consumerSignalled = mRemixComposite->takeConsumerSignalled();

        if (consumerSignalled)
            mRemixSignalOutstanding = false;

        // At most one signal outstanding, ever, and this is the invariant the earlier attempts broke.
        //
        // These are binary semaphores: signalling one that is already signalled, with no wait in between, is
        // invalid and leaves it in a state no later wait can recover from. It is not enough to check that the
        // handshake is healthy -- the question is whether anything is actually going to consume this signal.
        // It was not, and the failure looked nothing like the cause: the copy ran every frame from startup
        // while the composite stayed disabled through the whole main menu, because compositing only switches
        // on once a copy has succeeded. Dozens of unconsumed signals accumulated, and the first real wait
        // then failed with GL_INVALID_OPERATION -- which is exactly the error this path has been showing all
        // along, on a semaphore that was corrupt long before the wait was reached.
        //
        // So while a signal is still outstanding, copy without signalling. The image still updates; only the
        // semaphore traffic stops. Whenever the consumer does catch up it finds one pending signal, submitted
        // frames ago, and the handshake establishes itself from there without any special first-frame case.
        if (mRemixSignalOutstanding)
        {
            mRemixCopyOk = mRemix->copyOutput();
        }
        else
        {
            // One-way mode: Remix waits for us but does not signal back, because the composite cannot wait
            // on a GL-imported semaphore. Nothing would consume a copyComplete signal, and an unconsumed
            // binary semaphore stays signalled and poisons the next signal -- so it must be suppressed at
            // the source rather than ignored here.
            if (mRemixComposite->syncOneWay())
            {
                // Never armed, and never asking Remix to wait. This mode is unsynchronised, and the naming
                // is now the only thing "one-way" about it.
                //
                // Both directions of the GL/Vulkan semaphore handshake are broken on this driver, and they
                // fail differently. glWaitSemaphoreEXT rejects the wait outright with GL_INVALID_OPERATION
                // under every condition measured. glSignalSemaphoreEXT *accepts* the signal and returns
                // GL_NO_ERROR -- but Vulkan never observes it: arming the signal so that Remix's copy waited
                // on consumerDone hung the queue submission and the driver reset the device within a frame,
                // every single launch. GL_NO_ERROR meant the call was well-formed, nothing more.
                //
                // So the image is sampled with no ordering against Remix's copy. What makes that tolerable
                // rather than reckless is the deferral above: the composite samples during the draw
                // traversal and the copy is not issued until that traversal has returned, so the two are
                // separated by the whole of OpenMW's draw dispatch rather than racing inside one call. It is
                // a narrow window, not a guarantee, and it is a deliberate tradeoff for the readback's cost.
                //
                // The robust fix does not involve semaphores at all: two shared surfaces, Remix copying
                // into one while GL samples the other, alternating per frame. That removes the race by
                // construction instead of trying to order around it.
                mRemixCopyOk = mRemix->copyOutputWaitOnly(false);
                mRemixComposite->setSyncArmed(false);
            }
            else
            {
                mRemixCopyOk = mRemix->copyOutputSynced(consumerSignalled);

                if (mRemixCopyOk)
                    mRemixSignalOutstanding = true;

                // Armed when there is a signal to consume, rather than when a copy happened. Those came
                // apart during the menu and that difference was the bug.
                mRemixComposite->setSyncArmed(mRemixSignalOutstanding);
            }
        }

        mRemixTiming.mCopyMs += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - beforeDeferredCopy)
                                    .count();
    }

    // The last unmeasured thing in the frame. finishUpdate joins the Lua worker, so a script or a
    // synchronisation inside it blocks here and would otherwise be invisible.
    const auto beforeLuaFinish = std::chrono::steady_clock::now();
    mLuaWorker->finishUpdate(frameStart, frameNumber, *stats);
    mRemixLuaFinishMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - beforeLuaFinish)
              .count();

    return true;
}

OMW::Engine::Engine(Files::ConfigurationManager& configurationManager)
    : mWindow(nullptr)
    , mEncoding(ToUTF8::WINDOWS_1252)
    , mScreenCaptureOperation(nullptr)
    , mSelectDepthFormatOperation(new SceneUtil::SelectDepthFormatOperation())
    , mSelectColorFormatOperation(new SceneUtil::Color::SelectColorFormatOperation())
    , mStereoManager(nullptr)
    , mSkipMenu(false)
    , mUseSound(true)
    , mCompileAll(false)
    , mCompileAllDialogue(false)
    , mWarningsMode(1)
    , mScriptConsoleMode(false)
    , mActivationDistanceOverride(-1)
    , mGrab(true)
    , mExportFonts(false)
    , mRandomSeed(0)
    , mNewGame(false)
    , mCfgMgr(configurationManager)
    , mGlMaxTextureImageUnits(0)
{
#if SDL_VERSION_ATLEAST(2, 24, 0)
    SDL_SetHint(SDL_HINT_MAC_OPENGL_ASYNC_DISPATCH, "1");
#endif
    SDL_SetHint(SDL_HINT_ACCELEROMETER_AS_JOYSTICK, "0"); // We use only gamepads

    Uint32 flags
        = SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE | SDL_INIT_GAMECONTROLLER | SDL_INIT_JOYSTICK | SDL_INIT_SENSOR;
    if (SDL_WasInit(flags) == 0)
    {
        SDL_SetMainReady();
        if (SDL_Init(flags) != 0)
        {
            throw std::runtime_error("Could not initialize SDL! " + std::string(SDL_GetError()));
        }
    }
}

OMW::Engine::~Engine()
{
    if (mScreenCaptureOperation != nullptr)
    {
        mScreenCaptureOperation->stop();
        mScreenCaptureOperation = nullptr;
    }
    mScreenCaptureHandler = nullptr;

    mMechanicsManager = nullptr;
    mDialogueManager = nullptr;
    mJournal = nullptr;
    mWindowManager = nullptr;
    mScriptManager = nullptr;
    mWorld = nullptr;
    mStereoManager = nullptr;
    mSoundManager = nullptr;
    mInputManager = nullptr;
    mStateManager = nullptr;
    mLuaWorker = nullptr;
    mLuaManager = nullptr;
    mL10nManager = nullptr;

    mScriptContext = nullptr;

    mUnrefQueue = nullptr;
    mWorkQueue = nullptr;

    mViewer = nullptr;

    mResourceSystem.reset();

    mEncoder = nullptr;

    // Must precede SDL_DestroyWindow: the Remix device holds the HWND that Startup() was given.
    mRemix.reset();

    if (mWindow)
    {
        SDL_DestroyWindow(mWindow);
        mWindow = nullptr;
    }

    SDL_Quit();

    Log(Debug::Info) << "Quitting peacefully.";
}

// Set data dir

void OMW::Engine::setDataDirs(const Files::PathContainer& dataDirs)
{
    mDataDirs = dataDirs;
    mDataDirs.insert(mDataDirs.begin(), mResDir / "vfs");
    mFileCollections = Files::Collections(mDataDirs);
}

// Add BSA archive
void OMW::Engine::addArchive(const std::string& archive)
{
    mArchives.push_back(archive);
}

// Set resource dir
void OMW::Engine::setResourceDir(const std::filesystem::path& parResDir)
{
    mResDir = parResDir;
    if (!Version::checkResourcesVersion(mResDir))
        Log(Debug::Error) << "Resources dir " << mResDir
                          << " doesn't match OpenMW binary, the game may work incorrectly.";
}

// Set start cell name
void OMW::Engine::setCell(const std::string& cellName)
{
    mCellName = cellName;
}

void OMW::Engine::addContentFile(const std::string& file)
{
    mContentFiles.push_back(file);
}

void OMW::Engine::addGroundcoverFile(const std::string& file)
{
    mGroundcoverFiles.emplace_back(file);
}

void OMW::Engine::setSkipMenu(bool skipMenu, bool newGame)
{
    mSkipMenu = skipMenu;
    mNewGame = newGame;
}

void OMW::Engine::createWindow()
{
    const int screen = Settings::video().mScreen;
    const int width = Settings::video().mResolutionX;
    const int height = Settings::video().mResolutionY;
    const Settings::WindowMode windowMode = Settings::video().mWindowMode;
    const bool windowBorder = Settings::video().mWindowBorder;
    const SDLUtil::VSyncMode vsync = Settings::video().mVsyncMode;
    unsigned antialiasing = static_cast<unsigned>(Settings::video().mAntialiasing);

    int posX = SDL_WINDOWPOS_CENTERED_DISPLAY(screen);
    int posY = SDL_WINDOWPOS_CENTERED_DISPLAY(screen);

    if (windowMode == Settings::WindowMode::Fullscreen || windowMode == Settings::WindowMode::WindowedFullscreen)
    {
        posX = SDL_WINDOWPOS_UNDEFINED_DISPLAY(screen);
        posY = SDL_WINDOWPOS_UNDEFINED_DISPLAY(screen);
    }

    Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    if (windowMode == Settings::WindowMode::Fullscreen)
        flags |= SDL_WINDOW_FULLSCREEN;
    else if (windowMode == Settings::WindowMode::WindowedFullscreen)
        flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;

    // Allows for Windows snapping features to properly work in borderless window
    SDL_SetHint("SDL_BORDERLESS_WINDOWED_STYLE", "1");
    SDL_SetHint("SDL_BORDERLESS_RESIZABLE_STYLE", "1");

    if (!windowBorder)
        flags |= SDL_WINDOW_BORDERLESS;

    SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, Settings::video().mMinimizeOnFocusLoss ? "1" : "0");

    checkSDLError(SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8));
    checkSDLError(SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8));
    checkSDLError(SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8));
    checkSDLError(SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 0));
    checkSDLError(SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24));
    if (Debug::shouldDebugOpenGL())
        checkSDLError(SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG));

    if (antialiasing > 0)
    {
        checkSDLError(SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1));
        checkSDLError(SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, antialiasing));
    }

    osg::ref_ptr<SDLUtil::GraphicsWindowSDL2> graphicsWindow;
    while (!graphicsWindow || !graphicsWindow->valid())
    {
        while (!mWindow)
        {
            mWindow = SDL_CreateWindow("OpenMW", posX, posY, width, height, flags);
            if (!mWindow)
            {
                // Try with a lower AA
                if (antialiasing > 0)
                {
                    Log(Debug::Warning) << "Warning: " << antialiasing << "x antialiasing not supported, trying "
                                        << antialiasing / 2;
                    antialiasing /= 2;
                    Settings::video().mAntialiasing.set(antialiasing);
                    checkSDLError(SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, antialiasing));
                    continue;
                }
                else
                {
                    std::stringstream error;
                    error << "Failed to create SDL window: " << SDL_GetError();
                    throw std::runtime_error(error.str());
                }
            }
        }

        // Since we use physical resolution internally, we have to create the window with scaled resolution,
        // but we can't get the scale before the window exists, so instead we have to resize aftewards.
        int w, h;
        SDL_GetWindowSize(mWindow, &w, &h);
        int dw, dh;
        SDL_GL_GetDrawableSize(mWindow, &dw, &dh);
        if (dw != w || dh != h)
        {
            SDL_SetWindowSize(mWindow, width / (dw / w), height / (dh / h));
        }

        setWindowIcon();

        osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
        SDL_GetWindowPosition(mWindow, &traits->x, &traits->y);
        SDL_GL_GetDrawableSize(mWindow, &traits->width, &traits->height);
        traits->windowName = SDL_GetWindowTitle(mWindow);
        traits->windowDecoration = !(SDL_GetWindowFlags(mWindow) & SDL_WINDOW_BORDERLESS);
        traits->screenNum = SDL_GetWindowDisplayIndex(mWindow);
        traits->vsync = 0;
        traits->inheritedWindowData = new SDLUtil::GraphicsWindowSDL2::WindowData(mWindow);

        graphicsWindow = new SDLUtil::GraphicsWindowSDL2(traits, vsync);
        if (!graphicsWindow->valid())
            throw std::runtime_error("Failed to create GraphicsContext");

        if (traits->samples < antialiasing)
        {
            Log(Debug::Warning) << "Warning: Framebuffer MSAA level is only " << traits->samples << "x instead of "
                                << antialiasing << "x. Trying " << antialiasing / 2 << "x instead.";
            graphicsWindow->closeImplementation();
            SDL_DestroyWindow(mWindow);
            mWindow = nullptr;
            antialiasing /= 2;
            Settings::video().mAntialiasing.set(antialiasing);
            checkSDLError(SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, antialiasing));
            continue;
        }

        if (traits->red < 8)
            Log(Debug::Warning) << "Warning: Framebuffer only has a " << traits->red << " bit red channel.";
        if (traits->green < 8)
            Log(Debug::Warning) << "Warning: Framebuffer only has a " << traits->green << " bit green channel.";
        if (traits->blue < 8)
            Log(Debug::Warning) << "Warning: Framebuffer only has a " << traits->blue << " bit blue channel.";
        if (traits->depth < 24)
            Log(Debug::Warning) << "Warning: Framebuffer only has " << traits->depth << " bits of depth precision.";

        traits->alpha = 0; // set to 0 to stop ScreenCaptureHandler reading the alpha channel
    }

    // When Remix owns the screen, OpenMW renders but does not present.
    //
    // Remix path traces into a child window covering this window's whole client area. OpenMW's swap is a
    // blit across that same area, so with both running the two alternate and the raytraced image is
    // overwritten as fast as it arrives -- the symptom is simply seeing OpenMW, with no hint that anything
    // else was drawn. Rendering has to continue regardless: the Remix submission rides on OpenMW's cull,
    // which is also what keeps off-screen NPCs animating.
    if (remixPresentsToScreen())
    {
        graphicsWindow->setSwapEnabled(false);
        Log(Debug::Info) << "Remix: OpenMW will render without presenting -- Remix's child surface is what "
                            "reaches the screen, and OpenMW's swap would paint over it every frame";
    }

    osg::ref_ptr<osg::Camera> camera = mViewer->getCamera();
    camera->setGraphicsContext(graphicsWindow);
    camera->setViewport(0, 0, graphicsWindow->getTraits()->width, graphicsWindow->getTraits()->height);

    osg::ref_ptr<SceneUtil::OperationSequence> realizeOperations = new SceneUtil::OperationSequence(false);
    mViewer->setRealizeOperation(realizeOperations);
    osg::ref_ptr<IdentifyOpenGLOperation> identifyOp = new IdentifyOpenGLOperation();
    realizeOperations->add(identifyOp);
    realizeOperations->add(new SceneUtil::GetGLExtensionsOperation());

    if (Debug::shouldDebugOpenGL())
        realizeOperations->add(new Debug::EnableGLDebugOperation());

    realizeOperations->add(mSelectDepthFormatOperation);
    realizeOperations->add(mSelectColorFormatOperation);

    if (Stereo::getStereo())
    {
        Stereo::Settings settings;

        settings.mMultiview = Settings::stereo().mMultiview;
        settings.mAllowDisplayListsForMultiview = Settings::stereo().mAllowDisplayListsForMultiview;
        settings.mSharedShadowMaps = Settings::stereo().mSharedShadowMaps;

        if (Settings::stereo().mUseCustomView)
        {
            const osg::Vec3 leftEyeOffset(Settings::stereoView().mLeftEyeOffsetX,
                Settings::stereoView().mLeftEyeOffsetY, Settings::stereoView().mLeftEyeOffsetZ);

            const osg::Quat leftEyeOrientation(Settings::stereoView().mLeftEyeOrientationX,
                Settings::stereoView().mLeftEyeOrientationY, Settings::stereoView().mLeftEyeOrientationZ,
                Settings::stereoView().mLeftEyeOrientationW);

            const osg::Vec3 rightEyeOffset(Settings::stereoView().mRightEyeOffsetX,
                Settings::stereoView().mRightEyeOffsetY, Settings::stereoView().mRightEyeOffsetZ);

            const osg::Quat rightEyeOrientation(Settings::stereoView().mRightEyeOrientationX,
                Settings::stereoView().mRightEyeOrientationY, Settings::stereoView().mRightEyeOrientationZ,
                Settings::stereoView().mRightEyeOrientationW);

            settings.mCustomView = Stereo::CustomView{
                .mLeft = Stereo::View{
                    .pose = Stereo::Pose{
                        .position = leftEyeOffset,
                        .orientation = leftEyeOrientation,
                    },
                    .fov = Stereo::FieldOfView{
                        .angleLeft = Settings::stereoView().mLeftEyeFovLeft,
                        .angleRight = Settings::stereoView().mLeftEyeFovRight,
                        .angleUp = Settings::stereoView().mLeftEyeFovUp,
                        .angleDown = Settings::stereoView().mLeftEyeFovDown,
                    },
                },
                .mRight = Stereo::View{
                    .pose = Stereo::Pose{
                        .position = rightEyeOffset,
                        .orientation = rightEyeOrientation,
                    },
                    .fov = Stereo::FieldOfView{
                        .angleLeft = Settings::stereoView().mRightEyeFovLeft,
                        .angleRight = Settings::stereoView().mRightEyeFovRight,
                        .angleUp = Settings::stereoView().mRightEyeFovUp,
                        .angleDown = Settings::stereoView().mRightEyeFovDown,
                    },
                },
            };
        }

        if (Settings::stereo().mUseCustomEyeResolution)
            settings.mEyeResolution
                = osg::Vec2i(Settings::stereoView().mEyeResolutionX, Settings::stereoView().mEyeResolutionY);

        realizeOperations->add(new Stereo::InitializeStereoOperation(settings));
    }

    mViewer->realize();
    mGlMaxTextureImageUnits = identifyOp->getMaxTextureImageUnits();

    mViewer->getEventQueue()->getCurrentEventState()->setWindowRectangle(
        0, 0, graphicsWindow->getTraits()->width, graphicsWindow->getTraits()->height);
}

void OMW::Engine::setWindowIcon()
{
    std::ifstream windowIconStream;
    const auto windowIcon = mResDir / "openmw.png";
    windowIconStream.open(windowIcon, std::ios_base::in | std::ios_base::binary);
    if (windowIconStream.fail())
        Log(Debug::Error) << "Error: Failed to open " << windowIcon;
    osgDB::ReaderWriter* reader = osgDB::Registry::instance()->getReaderWriterForExtension("png");
    if (!reader)
    {
        Log(Debug::Error) << "Error: Failed to read window icon, no png readerwriter found";
        return;
    }
    osgDB::ReaderWriter::ReadResult result = reader->readImage(windowIconStream);
    if (!result.success())
        Log(Debug::Error) << "Error: Failed to read " << windowIcon << ": " << result.message() << " code "
                          << result.status();
    else
    {
        osg::ref_ptr<osg::Image> image = result.getImage();
        auto surface = SDLUtil::imageToSurface(image, true);
        SDL_SetWindowIcon(mWindow, surface.get());
    }
}

void OMW::Engine::prepareEngine()
{
    mStateManager = std::make_unique<MWState::StateManager>(mCfgMgr.getUserDataPath() / "saves", mContentFiles);
    mEnvironment.setStateManager(*mStateManager);

    const bool stereoEnabled = Settings::stereo().mStereoEnabled || osg::DisplaySettings::instance().get()->getStereo();
    mStereoManager = std::make_unique<Stereo::Manager>(
        mViewer, stereoEnabled, Settings::camera().mNearClip, Settings::camera().mViewingDistance);

    osg::ref_ptr<osg::Group> rootNode(new osg::Group);
    mViewer->setSceneData(rootNode);

    createWindow();

    // Bring Remix up directly after the window exists, because Startup() wants its native handle, and
    // before any asset loading, so that a runtime that cannot initialise fails fast and cheaply.
    // Failure is not fatal: mRemix simply reports not-ready and OpenMW renders normally.
    if (RemixRT::Runtime::requested())
    {
        mRemix = std::make_unique<RemixRT::Runtime>();
        if (!mRemix->initialize(mWindow))
        {
            Log(Debug::Warning) << "Remix: initialisation failed, continuing with normal rendering";
            mRemix.reset();
        }
        else
        {
            // Importing the shared image into GL has to happen on the thread owning the context, and
            // the realize operations have already run by now (mViewer->realize() is inside
            // createWindow()). Queueing on the context runs it there at the next frame instead.
            if (osg::GraphicsContext* gc = mViewer->getCamera()->getGraphicsContext())
            {
                mRemixImport = new RemixRT::ImportOperation(mRemix->outputImage(), mRemix->outputSync());
                gc->add(mRemixImport);

                // The composite goes in as scene content under its own camera, ordered after the world
                // and before the GUI.
                //
                // The order number is the whole point, and both neighbours matter:
                //
                //   POST_RENDER 0  MWRender::PostProcessor's HUD camera, which resolves OpenMW's
                //                  rendered scene to the screen. Always present, whether or not user
                //                  post-process shaders are enabled.
                //   POST_RENDER 1  this camera, replacing that image with Remix's.
                //   POST_RENDER 5  MyGUI's camera, drawing the interface on top.
                //
                // Two earlier placements were wrong in instructive ways. As the main camera's *final*
                // draw callback it ran after that camera's entire render-stage tree, the GUI's nested
                // camera included, so a full-frame overwrite erased every UI element. Moved to
                // POST_RENDER -1 it ran before the HUD camera, which then blitted the rasterised scene
                // straight over it -- the picture went back to looking like plain OpenMW. There is no
                // callback hook between those two points, which is why this is scene content in its own
                // camera rather than a callback anywhere.
                //
                // MyGUI's camera was moved off the default 0 to make this slot exist at all; see
                // myguirendermanager.cpp.
                mRemixComposite = new RemixRT::CompositeCallback(mRemixImport.get());

                osg::ref_ptr<osg::Camera> compositeCamera = new osg::Camera;
                compositeCamera->setName("RemixComposite");
                compositeCamera->setRenderOrder(osg::Camera::POST_RENDER, 1);
                compositeCamera->setReferenceFrame(osg::Camera::ABSOLUTE_RF);
                compositeCamera->setProjectionMatrix(osg::Matrix::identity());
                compositeCamera->setViewMatrix(osg::Matrix::identity());
                // Nothing to clear: this overwrites every pixel it covers, and clearing would throw
                // away the frame it is about to replace for no gain.
                compositeCamera->setClearMask(GL_NONE);
                compositeCamera->setAllowEventFocus(false);
                // The same viewport object, not a copy, so a window resize is picked up without this
                // camera needing to be told. The composite reads the viewport off the current camera to
                // set the GL one, so it has to be present and correct.
                compositeCamera->setViewport(mViewer->getCamera()->getViewport());
                compositeCamera->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
                compositeCamera->addChild(new RemixRT::CompositeDrawable(mRemixComposite));
                rootNode->addChild(compositeCamera);

                // An image for OpenMW's interface, when Remix is what reaches the screen.
                //
                // Only then, because otherwise OpenMW's own frame is on screen and its interface is
                // already in it; drawing it into a shared image as well would cost an allocation and a
                // clear per frame to produce something nothing looks at.
                //
                // Sized to Remix's render resolution rather than the window's, since that is the image it
                // composites over. The two are the same now the surface is a child window, but they were
                // not when it was a captioned top-level window clamped to 3840x2141, and tying it to the
                // wrong one would put the interface at the wrong scale.
                if (remixPresentsToScreen())
                {
                    const RemixRT::Runtime::ExternalImage& output = mRemix->outputImage();
                    if (mRemix->createOverlayImage(output.mWidth, output.mHeight))
                    {
                        // No semaphores. Passing an empty ExternalSync is deliberate rather than an
                        // oversight: the handshake does not work on this driver in either direction, and
                        // for an interface the cost of going without is a single frame of a half-drawn
                        // menu, which is nothing like the torn world frame the output path risks.
                        mRemixOverlayImport = new RemixRT::ImportOperation(mRemix->overlayImage(),
                            RemixRT::Runtime::ExternalSync{},
                            RemixRT::ImportOperation::Usage::OpenGLWrites);
                        gc->add(mRemixOverlayImport);

                        mRemixOverlayTarget = new RemixRT::GuiOverlayTarget(
                            mRemixOverlayImport.get(), output.mWidth, output.mHeight);
                    }
                }

                // Feeds OpenMW's scene graph to Remix. Without it the runtime has a camera and nothing
                // else, never enters its raytracing path, and produces no output at all.
                mRemixScene = std::make_unique<MWRender::RemixScene>(*mRemix);

                // Drives the runtime's own sky, sun, moons and fog from OpenMW's weather. Separate from
                // the scene because it submits no geometry: Remix builds the sun and each moon as distant
                // lights itself, and this only tells it where they are and what the weather is doing.
                mRemixSky = std::make_unique<MWRender::RemixSky>(*mRemix);
            }
            else
            {
                Log(Debug::Error) << "Remix: no graphics context to import the shared image into";
            }

            // Select the fork's own sky/atmosphere. Without it an empty scene path-traces to black and
            // there is nothing to tell "compositing is broken" apart from "nothing was submitted yet".
            // This also exercises the atmosphere and weather system, which is driven purely through
            // config and game-state values and so is host-agnostic.
            // OpenMW's world is right-handed with Z up; Remix assumes Y up unless told otherwise. Only
            // the sky and the free camera read this option -- world geometry and the camera transform are
            // unaffected -- which is exactly why the symptom was a sky rotated ninety degrees, with a
            // horizon running vertically down the screen, rather than an inverted world.
            mRemix->setConfigVariable("rtx.zUp", "1");

            // Centimetres per game unit. Morrowind is about 69.99 units to the metre, so 1.4288 cm each.
            //
            // This does not affect geometry, the camera or lighting -- it is not a global unit conversion
            // -- but it does drive the atmosphere's worldUnitsPerKm, cloud shadow positioning and the
            // precipitation particle scale. Left at 1 those all believe a kilometre is 100000 units when
            // for Morrowind it is 143000, a 43% error in every atmospheric distance.
            mRemix->setConfigVariable("rtx.sceneScale", "1.43");
            mRemix->setConfigVariable("rtx.skyMode", "1");
            mRemix->setGameValue("__weather.target", "clear");
            mRemix->setGameValue("__weather.blend_seconds", "0");
        }
    }

    mVFS = std::make_unique<VFS::Manager>();

    VFS::registerArchives(mVFS.get(), mFileCollections, mArchives, true, &mEncoder.get()->getStatelessEncoder());

    mResourceSystem = std::make_unique<Resource::ResourceSystem>(
        mVFS.get(), Settings::cells().mCacheExpiryDelay, &mEncoder.get()->getStatelessEncoder());
    mResourceSystem->getSceneManager()->getShaderManager().setMaxTextureUnits(mGlMaxTextureImageUnits);
    mResourceSystem->getSceneManager()->setUnRefImageDataAfterApply(
        false); // keep to Off for now to allow better state sharing
    mResourceSystem->getSceneManager()->setFilterSettings(Settings::general().mTextureMagFilter,
        Settings::general().mTextureMinFilter, Settings::general().mTextureMipmap,
        static_cast<float>(Settings::general().mAnisotropy));
    mEnvironment.setResourceSystem(*mResourceSystem);

    mWorkQueue = new SceneUtil::WorkQueue(Settings::cells().mPreloadNumThreads);
    mUnrefQueue = std::make_unique<SceneUtil::UnrefQueue>();

    mScreenCaptureOperation = new SceneUtil::AsyncScreenCaptureOperation(mWorkQueue,
        new SceneUtil::WriteScreenshotToFileOperation(mCfgMgr.getScreenshotPath(),
            Settings::general().mScreenshotFormat,
            Settings::general().mNotifyOnSavedScreenshot ? std::function<void(std::string)>(ScreenCaptureMessageBox{})
                                                         : std::function<void(std::string)>(IgnoreString{})));

    mScreenCaptureHandler = new osgViewer::ScreenCaptureHandler(mScreenCaptureOperation);

    mViewer->addEventHandler(mScreenCaptureHandler);

    mL10nManager = std::make_unique<L10n::Manager>(mVFS.get());
    mL10nManager->setPreferredLocales(Settings::general().mPreferredLocales, Settings::general().mGmstOverridesL10n);
    mEnvironment.setL10nManager(*mL10nManager);

    mLuaManager = std::make_unique<MWLua::LuaManager>(mVFS.get(), mResDir / "lua_libs");
    mEnvironment.setLuaManager(*mLuaManager);

    // Create input and UI first to set up a bootstrapping environment for
    // showing a loading screen and keeping the window responsive while doing so

    const auto keybinderUser = mCfgMgr.getUserConfigPath() / "input_v3.xml";
    bool keybinderUserExists = std::filesystem::exists(keybinderUser);
    if (!keybinderUserExists)
    {
        const auto input2 = (mCfgMgr.getUserConfigPath() / "input_v2.xml");
        if (std::filesystem::exists(input2))
        {
            keybinderUserExists = std::filesystem::copy_file(input2, keybinderUser);
            Log(Debug::Info) << "Loading keybindings file: " << keybinderUser;
        }
    }
    else
        Log(Debug::Info) << "Loading keybindings file: " << keybinderUser;

    const auto userdefault = mCfgMgr.getUserConfigPath() / "gamecontrollerdb.txt";
    const auto localdefault = mCfgMgr.getLocalPath() / "gamecontrollerdb.txt";

    std::filesystem::path userGameControllerdb;
    if (std::filesystem::exists(userdefault))
        userGameControllerdb = userdefault;

    std::filesystem::path gameControllerdb;
    if (std::filesystem::exists(localdefault))
        gameControllerdb = localdefault;
    else if (!mCfgMgr.getGlobalPath().empty())
    {
        const auto globaldefault = mCfgMgr.getGlobalPath() / "gamecontrollerdb.txt";
        if (std::filesystem::exists(globaldefault))
            gameControllerdb = globaldefault;
    }
    // else if it doesn't exist, pass in an empty path

    // gui needs our shaders path before everything else
    mResourceSystem->getSceneManager()->setShaderPath(mResDir / "shaders");

    osg::GLExtensions& exts = SceneUtil::getGLExtensions();

#if OSG_VERSION_LESS_THAN(3, 6, 6)
    // hack fix for https://github.com/openscenegraph/OpenSceneGraph/issues/1028
    if (!osg::isGLExtensionSupported(exts.contextID, "NV_framebuffer_multisample_coverage"))
        exts.glRenderbufferStorageMultisampleCoverageNV = nullptr;
#endif

    osg::ref_ptr<osg::Group> guiRoot = new osg::Group;
    guiRoot->setName("GUI Root");
    guiRoot->setNodeMask(MWRender::Mask_GUI);
    mStereoManager->disableStereoForNode(guiRoot);
    rootNode->addChild(guiRoot);

    mWindowManager = std::make_unique<MWGui::WindowManager>(mWindow, mViewer, guiRoot, mResourceSystem.get(),
        mWorkQueue.get(), mCfgMgr.getLogPath(), mScriptConsoleMode, mTranslationDataStorage, mEncoding, mExportFonts,
        Version::getOpenmwVersionDescription(), mCfgMgr);
    mEnvironment.setWindowManager(*mWindowManager);

    // Redirect the interface into Remix's overlay image, now that MyGUI's camera exists.
    //
    // Split from the allocation above because that happens right after the window is created, long before
    // there is a WindowManager to ask. The camera is the one MyGUI draws everything under, so this catches
    // menus, the HUD, inventory, dialogue and the console in one place.
    //
    // Bracketing the camera's drawing rather than giving OSG a render target: the target is a texture
    // aliasing memory Vulkan allocated, and OSG would have to be told it owns a GL object it did not
    // create. See GuiOverlayTarget.
    if (mRemixOverlayTarget != nullptr)
    {
        // Through MyGUI's own singleton rather than by widening WindowManager's interface. The render
        // manager is created by the WindowManager constructor that just ran, so it exists by now, and
        // reaching it this way keeps a Remix-specific concern out of a class that has nothing to do with
        // Remix.
        MyGUIPlatform::RenderManager* guiRenderManager = MyGUIPlatform::RenderManager::getInstancePtr();
        osg::Camera* guiCamera = guiRenderManager != nullptr ? guiRenderManager->getGuiCamera() : nullptr;
        if (guiCamera != nullptr)
        {
            guiCamera->setPreDrawCallback(new RemixRT::GuiOverlayBegin(mRemixOverlayTarget.get()));
            guiCamera->setPostDrawCallback(new RemixRT::GuiOverlayEnd(mRemixOverlayTarget.get()));

            // The camera clears nothing by default, because it was drawing over OpenMW's frame. Into its
            // own image it has to clear, or every frame accumulates on the last and closed windows linger
            // as ghosts. Done inside GuiOverlayTarget::begin rather than through setClearMask so the clear
            // colour is transparent black regardless of what OSG would have used.
            guiCamera->setClearMask(GL_NONE);

            // The vertical flip this needs is applied by Remix when it samples, not here.
            //
            // An OpenGL framebuffer's origin is bottom-left and Vulkan samples from top-left, so the
            // interface arrives mirrored. Flipping the camera's projection was the obvious fix and does
            // nothing: MyGUI's drawable emits vertices already in clip space and draws them through raw GL,
            // so the camera's projection matrix never reaches them. Remix's compositing shader takes a
            // flipV flag instead, set only for a shared image, since the uploaded overlay path is fed
            // top-down CPU pixels and must not be flipped.

            // Let the loops that drive the viewer themselves present too.
            //
            // Loading screens, the modal message-box loop and video playback each pump their own
            // traversals rather than going through Engine::frame, so the frames they draw were never
            // presented and never appeared -- correctly rendered into the overlay image and then discarded.
            //
            // Only the present, deliberately: those loops submit no scene, so Remix repeats whatever
            // geometry it last had, and the overlay drawn on top is what is meant to be looked at.
            // How much of a nested loop's wall time may go on presenting it, and how long the loop has to
            // have been running before the first presented frame.
            //
            // The loading screen asks for a frame on every progress tick, gated only by
            // 1/mTargetFrameRate -- 8.3 ms at its default of 120, and nothing lowers it unless a framerate
            // limit is set. Under a path tracer a presented frame does not cost 8.3 ms, so that gate never
            // engages and the loop presents as fast as it can. Measured on a cell load whose own work was
            // around 160 ms: seven presents, 10379.7 ms.
            //
            // A progress indicator costing sixty times the operation it reports is not reporting on the
            // operation, it is the operation. Stated as a share of the loop rather than as a frame rate,
            // because a frame rate cannot be chosen without knowing what a frame costs, and that is exactly
            // what varies. Overridable so the numbers can be measured rather than argued about.
            const double nestedShare = envDouble("OPENMW_REMIX_NESTED_PRESENT_SHARE", 0.25);
            const double nestedFirstDelayMs = envDouble("OPENMW_REMIX_NESTED_PRESENT_DELAY_MS", 200.0);

            RemixRT::setNestedFramePresenter([this, nestedShare, nestedFirstDelayMs]() {
                if (mRemix == nullptr || !mRemix->isReady())
                    return;

                RemixRT::NestedFrameStats& stats = RemixRT::nestedFrameStats();
                const auto now = std::chrono::steady_clock::now();
                const auto since = [&now](const std::chrono::steady_clock::time_point& stamp) {
                    return std::chrono::duration<double, std::milli>(now - stamp).count();
                };

                // These loops run in bursts -- one per cell load, message box or video -- and the ration is
                // per burst. A gap longer than any single presented frame means the previous burst ended, so
                // the next starts with a full allowance rather than inheriting a spent one.
                constexpr double burstGapMs = 500.0;
                if (mRemixNestedSequenceActive && since(mRemixNestedLastCall) > burstGapMs)
                    mRemixNestedSequenceActive = false;

                if (!mRemixNestedSequenceActive)
                {
                    mRemixNestedSequenceActive = true;
                    mRemixNestedSequenceStart = now;
                    mRemixNestedSequenceSpentMs = 0.0;
                }
                mRemixNestedLastCall = now;

                const double elapsedMs = since(mRemixNestedSequenceStart);

                // Nothing is shown for a loop that finishes before a human would notice it began. Without
                // this a load whose real work is 160 ms still pays for a whole presented frame, which is
                // most of what made short loads feel like long ones.
                if (elapsedMs < nestedFirstDelayMs || mRemixNestedSequenceSpentMs > nestedShare * elapsedMs)
                {
                    stats.mSkipped += 1;
                    return;
                }

                // The camera has to be re-sent, not just the present issued. A frame with no camera is one
                // Remix does not raytrace, and the screen-overlay composite is at the end of that path,
                // after tone mapping -- so without this the interface those loops draw is composited by
                // nothing. Geometry is deliberately not submitted: see RemixScene::resubmitCamera.
                const auto beforeNested = std::chrono::steady_clock::now();

                if (mRemixScene != nullptr)
                    mRemixScene->resubmitCamera();

                mRemix->present();

                const double presentMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - beforeNested)
                                             .count();

                mRemixNestedSequenceSpentMs += presentMs;

                if (stats.mFrames == 0)
                {
                    stats.mFirstPresentMs = presentMs;
                    stats.mLeastPresentMs = presentMs;
                }
                stats.mLeastPresentMs = std::min(stats.mLeastPresentMs, presentMs);
                stats.mWorstPresentMs = std::max(stats.mWorstPresentMs, presentMs);
                stats.mPresentMs += presentMs;
                stats.mFrames += 1;

                mRemixTiming.mNestedPresents += 1;
                mRemixTiming.mNestedPresentMs += presentMs;
            });

            if (mRemix != nullptr && mRemix->setOverlayEnabled(true, 1.0f))
            {
                Log(Debug::Info) << "Remix: compositing OpenMW's interface over the path-traced frame";
            }
            else
            {
                Log(Debug::Error) << "Remix: could not enable interface compositing; the interface will "
                                     "be drawn into the shared image but never shown";
            }
        }
        else
        {
            Log(Debug::Error) << "Remix: MyGUI has no camera to redirect, so the interface cannot be "
                                 "composited";
            mRemixOverlayTarget = nullptr;
        }
    }

    mInputManager = std::make_unique<MWInput::InputManager>(mWindow, mViewer, mScreenCaptureHandler, keybinderUser,
        keybinderUserExists, userGameControllerdb, gameControllerdb, mGrab);
    mEnvironment.setInputManager(*mInputManager);

    // Create sound system
    mSoundManager = std::make_unique<MWSound::SoundManager>(mVFS.get(), mUseSound);
    mEnvironment.setSoundManager(*mSoundManager);

    // Create the world
    mWorld = std::make_unique<MWWorld::World>(
        mResourceSystem.get(), mActivationDistanceOverride, mCellName, mCfgMgr.getUserDataPath());
    mEnvironment.setWorld(*mWorld);
    mEnvironment.setWorldModel(mWorld->getWorldModel());
    mEnvironment.setESMStore(mWorld->getStore());

    const MWWorld::Store<ESM::GameSetting>* gmst = &mWorld->getStore().get<ESM::GameSetting>();
    mL10nManager->setGmstLoader([gmst, misses = std::set<std::string, Misc::StringUtils::CiComp>()](
                                    std::string_view gmstName) mutable -> const std::string* {
        const ESM::GameSetting* res = gmst->search(gmstName);
        if (res && res->mValue.getType() == ESM::VT_String)
            return &res->mValue.getString();
        if (misses.emplace(gmstName).second)
            Log(Debug::Error) << "GMST " << gmstName << " not found";
        return nullptr;
    });

    mWindowManager->setStore(mWorld->getStore());

    // Load translation data
    mTranslationDataStorage.setEncoder(mEncoder.get());
    for (auto& mContentFile : mContentFiles)
        mTranslationDataStorage.loadTranslationData(mFileCollections, mContentFile);

    Compiler::registerExtensions(mExtensions);

    // Create script system
    mScriptContext = std::make_unique<MWScript::CompilerContext>(MWScript::CompilerContext::Type_Full);
    mScriptContext->setExtensions(&mExtensions);

    mScriptManager = std::make_unique<MWScript::ScriptManager>(mWorld->getStore(), *mScriptContext, mWarningsMode);
    mEnvironment.setScriptManager(*mScriptManager);

    // Create game mechanics system
    mMechanicsManager = std::make_unique<MWMechanics::MechanicsManager>();
    mEnvironment.setMechanicsManager(*mMechanicsManager);

    // Create dialog system
    mJournal = std::make_unique<MWDialogue::Journal>();
    mEnvironment.setJournal(*mJournal);

    mDialogueManager = std::make_unique<MWDialogue::DialogueManager>(mExtensions, mTranslationDataStorage);
    mEnvironment.setDialogueManager(*mDialogueManager);

    mLuaManager->loadPermanentStorage(mCfgMgr.getUserConfigPath());
    mLuaManager->initPreLoad();

    Loading::Listener* listener = MWBase::Environment::get().getWindowManager()->getLoadingScreen();
    Loading::AsyncListener asyncListener(*listener);
    auto dataLoading = std::async(std::launch::async,
        [&] { mWorld->loadData(mFileCollections, mContentFiles, mGroundcoverFiles, mEncoder.get(), &asyncListener); });

    if (!mSkipMenu)
    {
        std::string_view logo = Fallback::Map::getString("Movies_Company_Logo");
        if (!logo.empty())
            mWindowManager->playVideo(logo, true);
    }

    listener->loadingOn();
    {
        using namespace std::chrono_literals;
        while (dataLoading.wait_for(50ms) != std::future_status::ready)
            asyncListener.update();
        dataLoading.get();
    }
    listener->loadingOff();

    mWorld->init(mMaxRecastLogLevel, mViewer, std::move(rootNode), mWorkQueue.get(), *mUnrefQueue);
    mEnvironment.setWorldScene(mWorld->getWorldScene());
    mWorld->setupPlayer();
    mWorld->setRandomSeed(mRandomSeed);
    mWindowManager->initUI();
    mLuaManager->initPostLoad();

    // scripts
    if (mCompileAll)
    {
        std::pair<int, int> result = mScriptManager->compileAll();
        if (result.first)
            Log(Debug::Info) << "compiled " << result.second << " of " << result.first << " scripts ("
                             << 100 * static_cast<double>(result.second) / result.first << "%)";
    }
    if (mCompileAllDialogue)
    {
        std::pair<int, int> result = MWDialogue::ScriptTest::compileAll(&mExtensions, mWarningsMode);
        if (result.first)
            Log(Debug::Info) << "compiled " << result.second << " of " << result.first << " dialogue scripts ("
                             << 100 * static_cast<double>(result.second) / result.first << "%)";
    }

    // starts a separate lua thread if "lua num threads" > 0
    mLuaWorker = std::make_unique<MWLua::Worker>(*mLuaManager);
}

// Initialise and enter main loop.
void OMW::Engine::go()
{
    assert(!mContentFiles.empty());

    Log(Debug::Info) << "OSG version: " << osgGetVersion();
    SDL_version sdlVersion;
    SDL_GetVersion(&sdlVersion);
    Log(Debug::Info) << "SDL version: " << (int)sdlVersion.major << "." << (int)sdlVersion.minor << "."
                     << (int)sdlVersion.patch;

    Misc::Rng::init(mRandomSeed);

    Settings::ShaderManager::get().load(mCfgMgr.getUserConfigPath() / "shaders.yaml");

    MWClass::registerClasses();

    // Create encoder
    mEncoder = std::make_unique<ToUTF8::Utf8Encoder>(mEncoding);

    // Setup viewer
    mViewer = new osgViewer::Viewer;
    mViewer->setReleaseContextAtEndOfFrameHint(false);

    // Do not try to outsmart the OS thread scheduler (see bug #4785).
    mViewer->setUseConfigureAffinity(false);

    mEnvironment.setFrameRateLimit(Settings::video().mFramerateLimit);

    prepareEngine();

#ifdef _WIN32
    const auto* statsFile = _wgetenv(L"OPENMW_OSG_STATS_FILE");
#else
    const auto* statsFile = std::getenv("OPENMW_OSG_STATS_FILE");
#endif

    std::filesystem::path path;
    if (statsFile != nullptr)
        path = statsFile;

    std::ofstream stats;
    if (!path.empty())
    {
        stats.open(path, std::ios_base::out);
        if (stats.is_open())
            Log(Debug::Info) << "OSG stats will be written to: " << path;
        else
            Log(Debug::Warning) << "Failed to open file to write OSG stats \"" << path
                                << "\": " << std::generic_category().message(errno);
    }

    // Setup profiler
    osg::ref_ptr<Resource::Profiler> statsHandler = new Resource::Profiler(stats.is_open(), *mVFS);

    initStatsHandler(*statsHandler);

    mViewer->addEventHandler(statsHandler);

    osg::ref_ptr<Resource::StatsHandler> resourcesHandler = new Resource::StatsHandler(stats.is_open(), *mVFS);
    mViewer->addEventHandler(resourcesHandler);

    if (stats.is_open())
        Resource::collectStatistics(*mViewer);

    // Start the game
    if (!mSaveGameFile.empty())
    {
        mStateManager->loadGame(mSaveGameFile);
    }
    else if (!mSkipMenu)
    {
        // start in main menu
        mWindowManager->pushGuiMode(MWGui::GM_MainMenu);

        if (mVFS->exists(MWSound::titleMusic))
            mSoundManager->streamMusic(MWSound::titleMusic, MWSound::MusicType::Normal);
        else
            Log(Debug::Warning) << "Title music not found";

        std::string_view logo = Fallback::Map::getString("Movies_Morrowind_Logo");
        if (!logo.empty())
            mWindowManager->playVideo(logo, /*allowSkipping*/ true, /*overrideSounds*/ false);
    }
    else
    {
        mStateManager->newGame(!mNewGame);
    }

    if (!mStartupScript.empty() && mStateManager->getState() == MWState::StateManager::State_Running)
    {
        mWindowManager->executeInConsole(mStartupScript);
    }

    // Start the main rendering loop
    MWWorld::DateTimeManager& timeManager = *mWorld->getTimeManager();
    Misc::FrameRateLimiter frameRateLimiter = Misc::makeFrameRateLimiter(mEnvironment.getFrameRateLimit());
    const std::chrono::steady_clock::duration maxSimulationInterval(std::chrono::milliseconds(200));
    while (!mViewer->done() && !mStateManager->hasQuitRequest())
    {
        const double dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                              std::min(frameRateLimiter.getLastFrameDuration(), maxSimulationInterval))
                              .count()
            * timeManager.getSimulationTimeScale();

        mViewer->advance(timeManager.getRenderingSimulationTime());

        const unsigned frameNumber = mViewer->getFrameStamp()->getFrameNumber();

        if (!frame(frameNumber, static_cast<float>(dt)))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        timeManager.updateIsPaused();
        if (!timeManager.isPaused())
        {
            timeManager.setSimulationTime(timeManager.getSimulationTime() + dt);
            timeManager.setRenderingSimulationTime(timeManager.getRenderingSimulationTime() + dt);
        }

        if (stats)
        {
            // The delay is required because rendering happens in parallel to the main thread and stats from there is
            // available with delay.
            constexpr unsigned statsReportDelay = 3;
            if (frameNumber >= statsReportDelay)
            {
                // Viewer frame number can be different from frameNumber because of loading screens which render new
                // frames inside a simulation frame.
                const unsigned currentFrameNumber = mViewer->getFrameStamp()->getFrameNumber();
                for (unsigned i = frameNumber; i <= currentFrameNumber; ++i)
                    reportStats(i - statsReportDelay, *mViewer, stats);
            }
        }

        frameRateLimiter.limit();
    }

    // Before anything else in the shutdown, because from here on the heartbeat stops and a stopped heartbeat
    // reads as a stall. See stopStallSampler.
    stopStallSampler();

    mLuaWorker->join();

    // Save user settings
    Settings::Manager::saveUser(mCfgMgr.getUserConfigPath() / "settings.cfg");
    Settings::ShaderManager::get().save();
    mLuaManager->savePermanentStorage(mCfgMgr.getUserConfigPath());
}

void OMW::Engine::setCompileAll(bool all)
{
    mCompileAll = all;
}

void OMW::Engine::setCompileAllDialogue(bool all)
{
    mCompileAllDialogue = all;
}

void OMW::Engine::setSoundUsage(bool soundUsage)
{
    mUseSound = soundUsage;
}

void OMW::Engine::setEncoding(const ToUTF8::FromType& encoding)
{
    mEncoding = encoding;
}

void OMW::Engine::setScriptConsoleMode(bool enabled)
{
    mScriptConsoleMode = enabled;
}

void OMW::Engine::setStartupScript(const std::filesystem::path& path)
{
    mStartupScript = path;
}

void OMW::Engine::setActivationDistanceOverride(int distance)
{
    mActivationDistanceOverride = distance;
}

void OMW::Engine::setWarningsMode(int mode)
{
    mWarningsMode = mode;
}

void OMW::Engine::enableFontExport(bool exportFonts)
{
    mExportFonts = exportFonts;
}

void OMW::Engine::setSaveGameFile(const std::filesystem::path& savegame)
{
    mSaveGameFile = savegame;
}

void OMW::Engine::setRandomSeed(unsigned int seed)
{
    mRandomSeed = seed;
}
