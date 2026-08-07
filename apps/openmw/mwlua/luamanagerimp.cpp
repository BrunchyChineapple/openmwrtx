#include "luamanagerimp.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <MyGUI_InputManager.h>
#include <osg/Stats>

#include <sol/object.hpp>
#include <sol/table.hpp>
#include <sol/types.hpp>

#include <components/debug/debuglog.hpp>

#include <components/esm/luascripts.hpp>
#include <components/esm3/esmreader.hpp>
#include <components/esm3/esmwriter.hpp>

#include <components/settings/values.hpp>

#include <components/l10n/manager.hpp>

#include <components/lua_ui/registerscriptsettings.hpp>
#include <components/lua_ui/util.hpp>

#include "../mwbase/windowmanager.hpp"
#include "../mwbase/world.hpp"

#include "../mwrender/bonegroup.hpp"
#include "../mwrender/postprocessor.hpp"

#include "../mwworld/datetimemanager.hpp"
#include "../mwworld/esmstore.hpp"
#include "../mwworld/player.hpp"
#include "../mwworld/ptr.hpp"
#include "../mwworld/scene.hpp"
#include "../mwworld/worldmodel.hpp"

#include "luabindings.hpp"
#include "playerscripts.hpp"
#include "types/types.hpp"
#include "userdataserializer.hpp"

namespace MWLua
{
    namespace
    {
        struct BoolScopeGuard
        {
            bool& mValue;
            BoolScopeGuard(bool& value)
                : mValue(value)
            {
                mValue = true;
            }

            ~BoolScopeGuard() { mValue = false; }
        };

        LocalScripts* asLocal(const LuaUtil::ScriptsContainerWeakPtr& ptr)
        {
            auto scripts = static_cast<LocalScripts*>(*ptr);
            if (scripts == nullptr)
                Log(Debug::Warning) << "Found local Lua script that outlived its object";
            return scripts;
        }
    }

    static LuaUtil::LuaStateSettings createLuaStateSettings()
    {
        if (!Settings::lua().mLuaProfiler)
            LuaUtil::LuaState::disableProfiler();
        return { .mInstructionLimit = Settings::lua().mInstructionLimitPerCall,
            .mMemoryLimit = Settings::lua().mMemoryLimit,
            .mSmallAllocMaxSize = Settings::lua().mSmallAllocMaxSize,
            .mLogMemoryUsage = Settings::lua().mLogMemoryUsage };
    }

    LuaManager::LuaManager(const VFS::Manager* vfs, const std::filesystem::path& libsDir)
        : mLua(vfs, &mConfiguration, createLuaStateSettings())
    {
        Log(Debug::Info) << "Lua version: " << LuaUtil::getLuaVersion();
        mLua.addInternalLibSearchPath(libsDir);

        mGlobalSerializer = createUserdataSerializer(false);
        mLocalSerializer = createUserdataSerializer(true);
        mGlobalLoader = createUserdataSerializer(false, &mContentFileMapping);
        mLocalLoader = createUserdataSerializer(true, &mContentFileMapping);

        mGlobalScripts.setSerializer(mGlobalSerializer.get());
    }

    LuaManager::~LuaManager()
    {
        LuaUi::clearSettings();
    }

    void LuaManager::initConfiguration(bool reload)
    {
        mConfiguration.init(MWBase::Environment::get().getESMStore()->getLuaScriptsCfg(), reload);
        Log(Debug::Verbose) << "Lua scripts configuration (" << mConfiguration.size() << " scripts):";
        for (size_t i = 0; i < mConfiguration.size(); ++i)
            Log(Debug::Verbose) << "#" << i << " " << LuaUtil::scriptCfgToString(mConfiguration[i]);
        mMenuScripts.setAutoStartConf(mConfiguration.getMenuConf());
        mGlobalScripts.setAutoStartConf(mConfiguration.getGlobalConf());
    }

    void LuaManager::initPreLoad()
    {
        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            Context context;
            context.mType = Context::Load;
            context.mLuaManager = this;
            context.mLua = &mLua;

            for (const auto& [name, package] : initCommonPackages(context))
                mLua.addCommonPackage(name, package);

            for (const auto& [name, package] : initLoadPackages(context))
                mLoadScripts.addPackage(name, package);

            mLoadScripts.addPackage("openmw.storage", LuaUtil::LuaStorage::initLoadPackage(view, &mPlayerStorage));

            LuaUtil::LuaStorage::initLuaBindings(view);
        });
    }

    void LuaManager::contentFilesLoaded()
    {
        initConfiguration(false);
        mLoadScripts.setAutoStartConf(mConfiguration.getLoadConf());
        mLoadScripts.addAutoStartedScripts();
        mLoadScripts.contentFilesLoaded();
        mLoadScripts.removeAllScripts();
    }

    void LuaManager::initPostLoad()
    {
        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            Context globalContext;
            globalContext.mType = Context::Global;
            globalContext.mLuaManager = this;
            globalContext.mLua = &mLua;
            globalContext.mObjectLists = &mObjectLists;
            globalContext.mLuaEvents = &mLuaEvents;
            globalContext.mSerializer = mGlobalSerializer.get();

            Context localContext = globalContext;
            localContext.mType = Context::Local;
            localContext.mSerializer = mLocalSerializer.get();

            Context menuContext = globalContext;
            menuContext.mType = Context::Menu;

            for (const auto& [name, package] : initGlobalPackages(globalContext))
                mGlobalScripts.addPackage(name, package);
            for (const auto& [name, package] : initMenuPackages(menuContext))
                mMenuScripts.addPackage(name, package);

            mLocalPackages = initLocalPackages(localContext);

            mPlayerPackages = initPlayerPackages(localContext);
            mPlayerPackages.insert(mLocalPackages.begin(), mLocalPackages.end());

            mGlobalScripts.addPackage("openmw.storage", LuaUtil::LuaStorage::initGlobalPackage(view, &mGlobalStorage));
            mMenuScripts.addPackage(
                "openmw.storage", LuaUtil::LuaStorage::initMenuPackage(view, &mGlobalStorage, &mPlayerStorage));
            mLocalPackages["openmw.storage"] = LuaUtil::LuaStorage::initLocalPackage(view, &mGlobalStorage);
            mPlayerPackages["openmw.storage"]
                = LuaUtil::LuaStorage::initPlayerPackage(view, &mGlobalStorage, &mPlayerStorage);

            mPlayerStorage.setActive(true);
            mGlobalStorage.setActive(false);

            mInitialized = true;
            mMenuScripts.addAutoStartedScripts();
        });
    }

    void LuaManager::loadPermanentStorage(const std::filesystem::path& userConfigPath)
    {
        mPlayerStorage.setActive(true);
        mGlobalStorage.setActive(true);
        const auto globalPath = userConfigPath / "global_storage.bin";
        const auto playerPath = userConfigPath / "player_storage.bin";

        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            if (std::filesystem::exists(globalPath))
                mGlobalStorage.load(view.sol(), globalPath);
            if (std::filesystem::exists(playerPath))
                mPlayerStorage.load(view.sol(), playerPath);
        });
    }

    void LuaManager::savePermanentStorage(const std::filesystem::path& userConfigPath)
    {
        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            if (mGlobalScriptsStarted)
                mGlobalStorage.save(view.sol(), userConfigPath / "global_storage.bin");
            mPlayerStorage.save(view.sol(), userConfigPath / "player_storage.bin");
        });
    }

    void LuaManager::sendLocalEvent(
        const MWWorld::Ptr& target, const std::string& name, const std::optional<sol::table>& data)
    {
        LuaUtil::BinaryData binary = {};
        if (data)
        {
            binary = LuaUtil::serialize(*data, mLocalSerializer.get());
        }
        mLuaEvents.addLocalEvent({ getId(target), name, std::move(binary) });
    }

    void LuaManager::update()
    {
        // Wall-clock breakdown of this function, reported when it overruns.
        //
        // This runs on the Lua worker thread and the frame loop blocks in Worker::finishUpdate waiting for
        // it. The frame report measured that wait at 108-346 ms on ordinary walking frames, which on its
        // own accounted for nearly all of a 379 ms frame -- 32 ms of measured spans plus 346 ms of waiting
        // here. What no existing measurement could say was which part of the Lua update was slow.
        //
        // The spans below tile the whole function rather than sampling suspects, because the previous
        // round of this was spent ruling out things that were never in the frame. Per-script time comes
        // from ScriptStats::mFrameTimeMs, which exists because OpenMW's own per-script metric is an
        // averaged instruction count and a call into an engine binding is one instruction however long
        // the engine spends inside it.
        const auto updateStart = std::chrono::steady_clock::now();
        auto lapStamp = updateStart;
        // Milliseconds since the previous lap, so consecutive spans share one clock read at each boundary
        // instead of taking two each.
        const auto lap = [&lapStamp] {
            const auto now = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - lapStamp).count();
            lapStamp = now;
            return ms;
        };

        // Opens the window that callEngineHandlers accumulates per-script time into. Placed here so the
        // window is exactly the span the frame loop waits on, no wider and no narrower.
        LuaUtil::ScriptsContainer::beginProfilerFrame();

        UpdateBreakdown breakdown;

        if (const int steps = Settings::lua().mGcStepsPerFrame; steps > 0)
            lua_gc(mLua.unsafeState(), LUA_GCSTEP, steps);
        breakdown.mGarbageCollect = lap();

        if (mPlayer.isEmpty())
            return; // The game is not started yet.

        MWWorld::Ptr newPlayerPtr = MWBase::Environment::get().getWorld()->getPlayerPtr();
        if (!(getId(mPlayer) == getId(newPlayerPtr)))
            throw std::logic_error("Player RefNum was changed unexpectedly");
        if (!mPlayer.isInCell() || !newPlayerPtr.isInCell() || mPlayer.getCell() != newPlayerPtr.getCell())
        {
            mPlayer = newPlayerPtr; // player was moved to another cell, update ptr in registry
            MWBase::Environment::get().getWorldModel()->registerPtr(mPlayer);
        }

        mObjectLists.update();

        // Milliseconds this frame may spend instantiating local scripts that are not loaded yet.
        //
        // Instantiating a container runs the top level of its whole require graph, and that graph is content,
        // not engine: LuaState::runInNewSandbox gives every sandbox its own `loaded` table, so the tree is
        // re-executed once per object. A measured mod put 64 modules and 1.08 MB of Lua behind one NPC
        // script -- about 5 ms per NPC -- and entering a town activates sixty of them at once. That arrived
        // as a single 345 ms frame, twice over: once through the queue below on an NPC's first activation,
        // and again through processTimers when a returning NPC's container had been unloaded by
        // ScriptTracker in the meantime.
        //
        // Spending the same work over several frames rather than one is the same treatment the terrain
        // composite encoder and the reference id index already get. Nothing is skipped permanently: a
        // container that does not fit this frame keeps its place and is instantiated by a later one, and
        // because a loaded container never spends the budget again the queue always drains.
        //
        // Zero disables the budget and restores the previous behaviour.
        double loadBudgetMs = Settings::lua().mLocalScriptLoadBudgetMs;
        const bool loadBudgetActive = loadBudgetMs > 0.0;
        // Charges the wall time of a call that may have instantiated a container. Only ever called when the
        // container was unloaded beforehand, so no clock is read on the common path where it was not.
        const auto chargeLoad = [&loadBudgetMs](const std::chrono::steady_clock::time_point start) {
            loadBudgetMs -= std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                                .count();
        };

        {
            // Kept in the same order, with the tail retained rather than dropped.
            std::vector<LuaUtil::ScriptsContainerWeakPtr> deferred;
            for (std::size_t i = 0; i < mQueuedAutoStartedScripts.size(); ++i)
            {
                LocalScripts* scripts = asLocal(mQueuedAutoStartedScripts[i]);
                if (scripts == nullptr)
                    continue;
                if (loadBudgetActive && loadBudgetMs <= 0.0)
                {
                    deferred.push_back(mQueuedAutoStartedScripts[i]);
                    continue;
                }
                const auto started = std::chrono::steady_clock::now();
                scripts->addAutoStartedScripts();
                if (loadBudgetActive)
                    chargeLoad(started);
            }
            mQueuedAutoStartedScripts = std::move(deferred);
        }

        std::erase_if(mActiveLocalScripts, [](const LuaUtil::ScriptsContainerWeakPtr& ptr) {
            LocalScripts* l = asLocal(ptr);
            return l == nullptr || l->getPtrOrEmpty().isEmpty() || l->getPtrOrEmpty().mRef->isDeleted();
        });

        mGlobalScripts.statsNextFrame();
        for (const LuaUtil::ScriptsContainerWeakPtr& ptr : mActiveLocalScripts)
            asLocal(ptr)->statsNextFrame();
        breakdown.mBookkeeping = lap();
        breakdown.mActiveLocalScripts = static_cast<unsigned int>(mActiveLocalScripts.size());

        mLuaEvents.finalizeEventBatch();

        MWWorld::DateTimeManager& timeManager = *MWBase::Environment::get().getWorld()->getTimeManager();
        if (!timeManager.isPaused())
        {
            mMenuScripts.processTimers(timeManager.getSimulationTime(), timeManager.getGameTime());
            mGlobalScripts.processTimers(timeManager.getSimulationTime(), timeManager.getGameTime());
            for (const LuaUtil::ScriptsContainerWeakPtr& ptr : mActiveLocalScripts)
            {
                LocalScripts* scripts = asLocal(ptr);
                // processTimers calls ensureLoaded unconditionally, so for an unloaded container this is
                // where the whole module graph gets built -- which is why the measured stall landed in the
                // timers span rather than in the queue drain above. An unloaded container has no live timer
                // queues to poll, so deferring it costs nothing this frame beyond the deferral itself.
                if (!scripts->isLoaded())
                {
                    if (loadBudgetActive && loadBudgetMs <= 0.0)
                        continue;
                    const auto started = std::chrono::steady_clock::now();
                    scripts->processTimers(timeManager.getSimulationTime(), timeManager.getGameTime());
                    if (loadBudgetActive)
                        chargeLoad(started);
                    continue;
                }
                scripts->processTimers(timeManager.getSimulationTime(), timeManager.getGameTime());
            }
        }
        breakdown.mTimers = lap();

        // Run event handlers for events that were sent before `finalizeEventBatch`.
        //
        // Given its own budget rather than the remainder of the shared one, for the same reason
        // callEngineHandlers is: delivering an event to an object whose scripts are not loaded instantiates
        // them, and a door that runs after others is starved by a shared allowance.
        double eventHandlerLoadBudgetMs = Settings::lua().mLocalScriptLoadBudgetMs;
        mLuaEvents.callEventHandlers(eventHandlerLoadBudgetMs, loadBudgetActive);
        breakdown.mEventHandlers = lap();

        mLua.protectedCall([&](LuaUtil::LuaView& lua) {
            // Run queued callbacks
            for (CallbackWithData& c : mQueuedCallbacks)
                c.mCallback.tryCall(c.mArg);
            mQueuedCallbacks.clear();
            breakdown.mQueuedCallbacks = lap();

            // Run engine handlers. Shares the frame's instantiation budget with the loops below, because
            // this is the door most newly active objects come through: OnActive ends in
            // LocalScripts::setActive, which calls handlers, which loads the container.
            //
            // Bounding only the other three doors was worse than useless. It moved the same work into this
            // span rather than removing it -- bookkeeping and timers dropped from 346 ms and 612 ms to
            // roughly 5 ms each, and engine events rose to 512 ms in their place.
            // Its own budget rather than the remainder of the shared one, because this is the door most
            // newly active objects come through and it runs after three others that can legitimately empty
            // the shared allowance. Sharing starved it: the queue drain above spent the whole 4 ms, engine
            // events then fell back to its one-event floor, and a burst of a few hundred OnActive events
            // would have taken thousands of frames to clear.
            //
            // The cost of the separation is that a frame can spend up to two budgets on instantiation
            // instead of one. That is still bounded, still small next to a 30-40 ms frame, and far better
            // than a queue that never catches up.
            double eventLoadBudgetMs = Settings::lua().mLocalScriptLoadBudgetMs;
            mEngineEvents.callEngineHandlers(eventLoadBudgetMs, loadBudgetActive);
            breakdown.mEngineEvents = lap();
            bool isPaused = timeManager.isPaused();

            float frameDuration = MWBase::Environment::get().getFrameDuration();
            for (const LuaUtil::ScriptsContainerWeakPtr& ptr : mActiveLocalScripts)
            {
                LocalScripts* scripts = asLocal(ptr);
                // Same reasoning as the timer loop: callEngineHandlers begins with ensureLoaded, so an
                // unloaded container is instantiated here. Deferring one means its onUpdate starts a frame
                // or two later than it otherwise would, which is already true of every container that came
                // through mQueuedAutoStartedScripts.
                if (!scripts->isLoaded())
                {
                    if (loadBudgetActive && loadBudgetMs <= 0.0)
                        continue;
                    const auto started = std::chrono::steady_clock::now();
                    scripts->update(isPaused ? 0 : frameDuration);
                    if (loadBudgetActive)
                        chargeLoad(started);
                    continue;
                }
                scripts->update(isPaused ? 0 : frameDuration);
            }
            breakdown.mLocalScriptUpdates = lap();
            mGlobalScripts.update(isPaused ? 0 : frameDuration);
            breakdown.mGlobalScriptUpdates = lap();

            mScriptTracker.unloadInactiveScripts(lua);
            breakdown.mScriptUnloading = lap();
        });

        breakdown.mTotal
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - updateStart)
                  .count();
        reportSlowUpdate(breakdown);
    }

    void LuaManager::reportSlowUpdate(const UpdateBreakdown& breakdown) const
    {
        // Well above anything a frame's Lua update should cost, so this stays silent unless something is
        // genuinely wrong. Set below the smallest wait measured so far (108 ms) so none of them escape.
        constexpr double overrunMs = 100.0;
        if (breakdown.mTotal < overrunMs)
            return;

        Log(Debug::Warning) << "Lua update held the frame thread for " << breakdown.mTotal << " ms across "
                            << breakdown.mActiveLocalScripts
                            << " active local script container(s): garbage collect "
                            << breakdown.mGarbageCollect << " | bookkeeping " << breakdown.mBookkeeping
                            << " | timers " << breakdown.mTimers << " | event handlers "
                            << breakdown.mEventHandlers << " | queued callbacks "
                            << breakdown.mQueuedCallbacks << " | engine events "
                            << breakdown.mEngineEvents << " | local script updates "
                            << breakdown.mLocalScriptUpdates << " | global script updates "
                            << breakdown.mGlobalScriptUpdates << " | script unloading "
                            << breakdown.mScriptUnloading;

        // Which events made up the event-handler phase, worst first.
        //
        // The phase line above and the per-script list below leave one question open between them: a script
        // with thirty handlers is named, but not the handler. That gap cost several runs of guessing at a
        // 300 ms phase, so the answer is measured rather than inferred. An event sent to four hundred actors
        // arrives here as one name with four hundred calls, which is the shape that matters -- a handler
        // costing five milliseconds is unremarkable until something addresses every actor at once with it.
        {
            const auto& costs = mLuaEvents.lastFrameEventCosts();
            std::vector<const std::pair<const std::string, LuaEvents::EventCost>*> worst;
            worst.reserve(costs.size());
            for (const auto& entry : costs)
                worst.push_back(&entry);
            std::sort(worst.begin(), worst.end(), [](const auto* a, const auto* b) {
                return a->second.mMs > b->second.mMs;
            });

            // Enough to show a culprit and whatever is next behind it, without turning one late frame into
            // a page of log.
            constexpr std::size_t kReportedEvents = 5;
            for (std::size_t i = 0; i < worst.size() && i < kReportedEvents; ++i)
            {
                // Sub-millisecond events are noise here; the phase is only reported at all above 100 ms.
                if (worst[i]->second.mMs < 1.0)
                    break;
                Log(Debug::Warning) << "  event " << worst[i]->first << ": " << worst[i]->second.mMs
                                    << " ms across " << worst[i]->second.mCalls << " handler call(s)";
            }
        }

        // Per-script attribution for the same frame, covering every way Lua gets called. Sums every
        // container so a script running on four hundred actors is reported once with its total, which is
        // the number that matters here -- a cheap handler multiplied by the active actor count is a
        // perfectly ordinary way to lose 300 ms, and it looks nothing like one expensive script in a
        // per-instance view.
        using Stats = LuaUtil::ScriptsContainer::ScriptStats;
        std::vector<Stats> stats;
        mGlobalScripts.collectStats(stats);
        for (const LuaUtil::ScriptsContainerWeakPtr& ptr : mActiveLocalScripts)
        {
            if (LocalScripts* scripts = asLocal(ptr))
                scripts->collectStats(stats);
        }

        std::vector<std::size_t> order;
        order.reserve(stats.size());
        for (std::size_t id = 0; id < stats.size(); ++id)
        {
            if (stats[id].mFrameCalls > 0)
                order.push_back(id);
        }
        std::sort(order.begin(), order.end(),
            [&stats](std::size_t a, std::size_t b) { return stats[a].mFrameTimeMs > stats[b].mFrameTimeMs; });

        const LuaUtil::ScriptsConfiguration& configuration = mLua.getConfiguration();
        double attributed = 0.0;
        unsigned int calls = 0;
        for (const std::size_t id : order)
        {
            attributed += stats[id].mFrameTimeMs;
            calls += stats[id].mFrameCalls;
        }

        // Only the worst few. The point is to name a culprit, and a table of 206 rows in a log is a
        // thing nobody reads.
        constexpr std::size_t kReported = 8;
        for (std::size_t i = 0; i < order.size() && i < kReported; ++i)
        {
            const std::size_t id = order[i];
            Log msg(Debug::Warning);
            msg << "  " << stats[id].mFrameTimeMs << " ms across " << stats[id].mFrameCalls
                << " Lua call(s): ";
            // collectStats sizes its vector from the configuration, so this should always hold. Checked
            // anyway rather than indexed on faith, because the alternative is an out-of-range read in a
            // diagnostic that only runs when something is already going wrong.
            if (id < configuration.size())
                msg << configuration[id].mScriptPath;
            else
                msg << "<script id " << id << " not in the configuration>";
        }
        // A remainder now means time inside the engine rather than inside a script: script loading from
        // ensureLoaded, the object list rebuild, event deserialization, and the engine bindings a script
        // calls are all charged to the engine side of the boundary, not to Lua. The spans above say which.
        Log(Debug::Warning) << "  " << attributed << " ms of " << breakdown.mTotal
                            << " ms is script self time across " << calls << " Lua call(s) in "
                            << order.size() << " script(s); the rest is engine work in the spans above";
    }

    void LuaManager::objectTeleported(const MWWorld::Ptr& ptr)
    {
        if (ptr == mPlayer)
        {
            // For player run the onTeleported handler immediately,
            // so it can adjust camera position after teleporting.
            PlayerScripts* playerScripts = dynamic_cast<PlayerScripts*>(mPlayer.getRefData().getLuaScripts());
            if (playerScripts)
                playerScripts->onTeleported();
        }
        else
            mEngineEvents.addToQueue(EngineEvents::OnTeleported{ getId(ptr) });
    }

    void LuaManager::questUpdated(const ESM::RefId& questId, int stage)
    {
        if (mPlayer.isEmpty())
            return; // The game is not started yet.
        PlayerScripts* playerScripts = dynamic_cast<PlayerScripts*>(mPlayer.getRefData().getLuaScripts());
        if (playerScripts)
        {
            playerScripts->onQuestUpdate(questId.serializeText(), stage);
        }
    }

    void LuaManager::synchronizedUpdate()
    {
        mLua.protectedCall([&](LuaUtil::LuaView&) { synchronizedUpdateUnsafe(); });
    }

    void LuaManager::synchronizedUpdateUnsafe()
    {
        if (mNewGameStarted)
        {
            mNewGameStarted = false;
            // Run onNewGame handler in synchronizedUpdate (at the beginning of the frame), so it
            // can teleport the player to the starting location before the first frame is rendered.
            mGlobalScripts.newGameStarted();
        }
        BoolScopeGuard updateGuard(mRunningSynchronizedUpdates);

        MWBase::WindowManager* windowManager = MWBase::Environment::get().getWindowManager();
        PlayerScripts* playerScripts
            = mPlayer.isEmpty() ? nullptr : dynamic_cast<PlayerScripts*>(mPlayer.getRefData().getLuaScripts());
        // We apply input events in `synchronizedUpdate` rather than in `update` in order to reduce input latency.
        {
            BoolScopeGuard processingGuard(mProcessingInputEvents);

            for (const auto& event : mMenuInputEvents)
                mMenuScripts.processInputEvent(event);
            mMenuInputEvents.clear();
            if (playerScripts && !windowManager->containsMode(MWGui::GM_MainMenu))
            {
                for (const auto& event : mInputEvents)
                    playerScripts->processInputEvent(event);
            }
            mInputEvents.clear();
            mLuaEvents.callMenuEventHandlers();
            float frameDuration = MWBase::Environment::get().getWorld()->getTimeManager()->isPaused()
                ? 0.f
                : MWBase::Environment::get().getFrameDuration();
            mInputActions.update(frameDuration);
            mMenuScripts.onFrame(frameDuration);
            if (playerScripts)
                playerScripts->onFrame(frameDuration);
        }

        for (const auto& [message, mode] : mUIMessages)
            windowManager->messageBox(message, mode);
        mUIMessages.clear();
        for (auto& [msg, color] : mInGameConsoleMessages)
            windowManager->printToConsole(msg, "#" + color.toHex());
        mInGameConsoleMessages.clear();

        applyDelayedActions();

        if (mReloadAllScriptsRequested)
        {
            // Reloading right after `applyDelayedActions` to guarantee that no delayed actions are currently queued.
            reloadAllScriptsImpl();
            mReloadAllScriptsRequested = false;
        }

        if (mDelayedUiModeChangedArg)
        {
            if (playerScripts)
                playerScripts->uiModeChanged(*mDelayedUiModeChangedArg, true);
            mDelayedUiModeChangedArg = std::nullopt;
        }
    }

    void LuaManager::applyDelayedActions()
    {
        // The batch is timed as well as each action, because the two describe different problems and only
        // one of them was visible.
        //
        // A measured 769 ms stall sat inside LuaUi::Element::create under DelayedAction::apply, yet no
        // single action reported an overrun -- the queue was simply long. One four-hundred-millisecond
        // action and four hundred one-millisecond actions cost the same frame and want opposite fixes, and
        // per-action timing alone cannot tell them apart.
        constexpr double batchOverrunMs = 150.0;
        const auto batchStarted = std::chrono::steady_clock::now();
        const std::size_t queued = mActionQueue.size();

        // Tallied inside the loop and keyed by an owning string, not a view.
        //
        // The first version collected string_views into each action's mName and aggregated them after
        // mActionQueue.clear(), which destroys the strings those views point at. It printed recognisable
        // but corrupt names -- "Create UI" came out as " reate UI" -- because the freed small-string buffer
        // still held most of its old contents with the first byte reused. A use-after-free that produces
        // almost-correct output is worse than one that crashes, so this owns its keys.
        //
        // Aggregating unconditionally rather than only for slow batches: it is a map lookup per action
        // against a handful of distinct names, which is nothing next to the actions themselves, and the
        // alternative is keeping per-action state alive past the point where it is valid.
        std::map<std::string, std::pair<unsigned int, double>> byName;

        // The single slowest action of the batch, kept with its traceback. A batch report can say what it
        // was full of, but only a traceback can say which script queued it -- every ui.create is named
        // "Create UI" whoever called it.
        double slowestMs = -1.0;
        std::string slowestName;
        std::string slowestTraceback;

        {
            BoolScopeGuard applyingGuard(mApplyingDelayedActions);
            for (DelayedAction& action : mActionQueue)
            {
                const auto actionStarted = std::chrono::steady_clock::now();
                action.apply();
                const double actionMs
                    = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - actionStarted)
                          .count();

                auto& entry = byName[std::string(action.name())];
                entry.first += 1;
                entry.second += actionMs;

                if (actionMs > slowestMs)
                {
                    slowestMs = actionMs;
                    slowestName = action.name();
                    slowestTraceback = action.callerTraceback();
                }
            }
            mActionQueue.clear();

            if (mTeleportPlayerAction)
                mTeleportPlayerAction->apply();
            mTeleportPlayerAction.reset();
        }

        const double batchMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - batchStarted)
                  .count();
        if (batchMs < batchOverrunMs || queued == 0)
            return;

        std::vector<std::pair<std::string, std::pair<unsigned int, double>>> ranked(
            byName.begin(), byName.end());
        std::sort(ranked.begin(), ranked.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second.second > rhs.second.second; });

        Log(Debug::Warning) << "Lua actions held the main thread for " << batchMs << " ms across " << queued
                            << " queued action(s); worst kinds:";
        const std::size_t shown = std::min<std::size_t>(ranked.size(), 5);
        for (std::size_t i = 0; i < shown; ++i)
        {
            Log(Debug::Warning) << "  '" << ranked[i].first << "' x" << ranked[i].second.first << " = "
                                << ranked[i].second.second << " ms";
        }

        Log(Debug::Warning) << "  slowest single action '" << slowestName << "' " << slowestMs << " ms";
        if (slowestTraceback.empty())
            Log(Debug::Warning) << "  set 'lua debug = true' in settings.cfg to learn which script queued it";
        else
            Log(Debug::Warning) << "  queued by " << slowestTraceback;
    }

    void LuaManager::clear()
    {
        LuaUi::clearGameInterface();
        mUiResourceManager.clear();
        MWBase::Environment::get().getWorld()->getPostProcessor()->disableDynamicShaders();
        mActiveLocalScripts.clear();
        mLuaEvents.clear();
        mEngineEvents.clear();
        mInputEvents.clear();
        mMenuInputEvents.clear();
        mObjectLists.clear();
        mGlobalScripts.removeAllScripts();
        mGlobalScriptsStarted = false;
        mNewGameStarted = false;
        mDelayedUiModeChangedArg = std::nullopt;
        if (!mPlayer.isEmpty())
        {
            mPlayer.getCellRef().unsetRefNum();
            mPlayer.getRefData().setLuaScripts(nullptr);
            mPlayer = MWWorld::Ptr();
        }
        mGlobalStorage.setActive(true);
        mGlobalStorage.clearTemporaryAndRemoveCallbacks();
        mGlobalStorage.setActive(false);
        mPlayerStorage.clearTemporaryAndRemoveCallbacks();
        mInputActions.clear();
        mInputTriggers.clear();
        mQueuedAutoStartedScripts.clear();
        for (int i = 0; i < 5; ++i)
            lua_gc(mLua.unsafeState(), LUA_GCCOLLECT, 0);
    }

    void LuaManager::setupPlayer(const MWWorld::Ptr& ptr)
    {
        if (!mInitialized)
            return;
        if (!mPlayer.isEmpty())
            throw std::logic_error("Player is initialized twice");
        mObjectLists.objectAddedToScene(ptr);
        mObjectLists.setPlayer(ptr);
        mPlayer = ptr;
        LocalScripts* localScripts = ptr.getRefData().getLuaScripts();
        if (!localScripts)
        {
            localScripts = createLocalScripts(ptr);
            mQueuedAutoStartedScripts.push_back(localScripts->getWeakPointer());
        }
        mActiveLocalScripts.insert(localScripts->getWeakPointer());
        mEngineEvents.addToQueue(EngineEvents::OnActive{ getId(ptr) });
    }

    void LuaManager::newGameStarted()
    {
        mGlobalStorage.setActive(true);
        mInputEvents.clear();
        mGlobalScripts.addAutoStartedScripts();
        mGlobalScriptsStarted = true;
        mNewGameStarted = true;
    }

    void LuaManager::gameLoaded()
    {
        mGlobalStorage.setActive(true);
        if (!mGlobalScriptsStarted)
            mGlobalScripts.addAutoStartedScripts();
        mGlobalScriptsStarted = true;
        mMenuScripts.stateChanged();
    }

    void LuaManager::gameEnded()
    {
        // TODO: disable scripts and global storage when the game is actually unloaded
        // mGlobalStorage.setActive(false);
        mMenuScripts.stateChanged();
    }

    void LuaManager::noGame()
    {
        clear();
        mMenuScripts.stateChanged();
    }

    void LuaManager::uiModeChanged(const MWWorld::Ptr& arg)
    {
        if (mPlayer.isEmpty())
            return;
        ObjectId argId = arg.isEmpty() ? ObjectId() : getId(arg);
        if (mApplyingDelayedActions)
        {
            mDelayedUiModeChangedArg = argId;
            return;
        }
        PlayerScripts* playerScripts = dynamic_cast<PlayerScripts*>(mPlayer.getRefData().getLuaScripts());
        if (playerScripts)
            playerScripts->uiModeChanged(argId, false);
    }

    void LuaManager::viewportResized(int width, int height)
    {
        if (!mPlayer.isEmpty())
        {
            PlayerScripts* playerScripts = dynamic_cast<PlayerScripts*>(mPlayer.getRefData().getLuaScripts());
            if (playerScripts)
                playerScripts->onViewportResized(width, height);
        }

        mMenuScripts.onViewportResized(width, height);
    }

    void LuaManager::actorDied(const MWWorld::Ptr& actor)
    {
        if (actor.isEmpty())
            return;
        mLuaEvents.addLocalEvent({ getId(actor), "Died", {} });
    }

    void LuaManager::onDialogueResponse(
        const MWWorld::Ptr& actor, const ESM::DialInfo& info, const ESM::Dialogue& record)
    {
        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            sol::table data = view.newTable();
            data["actor"] = LObject(actor);
            if (record.mType == ESM::Dialogue::Type::Greeting)
                data["type"] = "greeting";
            else if (record.mType == ESM::Dialogue::Type::Journal)
                data["type"] = "journal";
            else if (record.mType == ESM::Dialogue::Type::Persuasion)
                data["type"] = "persuasion";
            else if (record.mType == ESM::Dialogue::Type::Topic)
                data["type"] = "topic";
            else if (record.mType == ESM::Dialogue::Type::Voice)
                data["type"] = "voice";
            data["infoId"] = info.mId.serializeText();
            data["recordId"] = record.mId.serializeText();
            sendLocalEvent(mPlayer, "DialogueResponse", data);
        });
    }

    void LuaManager::useItem(const MWWorld::Ptr& object, const MWWorld::Ptr& actor, bool force)
    {
        MWBase::Environment::get().getWorldModel()->registerPtr(object);
        mEngineEvents.addToQueue(EngineEvents::OnUseItem{ getId(actor), getId(object), force });
    }

    void LuaManager::objectDropped(
        const MWWorld::Ptr& object, const MWWorld::Ptr& actor, const osg::Vec3f& position, const osg::Quat& rotation)
    {
        MWBase::Environment::get().getWorldModel()->registerPtr(object);
        mEngineEvents.addToQueue(
            EngineEvents::OnDropped{ getId(object), getId(actor), position, LuaUtil::asTransform(rotation) });
    }

    void LuaManager::objectPlaced(
        const MWWorld::Ptr& object, const MWWorld::Ptr& actor, const osg::Vec3f& position, const osg::Quat& rotation)
    {
        MWBase::Environment::get().getWorldModel()->registerPtr(object);
        mEngineEvents.addToQueue(
            EngineEvents::OnPlaced{ getId(object), getId(actor), position, LuaUtil::asTransform(rotation) });
    }

    void LuaManager::animationTextKey(const MWWorld::Ptr& actor, const std::string& key)
    {
        auto pos = key.find(": ");
        if (pos != std::string::npos)
            mEngineEvents.addToQueue(
                EngineEvents::OnAnimationTextKey{ getId(actor), key.substr(0, pos), key.substr(pos + 2) });
    }

    void LuaManager::playAnimation(const MWWorld::Ptr& actor, const std::string& groupname,
        const MWRender::AnimPriority& priority, int blendMask, bool autodisable, float speedmult,
        std::string_view start, std::string_view stop, float startpoint, uint32_t loops, bool loopfallback)
    {
        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            sol::table options = view.newTable();
            options["blendMask"] = blendMask;
            options["autoDisable"] = autodisable;
            options["speed"] = speedmult;
            options["startKey"] = start;
            options["stopKey"] = stop;
            options["startPoint"] = startpoint;
            options["loops"] = loops;
            options["forceLoop"] = loopfallback;

            bool priorityAsTable = false;
            for (uint32_t i = 1; i < MWRender::sNumBlendMasks; i++)
                if (priority[static_cast<MWRender::BoneGroup>(i)] != priority[static_cast<MWRender::BoneGroup>(0)])
                    priorityAsTable = true;
            if (priorityAsTable)
            {
                sol::table priorityTable = view.newTable();
                for (uint32_t i = 0; i < MWRender::sNumBlendMasks; i++)
                    priorityTable[static_cast<MWRender::BoneGroup>(i)] = priority[static_cast<MWRender::BoneGroup>(i)];
                options["priority"] = priorityTable;
            }
            else
                options["priority"] = priority[MWRender::BoneGroup_LowerBody];

            // mEngineEvents.addToQueue(event);
            //  Has to be called immediately, otherwise engine details that depend on animations playing immediately
            //  break.
            if (auto* scripts = actor.getRefData().getLuaScripts())
                scripts->onPlayAnimation(groupname, options);
        });
    }

    void LuaManager::animationEnded(const MWWorld::Ptr& actor, std::string_view groupname, float time, float completion,
        std::string_view startKey, std::string_view stopKey)
    {
        mEngineEvents.addToQueue(EngineEvents::OnAnimationEnded{
            getId(actor), std::string(groupname), std::string(startKey), std::string(stopKey), time, completion });
    }

    void LuaManager::skillUse(const MWWorld::Ptr& actor, ESM::RefId skillId, int useType, float scale)
    {
        mEngineEvents.addToQueue(EngineEvents::OnSkillUse{ getId(actor), skillId.serializeText(), useType, scale });
    }

    void LuaManager::skillLevelUp(const MWWorld::Ptr& actor, ESM::RefId skillId, std::string_view source)
    {
        mEngineEvents.addToQueue(
            EngineEvents::OnSkillLevelUp{ getId(actor), skillId.serializeText(), std::string(source) });
    }

    void LuaManager::jailTimeServed(const MWWorld::Ptr& actor, int days)
    {
        mEngineEvents.addToQueue(EngineEvents::OnJailTimeServed{ getId(actor), days });
    }

    void LuaManager::onHit(const MWWorld::Ptr& attacker, const MWWorld::Ptr& victim, const MWWorld::Ptr& weapon,
        const MWWorld::Ptr& ammo, int attackType, float attackStrength, float attackWindUp, float damage, bool isHealth,
        const osg::Vec3f& hitPos, bool successful, MWMechanics::DamageSourceType sourceType)
    {
        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            sol::table damageTable = view.newTable();
            if (isHealth)
                damageTable["health"] = damage;
            else
                damageTable["fatigue"] = damage;

            sol::table data = view.newTable();
            if (!attacker.isEmpty())
                data["attacker"] = LObject(attacker);
            if (!weapon.isEmpty())
                data["weapon"] = LObject(weapon);
            if (!ammo.isEmpty())
                data["ammo"] = ammo.getCellRef().getRefId().serializeText();
            data["type"] = attackType;
            data["strength"] = attackStrength;
            data["windUp"] = attackWindUp;
            data["damage"] = damageTable;
            data["hitPos"] = hitPos;
            data["successful"] = successful;
            switch (sourceType)
            {
                case MWMechanics::DamageSourceType::Unspecified:
                    data["sourceType"] = "unspecified";
                    break;
                case MWMechanics::DamageSourceType::Melee:
                    data["sourceType"] = "melee";
                    break;
                case MWMechanics::DamageSourceType::Ranged:
                    data["sourceType"] = "ranged";
                    break;
                case MWMechanics::DamageSourceType::Magical:
                    data["sourceType"] = "magic";
                    break;
            }

            sendLocalEvent(victim, "Hit", data);
        });
    }

    void LuaManager::objectAddedToScene(const MWWorld::Ptr& ptr)
    {
        mObjectLists.objectAddedToScene(ptr); // assigns generated RefNum if it is not set yet.
        mEngineEvents.addToQueue(EngineEvents::OnActive{ getId(ptr) });

        LocalScripts* localScripts = ptr.getRefData().getLuaScripts();
        if (!localScripts)
        {
            LuaUtil::ScriptIdsWithInitializationData autoStartConf
                = mConfiguration.getLocalConf(getLiveCellRefType(ptr.mRef), ptr.getCellRef().getRefId(), getId(ptr));
            if (!autoStartConf.empty())
            {
                localScripts = createLocalScripts(ptr, std::move(autoStartConf));
                mQueuedAutoStartedScripts.push_back(localScripts->getWeakPointer());
            }
        }
        if (localScripts)
            mActiveLocalScripts.insert(localScripts->getWeakPointer());
    }

    void LuaManager::objectRemovedFromScene(const MWWorld::Ptr& ptr)
    {
        mObjectLists.objectRemovedFromScene(ptr);
        LocalScripts* localScripts = ptr.getRefData().getLuaScripts();
        if (localScripts)
        {
            // TODO replace with mActiveLocalScripts.erase(localScripts) when we switch to C++23
            auto it = mActiveLocalScripts.find(localScripts);
            if (it != mActiveLocalScripts.end())
                mActiveLocalScripts.erase(it);
            if (!MWBase::Environment::get().getWorldModel()->getPtr(getId(ptr)).isEmpty())
                mEngineEvents.addToQueue(EngineEvents::OnInactive{ getId(ptr) });
        }
    }

    void LuaManager::inputEvent(const InputEvent& event)
    {
        if (!MyGUI::InputManager::getInstance().isModalAny()
            && !MWBase::Environment::get().getWindowManager()->isConsoleMode())
        {
            mInputEvents.push_back(event);
        }
        mMenuInputEvents.push_back(event);
    }

    MWBase::LuaManager::ActorControls* LuaManager::getActorControls(const MWWorld::Ptr& ptr) const
    {
        LocalScripts* localScripts = ptr.getRefData().getLuaScripts();
        if (!localScripts)
            return nullptr;
        return localScripts->getActorControls();
    }

    void LuaManager::addCustomLocalScript(const MWWorld::Ptr& ptr, int scriptId, std::string_view initData)
    {
        LocalScripts* localScripts = ptr.getRefData().getLuaScripts();
        if (!localScripts)
        {
            localScripts = createLocalScripts(ptr);
            localScripts->addAutoStartedScripts();
            if (ptr.isInCell() && MWBase::Environment::get().getWorldScene()->isCellActive(*ptr.getCell()))
            {
                localScripts->setActive(true, false);
                mActiveLocalScripts.insert(localScripts->getWeakPointer());
            }
        }
        localScripts->addCustomScript(scriptId, initData);
    }

    LocalScripts* LuaManager::createLocalScripts(
        const MWWorld::Ptr& ptr, std::optional<LuaUtil::ScriptIdsWithInitializationData> autoStartConf)
    {
        assert(mInitialized);
        std::shared_ptr<LocalScripts> scripts;
        const uint32_t type = getLiveCellRefType(ptr.mRef);
        if (type == ESM::REC_STAT)
            throw std::runtime_error("Lua scripts on static objects are not allowed");
        else if (type == ESM::REC_INTERNAL_PLAYER)
        {
            scripts = std::make_shared<PlayerScripts>(&mLua, LObject(getId(ptr)));
            scripts->setAutoStartConf(mConfiguration.getPlayerConf());
            for (const auto& [name, package] : mPlayerPackages)
                scripts->addPackage(name, package);
        }
        else
        {
            scripts = std::make_shared<LocalScripts>(&mLua, LObject(getId(ptr)), &mScriptTracker);
            if (!autoStartConf.has_value())
                autoStartConf = mConfiguration.getLocalConf(type, ptr.getCellRef().getRefId(), getId(ptr));
            scripts->setAutoStartConf(std::move(*autoStartConf));
            for (const auto& [name, package] : mLocalPackages)
                scripts->addPackage(name, package);
        }
        scripts->setSerializer(mLocalSerializer.get());

        MWWorld::RefData& refData = ptr.getRefData();
        refData.setLuaScripts(std::move(scripts));
        return refData.getLuaScripts();
    }

    void LuaManager::write(ESM::ESMWriter& writer, Loading::Listener& progress)
    {
        writer.startRecord(ESM::REC_LUAM);

        writer.writeHNT<double>("LUAW", MWBase::Environment::get().getWorld()->getTimeManager()->getSimulationTime());
        writer.writeFormId(MWBase::Environment::get().getWorldModel()->getLastGeneratedRefNum(), true);
        mConfiguration.write(writer);
        ESM::LuaScripts globalScripts;
        mGlobalScripts.save(globalScripts);
        globalScripts.save(writer);
        mLuaEvents.save(writer);

        writer.endRecord(ESM::REC_LUAM);
    }

    void LuaManager::readRecord(ESM::ESMReader& reader, uint32_t type)
    {
        if (type != ESM::REC_LUAM)
            throw std::runtime_error("ESM::REC_LUAM is expected");

        double simulationTime;
        reader.getHNT(simulationTime, "LUAW");
        MWBase::Environment::get().getWorld()->getTimeManager()->setSimulationTime(simulationTime);
        ESM::FormId lastGenerated = reader.getFormId(true);
        if (lastGenerated.hasContentFile())
            throw std::runtime_error("Last generated RefNum is invalid");
        MWBase::Environment::get().getWorldModel()->setLastGeneratedRefNum(lastGenerated);

        mConfiguration.read(reader);

        // TODO: don't execute scripts right away, it will be necessary in multiplayer where global storage requires
        // initialization. For now just set global storage as active slightly before it would be set by gameLoaded()
        mGlobalStorage.setActive(true);

        ESM::LuaScripts globalScripts;
        globalScripts.load(reader);
        mLua.protectedCall([&](LuaUtil::LuaView& view) {
            mLuaEvents.load(view.sol(), reader, mContentFileMapping, mGlobalLoader.get());
        });

        mGlobalScripts.setSavedDataDeserializer(mGlobalLoader.get());
        mGlobalScripts.load(globalScripts);
        mGlobalScriptsStarted = true;
    }

    void LuaManager::saveLocalScripts(const MWWorld::Ptr& ptr, ESM::LuaScripts& data)
    {
        if (ptr.getRefData().getLuaScripts())
            ptr.getRefData().getLuaScripts()->save(data);
        else
            data.mScripts.clear();
    }

    void LuaManager::loadLocalScripts(const MWWorld::Ptr& ptr, const ESM::LuaScripts& data)
    {
        if (data.mScripts.empty())
        {
            if (ptr.getRefData().getLuaScripts())
                ptr.getRefData().setLuaScripts(nullptr);
            return;
        }

        MWBase::Environment::get().getWorldModel()->registerPtr(ptr);
        LocalScripts* scripts = createLocalScripts(ptr);

        scripts->setSerializer(mLocalSerializer.get());
        scripts->setSavedDataDeserializer(mLocalLoader.get());
        scripts->load(data);
    }

    void LuaManager::reloadAllScriptsImpl()
    {
        Log(Debug::Info) << "Reload Lua";

        LuaUi::clearGameInterface();
        LuaUi::clearMenuInterface();
        LuaUi::clearSettings();
        MWBase::Environment::get().getWindowManager()->setConsoleMode("");
        MWBase::Environment::get().getL10nManager()->dropCache();
        mUiResourceManager.clear();
        mLua.dropScriptCache();
        mInputActions.clear(true);
        mInputTriggers.clear(true);

        ESM::LuaScripts globalData;

        if (mGlobalScriptsStarted)
        {
            mGlobalScripts.setSavedDataDeserializer(mGlobalSerializer.get());
            mGlobalScripts.save(globalData);
            mGlobalStorage.clearTemporaryAndRemoveCallbacks();
        }

        std::unordered_map<ESM::RefNum, ESM::LuaScripts> localData;

        for (const auto& [id, ptr] : MWBase::Environment::get().getWorldModel()->getPtrRegistryView())
        {
            LocalScripts* scripts = ptr.getRefData().getLuaScripts();
            if (scripts == nullptr)
                continue;
            scripts->setSavedDataDeserializer(mLocalSerializer.get());
            ESM::LuaScripts data;
            scripts->save(data);
            localData[id] = std::move(data);
        }

        initConfiguration(true);

        mMenuScripts.removeAllScripts();

        mPlayerStorage.clearTemporaryAndRemoveCallbacks();

        mMenuScripts.addAutoStartedScripts();

        for (const auto& [id, ptr] : MWBase::Environment::get().getWorldModel()->getPtrRegistryView())
        {
            LocalScripts* scripts = ptr.getRefData().getLuaScripts();
            if (scripts == nullptr)
                continue;
            scripts->load(localData[id]);
        }

        for (const LuaUtil::ScriptsContainerWeakPtr& ptr : mActiveLocalScripts)
        {
            if (LocalScripts* scripts = asLocal(ptr))
                scripts->setActive(true);
        }

        if (mGlobalScriptsStarted)
        {
            mGlobalScripts.load(globalData);
        }
    }

    void LuaManager::handleConsoleCommand(
        const std::string& consoleMode, const std::string& command, const MWWorld::Ptr& selectedPtr)
    {
        PlayerScripts* playerScripts = nullptr;
        if (!mPlayer.isEmpty())
            playerScripts = dynamic_cast<PlayerScripts*>(mPlayer.getRefData().getLuaScripts());
        bool processed = mMenuScripts.consoleCommand(consoleMode, command);
        if (playerScripts)
        {
            sol::object selected = sol::nil;
            if (!selectedPtr.isEmpty())
                mLua.protectedCall([&](LuaUtil::LuaView& view) {
                    selected = sol::make_object(view.sol(), LObject(getId(selectedPtr)));
                });
            if (playerScripts->consoleCommand(consoleMode, command, selected))
                processed = true;
        }
        if (!processed)
            MWBase::Environment::get().getWindowManager()->printToConsole(
                "No Lua handlers for console\n", MWBase::WindowManager::sConsoleColor_Error);
    }

    LuaManager::DelayedAction::DelayedAction(LuaUtil::LuaState* state, std::function<void()> fn, std::string_view name)
        : mFn(std::move(fn))
        , mName(name)
    {
        if (Settings::lua().mLuaDebug)
            mCallerTraceback = state->debugTraceback();
    }

    void LuaManager::DelayedAction::apply() const
    {
        // Timed, and named when it overruns.
        //
        // These run on the main thread inside synchronizedUpdate, so a slow one is a stall with no other
        // symptom: the frame breakdown can only say "lua sync", which is true of every action there is.
        // Two measured stalls came through here -- 764 ms building a nested MyGUI widget tree via
        // LuaUi::Element::create, and 792 ms inside ActionTeleport::teleport waiting on terrain load -- and
        // neither could be attributed to a mod without this. mName is free; the traceback needs
        // 'lua debug = true', which is worth one run to identify a culprit.
        //
        // The threshold is deliberately well above anything an action should cost, so this stays silent
        // unless something is genuinely wrong.
        constexpr double overrunMs = 100.0;
        const auto started = std::chrono::steady_clock::now();

        try
        {
            mFn();
        }
        catch (const std::exception& e)
        {
            Log(Debug::Error) << "Error in DelayedAction " << mName << ": " << e.what();

            if (mCallerTraceback.empty())
                Log(Debug::Error) << "Set 'lua debug=true' in settings.cfg to enable action tracebacks";
            else
                Log(Debug::Error) << "Caller " << mCallerTraceback;
        }

        const double elapsedMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (elapsedMs >= overrunMs)
        {
            Log(Debug::Warning) << "Lua action '" << mName << "' held the main thread for " << elapsedMs
                                << " ms";
            if (mCallerTraceback.empty())
                Log(Debug::Warning) << "  set 'lua debug = true' in settings.cfg to learn which script "
                                       "queued it";
            else
                Log(Debug::Warning) << "  queued by " << mCallerTraceback;
        }
    }

    void LuaManager::addAction(std::function<void()> action, std::string_view name)
    {
        if (mApplyingDelayedActions)
            throw std::runtime_error("DelayedAction is not allowed to create another DelayedAction");
        mActionQueue.emplace_back(&mLua, std::move(action), name);
    }

    void LuaManager::addTeleportPlayerAction(std::function<void()> action)
    {
        mTeleportPlayerAction = DelayedAction(&mLua, std::move(action), "TeleportPlayer");
    }

    void LuaManager::reportStats(unsigned int frameNumber, osg::Stats& stats) const
    {
        stats.setAttribute(frameNumber, "Lua UsedMemory", static_cast<double>(mLua.getTotalMemoryUsage()));
    }

    std::string LuaManager::formatResourceUsageStats() const
    {
        if (!LuaUtil::LuaState::isProfilerEnabled())
            return "Lua profiler is disabled";

        std::stringstream out;

        constexpr unsigned nameW = 50;
        constexpr int valueW = 12;

        auto outMemSize = [&](size_t bytes) {
            constexpr size_t limit = 10000;
            out << std::right << std::setw(valueW - 3);
            if (bytes < limit)
                out << bytes << " B ";
            else if (bytes < limit * 1024)
                out << (bytes / 1024) << " KB";
            else if (bytes < limit * 1024 * 1024)
                out << (bytes / (1024 * 1024)) << " MB";
            else
                out << (bytes / (1024 * 1024 * 1024)) << " GB";
        };

        const uint64_t smallAllocSize = Settings::lua().mSmallAllocMaxSize;
        out << "Total memory usage:";
        outMemSize(mLua.getTotalMemoryUsage());
        out << "\n";
        out << "LuaUtil::ScriptsContainer count: " << LuaUtil::ScriptsContainer::getInstanceCount() << "\n";
        out << "\n";
        out << "small alloc max size = " << smallAllocSize << " (section [Lua] in settings.cfg)\n";
        out << "Smaller values give more information for the profiler, but increase performance overhead.\n";
        out << "  Memory allocations <= " << smallAllocSize << " bytes:";
        outMemSize(mLua.getSmallAllocMemoryUsage());
        out << " (not tracked)\n";
        out << "  Memory allocations >  " << smallAllocSize << " bytes:";
        outMemSize(mLua.getTotalMemoryUsage() - mLua.getSmallAllocMemoryUsage());
        out << " (see the table below)\n\n";

        using Stats = LuaUtil::ScriptsContainer::ScriptStats;

        std::vector<Stats> activeStats;
        mGlobalScripts.collectStats(activeStats);
        for (const LuaUtil::ScriptsContainerWeakPtr& ptr : mActiveLocalScripts)
        {
            if (LocalScripts* scripts = asLocal(ptr))
                scripts->collectStats(activeStats);
        }

        std::vector<Stats> selectedStats;
        MWWorld::Ptr selectedPtr = MWBase::Environment::get().getWindowManager()->getConsoleSelectedObject();
        LocalScripts* selectedScripts = nullptr;
        if (!selectedPtr.isEmpty())
        {
            selectedScripts = selectedPtr.getRefData().getLuaScripts();
            if (selectedScripts)
                selectedScripts->collectStats(selectedStats);
            out << "Profiled object (selected in the in-game console): " << selectedPtr.toString() << "\n";
        }
        else
            out << "No selected object. Use the in-game console to select an object for detailed profile.\n";
        out << "\n";

        out << "Legend\n";
        out << "  ops:        Averaged number of Lua instruction per frame;\n";
        out << "  memory:     Aggregated size of Lua allocations > " << smallAllocSize << " bytes;\n";
        out << "  [all]:      Sum over all instances of each script;\n";
        out << "  [active]:   Sum over all active (i.e. currently in scene) instances of each script;\n";
        out << "  [inactive]: Sum over all inactive instances of each script;\n";
        out << "  [for selected object]: Only for the object that is selected in the console;\n";
        out << "\n";

        out << std::left;
        out << " " << std::setw(nameW + 2) << "*** Resource usage per script";
        out << std::right;
        out << std::setw(valueW) << "ops";
        out << std::setw(valueW) << "memory";
        out << std::setw(valueW) << "memory";
        out << std::setw(valueW) << "ops";
        out << std::setw(valueW) << "memory";
        out << "\n";
        out << std::left << " " << std::setw(nameW + 2) << "[name]" << std::right;
        out << std::setw(valueW) << "[all]";
        out << std::setw(valueW) << "[active]";
        out << std::setw(valueW) << "[inactive]";
        out << std::setw(valueW * 2) << "[for selected object]";
        out << "\n";

        for (size_t i = 0; i < mConfiguration.size(); ++i)
        {
            bool isGlobal = mConfiguration[i].mFlags & ESM::LuaScriptCfg::sGlobal;
            bool isMenu = mConfiguration[i].mFlags & ESM::LuaScriptCfg::sMenu;

            out << std::left;
            out << " " << std::setw(nameW) << mConfiguration[i].mScriptPath.value();
            if (mConfiguration[i].mScriptPath.value().size() > nameW)
                out << "\n " << std::setw(nameW) << ""; // if path is too long, break line
            out << std::right;
            out << std::setw(valueW) << static_cast<int64_t>(activeStats[i].mAvgInstructionCount);
            outMemSize(static_cast<size_t>(activeStats[i].mMemoryUsage));
            outMemSize(mLua.getMemoryUsageByScriptIndex(static_cast<unsigned>(i))
                - static_cast<uint64_t>(activeStats[i].mMemoryUsage));

            if (isGlobal)
                out << std::setw(valueW * 2) << "NA (global script)";
            else if (isMenu && (!selectedScripts || !selectedScripts->hasScript(static_cast<int>(i))))
                out << std::setw(valueW * 2) << "NA (menu script)";
            else if (selectedPtr.isEmpty())
                out << std::setw(valueW * 2) << "NA (not selected) ";
            else if (!selectedScripts || !selectedScripts->hasScript(static_cast<int>(i)))
                out << std::setw(valueW * 2) << "NA";
            else
            {
                out << std::setw(valueW) << static_cast<int64_t>(selectedStats[i].mAvgInstructionCount);
                outMemSize(static_cast<size_t>(selectedStats[i].mMemoryUsage));
            }
            out << "\n";
        }

        return out.str();
    }
}
