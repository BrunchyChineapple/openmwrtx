#include "luaevents.hpp"

#include <chrono>

#include <components/debug/debuglog.hpp>

#include <components/esm/luascripts.hpp>
#include <components/esm3/esmreader.hpp>
#include <components/esm3/esmwriter.hpp>

#include <components/lua/serialization.hpp>

#include "../mwbase/environment.hpp"
#include "../mwworld/worldmodel.hpp"

#include "globalscripts.hpp"
#include "localscripts.hpp"
#include "menuscripts.hpp"

namespace MWLua
{

    void LuaEvents::clear()
    {
        mGlobalEventBatch.clear();
        mLocalEventBatch.clear();
        mNewGlobalEventBatch.clear();
        mNewLocalEventBatch.clear();
        mMenuEvents.clear();
    }

    void LuaEvents::finalizeEventBatch()
    {
        mNewGlobalEventBatch.swap(mGlobalEventBatch);
        mNewLocalEventBatch.swap(mLocalEventBatch);
        mNewGlobalEventBatch.clear();
        mNewLocalEventBatch.clear();
    }

    void LuaEvents::callEventHandlers(double& loadBudgetMs, bool enforceBudget)
    {
        // Two clock reads per event, against a phase that has been measured at 340 ms. Unconditional for
        // the same reason the per-script timing is: the report it feeds has to work on a default install,
        // not only when someone has already turned a profiler on.
        mFrameEventCosts.clear();
        const auto charge = [this](const std::string& name, const std::chrono::steady_clock::time_point& started) {
            EventCost& cost = mFrameEventCosts[name];
            cost.mMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            cost.mCalls += 1;
        };

        // Global events go to a single container that is loaded once, early, so there is no instantiation
        // here to budget for.
        for (const Global& e : mGlobalEventBatch)
        {
            const auto started = std::chrono::steady_clock::now();
            mGlobalScripts.receiveEvent(e.mEventName, e.mEventData);
            charge(e.mEventName, started);
        }
        mGlobalEventBatch.clear();

        // Delivering a local event is also a door for script instantiation, and it was the last one left
        // unbudgeted.
        //
        // ScriptsContainer::receiveEvent begins with ensureLoaded, so an event addressed to an object whose
        // scripts are not loaded yet instantiates every script attached to it before any handler runs --
        // including the ones with no handler for that event and no interest in it. The cost therefore has
        // nothing to do with the event that paid it, which is how one frame came to spend 346 ms across 16
        // deliveries of a Bardcraft event with 321 ms of it inside sitDownPlease's seeker: a 5,000-line
        // script being loaded onto sixteen actors, charged to whichever event happened to reach them first.
        //
        // Same treatment as the other three doors, and its own budget rather than the remainder of theirs,
        // for the reason given at callEngineHandlers: sharing starves whichever door runs last. A frame can
        // now spend up to three budgets on instantiation, which at the default of 4 ms is still bounded and
        // still small beside the stall it replaces.
        //
        // Deferring an event costs it a frame or two. Events are already batched a frame behind their
        // sender, and both remaining batches are saved, so nothing is lost or reordered -- the remainder is
        // put back ahead of anything queued during dispatch.
        const auto start = std::chrono::steady_clock::now();
        std::size_t processed = 0;
        for (std::size_t i = 0; i < mLocalEventBatch.size(); ++i)
        {
            const Local& e = mLocalEventBatch[i];
            MWWorld::Ptr ptr = MWBase::Environment::get().getWorldModel()->getPtr(e.mDest);
            LocalScripts* scripts = ptr.isEmpty() ? nullptr : ptr.getRefData().getLuaScripts();
            if (scripts)
            {
                const auto started = std::chrono::steady_clock::now();
                scripts->receiveEvent(e.mEventName, e.mEventData);
                charge(e.mEventName, started);
            }
            else
                Log(Debug::Debug) << "Ignored event " << e.mEventName << " to L" << e.mDest.toString()
                                  << ". Object not found or has no attached scripts";
            ++processed;

            if (!enforceBudget)
                continue;
            // One event always goes through even when the budget arrived already spent, so a frame whose
            // earlier phases used everything still drains the batch rather than stalling it indefinitely.
            if (processed == 1 && loadBudgetMs <= 0.0 && mLocalEventBatch.size() > 1)
                break;
            // Checked after dispatch rather than before, because whether an event instantiates anything is
            // not knowable from the event alone. The cost is overshooting by at most one event.
            const double spent
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (spent >= loadBudgetMs && processed < mLocalEventBatch.size())
                break;
        }
        if (enforceBudget)
            loadBudgetMs
                -= std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        if (processed < mLocalEventBatch.size())
        {
            std::vector<Local> remaining;
            remaining.reserve(mLocalEventBatch.size() - processed + mNewLocalEventBatch.size());
            for (std::size_t i = processed; i < mLocalEventBatch.size(); ++i)
                remaining.push_back(std::move(mLocalEventBatch[i]));
            for (Local& queued : mNewLocalEventBatch)
                remaining.push_back(std::move(queued));
            mNewLocalEventBatch.swap(remaining);
        }
        mLocalEventBatch.clear();
    }

    void LuaEvents::callMenuEventHandlers()
    {
        for (const Global& e : mMenuEvents)
            mMenuScripts.receiveEvent(e.mEventName, e.mEventData);
        mMenuEvents.clear();
    }

    template <typename Event>
    static void saveEvent(ESM::ESMWriter& esm, ESM::RefNum dest, const Event& event)
    {
        esm.writeHNString("LUAE", event.mEventName);
        esm.writeFormId(dest, true);
        if (!event.mEventData.empty())
            saveLuaBinaryData(esm, event.mEventData);
    }

    void LuaEvents::load(lua_State* lua, ESM::ESMReader& esm, const std::map<int, int>& contentFileMapping,
        const LuaUtil::UserdataSerializer* serializer)
    {
        clear();
        while (esm.isNextSub("LUAE"))
        {
            std::string name = esm.getHString();
            ESM::RefNum dest = esm.getFormId(true);
            std::string data = loadLuaBinaryData(esm);
            try
            {
                data = LuaUtil::serialize(LuaUtil::deserialize(lua, data, serializer), serializer);
            }
            catch (std::exception& e)
            {
                Log(Debug::Error) << "loadEvent: invalid event data: " << e.what();
            }
            if (dest.isSet())
            {
                auto it = contentFileMapping.find(dest.mContentFile);
                if (it != contentFileMapping.end())
                    dest.mContentFile = it->second;
                mLocalEventBatch.push_back({ dest, std::move(name), std::move(data) });
            }
            else
                mGlobalEventBatch.push_back({ std::move(name), std::move(data) });
        }
    }

    void LuaEvents::save(ESM::ESMWriter& esm) const
    {
        // Used as a marker of a global event.
        constexpr ESM::RefNum globalId;

        for (const Global& e : mGlobalEventBatch)
            saveEvent(esm, globalId, e);
        for (const Global& e : mNewGlobalEventBatch)
            saveEvent(esm, globalId, e);
        for (const Local& e : mLocalEventBatch)
            saveEvent(esm, e.mDest, e);
        for (const Local& e : mNewLocalEventBatch)
            saveEvent(esm, e.mDest, e);
    }

}
