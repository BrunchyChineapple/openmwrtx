#ifndef MWLUA_LUAEVENTS_H
#define MWLUA_LUAEVENTS_H

#include <map>
#include <string>

#include <components/esm3/cellref.hpp> // defines RefNum that is used as a unique id

struct lua_State;

namespace ESM
{
    class ESMReader;
    class ESMWriter;
}

namespace LuaUtil
{
    class UserdataSerializer;
}

namespace MWLua
{

    class GlobalScripts;
    class MenuScripts;

    class LuaEvents
    {
    public:
        explicit LuaEvents(GlobalScripts& globalScripts, MenuScripts& menuScripts)
            : mGlobalScripts(globalScripts)
            , mMenuScripts(menuScripts)
        {
        }

        struct Global
        {
            std::string mEventName;
            std::string mEventData;
        };
        struct Local
        {
            ESM::RefNum mDest;
            std::string mEventName;
            std::string mEventData;
        };

        void addGlobalEvent(Global event) { mNewGlobalEventBatch.push_back(std::move(event)); }
        void addMenuEvent(Global event) { mMenuEvents.push_back(std::move(event)); }
        void addLocalEvent(Local event) { mNewLocalEventBatch.push_back(std::move(event)); }

        void clear();
        void finalizeEventBatch();
        void callEventHandlers();
        void callMenuEventHandlers();

        /// What one event name cost during the last callEventHandlers.
        struct EventCost
        {
            double mMs = 0.0;
            unsigned int mCalls = 0;
        };

        /// Per-event-name cost of the last callEventHandlers, for the overrun report.
        ///
        /// The phase breakdown says "event handlers" and the per-script attribution says which script, but
        /// between them they still do not say which event -- and a script with thirty handlers is not a
        /// lead. This is the one place where the name and the call are both in hand, so it is measured here
        /// rather than reconstructed later.
        ///
        /// Summing is safe: an event sent from inside a handler lands in the next batch rather than nesting
        /// inside this one, so no cost is counted twice.
        const std::map<std::string, EventCost>& lastFrameEventCosts() const { return mFrameEventCosts; }

        void load(lua_State* lua, ESM::ESMReader& esm, const std::map<int, int>& contentFileMapping,
            const LuaUtil::UserdataSerializer* serializer);
        void save(ESM::ESMWriter& esm) const;

    private:
        GlobalScripts& mGlobalScripts;
        MenuScripts& mMenuScripts;
        std::vector<Global> mNewGlobalEventBatch;
        std::vector<Local> mNewLocalEventBatch;
        std::vector<Global> mGlobalEventBatch;
        std::vector<Local> mLocalEventBatch;
        std::vector<Global> mMenuEvents;
        std::map<std::string, EventCost> mFrameEventCosts;
    };

}

#endif // MWLUA_LUAEVENTS_H
