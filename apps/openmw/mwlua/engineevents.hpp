#ifndef MWLUA_ENGINEEVENTS_H
#define MWLUA_ENGINEEVENTS_H

#include <variant>

#include <osg/Vec3f>

#include <components/esm3/cellref.hpp> // defines RefNum that is used as a unique id
#include <components/lua/utilpackage.hpp>

#include "../mwworld/cellstore.hpp"

namespace MWLua
{
    class GlobalScripts;

    class EngineEvents
    {
    public:
        explicit EngineEvents(GlobalScripts& globalScripts)
            : mGlobalScripts(globalScripts)
        {
        }

        struct OnActive
        {
            ESM::RefNum mObject;
        };
        struct OnInactive
        {
            ESM::RefNum mObject;
        };
        struct OnTeleported
        {
            ESM::RefNum mObject;
        };
        struct OnActivate
        {
            ESM::RefNum mActor;
            ESM::RefNum mObject;
        };
        struct OnUseItem
        {
            ESM::RefNum mActor;
            ESM::RefNum mObject;
            bool mForce;
        };
        struct OnConsume
        {
            ESM::RefNum mActor;
            ESM::RefNum mConsumable;
        };
        struct OnNewExterior
        {
            MWWorld::CellStore& mCell;
        };
        struct OnAnimationTextKey
        {
            ESM::RefNum mActor;
            std::string mGroupname;
            std::string mKey;
        };
        struct OnAnimationEnded
        {
            ESM::RefNum mActor;
            std::string mGroupname;
            std::string mStartKey;
            std::string mStopKey;
            float mTime;
            float mCompletion;
        };
        struct OnSkillUse
        {
            ESM::RefNum mActor;
            std::string mSkill;
            int useType;
            float scale;
        };
        struct OnSkillLevelUp
        {
            ESM::RefNum mActor;
            std::string mSkill;
            std::string mSource;
        };
        struct OnJailTimeServed
        {
            ESM::RefNum mActor;
            int mDays;
        };
        struct OnDropped
        {
            ESM::RefNum mObject;
            ESM::RefNum mActor;
            osg::Vec3f mPosition;
            LuaUtil::TransformQ mRotation;
        };
        struct OnPlaced
        {
            ESM::RefNum mObject;
            ESM::RefNum mActor;
            osg::Vec3f mPosition;
            LuaUtil::TransformQ mRotation;
        };
        using Event = std::variant<OnActive, OnInactive, OnConsume, OnActivate, OnUseItem, OnNewExterior, OnTeleported,
            OnAnimationTextKey, OnAnimationEnded, OnSkillUse, OnSkillLevelUp, OnJailTimeServed, OnDropped, OnPlaced>;

        void clear() { mQueue.clear(); }
        void addToQueue(Event e) { mQueue.push_back(std::move(e)); }

        /// Dispatches queued events, spending at most \a budgetMs on it and deducting what it used.
        ///
        /// Needs a budget because OnActive reaches LocalScripts::setActive, which calls engine handlers,
        /// which begins with ensureLoaded -- so this is where a newly active object's scripts are
        /// instantiated, and instantiation runs the top level of the object's whole require graph. A mod
        /// whose NPC script pulls a megabyte of modules costs milliseconds per NPC, and entering a town
        /// activates dozens at once: measured at 512 ms in one frame.
        ///
        /// \a enforceBudget separates "the budget is switched off" from "the budget is used up", which are
        /// opposite instructions and cannot both be spelled with a zero. The first revision of this took
        /// only the remaining budget and treated <= 0 as no limit -- but the phases that run before this one
        /// routinely spend all of it, so an exhausted budget arrived here as permission to do unlimited
        /// work. Engine events then stayed at 148-459 ms while the spans either side of it sat at 4 ms.
        ///
        /// Whatever does not fit stays queued, in order, ahead of anything added later. The queue is
        /// drained from the front and always makes progress by at least one event, so nothing can be
        /// starved by a single expensive one.
        void callEngineHandlers(double& budgetMs, bool enforceBudget);

    private:
        class Visitor;

        GlobalScripts& mGlobalScripts;
        std::vector<Event> mQueue;
    };

}

#endif // MWLUA_ENGINEEVENTS_H
