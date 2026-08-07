#include "engineevents.hpp"

#include <chrono>
#include <iterator>

#include <components/debug/debuglog.hpp>
#include <components/settings/values.hpp>

#include "../mwbase/environment.hpp"
#include "../mwworld/class.hpp"
#include "../mwworld/worldmodel.hpp"

#include "globalscripts.hpp"
#include "localscripts.hpp"
#include "object.hpp"

namespace MWLua
{

    class EngineEvents::Visitor
    {
    public:
        explicit Visitor(GlobalScripts& globalScripts)
            : mGlobalScripts(globalScripts)
        {
        }

        void operator()(const OnActive& event) const
        {
            MWWorld::Ptr ptr = getPtr(event.mObject);
            if (ptr.isEmpty())
                return;
            if (ptr.getCellRef().getRefId() == "player")
                mGlobalScripts.playerAdded(GObject(ptr));
            else
            {
                mGlobalScripts.objectActive(GObject(ptr));
                const MWWorld::Class& objClass = ptr.getClass();
                if (objClass.isActor())
                    mGlobalScripts.actorActive(GObject(ptr));
                if (objClass.isItem(ptr))
                    mGlobalScripts.itemActive(GObject(ptr));
            }
            if (auto* scripts = getLocalScripts(ptr))
                scripts->setActive(true);
        }

        void operator()(const OnInactive& event) const
        {
            if (auto* scripts = getLocalScripts(event.mObject))
                scripts->setActive(false);
        }

        void operator()(const OnTeleported& event) const
        {
            if (auto* scripts = getLocalScripts(event.mObject))
                scripts->onTeleported();
        }

        void operator()(const OnActivate& event) const
        {
            MWWorld::Ptr obj = getPtr(event.mObject);
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (actor.isEmpty() || obj.isEmpty())
                return;
            mGlobalScripts.onActivate(GObject(obj), GObject(actor));
            if (auto* scripts = getLocalScripts(obj))
                scripts->onActivated(LObject(actor));
        }

        void operator()(const OnUseItem& event) const
        {
            MWWorld::Ptr obj = getPtr(event.mObject);
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (actor.isEmpty() || obj.isEmpty())
                return;
            mGlobalScripts.onUseItem(GObject(obj), GObject(actor), event.mForce);
        }

        void operator()(const OnConsume& event) const
        {
            MWWorld::Ptr actor = getPtr(event.mActor);
            MWWorld::Ptr consumable = getPtr(event.mConsumable);
            if (actor.isEmpty() || consumable.isEmpty())
                return;
            if (auto* scripts = getLocalScripts(actor))
                scripts->onConsume(LObject(consumable));
        }

        void operator()(const OnDropped& event) const
        {
            MWWorld::Ptr obj = getPtr(event.mObject);
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (obj.isEmpty() || actor.isEmpty())
                return;
            mGlobalScripts.onDropped(GObject(obj), GObject(actor), event.mPosition, event.mRotation);
        }

        void operator()(const OnPlaced& event) const
        {
            MWWorld::Ptr obj = getPtr(event.mObject);
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (obj.isEmpty() || actor.isEmpty())
                return;
            mGlobalScripts.onPlaced(GObject(obj), GObject(actor), event.mPosition, event.mRotation);
        }

        void operator()(const OnNewExterior& event) const { mGlobalScripts.onNewExterior(GCell{ &event.mCell }); }

        void operator()(const OnAnimationTextKey& event) const
        {
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (actor.isEmpty())
                return;
            if (auto* scripts = getLocalScripts(actor))
                scripts->onAnimationTextKey(event.mGroupname, event.mKey);
        }

        void operator()(const OnAnimationEnded& event) const
        {
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (actor.isEmpty())
                return;
            if (auto* scripts = getLocalScripts(actor))
                scripts->onAnimationEnded(
                    event.mGroupname, event.mStartKey, event.mStopKey, event.mTime, event.mCompletion);
        }

        void operator()(const OnSkillUse& event) const
        {
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (actor.isEmpty())
                return;
            if (auto* scripts = getLocalScripts(actor))
                scripts->onSkillUse(event.mSkill, event.useType, event.scale);
        }

        void operator()(const OnSkillLevelUp& event) const
        {
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (actor.isEmpty())
                return;
            if (auto* scripts = getLocalScripts(actor))
                scripts->onSkillLevelUp(event.mSkill, event.mSource);
        }

        void operator()(const OnJailTimeServed& event) const
        {
            MWWorld::Ptr actor = getPtr(event.mActor);
            if (actor.isEmpty())
                return;
            if (auto* scripts = getLocalScripts(actor))
                scripts->onJailTimeServed(event.mDays);
        }

    private:
        MWWorld::Ptr getPtr(ESM::RefNum id) const
        {
            MWWorld::Ptr res = mWorldModel->getPtr(id);
            if (res.isEmpty() && Settings::lua().mLuaDebug)
                Log(Debug::Verbose) << "Can not find object" << id.toString() << " when calling engine hanglers";
            return res;
        }

        LocalScripts* getLocalScripts(const MWWorld::Ptr& ptr) const
        {
            if (ptr.isEmpty())
                return nullptr;
            else
                return ptr.getRefData().getLuaScripts();
        }

        LocalScripts* getLocalScripts(ESM::RefNum id) const { return getLocalScripts(getPtr(id)); }

        GlobalScripts& mGlobalScripts;
        MWWorld::WorldModel* mWorldModel = MWBase::Environment::get().getWorldModel();
    };

    void EngineEvents::callEngineHandlers(double& budgetMs, bool enforceBudget)
    {
        Visitor vis(mGlobalScripts);

        // Swapped out before dispatch rather than iterated in place. A handler can queue another engine
        // event, and the previous form walked mQueue with a range-for while that was possible; retaining a
        // tail makes that hazard sharper, so the batch is detached first and the remainder put back
        // afterwards -- ahead of anything the handlers added, which keeps the queue in order.
        std::vector<Event> batch;
        batch.swap(mQueue);

        if (!enforceBudget)
        {
            for (const Event& event : batch)
                std::visit(vis, event);
            return;
        }

        const auto start = std::chrono::steady_clock::now();
        std::size_t processed = 0;
        for (const Event& event : batch)
        {
            std::visit(vis, event);
            ++processed;
            // One event always goes through even when the budget arrived already spent, so a frame whose
            // earlier phases used everything still drains the queue rather than stalling it indefinitely.
            if (processed == 1 && budgetMs <= 0.0 && batch.size() > 1)
                break;
            // Checked after dispatch, not before, because whether an event instantiates anything is not
            // knowable from the event alone -- OnActive only costs something when the object's scripts are
            // not loaded yet. The cost of that is overshooting the budget by at most one event.
            const double spent
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (spent >= budgetMs && processed < batch.size())
                break;
        }
        budgetMs -= std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        if (processed < batch.size())
        {
            // Rebuilt by move-construction rather than inserted at the front. OnNewExterior holds a
            // CellStore reference, so Event has no move assignment, and vector::insert needs assignment to
            // shift the elements already present.
            std::vector<Event> remaining;
            remaining.reserve(batch.size() - processed + mQueue.size());
            for (std::size_t i = processed; i < batch.size(); ++i)
                remaining.push_back(std::move(batch[i]));
            // Anything a handler queued during this pass goes after the deferred tail, preserving order.
            for (Event& queued : mQueue)
                remaining.push_back(std::move(queued));
            mQueue = std::move(remaining);
        }
    }

}
