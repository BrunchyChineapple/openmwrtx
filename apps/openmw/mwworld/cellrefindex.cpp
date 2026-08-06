#include "cellrefindex.hpp"

#include <algorithm>
#include <chrono>
#include <exception>

#include <components/debug/debuglog.hpp>
#include <components/esm3/cellref.hpp>
#include <components/esm3/loadcell.hpp>
#include <components/esm3/readerscache.hpp>

#include "esmstore.hpp"
#include "store.hpp"

namespace MWWorld
{
    bool CellRefIndex::Query::cannotContain(const ESM::RefId& cellId) const
    {
        if (mIndex == nullptr)
            return false;

        // Per cell, not per index. An unfinished index is still the truth about the cells it has read, and
        // this measured 39,342 cells on one load order -- a job far too big to be worth nothing until the
        // last cell is done. Answering per cell means the very first search skips whatever has been read so
        // far and only pays for the remainder, so there is no window in which the index exists but helps
        // nobody.
        if (mIndex->mIndexedCells.find(cellId) == mIndex->mIndexedCells.end())
            return false;

        // A cell that could not be read is not a cell we know anything about.
        if (mIndex->mUnreadable.find(cellId) != mIndex->mUnreadable.end())
            return false;

        // No indexed cell declares this id at all, so this one, being indexed, does not either.
        if (mCells == nullptr)
            return true;

        // Kept sorted on insertion rather than sorted at the end, because the answers are wanted while the
        // build is still running. The lists are short, so an ordered insert costs less than the searches it
        // saves for an id that appears in hundreds of cells.
        return !std::binary_search(mCells->begin(), mCells->end(), cellId);
    }

    CellRefIndex::Query CellRefIndex::query(const ESM::RefId& refId) const
    {
        Query result;
        if (!mStarted)
            return result;

        result.mIndex = this;
        const auto it = mCellsByRefId.find(refId);
        if (it != mCellsByRefId.end())
            result.mCells = &it->second;
        return result;
    }

    void CellRefIndex::advance(const ESMStore& store, ESM::ReadersCache& readers, double budgetMs)
    {
        if (mComplete || budgetMs <= 0.0)
            return;

        if (!mStarted)
        {
            mStarted = true;
            mFirstAdvance = std::chrono::steady_clock::now();

            // Exteriors then interiors, the order the fallback visits them in. The index is a set rather
            // than an ordering, so this is only so progress reads comparably to the work it replaces.
            const Store<ESM::Cell>& cells = store.get<ESM::Cell>();
            mPending.reserve(cells.getSize());
            for (auto it = cells.extBegin(); it != cells.extEnd(); ++it)
                mPending.push_back(&*it);
            for (auto it = cells.intBegin(); it != cells.intEnd(); ++it)
                mPending.push_back(&*it);

            Log(Debug::Info) << "Indexing reference ids across " << mPending.size()
                             << " cells, so that looking an object up by id need not read them";
        }

        const auto started = std::chrono::steady_clock::now();
        const auto elapsedMs = [&started] {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        };



        // At least one cell per call, so a budget smaller than a single cell still makes progress instead of
        // stalling the build forever.
        do
        {
            indexCell(*mPending[mCursor], readers);
            ++mCursor;
        } while (mCursor < mPending.size() && elapsedMs() < budgetMs);

        // Accumulated so completion can report both how much work there was and how long the chosen budget
        // took to get through it. The budget is a guess until measured once, and the gap between the two
        // numbers is how much of a session still runs on the old behaviour.
        mWorkMs += elapsedMs();

        if (mCursor < mPending.size())
        {
            // Progress, at a granularity that says something without filling the log. Worth reporting at all
            // because the job turned out to be twelve times bigger than first assumed, and a budget cannot be
            // sized without knowing the rate it achieves.
            constexpr std::size_t reportEvery = 4000;
            if (mCursor / reportEvery > mReportedAt / reportEvery)
            {
                mReportedAt = mCursor;
                Log(Debug::Info) << "Reference id index: " << mCursor << " of " << mPending.size()
                                 << " cells, " << mCellsByRefId.size() << " distinct ids so far, " << mWorkMs
                                 << " ms of work";
            }
            return;
        }

        // No final sort: the lists are kept ordered as they are built, so that a half-built index is still
        // searchable.
        for (auto& [refId, cellIds] : mCellsByRefId)
            cellIds.shrink_to_fit();

        mComplete = true;
        mPending.clear();
        mPending.shrink_to_fit();

        const double wallMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mFirstAdvance)
                  .count();
        Log(Debug::Info) << "Reference id index complete: " << mCursor << " cells, " << mCellsByRefId.size()
                         << " distinct ids, " << mUnreadable.size() << " cells unreadable and therefore never "
                         << "skipped; " << mWorkMs << " ms of work spread over " << (wallMs / 1000.0)
                         << " s of play";
    }

    void CellRefIndex::indexCell(const ESM::Cell& cell, ESM::ReadersCache& readers)
    {
        const auto record = [this, &cell](const ESM::RefId& refId) {
            // Held sorted so a partially built index can still be binary searched. Only membership matters
            // here, not how many instances the cell holds, so a repeat is dropped.
            std::vector<ESM::RefId>& cellIds = mCellsByRefId[refId];
            const auto at = std::lower_bound(cellIds.begin(), cellIds.end(), cell.mId);
            if (at == cellIds.end() || *at != cell.mId)
                cellIds.insert(at, cell.mId);
        };

        // An empty context list means a dynamically generated cell with nothing on disk to read, which
        // listRefs also declines to walk.
        for (std::size_t i = 0; i < cell.mContextList.size(); ++i)
        {
            try
            {
                const ESM::ReadersCache::BusyItem reader
                    = readers.get(static_cast<std::size_t>(cell.mContextList[i].index));
                cell.restore(*reader, i);

                ESM::CellRef ref;
                ESM::MovedCellRef movedRef;
                bool deleted = false;
                bool moved = false;
                while (ESM::Cell::getNextRef(
                    *reader, ref, deleted, movedRef, moved, ESM::Cell::GetNextRefMode::LoadOnlyNotMoved))
                {
                    if (deleted || moved)
                        continue;

                    // A reference moved out of this cell belongs to wherever it went, not here. The same
                    // test listRefs makes, and the reason this has to mirror it rather than approximate it.
                    if (std::find(cell.mMovedRefs.begin(), cell.mMovedRefs.end(), ref.mRefNum)
                        != cell.mMovedRefs.end())
                        continue;

                    record(ref.mRefID);
                }
            }
            catch (const std::exception& e)
            {
                // listRefs reports and carries on, leaving a partial id list. Doing the same here would be
                // unsafe in a way it is not there: a missing id would read as proof of absence and the
                // fallback would skip a cell it should have searched. So the cell is marked unknown and
                // never skipped, which costs the old behaviour for that one cell and nothing else.
                Log(Debug::Error) << "Could not index references for cell " << cell.mId << ": " << e.what();
                mUnreadable.insert(cell.mId);
            }
        }

        // References leased into this cell from another, which listRefs lists here too.
        for (const auto& [ref, deleted] : cell.mLeasedRefs)
            if (!deleted)
                record(ref.mRefID);

        // Recorded last, and only once everything above has run, so a cell is never reported as known until
        // it actually is. This is the marker the filter trusts.
        mIndexedCells.insert(cell.mId);
    }

    void CellRefIndex::clear()
    {
        mPending.clear();
        mPending.shrink_to_fit();
        mCellsByRefId.clear();
        mUnreadable.clear();
        mIndexedCells.clear();
        mCursor = 0;
        mReportedAt = 0;
        mWorkMs = 0.0;
        mStarted = false;
        mComplete = false;
    }
}
