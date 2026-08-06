#ifndef GAME_MWWORLD_CELLREFINDEX_H
#define GAME_MWWORLD_CELLREFINDEX_H

#include <chrono>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <components/esm/refid.hpp>

namespace ESM
{
    class ReadersCache;
    struct Cell;
}

namespace MWWorld
{
    class ESMStore;

    /// Which content-file cells declare each reference id, so a search by id need not read cells that
    /// cannot contain what it is looking for.
    ///
    /// WorldModel::getPtrByRefId ends in a fallback over every exterior and then every interior cell in the
    /// content files, constructing a CellStore for each and parsing its reference list off disk. Measured on
    /// a heavily modded install, one such call took 4.2 seconds and read 786 MiB across 249,000 operations,
    /// inside a single frame, reached from AiPackage::getTarget in the per-frame combat update. Nothing
    /// caches it away either: getTarget caches the answer, so the cost is paid once per AI package, but once
    /// is a four-second freeze.
    ///
    /// Why skipping on this index cannot change behaviour:
    ///  - The fallback only visits cells absent from WorldModel::mCells.
    ///  - Every cell a save game mentions is instantiated into mCells by WorldModel::readRecord when the save
    ///    loads, and anything instantiated at runtime is there too. So the cells the fallback visits have
    ///    purely content-file reference lists.
    ///  - For such a cell CellStore::getPtr searches CellStore::mIds, which is what CellStore::listRefs
    ///    produced.
    ///  - indexCell mirrors listRefs exactly: same GetNextRefMode, same deleted and moved rejections, same
    ///    mMovedRefs exclusion, same mLeasedRefs inclusion.
    /// So "the index says this cell does not declare it" and "getPtr would have returned empty" are the same
    /// statement.
    ///
    /// Built incrementally against a per-frame time budget rather than on a worker thread, because
    /// ESM::ReadersCache holds no lock and sharing it with a second thread would be a data race for the sake
    /// of work paid once per session. Until the build completes the index answers nothing and the fallback
    /// behaves exactly as before, so there is no window in which it is half-trusted.
    class CellRefIndex
    {
    public:
        /// A prepared answer for one reference id.
        ///
        /// Separate from the index so a caller walking thousands of cells does one hash lookup rather than
        /// one per cell, and so the "not ready" case cannot be forgotten: a default-constructed Query
        /// permits nothing.
        class Query
        {
        public:
            /// True only when \a cellId provably cannot contain the id this query was built for. False
            /// whenever there is any doubt at all -- index incomplete, cell unreadable, or the cell does
            /// declare the id.
            bool cannotContain(const ESM::RefId& cellId) const;

        private:
            friend class CellRefIndex;

            const CellRefIndex* mIndex = nullptr;
            const std::vector<ESM::RefId>* mCells = nullptr;
        };

        /// Spends at most \a budgetMs advancing the build. Returns immediately once complete.
        void advance(const ESMStore& store, ESM::ReadersCache& readers, double budgetMs);

        /// A query for \a refId. Answers for whatever has been indexed so far and nothing more.
        Query query(const ESM::RefId& refId) const;

        /// True once every cell has been read. The filter does not wait for this -- see Query -- so it only
        /// means there is no more work to do.
        bool complete() const { return mComplete; }

        void clear();

        std::size_t indexedCells() const { return mCursor; }

        /// How many cells the build will read in total. mPending is emptied on completion, so that case
        /// answers from the cursor instead.
        std::size_t totalCells() const { return mComplete ? mCursor : mPending.size(); }

    private:
        void indexCell(const ESM::Cell& cell, ESM::ReadersCache& readers);

        /// Cells still to read. Pointers into the store, which is immutable once content is loaded.
        std::vector<const ESM::Cell*> mPending;
        std::unordered_map<ESM::RefId, std::vector<ESM::RefId>> mCellsByRefId;
        /// Cells whose reference list could not be read. Never skipped: absence from the index would
        /// otherwise be indistinguishable from "this cell does not contain it", which would lose an object
        /// rather than merely be slow.
        std::unordered_set<ESM::RefId> mUnreadable;
        /// Cells actually read. What makes a partial index usable: the filter trusts this rather than
        /// completion, so the first search benefits from however much has been done.
        std::unordered_set<ESM::RefId> mIndexedCells;
        std::size_t mCursor = 0;
        std::size_t mReportedAt = 0;
        bool mStarted = false;
        bool mComplete = false;
        /// Work spent so far, and when the build began, so completion can report both. Without the pair there
        /// is no way to tell a budget that is too small from a job that is bigger than expected.
        double mWorkMs = 0.0;
        std::chrono::steady_clock::time_point mFirstAdvance;
    };
}

#endif
