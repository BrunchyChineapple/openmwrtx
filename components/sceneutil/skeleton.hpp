#ifndef OPENMW_COMPONENTS_NIFOSG_SKELETON_H
#define OPENMW_COMPONENTS_NIFOSG_SKELETON_H

#include <osg/Group>

#include <memory>
#include <unordered_map>

namespace SceneUtil
{

    /// @brief Defines a Bone hierarchy, used for updating of skeleton-space bone matrices.
    /// @note To prevent unnecessary updates, only bones that are used for skinning will be added to this hierarchy.
    class Bone
    {
    public:
        Bone();

        osg::Matrixf mMatrixInSkeletonSpace;

        osg::MatrixTransform* mNode;

        std::vector<std::unique_ptr<Bone>> mChildren;

        /// Update the skeleton-space matrix of this bone and all its children.
        void update(const osg::Matrixf* parentMatrixInSkeletonSpace);
    };

    /// @brief Handles the bone matrices for any number of child RigGeometries.
    /// @par Bones should be created as osg::MatrixTransform children of the skeleton.
    /// To be a referenced by a RigGeometry, a bone needs to have a unique name.
    class Skeleton : public osg::Group
    {
    public:
        Skeleton();
        Skeleton(const Skeleton& copy, const osg::CopyOp& copyop);

        META_Node(SceneUtil, Skeleton)

        /// Retrieve a bone by name.
        Bone* getBone(const std::string& name);

        /// Request an update of bone matrices. May be a no-op if already updated in this frame.
        void updateBoneMatrices(unsigned int traversalNumber);

        /// Recompute bone matrices for a consumer that is neither a cull nor an update traversal, at most
        /// once per supplied frame.
        ///
        /// Needed because updateBoneMatrices is keyed on the traversal number and the active flag, both of
        /// which belong to OpenMW's own render. A traversal that only reads the pose -- submitting it to an
        /// external renderer, say -- has no traversal number of its own and would otherwise see whatever
        /// was last computed for a different view, which looks like the animation freezing whenever that
        /// view stops refreshing it.
        ///
        /// Kept separate from updateBoneMatrices rather than folded into it so it cannot disturb
        /// mLastFrameNumber, which the cull and update paths rely on to decide whether work is needed.
        void refreshBoneMatrices(unsigned int frame);

        enum ActiveType
        {
            Inactive = 0,
            SemiActive, /// Like Active, but don't bother with Update (including new bounding box) if we're off-screen
            Active
        };

        /// Set the skinning active flag. Inactive skeletons will not have their child rigs updated.
        /// You should set this flag to false if you know that bones are not currently moving.
        void setActive(ActiveType active);

        bool getActive() const;



        void traverse(osg::NodeVisitor& nv) override;

        void markDirty();

        void childInserted(unsigned int) override;
        void childRemoved(unsigned int, unsigned int) override;

    private:
        // The root bone is not a "real" bone, it has no corresponding node in the scene graph.
        // As far as the scene graph goes we support multiple root bones.
        std::unique_ptr<Bone> mRootBone;

        typedef std::unordered_map<std::string, std::vector<osg::MatrixTransform*>> BoneCache;
        BoneCache mBoneCache;
        bool mBoneCacheInit;

        bool mNeedToUpdateBoneMatrices;

        ActiveType mActive;

        unsigned int mLastFrameNumber;
        unsigned int mLastCullFrameNumber;

        /// Frame of the last refreshBoneMatrices call. Deliberately a separate counter from
        /// mLastFrameNumber so a read-only consumer cannot convince the cull or update path that its work
        /// is already done. Starts at a value no frame counter will produce so the first call always runs.
        unsigned int mRefreshedFrame = ~0u;
    };

}

#endif
