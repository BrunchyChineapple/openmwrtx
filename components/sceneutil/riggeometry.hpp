#ifndef OPENMW_COMPONENTS_NIFOSG_RIGGEOMETRY_H
#define OPENMW_COMPONENTS_NIFOSG_RIGGEOMETRY_H

#include <osg/Geometry>
#include <osg/Matrixf>

#include <string_view>

namespace SceneUtil
{
    class Skeleton;
    class Bone;

    // TODO: This class has a lot of issues.
    // - We require too many workarounds to ensure safety.
    // - mSourceGeometry should be const, but can not be const because of a use case in shadervisitor.cpp.
    // - We create useless mGeometry clones in template RigGeometries.
    // - We do not support compileGLObjects.
    // - We duplicate some code in MorphGeometry.

    /// @brief Mesh skinning implementation.
    /// @note A RigGeometry may be attached directly to a Skeleton, or somewhere below a Skeleton.
    /// Note though that the RigGeometry ignores any transforms below the Skeleton, so the attachment point is not that
    /// important.
    /// @note The internal Geometry used for rendering is double buffered, this allows updates to be done in a thread
    /// safe way while not compromising rendering performance. This is crucial when using osg's default threading model
    /// of DrawThreadPerContext.
    class RigGeometry : public osg::Drawable
    {
    public:
        RigGeometry();
        RigGeometry(const RigGeometry& copy, const osg::CopyOp& copyop);

        META_Object(SceneUtil, RigGeometry)

        // Currently empty as this is difficult to implement. Technically we would need to compile both internal
        // geometries in separate frames but this method is only called once. Alternatively we could compile just the
        // static parts of the model.
        void compileGLObjects(osg::RenderInfo& renderInfo) const override {}

        struct BoneInfo
        {
            std::string mName;
            osg::BoundingSpheref mBoundSphere;
            osg::Matrixf mInvBindMatrix;
        };

        using BoneWeight = std::pair<size_t, float>;
        using BoneWeights = std::vector<BoneWeight>;
        using VertexList = std::vector<unsigned short>;

        void setBoneInfo(std::vector<BoneInfo>&& bones);
        // Convert influences in bone and weight list per vertex format
        void setInfluences(const std::vector<BoneWeights>& influences);

        /// Initialize this geometry from the source geometry.
        /// @note The source geometry will not be modified.
        void setSourceGeometry(osg::ref_ptr<osg::Geometry> sourceGeom);

        void setTransform(osg::Matrixf&& transform);

        void setRootBone(std::string_view name);

        osg::ref_ptr<osg::Geometry> getSourceGeometry() const;

        /// Per-vertex bone influences, grouped by identical weight set, or null before influences are set.
        ///
        /// Each entry pairs one set of (bone index, weight) pairs with every vertex that shares it. The
        /// grouping is a real space saving on a character mesh -- most vertices share a handful of distinct
        /// weight sets -- but it means a consumer wanting per-vertex data has to expand it. Vertices with
        /// no influences at all are absent from the list entirely, and those keep their bind-pose position:
        /// the deformation writes only the vertices it finds here.
        ///
        /// Exposed for consumers that deform the mesh themselves rather than reading the result. The
        /// deformed geometry this class produces is double buffered and private on purpose, so it is not
        /// something an outside caller can safely hold on to.
        const std::vector<std::pair<BoneWeights, VertexList>>* getInfluences() const;

        /// Bones the influence indices refer to. Available as soon as the influences are, which is before
        /// a parent skeleton has been found -- the influence data is static, only the matrices are not.
        size_t getBoneCount() const;

        /// Fills \a boneMatrices with one matrix per bone, ready to be applied to bind-pose vertices.
        ///
        /// The matrices are exactly what cull() blends: bind-pose vertex to this drawable's local space,
        /// with the skin transform already folded in. Folding it in per bone rather than leaving it to the
        /// caller is valid because the weights of a vertex sum to one, so
        /// `sum(w_i * B_i) * T == sum(w_i * (B_i * T))` -- and it means a GPU consumer needs nothing but
        /// this array and the weights.
        ///
        /// A bone the skeleton could not resolve gets an all-zero matrix, which contributes nothing when
        /// blended. That reproduces what cull() does by skipping the influence: the vertex is pulled
        /// toward the origin in proportion to the missing weight. It looks wrong because the skin *is*
        /// wrong, and silently substituting an identity would hide a broken mesh rather than show it.
        ///
        /// @return false if the skin is not ready -- no parent skeleton found yet, or no influence data --
        ///         in which case \a boneMatrices is untouched.
        bool getBoneMatrices(std::vector<osg::Matrixf>& boneMatrices) const;

        /// Finds the parent skeleton along \a path and binds this skin's bones to it.
        ///
        /// Split from the NodeVisitor overload because the visitor was never needed for anything but its
        /// node path, and a traversal that has a path but is neither a cull nor an update visit -- the Remix
        /// submission, for one -- could not initialise a skin without it.
        bool initFromParentSkeleton(const osg::NodePath& path);

        /// Brings the pose getBoneMatrices reads up to date from a traversal that is neither a cull nor an
        /// update visit.
        ///
        /// accept() only refreshes anything for those two visitor types; anything else falls through to a
        /// plain apply() and the pose stays as whatever the last cull or update left behind. Both halves
        /// matter here -- the bone matrices themselves, and the skin-to-skeleton transform that
        /// skinTransform() folds into every one of them -- so refreshing only the bones would still leave
        /// the skin anchored to a stale place.
        ///
        /// Safe to call every frame per rig: the skeleton collapses repeat calls for the same frame, so
        /// several skins sharing one skeleton cost one recomputation between them.
        void refreshPose(const osg::NodePath& nodePath, unsigned int frame);



        void accept(osg::NodeVisitor& nv) override;
        bool supports(const osg::PrimitiveFunctor&) const override { return true; }
        void accept(osg::PrimitiveFunctor&) const override;

        struct CopyBoundingBoxCallback : osg::Drawable::ComputeBoundingBoxCallback
        {
            osg::BoundingBox boundingBox;

            osg::BoundingBox computeBound(const osg::Drawable&) const override { return boundingBox; }
        };

        struct CopyBoundingSphereCallback : osg::Node::ComputeBoundingSphereCallback
        {
            osg::BoundingSphere boundingSphere;

            osg::BoundingSphere computeBound(const osg::Node&) const override { return boundingSphere; }
        };

    private:
        void cull(osg::NodeVisitor* nv);
        void updateBounds(osg::NodeVisitor* nv);

        osg::ref_ptr<osg::Geometry> mGeometry[2];
        osg::Geometry* getGeometry(unsigned int frame) const;

        osg::ref_ptr<osg::Geometry> mSourceGeometry;
        osg::ref_ptr<const osg::Vec4Array> mSourceTangents;
        Skeleton* mSkeleton{ nullptr };

        osg::ref_ptr<osg::RefMatrix> mSkinToSkelMatrix;

        struct InfluenceData : public osg::Referenced
        {
            std::vector<BoneInfo> mBones;
            std::vector<std::pair<BoneWeights, VertexList>> mInfluences;
            osg::Matrixf mTransform;
            std::string mRootBone;
        };
        osg::ref_ptr<InfluenceData> mData;
        std::vector<Bone*> mNodes;

        unsigned int mLastFrameNumber{ 0 };
        bool mBoundsFirstFrame{ true };

        bool initFromParentSkeleton(osg::NodeVisitor* nv);

        void updateSkinToSkelMatrix(const osg::NodePath& nodePath);

        /// Skeleton space to this drawable's local space. Requires mData.
        osg::Matrixf skinTransform() const;
    };

}

#endif
