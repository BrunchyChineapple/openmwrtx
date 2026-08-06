#ifndef OPENMW_COMPONENTS_TERRAIN_DRAWABLE_H
#define OPENMW_COMPONENTS_TERRAIN_DRAWABLE_H

#include <osg/Geometry>
#include <osg/Vec2f>

namespace osg
{
    class ClusterCullingCallback;
}

namespace osgUtil
{
    class CullVisitor;
}

namespace SceneUtil
{
    class LightListCallback;
}

namespace Terrain
{

    class CompositeMap;
    class CompositeMapRenderer;

    /**
     * Subclass of Geometry that supports built in multi-pass rendering and built in LightListCallback.
     */
    class TerrainDrawable : public osg::Geometry
    {
    public:
        osg::Object* cloneType() const override { return new TerrainDrawable(); }
        osg::Object* clone(const osg::CopyOp& copyop) const override { return new TerrainDrawable(*this, copyop); }
        bool isSameKindAs(const osg::Object* obj) const override
        {
            return dynamic_cast<const TerrainDrawable*>(obj) != nullptr;
        }
        const char* className() const override { return "TerrainDrawable"; }
        const char* libraryName() const override { return "Terrain"; }

        TerrainDrawable();
        ~TerrainDrawable(); // has to be defined in the cpp file because we only forward declared some members.
        TerrainDrawable(const TerrainDrawable& copy, const osg::CopyOp& copyop);

        void accept(osg::NodeVisitor& nv) override;
        void cull(osgUtil::CullVisitor* cv);

        typedef std::vector<osg::ref_ptr<osg::StateSet>> PassVector;
        void setPasses(const PassVector& passes);
        const PassVector& getPasses() const { return mPasses; }

        void setLightListCallback(SceneUtil::LightListCallback* lightListCallback);

        void createClusterCullingCallback();

        void compileGLObjects(osg::RenderInfo& renderInfo) const override;

        void setupWaterBoundingBox(float waterheight, float margin);
        const osg::BoundingBox& getWaterBoundingBox() const { return mWaterBoundingBox; }

        void setCompositeMap(CompositeMap* map) { mCompositeMap = map; }
        CompositeMap* getCompositeMap() const { return mCompositeMap; }
        void setCompositeMapRenderer(CompositeMapRenderer* renderer) { mCompositeMapRenderer = renderer; }

        /// Records the quadtree chunk this drawable represents for consumers that need a stable
        /// world-space terrain identity rather than the drawable's transient geometry hash.
        ///
        /// The centre and size are in world/game units. Keeping this on the drawable means a later scene
        /// traversal can recover the semantic footprint after the quadtree has selected its current LOD.
        void setChunkMetadata(
            const osg::Vec2f& worldCenter, float worldSize, unsigned char lod, bool defaultWorldspace)
        {
            mChunkWorldCenter = worldCenter;
            mChunkWorldSize = worldSize;
            mChunkLod = lod;
            mDefaultWorldspace = defaultWorldspace;
            mHasChunkMetadata = true;
        }
        bool hasChunkMetadata() const { return mHasChunkMetadata; }
        const osg::Vec2f& getChunkWorldCenter() const { return mChunkWorldCenter; }
        float getChunkWorldSize() const { return mChunkWorldSize; }
        unsigned char getChunkLod() const { return mChunkLod; }
        bool isDefaultWorldspaceChunk() const { return mDefaultWorldspace; }

    private:
        osg::BoundingBox mWaterBoundingBox;
        PassVector mPasses;

        osg::ref_ptr<osg::ClusterCullingCallback> mClusterCullingCallback;

        osg::ref_ptr<SceneUtil::LightListCallback> mLightListCallback;
        osg::ref_ptr<CompositeMap> mCompositeMap;
        osg::ref_ptr<CompositeMapRenderer> mCompositeMapRenderer;

        osg::Vec2f mChunkWorldCenter;
        float mChunkWorldSize = 0.0f;
        unsigned char mChunkLod = 0;
        bool mDefaultWorldspace = false;
        bool mHasChunkMetadata = false;
    };

}

#endif
