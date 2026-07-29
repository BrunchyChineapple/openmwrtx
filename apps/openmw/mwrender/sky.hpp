#ifndef OPENMW_MWRENDER_SKY_H
#define OPENMW_MWRENDER_SKY_H

#include <memory>
#include <string>
#include <vector>

#include <osg/Vec4f>
#include <osg/ref_ptr>

#include <components/vfs/pathutil.hpp>

#include "precipitationocclusion.hpp"
#include "skyutil.hpp"

namespace osg
{
    class Group;
    class Node;
    class Material;
    class PositionAttitudeTransform;
    class Camera;
}

namespace osgParticle
{
    class ParticleSystem;
    class BoxPlacer;
}

namespace Resource
{
    class SceneManager;
}

namespace SceneUtil
{
    class RTTNode;
}

namespace MWRender
{
    ///@brief The SkyManager handles rendering of the sky domes, celestial bodies as well as other objects that need to
    /// be rendered
    /// relative to the camera (e.g. weather particle effects)
    class SkyManager
    {
    public:
        SkyManager(osg::Group* parentNode, osg::Group* rootNode, osg::Camera* camera,
            Resource::SceneManager* sceneManager, bool enableSkyRTT);
        ~SkyManager();

        void update(float duration);

        void setEnabled(bool enabled);

        int getMasserPhase() const;
        ///< 0 new moon, 1 waxing or waning cresecent, 2 waxing or waning half,
        /// 3 waxing or waning gibbous, 4 full moon

        int getSecundaPhase() const;
        ///< 0 new moon, 1 waxing or waning cresecent, 2 waxing or waning half,
        /// 3 waxing or waning gibbous, 4 full moon

        void setMoonColour(bool red);
        ///< change Secunda colour to red

        void setWeather(const WeatherResult& weather);

        void sunEnable();

        void sunDisable();

        bool isEnabled();

        bool hasRain() const;

        bool getRainRipplesEnabled() const;

        float getPrecipitationAlpha() const;

        void setStormParticleDirection(const osg::Vec3f& direction);

        void setSunDirection(const osg::Vec3f& direction);

        void setMasserState(const MoonState& state);
        void setSecundaState(const MoonState& state);

        void setGlareTimeOfDayFade(float val);

        /// Enable or disable the water plane (used to remove underwater weather particles)
        void setWaterEnabled(bool enabled);

        /// Set height of water plane (used to remove underwater weather particles)
        void setWaterHeight(float height);

        void listAssetsToPreload(
            std::vector<VFS::Path::Normalized>& models, std::vector<VFS::Path::Normalized>& textures);

        float getBaseWindSpeed() const;

        void setSunglare(bool enabled);

        SceneUtil::RTTNode* getSkyRTT() { return mSkyRTT.get(); }

        osg::Vec4f getSkyColor() const { return mSkyColour; }

        /// The sky state as last set, for consumers that need the values rather than the rendered result.
        ///
        /// Everything here already passes through SkyManager on its way to the sky domes and celestial
        /// bodies; this only retains it. The Remix backend uses it to drive the runtime's own atmosphere,
        /// which has a real sun and moons of its own and needs the numbers rather than the geometry.
        ///
        /// Directions are the corrected, Z-up, point-at-the-body vectors SkyManager is handed, not the raw
        /// orbit constants WeatherManager starts from -- RenderingManager::setSunDirection rewrites the
        /// sun's Z before it arrives here.
        struct State
        {
            /// Direction toward the sun, normalised. Never below the horizon: OpenMW's orbit model sets
            /// z = 400 - |x|, so a consumer that needs a night sun has to use mNight to put it there.
            osg::Vec3f mSunDirection{ 0.f, 0.f, 1.f };
            bool mHaveSunDirection = false;
            MoonState mMasser{};
            bool mHaveMasser = false;
            MoonState mSecunda{};
            bool mHaveSecunda = false;
            /// Straight out of the last WeatherResult, already blended across a weather transition.
            osg::Vec4f mFogColor{ 0.f, 0.f, 0.f, 1.f };
            osg::Vec4f mSkyColor{ 0.f, 0.f, 0.f, 1.f };
            osg::Vec4f mSunColor{ 0.f, 0.f, 0.f, 1.f };
            osg::Vec4f mAmbientColor{ 0.f, 0.f, 0.f, 1.f };
            float mFogDepth = 0.f;
            float mWindSpeed = 0.f;
            float mCloudSpeed = 0.f;
            bool mNight = false;
            bool mIsStorm = false;
            osg::Vec3f mStormDirection{ 0.f, 1.f, 0.f };
            bool mHaveWeather = false;
        };

        const State& getState() const { return mState; }

    private:
        void create();
        ///< no need to call this, automatically done on first enable()

        void createRain();
        void destroyRain();
        void switchUnderwaterRain();
        void updateRainParameters();

        Resource::SceneManager* mSceneManager;

        osg::Camera* mCamera;

        osg::ref_ptr<CameraRelativeTransform> mSkyRootNode;
        osg::ref_ptr<osg::Group> mSkyNode;
        osg::ref_ptr<osg::Group> mEarlyRenderBinRoot;

        osg::ref_ptr<osg::PositionAttitudeTransform> mParticleNode;
        osg::ref_ptr<osg::Node> mParticleEffect;
        osg::ref_ptr<UnderwaterSwitchCallback> mUnderwaterSwitch;

        osg::ref_ptr<osg::Group> mCloudNode;

        osg::ref_ptr<CloudUpdater> mCloudUpdater;
        osg::ref_ptr<CloudUpdater> mNextCloudUpdater;
        osg::ref_ptr<osg::PositionAttitudeTransform> mCloudMesh;
        osg::ref_ptr<osg::PositionAttitudeTransform> mNextCloudMesh;

        osg::ref_ptr<osg::Node> mAtmosphereDay;

        osg::ref_ptr<osg::PositionAttitudeTransform> mAtmosphereNightNode;
        float mAtmosphereNightRoll;
        osg::ref_ptr<AtmosphereNightUpdater> mAtmosphereNightUpdater;

        osg::ref_ptr<AtmosphereUpdater> mAtmosphereUpdater;

        std::unique_ptr<Sun> mSun;
        std::unique_ptr<Moon> mMasser;
        std::unique_ptr<Moon> mSecunda;

        osg::ref_ptr<osg::Group> mRainNode;
        osg::ref_ptr<osgParticle::ParticleSystem> mRainParticleSystem;
        osg::ref_ptr<osgParticle::BoxPlacer> mPlacer;
        osg::ref_ptr<RainCounter> mCounter;
        osg::ref_ptr<RainShooter> mRainShooter;

        bool mPrecipitationOcclusion = false;
        std::unique_ptr<PrecipitationOccluder> mPrecipitationOccluder;

        bool mCreated;

        bool mIsStorm;

        bool mTimescaleClouds;
        float mCloudAnimationTimer;

        // particle system rotation is independent of cloud rotation internally
        osg::Vec3f mStormParticleDirection;
        osg::Vec3f mStormDirection;
        osg::Vec3f mNextStormDirection;

        // remember some settings so we don't have to apply them again if they didn't change
        std::string mClouds;
        std::string mNextClouds;
        float mCloudBlendFactor;
        float mCloudSpeed;
        float mStarsOpacity;
        osg::Vec4f mCloudColour;
        osg::Vec4f mSkyColour;
        osg::Vec4f mFogColour;

        VFS::Path::Normalized mCurrentParticleEffect;

        std::string mRainEffect;
        float mRainSpeed;
        float mRainDiameter;
        float mRainMinHeight;
        float mRainMaxHeight;
        float mRainEntranceSpeed;
        int mRainMaxRaindrops;
        bool mRainRipplesEnabled;
        bool mSnowRipplesEnabled;
        float mWindSpeed;
        float mBaseWindSpeed;

        bool mEnabled;
        bool mSunglareEnabled;

        float mPrecipitationAlpha;
        bool mDirtyParticlesEffect;

        osg::Vec4f mMoonScriptColor;

        osg::ref_ptr<SceneUtil::RTTNode> mSkyRTT;

        /// Retained copy of what has been set, exposed through getState(). Written by the setters
        /// unconditionally -- before mCreated as well as after -- because a consumer outside the render
        /// path has no reason to care whether the sky domes exist yet, and those setters return early
        /// while !mCreated.
        State mState;
    };
}

#endif
