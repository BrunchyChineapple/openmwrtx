#include "soundbuffer.hpp"

#include "../mwbase/environment.hpp"
#include "../mwworld/esmstore.hpp"

#include <components/debug/debuglog.hpp>
#include <components/esm3/loadsoun.hpp>
#include <components/esm4/loadsndr.hpp>
#include <components/esm4/loadsoun.hpp>
#include <components/misc/resourcehelpers.hpp>
#include <components/resource/resourcesystem.hpp>
#include <components/settings/values.hpp>
#include <components/vfs/pathutil.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace MWSound
{
    namespace
    {
        constexpr VFS::Path::NormalizedView soundDir("sound");
        constexpr VFS::Path::ExtensionView mp3("mp3");

        struct AudioParams
        {
            float mAudioDefaultMinDistance;
            float mAudioDefaultMaxDistance;
            float mAudioMinDistanceMult;
            float mAudioMaxDistanceMult;
        };

        AudioParams makeAudioParams(const MWWorld::Store<ESM::GameSetting>& settings)
        {
            AudioParams params;
            params.mAudioDefaultMinDistance = settings.find("fAudioDefaultMinDistance")->mValue.getFloat();
            params.mAudioDefaultMaxDistance = settings.find("fAudioDefaultMaxDistance")->mValue.getFloat();
            params.mAudioMinDistanceMult = settings.find("fAudioMinDistanceMult")->mValue.getFloat();
            params.mAudioMaxDistanceMult = settings.find("fAudioMaxDistanceMult")->mValue.getFloat();
            return params;
        }
    }

    SoundBufferPool::SoundBufferPool(SoundOutput& output)
        : mOutput(&output)
        , mBufferCacheMax(Settings::sound().mBufferCacheMax * 1024 * 1024)
        , mBufferCacheMin(
              std::min(static_cast<std::size_t>(Settings::sound().mBufferCacheMin) * 1024 * 1024, mBufferCacheMax))
    {
    }

    SoundBufferPool::~SoundBufferPool()
    {
        clear();
    }

    SoundBuffer* SoundBufferPool::lookup(const ESM::RefId& soundId) const
    {
        const auto it = mBufferNameMap.find(soundId);
        if (it != mBufferNameMap.end())
        {
            SoundBuffer* sfx = it->second;
            if (sfx->getHandle() != nullptr)
                return sfx;
        }
        return nullptr;
    }

    SoundBuffer* SoundBufferPool::lookup(VFS::Path::NormalizedView fileName) const
    {
        const auto it = mBufferFileNameMap.find(fileName);
        if (it != mBufferFileNameMap.end())
        {
            SoundBuffer* sfx = it->second;
            if (sfx->getHandle() != nullptr)
                return sfx;
        }
        return nullptr;
    }

    SoundBuffer* SoundBufferPool::loadSfx(SoundBuffer* sfx)
    {
        if (sfx->getHandle() != nullptr)
            return sfx;

        // Timed, and named when it overruns.
        //
        // This decodes the whole file to PCM on the frame thread, so the cost scales with the duration of
        // the sound rather than with anything the player can see. That is fine for the sound effects this
        // pool is meant to hold, and much less fine when a mod plays a long music track through
        // playSoundFile3d: thirteen minutes of 48 kHz stereo is 144 MB of PCM and seconds of decoding, and
        // nothing downstream would attribute the resulting hitch to a sound. Naming the file and its
        // decoded size is enough for whoever hits it to see what they asked for.
        constexpr double overrunMs = 20.0;
        const auto started = std::chrono::steady_clock::now();

        auto [handle, size] = mOutput->loadSound(sfx->getResourceName());
        if (handle == nullptr)
            return {};

        const double elapsedMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (elapsedMs >= overrunMs)
            Log(Debug::Warning) << "Decoding " << sfx->getResourceName() << " held the frame thread for "
                                << elapsedMs << " ms and produced " << (size / (1024 * 1024)) << " MB of audio";

        sfx->mHandle = handle;
        sfx->mSize = size;

        mBufferCacheSize += size;

        // Budget the buffers that can actually be freed, rather than every loaded byte.
        //
        // Buffers still in use cannot be evicted, so counting them towards the ceiling made the eviction
        // target unreachable whenever they exceeded 'buffer cache min' on their own: unloadUnused would run
        // to exhaustion, empty the reclaimable pool, and still be over budget. Every later load then found
        // an empty pool, freed whatever little had accumulated since, and left the next play of each of
        // those sounds to decode from scratch -- so one large sound that keeps playing disabled caching for
        // every other sound in the game, for as long as it played. A thirteen-minute music track played
        // through playSoundFile3d decodes to 144 MB against a 64 MB ceiling, which is all it takes.
        //
        // Budgeting the reclaimable pool keeps the setting's meaning for ordinary content, where sounds are
        // idle most of the time and nearly all of the cache is reclaimable. What changes is that memory held
        // by sounds that are currently playing is now treated as the cost of playing them, which it is,
        // instead of as a reason to throw away unrelated cached sounds to no benefit.
        if (mUnusedSize + size > mBufferCacheMax)
            unloadUnused(size);

        // Report a working set larger than the cache.
        //
        // This is no longer harmful -- it does not purge anything now -- but it does mean the cache cannot
        // hold what is being played, so it is worth saying once. It is also the only place a caller learns
        // that a single sound is disproportionately large.
        //
        // The previous form of this test also required the unused pool to be non-empty while over the
        // ceiling, which unloadUnused's two exit conditions make impossible, so it never fired.
        const std::size_t inUse = mBufferCacheSize - mUnusedSize;
        if (inUse > mBufferCacheMax)
        {
            if (!mReportedOverBudget)
            {
                Log(Debug::Warning) << "Sound buffers in use total " << (inUse / (1024 * 1024)) << " MB, over the "
                                    << (mBufferCacheMax / (1024 * 1024)) << " MB 'buffer cache max'";
                mReportedOverBudget = true;
            }
        }
        else
            mReportedOverBudget = false;

        mUnusedBuffers.push_front(sfx);
        mUnusedSize += size;

        return sfx;
    }

    SoundBuffer* SoundBufferPool::load(const ESM::RefId& soundId)
    {
        if (mBufferNameMap.empty())
        {
            const MWWorld::ESMStore* esmstore = MWBase::Environment::get().getESMStore();
            for (const ESM::Sound& sound : esmstore->get<ESM::Sound>())
                insertSound(sound.mId, sound);
            for (const ESM4::Sound& sound : esmstore->get<ESM4::Sound>())
                insertSound(sound.mId, sound);
            for (const ESM4::SoundReference& sound : esmstore->get<ESM4::SoundReference>())
                insertSound(sound.mId, sound);
        }

        SoundBuffer* sfx;
        const auto it = mBufferNameMap.find(soundId);
        if (it != mBufferNameMap.end())
            sfx = it->second;
        else
        {
            const ESM::Sound* sound = MWBase::Environment::get().getESMStore()->get<ESM::Sound>().search(soundId);
            if (sound == nullptr)
                return {};
            sfx = insertSound(soundId, *sound);
        }

        return loadSfx(sfx);
    }

    SoundBuffer* SoundBufferPool::load(VFS::Path::NormalizedView fileName)
    {
        SoundBuffer* sfx;
        const auto it = mBufferFileNameMap.find(fileName);
        if (it != mBufferFileNameMap.end())
            sfx = it->second;
        else
            sfx = insertSound(fileName);

        return loadSfx(sfx);
    }

    void SoundBufferPool::clear()
    {
        for (auto& sfx : mSoundBuffers)
        {
            if (sfx.mHandle)
                mOutput->unloadSound(sfx.mHandle);
            sfx.mHandle = nullptr;
            sfx.mSize = 0;
        }

        mBufferFileNameMap.clear();
        mBufferNameMap.clear();
        mUnusedBuffers.clear();
        mBufferCacheSize = 0;
        mUnusedSize = 0;
    }

    SoundBuffer* SoundBufferPool::insertSound(VFS::Path::NormalizedView fileName)
    {
        static const AudioParams audioParams
            = makeAudioParams(MWBase::Environment::get().getESMStore()->get<ESM::GameSetting>());

        float volume = 1.f;
        float min = std::max(audioParams.mAudioDefaultMinDistance * audioParams.mAudioMinDistanceMult, 1.f);
        float max = std::max(min, audioParams.mAudioDefaultMaxDistance * audioParams.mAudioMaxDistanceMult);

        min = std::max(min, 1.0f);
        max = std::max(min, max);

        SoundBuffer& sfx = mSoundBuffers.emplace_back(fileName, volume, min, max);

        mBufferFileNameMap.emplace(fileName, &sfx);
        return &sfx;
    }

    SoundBuffer* SoundBufferPool::insertSound(const ESM::RefId& soundId, const ESM::Sound& sound)
    {
        static const AudioParams audioParams
            = makeAudioParams(MWBase::Environment::get().getESMStore()->get<ESM::GameSetting>());

        float volume = static_cast<float>(std::pow(10.0, (sound.mData.mVolume / 255.0 * 3348.0 - 3348.0) / 2000.0));
        float min = sound.mData.mMinRange;
        float max = sound.mData.mMaxRange;
        if (min == 0 && max == 0)
        {
            min = audioParams.mAudioDefaultMinDistance;
            max = audioParams.mAudioDefaultMaxDistance;
        }

        min *= audioParams.mAudioMinDistanceMult;
        max *= audioParams.mAudioMaxDistanceMult;
        min = std::max(min, 1.0f);
        max = std::max(min, max);

        SoundBuffer& sfx = mSoundBuffers.emplace_back(
            Misc::ResourceHelpers::correctSoundPath(VFS::Path::toNormalized(sound.mSound)), volume, min, max);

        mBufferNameMap.emplace(soundId, &sfx);
        return &sfx;
    }

    SoundBuffer* SoundBufferPool::insertSound(const ESM::RefId& soundId, const ESM4::Sound& sound)
    {
        VFS::Path::Normalized path = Misc::ResourceHelpers::correctResourcePath({ { soundDir } },
            VFS::Path::toNormalized(sound.mSoundFile), *MWBase::Environment::get().getResourceSystem()->getVFS(), mp3);
        float volume = 1, min = 1, max = 255; // TODO: needs research
        SoundBuffer& sfx = mSoundBuffers.emplace_back(std::move(path), volume, min, max);
        mBufferNameMap.emplace(soundId, &sfx);
        return &sfx;
    }

    SoundBuffer* SoundBufferPool::insertSound(const ESM::RefId& soundId, const ESM4::SoundReference& sound)
    {
        VFS::Path::Normalized path = Misc::ResourceHelpers::correctResourcePath({ { soundDir } },
            VFS::Path::toNormalized(sound.mSoundFile), *MWBase::Environment::get().getResourceSystem()->getVFS(), mp3);
        float volume = 1, min = 1, max = 255; // TODO: needs research
        // TODO: sound.mSoundId can link to another SoundReference, probably we will need to add additional lookups to
        // ESMStore.
        SoundBuffer& sfx = mSoundBuffers.emplace_back(std::move(path), volume, min, max);
        mBufferNameMap.emplace(soundId, &sfx);
        return &sfx;
    }

    void SoundBufferPool::unloadUnused(std::size_t reserve)
    {
        while (!mUnusedBuffers.empty() && mUnusedSize + reserve > mBufferCacheMin)
        {
            SoundBuffer* const unused = mUnusedBuffers.back();

            mBufferCacheSize -= mOutput->unloadSound(unused->getHandle());
            mUnusedSize -= unused->mSize;
            unused->mHandle = nullptr;
            unused->mSize = 0;

            mUnusedBuffers.pop_back();
        }
    }
}
