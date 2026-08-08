#include "assethash.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// The only place this header is included. It is ~200KB of macro-heavy single-header library, and
// XXH_INLINE_ALL puts every definition in this translation unit so no separate xxhash.c has to be built.
#define XXH_INLINE_ALL
#include <xxHash/xxhash.h>

namespace RemixRT
{
    namespace AssetHash
    {
        std::uint64_t bytes(const void* data, std::size_t size)
        {
            if (data == nullptr || size == 0)
                return 0;
            return static_cast<std::uint64_t>(XXH3_64bits(data, size));
        }

        std::uint64_t bytesSeeded(const void* data, std::size_t size, std::uint64_t seed)
        {
            if (data == nullptr || size == 0)
                return seed;
            return static_cast<std::uint64_t>(XXH3_64bits_withSeed(data, size, seed));
        }

        std::uint64_t combine(std::uint64_t value, std::uint64_t seed)
        {
            return static_cast<std::uint64_t>(XXH64(&value, sizeof(value), seed));
        }

        std::uint64_t fold(const void* data, std::size_t size, std::uint64_t seed)
        {
            if (data == nullptr || size == 0)
                return seed;
            return static_cast<std::uint64_t>(XXH64(data, size, seed));
        }

        std::uint64_t d3d9SphereLight(const float position[3], float radius)
        {
            // Reproduces RtSphereLight::updateCachedHash from the runtime, term for term:
            //
            //   XXH64_hash_t h = (XXH64_hash_t)RtLightType::Sphere;
            //   h = XXH64(&m_position[0], sizeof(m_position), h);
            //   h = XXH64(&m_radius, sizeof(m_radius), h);
            //   h = XXH64(&h, sizeof(h), m_shaping.getHash());
            //
            // Three constants are folded in from the runtime rather than passed, because for a light
            // submitted through the API they cannot be anything else:
            //
            //  - The seed is lightTypeSphere, which is 0 (light_types.h). So the chain starts at zero.
            //  - m_position is a Vector3, so exactly twelve bytes, and m_radius a float, so four.
            //  - The final fold is seeded with RtLightShaping::getHash(), which returns 0 whenever
            //    shaping is disabled -- and it is, since createSphereLight submits no shaping. That
            //    makes the last step identical to combine(h, 0).
            //
            // Radiance is deliberately absent, and that is the property being bought here: the runtime's
            // comment says it is excluded "to somewhat uniquely identify lights when constructed from
            // D3D9 Lights", which is what lets a light keep its identity while it flickers, dims with an
            // actor's fade, or changes colour with the time of day.
            //
            // Radius is present, so it is part of the identity. Retuning the light radius therefore
            // renames every light and invalidates edits authored against the old names. That is not a
            // quirk of this implementation -- the runtime has the same property for D3D9 lights, where
            // the radius comes from rtx.lightConversionSphereLightFixedRadius -- so matching it is the
            // point rather than a cost.
            std::uint64_t h = 0; // lightTypeSphere
            h = fold(position, sizeof(float) * 3, h);
            h = fold(&radius, sizeof(radius), h);
            return combine(h, 0);
        }

        std::uint64_t d3d9Geometry(const void* positions, std::size_t positionStride,
            std::uint32_t vertexCount, const std::uint32_t* indices, std::uint32_t indexCount)
        {
            // Values from the Vulkan enums the runtime records in its geometry descriptor. Hard-coded
            // rather than included, because pulling vulkan headers into OpenMW for two integers is a poor
            // trade, and these two are the only combination a Morrowind draw ever used across the capture
            // that was checked.
            constexpr std::uint32_t kTriangleList = 3; // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
            constexpr std::uint32_t kIndexTypeUint16 = 0; // VK_INDEX_TYPE_UINT16

            if (positions == nullptr || indices == nullptr || vertexCount == 0 || indexCount == 0)
                return 0;
            if (indexCount % 3 != 0)
                return 0;
            // A 16-bit index reaches 65535, so this many vertices is the most that can be described the
            // way Morrowind described them. Meshes above it exist but are rare -- 12 of 720 in the sample
            // -- and are left to the caller's fallback rather than guessed at.
            if (vertexCount > 0x10000u)
                return 0;

            // Positions, over the referenced vertices only and in ascending unique index order. The
            // runtime reaches this order via deduplicateSortIndices; an unreferenced vertex contributes
            // nothing, so a mesh carrying spare vertices still hashes the same.
            //
            // Marked in a bitmap and walked in order rather than sorted and deduplicated. The result is
            // identical -- ascending, each vertex once -- but the cost is O(indices + vertices) instead of
            // O(indices log indices), with no copy of the index buffer. That matters because this runs for
            // every new mesh during a cell load, and the per-vertex hash below already cannot be batched:
            // the runtime chains each vertex into the previous result, so matching it means one call per
            // vertex no matter what. Sorting a few million indices per cell on top of that was pure waste.
            //
            // The buffers are members reused across calls, so a cell load does not allocate per mesh.
            thread_local std::vector<bool> mReferencedMask;
            thread_local std::vector<std::uint16_t> mNarrowedIndices;

            mReferencedMask.assign(vertexCount, false);
            for (std::uint32_t i = 0; i < indexCount; ++i)
            {
                if (indices[i] >= vertexCount)
                    return 0;
                mReferencedMask[indices[i]] = true;
            }

            const auto* base = static_cast<const std::uint8_t*>(positions);
            std::uint64_t positionsHash = 0;
            for (std::uint32_t index = 0; index < vertexCount; ++index)
            {
                if (!mReferencedMask[index])
                    continue;
                positionsHash = bytesSeeded(
                    base + static_cast<std::size_t>(index) * positionStride, 3 * sizeof(float), positionsHash);
            }

            // Indices, narrowed to 16-bit and otherwise verbatim. This buffer already agrees with
            // Morrowind's; see the header for how that was established and why reversing it here was wrong.
            mNarrowedIndices.resize(indexCount);
            for (std::uint32_t i = 0; i < indexCount; ++i)
                mNarrowedIndices[i] = static_cast<std::uint16_t>(indices[i]);
            const std::uint64_t indicesHash
                = bytes(mNarrowedIndices.data(), static_cast<std::size_t>(indexCount) * sizeof(std::uint16_t));

            std::uint64_t descriptorHash = bytesSeeded(&indexCount, sizeof(indexCount), 0);
            descriptorHash = bytesSeeded(&vertexCount, sizeof(vertexCount), descriptorHash);
            descriptorHash = bytesSeeded(&kTriangleList, sizeof(kTriangleList), descriptorHash);
            descriptorHash = bytesSeeded(&kIndexTypeUint16, sizeof(kIndexTypeUint16), descriptorHash);

            // Combined the way GeometryHashes::getHashForRuleImpl does: the first selected component
            // verbatim, each later one folded in, walked in HashComponents enum order. For this rule that
            // order is positions, then indices, then the descriptor.
            std::uint64_t hash = positionsHash;
            hash = combine(indicesHash, hash);
            hash = combine(descriptorHash, hash);
            return hash;
        }
    }
}