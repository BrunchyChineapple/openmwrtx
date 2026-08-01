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
            std::vector<std::uint32_t> referenced(indices, indices + indexCount);
            std::sort(referenced.begin(), referenced.end());
            referenced.erase(std::unique(referenced.begin(), referenced.end()), referenced.end());

            const auto* base = static_cast<const std::uint8_t*>(positions);
            std::uint64_t positionsHash = 0;
            for (const std::uint32_t index : referenced)
            {
                if (index >= vertexCount)
                    return 0;
                positionsHash = bytesSeeded(
                    base + static_cast<std::size_t>(index) * positionStride, 3 * sizeof(float), positionsHash);
            }

            // Indices, narrowed to 16-bit and otherwise verbatim. This buffer already agrees with
            // Morrowind's; see the header for how that was established and why reversing it here was wrong.
            std::vector<std::uint16_t> narrowed(indexCount);
            for (std::uint32_t i = 0; i < indexCount; ++i)
                narrowed[i] = static_cast<std::uint16_t>(indices[i]);
            const std::uint64_t indicesHash = bytes(narrowed.data(), narrowed.size() * sizeof(std::uint16_t));

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