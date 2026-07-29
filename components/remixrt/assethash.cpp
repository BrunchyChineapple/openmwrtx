#include "assethash.hpp"

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
    }
}