#ifndef OPENMW_COMPONENTS_REMIXRT_ASSETHASH_H
#define OPENMW_COMPONENTS_REMIXRT_ASSETHASH_H

#include <cstddef>
#include <cstdint>

namespace RemixRT
{
    /// Hashing for asset identity, matching what the Remix runtime does to its own assets.
    ///
    /// Remix has no notion of an asset name. A mesh, a material and a texture are each identified purely
    /// by a 64-bit value the host supplies, and that value is the key a USD replacement is authored
    /// against and the key a texture tag is stored under in rtx.conf. So the hash is not an
    /// implementation detail that can be chosen for convenience -- it *is* the asset's public identity,
    /// and it has to be derived from content so that it survives a restart.
    ///
    /// This is why deriving identity from a pointer was a defect rather than a shortcut: heap addresses
    /// differ between runs, so every replacement missed, every texture tag written to rtx.conf became
    /// dead on the next launch, and every capture named its meshes differently.
    ///
    /// The algorithms are XXH3-64 and XXH64, from the same vendored xxhash.h the runtime compiles
    /// against, because Remix hashes D3D9 textures with XXH3_64bits over the raw staging buffer and folds
    /// geometry components with XXH64 seeding. Reproducing hashes captured from another host is only
    /// possible at all if both sides agree bit for bit.
    namespace AssetHash
    {
        /// XXH3-64 over a memory range. This is Remix's texture hash primitive.
        std::uint64_t bytes(const void* data, std::size_t size);

        /// XXH3-64 over a memory range with a seed, for chaining across several regions.
        std::uint64_t bytesSeeded(const void* data, std::size_t size, std::uint64_t seed);

        /// Folds one 64-bit value into a running hash as XXH64(&value, 8, seed).
        ///
        /// Deliberately the same fold Remix uses to combine geometry hash components
        /// (GeometryHashes::getHashForRuleImpl), so a component-wise hash can be assembled here and
        /// still agree with the runtime's own combination of the same parts.
        std::uint64_t combine(std::uint64_t value, std::uint64_t seed);

        /// Zero is not a usable asset identity: the API rejects a null handle, and this code already
        /// treats a zero hash as "creation failed". Substituted rather than forced odd, because forcing a
        /// bit would perturb every hash and destroy any chance of matching a value captured elsewhere.
        inline std::uint64_t avoidZero(std::uint64_t hash)
        {
            return hash != 0 ? hash : 0x9E3779B97F4A7C15ull;
        }
    }
}

#endif
