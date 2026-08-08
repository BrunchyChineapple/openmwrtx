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

        /// XXH64 over a memory range with a seed.
        ///
        /// Distinct from bytesSeeded, which is XXH3. The runtime hashes light properties with plain XXH64
        /// over spans that are not eight bytes wide, so neither existing primitive can express it.
        std::uint64_t fold(const void* data, std::size_t size, std::uint64_t seed);

        /// The identity Remix's own D3D9 path would give a sphere light at this position and radius.
        ///
        /// A light's hash is its identity and its handle both: it is the key a USD light replacement is
        /// authored against, and the name the toolkit shows and stores edits under. Deriving it from
        /// OpenMW's light-source id -- a counter handed out as cells stream in -- made it unique among live
        /// lights but different on every run and after every cell reload, so no toolkit edit could ever
        /// refer to the same light twice.
        ///
        /// Position and radius are content, so this is stable across sessions. It is also the formula
        /// MGE-XE's lights hash under, which is what gives a replacement pack authored against a Morrowind
        /// capture a chance of binding -- the pack's light keys were authored at
        /// rtx.lightConversionSphereLightFixedRadius, so matching them additionally requires submitting
        /// that same radius.
        std::uint64_t d3d9SphereLight(const float position[3], float radius);

        /// The geometry identity Remix's own D3D9 path would give this mesh, under the default
        /// rtx.geometryAssetHashRule of "positions,indices,geometrydescriptor".
        ///
        /// This is what makes a replacement pack authored against a Morrowind capture bind to geometry
        /// submitted through the API instead: the API takes the mesh hash from the caller verbatim
        /// (remixapi_CreateMesh casts it straight to the handle) and the replacement lookup keys on that
        /// handle, so reproducing the number the D3D9 path would have produced is sufficient. Nothing
        /// about the submitted buffers has to match -- only this value.
        ///
        /// Each part was verified against ground truth before being written here, by recomputing it from
        /// the geometry inside a Morrowind capture and comparing with the per-component hashes the
        /// capturer records in each mesh layer's customLayerData: positions reproduced 40 of 40, indices
        /// 39 of 40, the descriptor 40 of 40, and the fully combined value including the material XOR 60
        /// of 60. Two details are not guessable from the runtime source and cost an earlier attempt:
        ///
        ///  - Morrowind's index buffers are 16-bit, so the indices are hashed narrowed and the descriptor
        ///    carries VK_INDEX_TYPE_UINT16. Hashing 32-bit indices corrupts both components at once, and
        ///    because the combiner chains them, one wrong component ruins the result.
        ///  - The index buffer is hashed verbatim. It already agrees with Morrowind's, which is not what a
        ///    comparison of the two engines' captures appears to show, and the discrepancy is a trap worth
        ///    stating plainly: Remix writes a D3D9 mesh's indices to a capture unchanged, but reverses the
        ///    winding of an API-submitted mesh's. So OpenMW's captured indices are the reverse of what it
        ///    actually submitted, while MGE-XE's are not, and comparing the two captures suggests a
        ///    reversal that does not exist between the submitted buffers. Reversing here on that basis made
        ///    every mesh hash wrong. Three measurements pin it down: the runtime's indices component,
        ///    computed over reverse(submitted), is reproduced by hashing the captured buffer verbatim on 94
        ///    of 94 meshes joined by position checksum; reverse(captured) reproduces MGE-XE's stored
        ///    indices hash on 708 of 720; therefore submitted equals what Morrowind submitted.
        ///
        /// Positions are hashed as submitted, with no transform. Verified the same way, on the same 94
        /// meshes: unlike indices, a capture does record positions faithfully, and the runtime's positions
        /// component matched the offline value on every one.
        ///
        /// Returns 0 if the geometry cannot be one of Remix's D3D9 draws -- an index count that is not a
        /// multiple of three, or more vertices than a 16-bit index can reach -- so the caller can fall
        /// back rather than bind a value that is wrong.
        ///
        /// \param positionStride byte stride between vertices; positions are read as the first three
        ///        floats of each, matching the interleaved vertex the runtime is handed.
        std::uint64_t d3d9Geometry(const void* positions, std::size_t positionStride,
            std::uint32_t vertexCount, const std::uint32_t* indices, std::uint32_t indexCount);

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
