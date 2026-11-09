#pragma once

#include "bfr/core/handle.h"

#include <cstdint>

namespace bfr::rhi {

/// Memory the GPU reads and writes. Only named, by a BufferHandle.
struct Buffer;

using BufferHandle = core::Handle<Buffer>;

/// Where a buffer's memory is, and whether the CPU maps it.
enum class BufferMemory : std::uint8_t {
    /// On or nearest the GPU; the CPU does not map it.
    Device,
    /// The CPU writes it and the GPU reads it.
    Upload,
    /// The GPU writes it and the CPU reads it.
    Readback,
};

/// What the GPU does with a buffer. Set every use it will have.
struct BufferUsage {
    bool copySrc = false;
    bool copyDst = false;
    /// Shaders read it as constants.
    bool uniform = false;
    /// Shaders read and write it. D3D12 views it in 32-bit values, so there
    /// its size must be a multiple of 4.
    bool storage = false;
    /// Indexed draws read it.
    bool index = false;
    /// Indirect draws, dispatches and traces read their arguments from it.
    bool indirect = false;
    /// Acceleration structure builds read vertices, indices, transforms,
    /// boxes or instances from it. It starts at an address of the alignment
    /// those take.
    bool accelerationStructureInput = false;
    /// Acceleration structure builds use it as scratch. It starts at an
    /// address that is a multiple of RayTracingLimits::scratchAlignment.
    bool accelerationStructureScratch = false;
    /// It holds shader binding table records. It starts at an address that is
    /// a multiple of RayTracingLimits::tableAlignment.
    bool shaderBindingTable = false;
    /// serializeAccelerationStructure writes it and
    /// deserializeAccelerationStructure reads it. It starts at an address
    /// that is a multiple of 256.
    bool accelerationStructureData = false;
};

/// How to create a buffer.
struct BufferDesc {
    /// Bytes.
    std::uint64_t size = 0;
    BufferMemory memory = BufferMemory::Device;
    BufferUsage usage;
    /// Made with no memory: tiles of SparseMemory are mapped into it by
    /// Device::updateTileMappings, as Caps::sparse says. memory is Device.
    bool sparse = false;
    /// Its tiles may be mapped to memory other sparse resources' tiles are
    /// mapped to, where Sparse::aliased.
    bool sparseAliased = false;
    /// One of Sparse::tileBytes, or 0 where that list is empty; only when sparse.
    std::uint32_t tileBytes = 0;
};

/// One range copied from a source buffer to a destination buffer, in bytes.
struct BufferCopy {
    std::uint64_t srcOffset = 0;
    std::uint64_t dstOffset = 0;
    std::uint64_t size = 0;
};

} // namespace bfr::rhi
