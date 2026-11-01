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
    /// Shaders read and write it.
    bool storage = false;
    /// Indexed draws read it.
    bool index = false;
    /// Indirect draws and dispatches read their arguments from it.
    bool indirect = false;
};

/// How to create a buffer.
struct BufferDesc {
    /// Bytes.
    std::uint64_t size = 0;
    BufferMemory memory = BufferMemory::Device;
    BufferUsage usage;
};

} // namespace bfr::rhi
