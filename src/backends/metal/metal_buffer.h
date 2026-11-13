#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "metal/metal_util.h"
#include "rhi/bindless_slots.h"

#include <Metal/Metal.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace bfr::rhi::metal {

/// Keeps an allocation in a residency set, which Metal 4 needs for every
/// allocation a command buffer uses, and takes it out when it goes. The set
/// is not thread-safe: whoever makes, moves or destroys one holds the lock
/// the set's other users take.
class Resident {
public:
    Resident() = default;
    Resident(Ref<MTL::ResidencySet> set, const MTL::Allocation* allocation);
    ~Resident();
    Resident(const Resident&) = delete;
    Resident& operator=(const Resident&) = delete;
    Resident(Resident&& other) noexcept;
    Resident& operator=(Resident&&) = delete;

private:
    Ref<MTL::ResidencySet> set_;
    const MTL::Allocation* allocation_ = nullptr;
};

/// A Metal buffer, released when the last Ref to it goes, and resident while
/// it lives. mapped is where the CPU reaches upload and readback memory,
/// empty for device memory. A sparse one has the page size it was made with,
/// and tileBytes, what Metal says that page's tiles are.
struct MetalBuffer {
    Ref<MTL::Buffer> buffer;
    BufferMemory memory = BufferMemory::Device;
    std::span<std::byte> mapped;
    /// Its places in the storage and uniform buffer tables, kEnd in a table
    /// its usage does not set; the device gives them back when the buffer is
    /// collected.
    BufferSlotsHeld slots;
    std::optional<MTL::SparsePageSize> page;
    std::uint64_t tileBytes = 0;
    // Declared after buffer, so it leaves the set before it is released.
    Resident resident;
};

/// A buffer on device as desc says, with no memory when it is sparse, added
/// to residency, which the caller commits under its lock. Fails with
/// RhiInvalidCall for a sparse buffer not in device memory, RhiUnsupported for
/// sparseAliased, which Metal has no form for, or a tileBytes no page size
/// has, and RhiDeviceFailed when Metal gives no buffer.
[[nodiscard]] core::Result<MetalBuffer> createMetalBuffer(MTL::Device& device, const BufferDesc& desc,
                                                          Ref<MTL::ResidencySet> residency);

} // namespace bfr::rhi::metal
