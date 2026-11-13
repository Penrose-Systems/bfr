#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/texture.h"
#include "core/slot_chains.h"
#include "rhi/device_limits.h"

#include <cstdint>

namespace bfr::rhi {

/// The slots a buffer holds, kEnd in a table its usage does not set.
struct BufferSlotsHeld {
    std::uint32_t storage = core::SlotChains::kEnd;
    std::uint32_t uniform = core::SlotChains::kEnd;
};

/// The slots a texture holds, kEnd in a table its usage does not set.
struct TextureSlotsHeld {
    std::uint32_t sampled = core::SlotChains::kEnd;
    std::uint32_t storage = core::SlotChains::kEnd;
};

/// The bindless tables' slots, as many as the DeviceDesc asked, which every
/// backend hands out the same way: a resource takes a slot in each table its
/// usage names when it is created, holds it until it is collected, and gives
/// it back then. Made with the device, so nothing here allocates.
class BindlessSlots {
public:
    explicit BindlessSlots(const DeviceDesc& desc)
        : desc_(heldPart(desc)),
          textures_(desc.maxBindlessTextures),
          storageTextures_(desc.maxBindlessStorageTextures),
          storageBuffers_(desc.maxBindlessStorageBuffers),
          samplers_(desc.maxBindlessSamplers),
          uniformBuffers_(desc.maxBindlessUniformBuffers),
          accelerationStructures_(desc.maxBindlessAccelerationStructures),
          partitionedTops_(desc.maxBindlessPartitionedTops) {}

    /// A slot in each table desc's usage names. Fails with RhiLimitReached
    /// when one is full, taking nothing.
    [[nodiscard]] core::Result<BufferSlotsHeld> takeBuffer(const BufferDesc& desc) {
        BufferSlotsHeld held;
        if (desc.usage.storage) {
            const auto slot = storageBuffers_.take(1);
            if (!slot) {
                return core::failure(core::ErrorCode::RhiLimitReached,
                                     "the storage buffer table holds at most {} buffers",
                                     desc_.maxBindlessStorageBuffers);
            }
            held.storage = *slot;
        }
        if (desc.usage.uniform) {
            const auto slot = uniformBuffers_.take(1);
            if (!slot) {
                storageBuffers_.give(held.storage);
                return core::failure(core::ErrorCode::RhiLimitReached,
                                     "the uniform buffer table holds at most {} buffers",
                                     desc_.maxBindlessUniformBuffers);
            }
            held.uniform = *slot;
        }
        return held;
    }

    void giveBuffer(const BufferSlotsHeld& held) {
        storageBuffers_.give(held.storage);
        uniformBuffers_.give(held.uniform);
    }

    /// A slot in each table desc's usage names. Fails with RhiLimitReached
    /// when one is full, taking nothing.
    [[nodiscard]] core::Result<TextureSlotsHeld> takeTexture(const TextureDesc& desc) {
        TextureSlotsHeld held;
        if (desc.usage.sampled) {
            const auto slot = textures_.take(1);
            if (!slot) {
                return core::failure(core::ErrorCode::RhiLimitReached, "the texture table holds at most {} textures",
                                     desc_.maxBindlessTextures);
            }
            held.sampled = *slot;
        }
        if (desc.usage.storage) {
            const auto slot = storageTextures_.take(1);
            if (!slot) {
                giveTexture(held);
                return core::failure(core::ErrorCode::RhiLimitReached,
                                     "the storage texture table holds at most {} textures",
                                     desc_.maxBindlessStorageTextures);
            }
            held.storage = *slot;
        }
        return held;
    }

    void giveTexture(const TextureSlotsHeld& held) {
        textures_.give(held.sampled);
        storageTextures_.give(held.storage);
    }

    /// A sampler table slot, which every sampler takes. Fails with
    /// RhiLimitReached when the table is full.
    [[nodiscard]] core::Result<std::uint32_t> takeSampler() {
        const auto slot = samplers_.take(1);
        if (!slot) {
            return core::failure(core::ErrorCode::RhiLimitReached, "the sampler table holds at most {} samplers",
                                 desc_.maxBindlessSamplers);
        }
        return *slot;
    }

    void giveSampler(std::uint32_t slot) { samplers_.give(slot); }

    /// A slot for a top structure in the acceleration structure table, or for
    /// a partitioned top one in the partitioned top table, where the device
    /// has that table, else kEnd: shaders trace rays against these only, and
    /// reach them without the table where the backend lets them. Fails with
    /// RhiLimitReached when the table is full.
    [[nodiscard]] core::Result<std::uint32_t> takeAccelerationStructure(AccelerationStructureType type) {
        const bool partitioned = type == AccelerationStructureType::PartitionedTop;
        const std::uint32_t slots = partitioned ? desc_.maxBindlessPartitionedTops
                                                : desc_.maxBindlessAccelerationStructures;
        if ((type != AccelerationStructureType::Top && !partitioned) || slots == 0) {
            return core::SlotChains::kEnd;
        }
        const auto slot = (partitioned ? partitionedTops_ : accelerationStructures_).take(1);
        if (!slot) {
            return core::failure(core::ErrorCode::RhiLimitReached, "the {} table holds at most {} structures",
                                 partitioned ? "partitioned top structure" : "acceleration structure", slots);
        }
        return *slot;
    }

    /// Gives back slot, which a structure of type took.
    void giveAccelerationStructure(AccelerationStructureType type, std::uint32_t slot) {
        (type == AccelerationStructureType::PartitionedTop ? partitionedTops_ : accelerationStructures_).give(slot);
    }

    /// slot as accelerationStructureSlot reports it. Fails with
    /// RhiInvalidCall for kEnd: a structure in no table, or one on a device
    /// with no slots in its table.
    [[nodiscard]] static core::Result<std::uint32_t> reportedStructure(std::uint32_t slot) {
        if (slot == core::SlotChains::kEnd) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "the acceleration structure has no slot: only top and partitioned top ones are in "
                                 "tables, on a device made with slots in them");
        }
        return slot;
    }

    /// held as the API reports it: kNoSlot, the chains' kEnd, for a table
    /// the texture is not in.
    [[nodiscard]] static TextureSlots reported(const TextureSlotsHeld& held) {
        static_assert(core::SlotChains::kEnd == kNoSlot);
        return {.sampled = held.sampled, .storage = held.storage};
    }

    /// held as bufferSlots reports it: kNoSlot, the chains' kEnd, for a table
    /// the buffer is not in.
    [[nodiscard]] static BufferSlots reported(const BufferSlotsHeld& held) {
        static_assert(core::SlotChains::kEnd == kNoSlot);
        return {.storage = held.storage, .uniform = held.uniform};
    }

private:
    DeviceDesc desc_;
    core::SlotChains textures_;
    core::SlotChains storageTextures_;
    core::SlotChains storageBuffers_;
    core::SlotChains samplers_;
    core::SlotChains uniformBuffers_;
    core::SlotChains accelerationStructures_;
    core::SlotChains partitionedTops_;
};

} // namespace bfr::rhi
