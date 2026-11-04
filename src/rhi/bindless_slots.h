#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/texture.h"
#include "core/slot_chains.h"

#include <cstdint>

namespace bfr::rhi {

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
        : desc_(desc),
          textures_(desc.maxBindlessTextures),
          storageTextures_(desc.maxBindlessStorageTextures),
          storageBuffers_(desc.maxBindlessStorageBuffers) {}

    /// A storage buffer slot when desc's usage is storage, else kEnd. Fails
    /// with RhiLimitReached when the table is full.
    [[nodiscard]] core::Result<std::uint32_t> takeBuffer(const BufferDesc& desc) {
        if (!desc.usage.storage) {
            return core::SlotChains::kEnd;
        }
        const auto slot = storageBuffers_.take(1);
        if (!slot) {
            return core::failure(core::ErrorCode::RhiLimitReached, "the storage buffer table holds at most {} buffers",
                                 desc_.maxBindlessStorageBuffers);
        }
        return *slot;
    }

    void giveBuffer(std::uint32_t slot) { storageBuffers_.give(slot); }

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

    /// held as the API reports it: 0 for a table the texture is not in.
    [[nodiscard]] static TextureSlots reported(const TextureSlotsHeld& held) {
        const auto slotOf = [](std::uint32_t slot) {
            return slot == core::SlotChains::kEnd ? 0 : slot;
        };
        return {.sampled = slotOf(held.sampled), .storage = slotOf(held.storage)};
    }

    /// slot as bufferSlot reports it. Fails with RhiInvalidCall for kEnd: a
    /// buffer without storage usage.
    [[nodiscard]] static core::Result<std::uint32_t> reported(std::uint32_t slot) {
        if (slot == core::SlotChains::kEnd) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer has no storage usage, so no slot");
        }
        return slot;
    }

private:
    DeviceDesc desc_;
    core::SlotChains textures_;
    core::SlotChains storageTextures_;
    core::SlotChains storageBuffers_;
};

} // namespace bfr::rhi
