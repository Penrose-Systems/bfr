#pragma once

#include "bfr/core/error.h"
#include "bfr/core/handle.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"

#include <cstdint>
#include <format>
#include <tuple>
#include <utility>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when a maximum in desc is more than a handle can
/// name, so a device holds exactly what it was asked for; 0 is none. Each
/// backend's createDevice calls it first.
[[nodiscard]] inline core::Result<void> checkLimits(const DeviceDesc& desc) {
    constexpr std::uint32_t kMost = core::Handle<void>::kMaxIndices;
    for (const auto& [name, value] :
         {std::pair{"maxCommandLists", desc.maxCommandLists}, std::pair{"maxCommandPools", desc.maxCommandPools},
          std::pair{"maxSwapchains", desc.maxSwapchains}, std::pair{"maxBindlessTextures", desc.maxBindlessTextures},
          std::pair{"maxBindlessStorageTextures", desc.maxBindlessStorageTextures},
          std::pair{"maxBindlessStorageBuffers", desc.maxBindlessStorageBuffers},
          std::pair{"maxBindlessSamplers", desc.maxBindlessSamplers}}) {
        if (value > kMost) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "{} is {}; it is at most {}", name, value, kMost);
        }
    }
    return {};
}

/// Fails with RhiLimitReached when a bindless table in desc is larger than
/// the adapter holds, as limits says. Each backend's createDevice calls it
/// after checkLimits, so every backend refuses the same desc the same way.
[[nodiscard]] inline core::Result<void> checkBindlessLimits(const DeviceDesc& desc, const BindlessLimits& limits) {
    for (const auto& [name, asked, held] :
         {std::tuple{"maxBindlessTextures", desc.maxBindlessTextures, limits.maxTextures},
          std::tuple{"maxBindlessStorageTextures", desc.maxBindlessStorageTextures, limits.maxStorageTextures},
          std::tuple{"maxBindlessStorageBuffers", desc.maxBindlessStorageBuffers, limits.maxStorageBuffers},
          std::tuple{"maxBindlessSamplers", desc.maxBindlessSamplers, limits.maxSamplers}}) {
        if (asked > held) {
            return core::failure(core::ErrorCode::RhiLimitReached, "{} is {}; the adapter holds {}", name, asked, held);
        }
    }
    return {};
}

} // namespace bfr::rhi
