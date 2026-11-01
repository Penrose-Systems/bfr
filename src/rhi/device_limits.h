#pragma once

#include "bfr/core/error.h"
#include "bfr/core/handle.h"
#include "bfr/rhi/command_list.h"

#include <cstdint>
#include <format>
#include <utility>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when a maximum in desc is 0 or more than a handle
/// can name, so a device holds exactly what it was asked for. Each backend's
/// createDevice calls it first.
[[nodiscard]] inline core::Result<void> checkLimits(const DeviceDesc& desc) {
    constexpr std::uint32_t kMost = core::Handle<void>::kMaxIndices;
    for (const auto& [name, value] :
         {std::pair{"maxCommandLists", desc.maxCommandLists}, std::pair{"maxCommandPools", desc.maxCommandPools},
          std::pair{"maxSwapchains", desc.maxSwapchains}}) {
        if (value == 0 || value > kMost) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 std::format("{} is {}; it must be 1 to {}", name, value, kMost));
        }
    }
    return {};
}

} // namespace bfr::rhi
