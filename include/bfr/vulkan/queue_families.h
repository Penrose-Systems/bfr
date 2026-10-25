#pragma once

#include <volk.h>

#include <cstdint>
#include <optional>
#include <span>

namespace bfr::rhi::vulkan {

/// The family each queue role takes, by index into the adapter's families.
/// present is nullopt when no family presents to the surface.
struct QueueFamilies {
    std::uint32_t graphics;
    std::uint32_t compute;
    std::uint32_t transfer;
    std::optional<std::uint32_t> present;
};

/// Chooses each role's family from an adapter's families; nullopt when none
/// runs graphics. presents says, index-aligned with families, which present to
/// the device's surface; empty when it has none, and graphics then presents.
[[nodiscard]] std::optional<QueueFamilies> chooseQueueFamilies(std::span<const VkQueueFamilyProperties> families,
                                                               std::span<const VkBool32> presents);

} // namespace bfr::rhi::vulkan
