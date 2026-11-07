#pragma once

#include <volk.h>

#include <cstdint>
#include <optional>
#include <span>

namespace bfr::rhi::vulkan {

/// The family each queue role takes, by index into the adapter's families;
/// nullopt for a role no family runs.
struct QueueFamilies {
    std::optional<std::uint32_t> graphics;
    std::optional<std::uint32_t> compute;
    std::optional<std::uint32_t> transfer;
};

/// Chooses each role's family from an adapter's families. presents says,
/// index-aligned with families, which present to the device's surface; empty
/// when it has none.
[[nodiscard]] QueueFamilies chooseQueueFamilies(std::span<const VkQueueFamilyProperties> families,
                                                std::span<const VkBool32> presents);

/// The queues of each kind a device may open on chosen, families' choice: the
/// graphics family's count; for compute and copy, their family's count, or
/// none when it is a family an earlier kind takes, since their queues are
/// numbered from 0 and would be that kind's. None for a role with no family.
struct QueueCounts {
    std::uint32_t graphics{};
    std::uint32_t compute{};
    std::uint32_t copy{};
};

[[nodiscard]] QueueCounts countQueues(const QueueFamilies& chosen, std::span<const VkQueueFamilyProperties> families);

} // namespace bfr::rhi::vulkan
