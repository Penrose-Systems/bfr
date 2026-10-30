#include "vulkan/queue_families.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <optional>
#include <span>

namespace bfr::rhi::vulkan {

std::optional<QueueFamilies> chooseQueueFamilies(std::span<const VkQueueFamilyProperties> families,
                                                 std::span<const VkBool32> presents) {
    // Without a surface, every family counts as presenting, so graphics presents.
    const auto presentsTo = [presents](std::uint32_t index) {
        return presents.empty() || presents[index] == VK_TRUE;
    };
    // The first family that runs every operation in has and none in lacks, and
    // presents to the surface when presenting.
    const auto first = [families, &presentsTo](VkQueueFlags has, VkQueueFlags lacks,
                                               bool presenting = false) -> std::optional<std::uint32_t> {
        for (std::uint32_t index = 0; index < families.size(); ++index) {
            const VkQueueFlags flags = families[index].queueFlags;
            if ((flags & has) == has && (flags & lacks) == 0 && (!presenting || presentsTo(index))) {
                return index;
            }
        }
        return std::nullopt;
    };
    constexpr bool kPresenting = true;
    // Graphics also runs compute where a family can, so a frame's compute work
    // can share its queue, and presents where a family can, so the frame
    // presents from its queue.
    const auto graphics = first(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, 0, kPresenting)
                              .or_else([&first] { return first(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, 0); })
                              .or_else([&first] { return first(VK_QUEUE_GRAPHICS_BIT, 0, kPresenting); })
                              .or_else([&first] { return first(VK_QUEUE_GRAPHICS_BIT, 0); });
    if (!graphics) {
        return std::nullopt;
    }
    const std::uint32_t compute = first(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT).value_or(*graphics);
    // Without a transfer-only family, every family left that transfers and
    // lacks graphics runs compute, so the compute family is the first of them.
    // It transfers even when it doesn't report the transfer bit.
    return QueueFamilies{
        .graphics = *graphics,
        .compute = compute,
        .transfer = first(VK_QUEUE_TRANSFER_BIT, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT).value_or(compute),
        .present = presentsTo(*graphics) ? graphics : first(0, 0, kPresenting),
    };
}

} // namespace bfr::rhi::vulkan
