#include "vulkan/queue_families.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <optional>
#include <span>

namespace bfr::rhi::vulkan {

QueueFamilies chooseQueueFamilies(std::span<const VkQueueFamilyProperties> families,
                                  std::span<const VkBool32> presents) {
    // Without a surface, every family counts as presenting.
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
    // Each kind takes a family that presents where one of its kind does, so
    // a window can be presented to from its queues.
    // Without a family that computes but not graphics, compute is the graphics
    // family, whose queues are graphics', so the adapter has no compute queue.
    const auto compute = first(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT, kPresenting)
                             .or_else([&first] { return first(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT); })
                             .or_else([graphics] { return graphics; });
    // Without a transfer-only family, every family left that transfers and
    // lacks graphics runs compute, so the compute family is the first of them.
    // A compute or graphics family transfers even when it doesn't report the
    // transfer bit.
    const auto transfer = first(VK_QUEUE_TRANSFER_BIT, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, kPresenting)
                              .or_else([&first] {
                                  return first(VK_QUEUE_TRANSFER_BIT, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
                              })
                              .or_else([compute] { return compute; })
                              .or_else([graphics] { return graphics; })
                              .or_else([&first] { return first(VK_QUEUE_TRANSFER_BIT, 0); });
    return QueueFamilies{
        .graphics = graphics,
        .compute = compute,
        .transfer = transfer,
    };
}

QueueCounts countQueues(const QueueFamilies& chosen, std::span<const VkQueueFamilyProperties> families) {
    const auto count = [families](std::optional<std::uint32_t> family, bool shared) {
        return !family || shared ? 0 : families[*family].queueCount;
    };
    return {
        .graphics = count(chosen.graphics, false),
        .compute = count(chosen.compute, chosen.compute == chosen.graphics),
        .copy = count(chosen.transfer, chosen.transfer == chosen.graphics || chosen.transfer == chosen.compute),
    };
}

} // namespace bfr::rhi::vulkan
