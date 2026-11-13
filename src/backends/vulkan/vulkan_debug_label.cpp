#include "bfr/core/error.h"
#include "bfr/rhi/debug_label.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_instance.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

constexpr char kEnd = '\0';

// label as Vulkan takes it, its name copied into room with the '\0' Vulkan
// reads as its end (VUID-VkDebugUtilsLabelEXT-pLabelName-parameter). Fails
// with RhiLimitReached when room can't hold it, and RhiInvalidCall when the
// name holds a '\0', which would end it early.
core::Result<VkDebugUtilsLabelEXT> labelIn(std::span<char> room, const DebugLabel& label) {
    if (label.name.size() >= room.size()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the label's name is {} bytes; DeviceDesc::maxDebugLabelLength is {}", label.name.size(),
                             room.size() - 1);
    }
    if (label.name.contains(kEnd)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the label's name holds a '\\0'");
    }
    *std::ranges::copy(label.name, room.begin()).out = kEnd;
    VkDebugUtilsLabelEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    info.pLabelName = room.data();
    std::ranges::copy(label.color, std::begin(info.color));
    return info;
}

core::Result<void> checkLabels(bool labels) {
    if (!labels) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the instance did not enable VK_EXT_debug_utils (Features::debugLabels)");
    }
    return {};
}

} // namespace

core::Result<VulkanCommandList*> VulkanDevice::labelledList(CommandListHandle commandList) {
    if (auto checked = checkLabels(features_.debugLabels); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return commandListFor(commandList);
}

std::span<char> VulkanDevice::listLabel(std::uint32_t slot) {
    const std::size_t room = std::size_t{desc_.maxDebugLabelLength} + 1;
    return std::span{listLabels_}.subspan(slot * room, room);
}

core::Result<void> VulkanDevice::beginDebugLabel(CommandListHandle commandList, const DebugLabel& label) {
    auto list = labelledList(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto info = labelIn(listLabel((*list)->slot), label);
    if (!info) {
        return std::unexpected(std::move(info.error()));
    }
    instance_->instanceTable().vkCmdBeginDebugUtilsLabelEXT((*list)->buffer, &*info);
    return {};
}

core::Result<void> VulkanDevice::endDebugLabel(CommandListHandle commandList) {
    auto list = labelledList(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    instance_->instanceTable().vkCmdEndDebugUtilsLabelEXT((*list)->buffer);
    return {};
}

core::Result<void> VulkanDevice::insertDebugLabel(CommandListHandle commandList, const DebugLabel& label) {
    auto list = labelledList(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto info = labelIn(listLabel((*list)->slot), label);
    if (!info) {
        return std::unexpected(std::move(info.error()));
    }
    instance_->instanceTable().vkCmdInsertDebugUtilsLabelEXT((*list)->buffer, &*info);
    return {};
}

core::Result<VulkanQueue*> VulkanDevice::labelledQueue(std::uint32_t queue) {
    if (auto checked = checkLabels(features_.queueDebugLabels); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return queues_[queue].get();
}

// VK_EXT_debug_utils is an instance extension, so its functions are in the
// instance's table. The queue is externally synchronized (vkQueueBeginDebugUtilsLabelEXT), so
// each call takes the lock submit takes.

core::Result<void> VulkanDevice::beginQueueDebugLabel(std::uint32_t queue, const DebugLabel& label) {
    auto found = labelledQueue(queue);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const std::scoped_lock held((*found)->lock);
    auto info = labelIn((*found)->label, label);
    if (!info) {
        return std::unexpected(std::move(info.error()));
    }
    instance_->instanceTable().vkQueueBeginDebugUtilsLabelEXT((*found)->queue, &*info);
    return {};
}

core::Result<void> VulkanDevice::endQueueDebugLabel(std::uint32_t queue) {
    auto found = labelledQueue(queue);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const std::scoped_lock held((*found)->lock);
    instance_->instanceTable().vkQueueEndDebugUtilsLabelEXT((*found)->queue);
    return {};
}

core::Result<void> VulkanDevice::insertQueueDebugLabel(std::uint32_t queue, const DebugLabel& label) {
    auto found = labelledQueue(queue);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const std::scoped_lock held((*found)->lock);
    auto info = labelIn((*found)->label, label);
    if (!info) {
        return std::unexpected(std::move(info.error()));
    }
    instance_->instanceTable().vkQueueInsertDebugUtilsLabelEXT((*found)->queue, &*info);
    return {};
}

} // namespace bfr::rhi::vulkan
