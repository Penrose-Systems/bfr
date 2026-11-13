#pragma once

#include "bfr/rhi/query.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

namespace bfr::rhi::vulkan {

/// A VkQueryPool and where its results land in a buffer
/// vkCmdCopyQueryPoolResults writes, destroyed when this goes.
class VulkanQueryPool {
public:
    VulkanQueryPool(const VolkDeviceTable& table, VkDevice device, VkQueryPool pool, const QueryLayout& layout);
    ~VulkanQueryPool();
    VulkanQueryPool(const VulkanQueryPool&) = delete;
    VulkanQueryPool& operator=(const VulkanQueryPool&) = delete;
    VulkanQueryPool(VulkanQueryPool&& other) noexcept;
    VulkanQueryPool& operator=(VulkanQueryPool&&) = delete;

    [[nodiscard]] VkQueryPool pool() const { return pool_; }
    [[nodiscard]] const QueryLayout& layout() const { return layout_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkQueryPool pool_;
    QueryLayout layout_;
};

} // namespace bfr::rhi::vulkan
