#pragma once

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace bfr::rhi::vulkan {

/// A VkQueryPool of timestamps, destroyed when this goes.
class VulkanTimestampPool {
public:
    VulkanTimestampPool(const VolkDeviceTable& table, VkDevice device, VkQueryPool pool);
    ~VulkanTimestampPool();
    VulkanTimestampPool(const VulkanTimestampPool&) = delete;
    VulkanTimestampPool& operator=(const VulkanTimestampPool&) = delete;
    VulkanTimestampPool(VulkanTimestampPool&& other) noexcept;
    VulkanTimestampPool& operator=(VulkanTimestampPool&&) = delete;

    [[nodiscard]] VkQueryPool pool() const { return pool_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkQueryPool pool_;
};

} // namespace bfr::rhi::vulkan
