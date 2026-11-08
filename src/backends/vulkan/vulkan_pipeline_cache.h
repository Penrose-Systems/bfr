#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/pipeline.h"

#include <volk.h>

#include <cstddef>
#include <span>

namespace bfr::rhi::vulkan {

/// A VkPipelineCache, destroyed when this goes. Vulkan synchronises its use
/// inside the calls that take it, so pipelines on several threads share it.
class VulkanPipelineCache {
public:
    VulkanPipelineCache(const VolkDeviceTable& table, VkDevice device, VkPipelineCache cache);
    ~VulkanPipelineCache();
    VulkanPipelineCache(const VulkanPipelineCache&) = delete;
    VulkanPipelineCache& operator=(const VulkanPipelineCache&) = delete;
    VulkanPipelineCache(VulkanPipelineCache&& other) noexcept;
    VulkanPipelineCache& operator=(VulkanPipelineCache&&) = delete;

    [[nodiscard]] VkPipelineCache cache() const { return cache_; }

    /// The bytes vkGetPipelineCacheData has now.
    [[nodiscard]] core::Result<std::size_t> size() const;

    /// Writes as many of the bytes as data holds, and whether Vulkan says
    /// that is all of them.
    [[nodiscard]] core::Result<PipelineCacheWrite> write(std::span<std::byte> data) const;

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkPipelineCache cache_;
};

/// A pipeline cache that starts with data, which Vulkan reads during the
/// call; data it finds incompatible makes an empty one. Fails with
/// RhiDeviceFailed when the API refuses.
[[nodiscard]] core::Result<VulkanPipelineCache> createVulkanPipelineCache(const VolkDeviceTable& table, VkDevice device,
                                                                          std::span<const std::byte> data);

} // namespace bfr::rhi::vulkan
