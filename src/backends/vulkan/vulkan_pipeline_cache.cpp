#include "vulkan/vulkan_pipeline_cache.h"

#include "bfr/core/error.h"
#include "bfr/rhi/pipeline.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <expected>
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

VulkanPipelineCache::VulkanPipelineCache(const VolkDeviceTable& table, VkDevice device, VkPipelineCache cache)
    : table_(&table), device_(device), cache_(cache) {}

VulkanPipelineCache::VulkanPipelineCache(VulkanPipelineCache&& other) noexcept
    : table_(other.table_), device_(other.device_), cache_(std::exchange(other.cache_, VK_NULL_HANDLE)) {}

VulkanPipelineCache::~VulkanPipelineCache() {
    if (cache_ != VK_NULL_HANDLE) {
        table_->vkDestroyPipelineCache(device_, cache_, nullptr);
    }
}

core::Result<std::size_t> VulkanPipelineCache::size() const {
    std::size_t size = 0;
    if (const VkResult result = table_->vkGetPipelineCacheData(device_, cache_, &size, nullptr); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkGetPipelineCacheData", .result = result});
    }
    return size;
}

core::Result<PipelineCacheWrite> VulkanPipelineCache::write(std::span<std::byte> data) const {
    // A null pData asks the size instead, so an empty data writes nothing,
    // all of a cache with no bytes.
    if (data.empty()) {
        auto total = size();
        if (!total) {
            return std::unexpected(std::move(total.error()));
        }
        return PipelineCacheWrite{.written = 0, .complete = *total == 0};
    }
    std::size_t size = data.size();
    // VK_INCOMPLETE says data held only part, which is written and valid.
    const VkResult result = table_->vkGetPipelineCacheData(device_, cache_, &size, data.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkGetPipelineCacheData", .result = result});
    }
    return PipelineCacheWrite{.written = size, .complete = result == VK_SUCCESS};
}

core::Result<VulkanPipelineCache> createVulkanPipelineCache(const VolkDeviceTable& table, VkDevice device,
                                                            std::span<const std::byte> data) {
    VkPipelineCacheCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    info.initialDataSize = data.size();
    info.pInitialData = data.data();
    VkPipelineCache cache = VK_NULL_HANDLE;
    if (const VkResult result = table.vkCreatePipelineCache(device, &info, nullptr, &cache); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreatePipelineCache", .result = result});
    }
    return VulkanPipelineCache{table, device, cache};
}

core::Result<PipelineCacheHandle> VulkanDevice::createPipelineCache(const PipelineCacheDesc& desc) {
    auto made = createVulkanPipelineCache(table_, device_, desc.data);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineCacheHandle cache = pipelineCaches_.emplace(std::move(*made));
    if (cache.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipeline caches",
                             desc_.maxPipelineCaches);
    }
    return cache;
}

core::Result<std::size_t> VulkanDevice::pipelineCacheSize(PipelineCacheHandle cache) {
    const VulkanPipelineCache* held = pipelineCaches_.find(cache);
    if (held == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    return held->size();
}

core::Result<PipelineCacheWrite> VulkanDevice::pipelineCacheData(PipelineCacheHandle cache, std::span<std::byte> data) {
    const VulkanPipelineCache* held = pipelineCaches_.find(cache);
    if (held == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    return held->write(data);
}

core::Result<void> VulkanDevice::destroyPipelineCache(PipelineCacheHandle cache) {
    if (cache.isNull()) {
        return {};
    }
    // Vulkan reads a cache only during the calls given it, so it goes at once.
    if (!pipelineCaches_.erase(cache)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    return {};
}

core::Result<VkPipelineCache> VulkanDevice::pipelineCacheFor(PipelineCacheHandle cache) {
    if (cache.isNull()) {
        return VK_NULL_HANDLE;
    }
    const VulkanPipelineCache* held = pipelineCaches_.find(cache);
    if (held == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    return held->cache();
}

} // namespace bfr::rhi::vulkan
