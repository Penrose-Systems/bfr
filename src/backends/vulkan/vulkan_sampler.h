#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/sampler.h"
#include "vulkan/vulkan_caps.h"

#include <volk.h>

#include <cstdint>

namespace bfr::rhi::vulkan {

/// A VkSampler, destroyed when this goes, and its slot in the sampler table,
/// which the device gives back when it is collected.
class VulkanSampler {
public:
    VulkanSampler(const VolkDeviceTable& table, VkDevice device, VkSampler sampler, std::uint32_t slot);
    ~VulkanSampler();
    VulkanSampler(const VulkanSampler&) = delete;
    VulkanSampler& operator=(const VulkanSampler&) = delete;
    VulkanSampler(VulkanSampler&& other) noexcept;
    VulkanSampler& operator=(VulkanSampler&&) = delete;

    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] std::uint32_t slot() const { return slot_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkSampler sampler_;
    std::uint32_t slot_;
};

/// A sampler as desc says, holding slot. table is device's and outlives it.
/// Fails with RhiUnsupported when desc needs a feature of features the
/// device lacks, or LOD averaging, which Vulkan has no setting for, and
/// RhiDeviceFailed when the API refuses.
[[nodiscard]] core::Result<VulkanSampler> createVulkanSampler(const VolkDeviceTable& table, VkDevice device,
                                                              const SamplerFeatures& features, const SamplerDesc& desc,
                                                              std::uint32_t slot);

} // namespace bfr::rhi::vulkan
