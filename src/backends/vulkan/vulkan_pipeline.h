#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"

#include <volk.h>

#include <cstdint>

namespace bfr::rhi::vulkan {

/// The bindings of descriptor set 0, which api/bfr.yaml states under
/// bfr_pipeline and every shader declares.
inline constexpr std::uint32_t kSampledTextureBinding = 0;
inline constexpr std::uint32_t kStorageTextureBinding = 1;
inline constexpr std::uint32_t kStorageBufferBinding = 2;
inline constexpr std::uint32_t kSamplerBinding = 3;

/// The viewports, and scissors, a pipeline has and setViewport and setScissor
/// set: one each, so the caps report that many.
inline constexpr std::uint32_t kViewports = 1;

/// op as Vulkan's, for depth and stencil tests and compare samplers.
[[nodiscard]] VkCompareOp compareFor(CompareOp op);

/// The one resource layout every pipeline shares: the descriptor set layout
/// of the four tables, sized as the DeviceDesc asked, the set holding them,
/// allocated from a pool of its own, and the pipeline layout over the set and
/// the push constants. Made with the device and destroyed with it, after
/// every pipeline.
class VulkanBindlessLayout {
public:
    /// Fails with RhiLimitReached when the API says it cannot hold a set of
    /// desc's sizes, and RhiDeviceFailed when it refuses otherwise. table is
    /// device's and outlives this. The push constant range is
    /// pushConstantsSize bytes over every stage.
    [[nodiscard]] static core::Result<VulkanBindlessLayout> create(const VolkDeviceTable& table, VkDevice device,
                                                                   const DeviceDesc& desc,
                                                                   std::uint32_t pushConstantsSize);

    VulkanBindlessLayout() = default;
    ~VulkanBindlessLayout();
    VulkanBindlessLayout(const VulkanBindlessLayout&) = delete;
    VulkanBindlessLayout& operator=(const VulkanBindlessLayout&) = delete;
    VulkanBindlessLayout(VulkanBindlessLayout&& other) noexcept;
    VulkanBindlessLayout& operator=(VulkanBindlessLayout&& other) noexcept;

    [[nodiscard]] VkPipelineLayout pipelineLayout() const { return pipelineLayout_; }
    [[nodiscard]] VkDescriptorSet set() const { return set_; }

    /// Points slot of the storage buffer table at the whole of buffer.
    void writeStorageBuffer(std::uint32_t slot, VkBuffer buffer) const;
    /// Points slot of the texture table at view, read in layout
    /// SHADER_READ_ONLY_OPTIMAL.
    void writeSampledImage(std::uint32_t slot, VkImageView view) const;
    /// Points slot of the storage texture table at view, in layout GENERAL.
    void writeStorageImage(std::uint32_t slot, VkImageView view) const;
    /// Points slot of the sampler table at sampler.
    void writeSampler(std::uint32_t slot, VkSampler sampler) const;

private:
    void destroy();

    const VolkDeviceTable* table_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
};

/// A VkPipeline, destroyed when this goes, with where it binds and the list
/// topology of its class, which a draw sets.
class VulkanPipeline {
public:
    VulkanPipeline(const VolkDeviceTable& table, VkDevice device, VkPipeline pipeline, VkPipelineBindPoint bindPoint,
                   VkPrimitiveTopology topology);
    ~VulkanPipeline();
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;
    VulkanPipeline(VulkanPipeline&& other) noexcept;
    VulkanPipeline& operator=(VulkanPipeline&&) = delete;

    [[nodiscard]] VkPipeline pipeline() const { return pipeline_; }
    [[nodiscard]] VkPipelineBindPoint bindPoint() const { return bindPoint_; }
    [[nodiscard]] VkPrimitiveTopology topology() const { return topology_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkPipeline pipeline_;
    VkPipelineBindPoint bindPoint_;
    VkPrimitiveTopology topology_;
};

/// A compute pipeline on layout as desc says, whose code is SPIR-V. Fails with
/// RhiInvalidCall when the code is not whole words, and RhiDeviceFailed when
/// the API refuses the module or the pipeline.
[[nodiscard]] core::Result<VulkanPipeline> createVulkanComputePipeline(const VolkDeviceTable& table, VkDevice device,
                                                                       VkPipelineLayout layout,
                                                                       const ComputePipelineDesc& desc);

/// A graphics pipeline on layout as desc says, whose code is SPIR-V, for
/// dynamic rendering to desc's formats with a dynamic viewport, scissor,
/// topology of the desc's class, blend constants and stencil reference. Fails
/// with RhiInvalidCall when code is not whole words, and RhiDeviceFailed when
/// the API refuses a module or the pipeline.
[[nodiscard]] core::Result<VulkanPipeline> createVulkanGraphicsPipeline(const VolkDeviceTable& table, VkDevice device,
                                                                        VkPipelineLayout layout,
                                                                        const GraphicsPipelineDesc& desc);

} // namespace bfr::rhi::vulkan
