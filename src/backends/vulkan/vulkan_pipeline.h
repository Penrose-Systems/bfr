#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"

#include <volk.h>

#include <cstdint>
#include <span>
#include <string>

namespace bfr::rhi::vulkan {

/// The bindings of descriptor set 0, which api/bfr.yaml states under
/// bfr_pipeline and every shader declares.
inline constexpr std::uint32_t kSampledTextureBinding = 0;
inline constexpr std::uint32_t kStorageTextureBinding = 1;
inline constexpr std::uint32_t kStorageBufferBinding = 2;
inline constexpr std::uint32_t kSamplerBinding = 3;
/// Only on a device with an acceleration structure table.
inline constexpr std::uint32_t kAccelerationStructureBinding = 4;
/// Only on a device with partitioned top structures.
inline constexpr std::uint32_t kPartitionedTopBinding = 5;

/// The viewports, and scissors, a pipeline has and setViewport and setScissor
/// set: one each, so the caps report that many.
inline constexpr std::uint32_t kViewports = 1;

/// op as Vulkan's, for depth and stencil tests and compare samplers.
[[nodiscard]] VkCompareOp compareFor(CompareOp op);

/// The Vulkan topology topology is, which a pipeline is made with and
/// setTopology sets.
[[nodiscard]] VkPrimitiveTopology topologyFor(Topology topology);

/// The one resource layout every pipeline shares: the descriptor set layout
/// of the four tables, and the acceleration structure and partitioned top
/// structure tables where the device has them, sized as the DeviceDesc asked, the set holding them,
/// allocated from a pool of its own, and the pipeline layout over the set and
/// the push constants. Made with the device and destroyed with it, after
/// every pipeline.
class VulkanBindlessLayout {
public:
    /// Fails with RhiLimitReached when the API says it cannot hold a set of
    /// desc's sizes, and RhiDeviceFailed when it refuses otherwise. The
    /// acceleration structure table is a binding only with
    /// accelerationStructures, and the partitioned top one only with
    /// partitionedTops. table is device's and outlives this. The push
    /// constant range is pushConstantsSize bytes over every stage.
    [[nodiscard]] static core::Result<VulkanBindlessLayout> create(const VolkDeviceTable& table, VkDevice device,
                                                                   const DeviceDesc& desc,
                                                                   std::uint32_t pushConstantsSize,
                                                                   bool accelerationStructures, bool partitionedTops);

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
    /// Points slot of the texture table at view, read in layout.
    void writeSampledImage(std::uint32_t slot, VkImageView view, VkImageLayout layout) const;
    /// Points slot of the storage texture table at view, in layout GENERAL.
    void writeStorageImage(std::uint32_t slot, VkImageView view) const;
    /// Points slot of the sampler table at sampler.
    void writeSampler(std::uint32_t slot, VkSampler sampler) const;
    /// Points slot of the acceleration structure table at structure.
    void writeAccelerationStructure(std::uint32_t slot, VkAccelerationStructureKHR structure) const;
    /// Points slot of the partitioned top table at the structure at address
    /// structure.
    void writePartitionedTop(std::uint32_t slot, VkDeviceAddress structure) const;

private:
    void destroy();

    const VolkDeviceTable* table_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
};

/// A VkPipeline, destroyed when this goes, with where it binds and its
/// topology, or a ray tracing pipeline's stack size, 0 for its own, which
/// binding it sets, and whether it was made with allowMotion.
class VulkanPipeline {
public:
    VulkanPipeline(const VolkDeviceTable& table, VkDevice device, VkPipeline pipeline, VkPipelineBindPoint bindPoint,
                   VkPrimitiveTopology topology, std::uint64_t stackSize = 0);
    ~VulkanPipeline();
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;
    VulkanPipeline(VulkanPipeline&& other) noexcept;
    VulkanPipeline& operator=(VulkanPipeline&&) = delete;

    [[nodiscard]] VkPipeline pipeline() const { return pipeline_; }
    [[nodiscard]] VkPipelineBindPoint bindPoint() const { return bindPoint_; }
    [[nodiscard]] VkPrimitiveTopology topology() const { return topology_; }
    [[nodiscard]] std::uint64_t stackSize() const { return stackSize_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkPipeline pipeline_;
    VkPipelineBindPoint bindPoint_;
    VkPrimitiveTopology topology_;
    std::uint64_t stackSize_;
};

/// A VkShaderModule and the entry point its pipelines run, destroyed when
/// this goes: pipelines made from it keep what they need of it.
class VulkanShader {
public:
    VulkanShader(const VolkDeviceTable& table, VkDevice device, VkShaderModule module, std::string entryPoint);
    ~VulkanShader();
    VulkanShader(const VulkanShader&) = delete;
    VulkanShader& operator=(const VulkanShader&) = delete;
    VulkanShader(VulkanShader&& other) noexcept;
    VulkanShader& operator=(VulkanShader&&) = delete;

    [[nodiscard]] VkShaderModule module() const { return module_; }
    [[nodiscard]] const std::string& entryPoint() const { return entryPoint_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkShaderModule module_;
    std::string entryPoint_;
};

/// A shader of code, which is SPIR-V. Fails with RhiInvalidCall when the
/// code is not whole words, and RhiDeviceFailed when the API refuses it.
[[nodiscard]] core::Result<VulkanShader> createVulkanShader(const VolkDeviceTable& table, VkDevice device,
                                                            const ShaderCode& code);

/// A ray tracing pipeline on layout of stages, which point at the shaders'
/// modules, and desc's groups and recursion depth, linking libraries, with
/// desc's replay identifiers of replaySize bytes each, using cache as
/// createVulkanComputePipeline does. Fails with RhiDeviceFailed when the API
/// refuses it.
[[nodiscard]] core::Result<VulkanPipeline> createVulkanRayTracingPipeline(
    const VolkDeviceTable& table, VkDevice device, VkPipelineLayout layout, VkPipelineCache cache,
    std::span<const VkPipelineShaderStageCreateInfo> stages, std::span<const VkPipeline> libraries,
    std::uint32_t replaySize, const RayTracingPipelineDesc& desc);

/// The Vulkan stage of a ray tracing shader at stage.
[[nodiscard]] VkShaderStageFlagBits rayTracingStageFor(RayTracingStage stage);

/// A compute pipeline on layout as desc says, whose code is SPIR-V, looked
/// for in cache and added to it, where cache is not VK_NULL_HANDLE. Fails with
/// RhiInvalidCall when the code is not whole words, and RhiDeviceFailed when
/// the API refuses the module or the pipeline.
[[nodiscard]] core::Result<VulkanPipeline> createVulkanComputePipeline(const VolkDeviceTable& table, VkDevice device,
                                                                       VkPipelineLayout layout, VkPipelineCache cache,
                                                                       const ComputePipelineDesc& desc);

/// A graphics pipeline on layout as desc says, whose code is SPIR-V, for
/// dynamic rendering to desc's formats with a dynamic viewport, scissor,
/// topology, blend constants and stencil reference, using cache as
/// createVulkanComputePipeline does. Fails
/// with RhiInvalidCall when code is not whole words, and RhiDeviceFailed when
/// the API refuses a module or the pipeline.
[[nodiscard]] core::Result<VulkanPipeline> createVulkanGraphicsPipeline(const VolkDeviceTable& table, VkDevice device,
                                                                        VkPipelineLayout layout, VkPipelineCache cache,
                                                                        bool dynamicTopology,
                                                                        const GraphicsPipelineDesc& desc);

} // namespace bfr::rhi::vulkan
