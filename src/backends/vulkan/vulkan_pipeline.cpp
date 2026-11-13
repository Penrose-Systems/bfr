#include "vulkan/vulkan_pipeline.h"

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"
#include "rhi/pipeline_check.h"
#include "rhi/ray_tracing_check.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bfr::rhi::vulkan {

VkPrimitiveTopology topologyFor(Topology topology) {
    switch (topology) {
        case Topology::TriangleStrip:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case Topology::TriangleFan:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        case Topology::LineList:
            return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case Topology::LineStrip:
            return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case Topology::PointList:
            return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case Topology::TriangleList:
        default:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

namespace {

// Bits in one VkSampleMask word; pSampleMask holds as many words as samples need.
constexpr std::uint32_t kSampleMaskBits = std::numeric_limits<VkSampleMask>::digits;

constexpr std::uint32_t kBindlessSet = 0;
// The four tables every device has, the acceleration structure table, the
// partitioned top structure table and the uniform buffer table.
constexpr std::uint32_t kBindingCount = 7;
constexpr VkDescriptorBindingFlags kTableFlags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                                 VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

struct Table {
    std::uint32_t binding;
    VkDescriptorType type;
    std::uint32_t count;
};

// The tables, in binding order, sized as desc asks.
std::array<Table, kBindingCount> tablesFor(const DeviceDesc& desc) {
    return {{
        {.binding = kSampledTextureBinding,
         .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .count = desc.maxBindlessTextures},
        {.binding = kStorageTextureBinding,
         .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .count = desc.maxBindlessStorageTextures},
        {.binding = kStorageBufferBinding,
         .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .count = desc.maxBindlessStorageBuffers},
        {.binding = kSamplerBinding, .type = VK_DESCRIPTOR_TYPE_SAMPLER, .count = desc.maxBindlessSamplers},
        {.binding = kAccelerationStructureBinding,
         .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
         .count = desc.maxBindlessAccelerationStructures},
        {.binding = kPartitionedTopBinding,
         .type = VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV,
         .count = desc.maxBindlessPartitionedTops},
        {.binding = kUniformBufferBinding,
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .count = desc.maxBindlessUniformBuffers},
    }};
}

// SPIR-V as the API takes it: whole words, aligned. The caller's bytes may be
// neither, so they are copied; pipeline creation is cold.
std::vector<std::uint32_t> wordsOf(std::span<const std::byte> code) {
    std::vector<std::uint32_t> words((code.size() + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    return words;
}

// A shader module from code, which the caller destroys once its pipeline is
// made: a pipeline keeps what it needs of it.
core::Result<VkShaderModule> createModule(const VolkDeviceTable& table, VkDevice device,
                                          std::span<const std::byte> code) {
    const auto words = wordsOf(code);
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (const VkResult result = table.vkCreateShaderModule(device, &info, nullptr, &module); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateShaderModule", .result = result});
    }
    return module;
}

// A stage's create info over module. entryPoint is kept by the caller as a
// string the info points into.
VkPipelineShaderStageCreateInfo stageFor(VkShaderStageFlagBits stage, VkShaderModule module, const char* entryPoint) {
    VkPipelineShaderStageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage = stage;
    info.module = module;
    info.pName = entryPoint;
    return info;
}

// The shader modules of a pipeline's stages and the stage infos pointing at
// them, destroyed once the pipeline is made from them.
class StageModules {
public:
    StageModules(const VolkDeviceTable& table, VkDevice device) : table_(&table), device_(device) {}
    ~StageModules() {
        for (VkShaderModule module : std::span{modules_}.first(count_)) {
            table_->vkDestroyShaderModule(device_, module, nullptr);
        }
    }
    StageModules(const StageModules&) = delete;
    StageModules& operator=(const StageModules&) = delete;
    StageModules(StageModules&&) = delete;
    StageModules& operator=(StageModules&&) = delete;

    // Adds stage, run from shader's code. Fails as createModule does.
    core::Result<void> add(VkShaderStageFlagBits stage, const ShaderCode& shader) {
        auto module = createModule(*table_, device_, shader.code);
        if (!module) {
            return std::unexpected(std::move(module.error()));
        }
        entryPoints_.at(count_) = shader.entryPoint;
        modules_.at(count_) = *module;
        stages_.at(count_) = stageFor(stage, *module, entryPoints_.at(count_).c_str());
        ++count_;
        return {};
    }

    [[nodiscard]] std::span<const VkPipelineShaderStageCreateInfo> stages() const {
        return std::span{stages_}.first(count_);
    }

private:
    // Task, mesh and fragment, the most a pipeline has.
    static constexpr std::size_t kMaxStages = 3;
    const VolkDeviceTable* table_;
    VkDevice device_;
    std::size_t count_ = 0;
    std::array<std::string, kMaxStages> entryPoints_{};
    std::array<VkShaderModule, kMaxStages> modules_{};
    std::array<VkPipelineShaderStageCreateInfo, kMaxStages> stages_{};
};

VkCullModeFlags cullFor(CullMode cull) {
    switch (cull) {
        case CullMode::Front:
            return VK_CULL_MODE_FRONT_BIT;
        case CullMode::Back:
            return VK_CULL_MODE_BACK_BIT;
        case CullMode::None:
        default:
            return VK_CULL_MODE_NONE;
    }
}

} // namespace

VkCompareOp compareFor(CompareOp op) {
    switch (op) {
        case CompareOp::Never:
            return VK_COMPARE_OP_NEVER;
        case CompareOp::Less:
            return VK_COMPARE_OP_LESS;
        case CompareOp::Equal:
            return VK_COMPARE_OP_EQUAL;
        case CompareOp::LessEqual:
            return VK_COMPARE_OP_LESS_OR_EQUAL;
        case CompareOp::Greater:
            return VK_COMPARE_OP_GREATER;
        case CompareOp::NotEqual:
            return VK_COMPARE_OP_NOT_EQUAL;
        case CompareOp::GreaterEqual:
            return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case CompareOp::Always:
        default:
            return VK_COMPARE_OP_ALWAYS;
    }
}

namespace {

VkStencilOp stencilFor(StencilOp op) {
    switch (op) {
        case StencilOp::Zero:
            return VK_STENCIL_OP_ZERO;
        case StencilOp::Replace:
            return VK_STENCIL_OP_REPLACE;
        case StencilOp::IncrementClamp:
            return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case StencilOp::DecrementClamp:
            return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case StencilOp::Invert:
            return VK_STENCIL_OP_INVERT;
        case StencilOp::IncrementWrap:
            return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case StencilOp::DecrementWrap:
            return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        case StencilOp::Keep:
        default:
            return VK_STENCIL_OP_KEEP;
    }
}

VkStencilOpState stencilFaceFor(const StencilFace& face) {
    // The reference is dynamic, set when recording.
    return {.failOp = stencilFor(face.fail),
            .passOp = stencilFor(face.pass),
            .depthFailOp = stencilFor(face.depthFail),
            .compareOp = compareFor(face.compare),
            .compareMask = face.readMask,
            .writeMask = face.writeMask,
            .reference = 0};
}

VkBlendFactor blendFactorFor(BlendFactor factor) {
    switch (factor) {
        case BlendFactor::One:
            return VK_BLEND_FACTOR_ONE;
        case BlendFactor::SrcColor:
            return VK_BLEND_FACTOR_SRC_COLOR;
        case BlendFactor::OneMinusSrcColor:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case BlendFactor::DstColor:
            return VK_BLEND_FACTOR_DST_COLOR;
        case BlendFactor::OneMinusDstColor:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case BlendFactor::SrcAlpha:
            return VK_BLEND_FACTOR_SRC_ALPHA;
        case BlendFactor::OneMinusSrcAlpha:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case BlendFactor::DstAlpha:
            return VK_BLEND_FACTOR_DST_ALPHA;
        case BlendFactor::OneMinusDstAlpha:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case BlendFactor::ConstantColor:
            return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case BlendFactor::OneMinusConstantColor:
            return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case BlendFactor::ConstantAlpha:
            return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case BlendFactor::OneMinusConstantAlpha:
            return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        case BlendFactor::SrcAlphaSaturate:
            return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case BlendFactor::Zero:
        default:
            return VK_BLEND_FACTOR_ZERO;
    }
}

VkBlendOp blendOpFor(BlendOp op) {
    switch (op) {
        case BlendOp::Subtract:
            return VK_BLEND_OP_SUBTRACT;
        case BlendOp::ReverseSubtract:
            return VK_BLEND_OP_REVERSE_SUBTRACT;
        case BlendOp::Min:
            return VK_BLEND_OP_MIN;
        case BlendOp::Max:
            return VK_BLEND_OP_MAX;
        case BlendOp::Add:
        default:
            return VK_BLEND_OP_ADD;
    }
}

// The write mask's bits are the API's, so it passes through.
constexpr VkColorComponentFlags kAllComponents = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
static_assert(VK_COLOR_COMPONENT_R_BIT == kWriteRed && VK_COLOR_COMPONENT_G_BIT == kWriteGreen &&
              VK_COLOR_COMPONENT_B_BIT == kWriteBlue && VK_COLOR_COMPONENT_A_BIT == kWriteAlpha);

VkPipelineColorBlendAttachmentState blendFor(const BlendState& blend) {
    return {.blendEnable = blend.enable ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = blendFactorFor(blend.srcColor),
            .dstColorBlendFactor = blendFactorFor(blend.dstColor),
            .colorBlendOp = blendOpFor(blend.colorOp),
            .srcAlphaBlendFactor = blendFactorFor(blend.srcAlpha),
            .dstAlphaBlendFactor = blendFactorFor(blend.dstAlpha),
            .alphaBlendOp = blendOpFor(blend.alphaOp),
            .colorWriteMask = blend.writeMask & kAllComponents};
}

} // namespace

core::Result<VulkanBindlessLayout> VulkanBindlessLayout::create(const VolkDeviceTable& table, VkDevice device,
                                                                const DeviceDesc& desc, std::uint32_t pushConstantsSize,
                                                                bool accelerationStructures, bool partitionedTops,
                                                                bool uniformBuffers) {
    // The acceleration structure tables' types exist only with their
    // extensions, and the first's update after bind only with its feature, as
    // the uniform buffer table's does
    // (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingUniformBufferUpdateAfterBind-03005),
    // so a device without them has no such binding; checkTables refused slots
    // for it.
    const std::array<Table, kBindingCount> tables = tablesFor(desc);
    std::array<VkDescriptorSetLayoutBinding, kBindingCount> bindings{};
    std::array<VkDescriptorBindingFlags, kBindingCount> flags{};
    std::array<VkDescriptorPoolSize, kBindingCount> sizes{};
    std::uint32_t bindingCount = 0;
    std::uint32_t sizeCount = 0;
    for (const Table& held : tables) {
        if ((held.binding == kAccelerationStructureBinding && !accelerationStructures) ||
            (held.binding == kPartitionedTopBinding && !partitionedTops) ||
            (held.binding == kUniformBufferBinding && !uniformBuffers)) {
            continue;
        }
        bindings.at(bindingCount) = {.binding = held.binding,
                                     .descriptorType = held.type,
                                     .descriptorCount = held.count,
                                     .stageFlags = VK_SHADER_STAGE_ALL,
                                     .pImmutableSamplers = nullptr};
        flags.at(bindingCount++) = kTableFlags;
        // A table of 0 slots is none: the pool makes room for no descriptor of it.
        if (held.count != 0) {
            sizes.at(sizeCount++) = {.type = held.type, .descriptorCount = held.count};
        }
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{};
    bindingFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlags.bindingCount = bindingCount;
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlags;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = bindingCount;
    layoutInfo.pBindings = bindings.data();

    // The API says first whether it can hold a set this large; the per-type
    // limits are the caller's to keep within the caps, but the whole set has
    // its own.
    VkDescriptorSetLayoutSupport support{};
    support.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT;
    table.vkGetDescriptorSetLayoutSupport(device, &layoutInfo, &support);
    if (support.supported == VK_FALSE) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the adapter cannot hold bindless tables of {} textures, {} storage textures, {} "
                             "storage buffers, {} samplers, {} acceleration structures and {} partitioned top "
                             "structures together",
                             desc.maxBindlessTextures, desc.maxBindlessStorageTextures, desc.maxBindlessStorageBuffers,
                             desc.maxBindlessSamplers, desc.maxBindlessAccelerationStructures,
                             desc.maxBindlessPartitionedTops);
    }

    VulkanBindlessLayout made;
    made.table_ = &table;
    made.device_ = device;
    if (const VkResult result = table.vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &made.setLayout_);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateDescriptorSetLayout", .result = result});
    }

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = sizeCount;
    poolInfo.pPoolSizes = sizes.data();
    if (const VkResult result = table.vkCreateDescriptorPool(device, &poolInfo, nullptr, &made.pool_);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateDescriptorPool", .result = result});
    }

    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = made.pool_;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &made.setLayout_;
    if (const VkResult result = table.vkAllocateDescriptorSets(device, &setInfo, &made.set_); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkAllocateDescriptorSets", .result = result});
    }

    const VkPushConstantRange pushConstants{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = pushConstantsSize};
    VkPipelineLayoutCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineInfo.setLayoutCount = 1;
    pipelineInfo.pSetLayouts = &made.setLayout_;
    pipelineInfo.pushConstantRangeCount = 1;
    pipelineInfo.pPushConstantRanges = &pushConstants;
    if (const VkResult result = table.vkCreatePipelineLayout(device, &pipelineInfo, nullptr, &made.pipelineLayout_);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreatePipelineLayout", .result = result});
    }
    return made;
}

VulkanBindlessLayout::~VulkanBindlessLayout() {
    destroy();
}

namespace {

VkWriteDescriptorSet writeFor(VkDescriptorSet set, std::uint32_t binding, std::uint32_t slot, VkDescriptorType type) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = type;
    return write;
}

} // namespace

void VulkanBindlessLayout::writeStorageBuffer(std::uint32_t slot, VkBuffer buffer) const {
    const VkDescriptorBufferInfo info{.buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet write = writeFor(set_, kStorageBufferBinding, slot, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    write.pBufferInfo = &info;
    table_->vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanBindlessLayout::writeUniformBuffer(std::uint32_t slot, VkBuffer buffer) const {
    const VkDescriptorBufferInfo info{.buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet write = writeFor(set_, kUniformBufferBinding, slot, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    write.pBufferInfo = &info;
    table_->vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanBindlessLayout::writeSampledImage(std::uint32_t slot, VkImageView view, VkImageLayout layout) const {
    const VkDescriptorImageInfo info{.sampler = VK_NULL_HANDLE, .imageView = view, .imageLayout = layout};
    VkWriteDescriptorSet write = writeFor(set_, kSampledTextureBinding, slot, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    write.pImageInfo = &info;
    table_->vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanBindlessLayout::writeStorageImage(std::uint32_t slot, VkImageView view) const {
    const VkDescriptorImageInfo info{.sampler = VK_NULL_HANDLE,
                                     .imageView = view,
                                     .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write = writeFor(set_, kStorageTextureBinding, slot, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    write.pImageInfo = &info;
    table_->vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanBindlessLayout::writeSampler(std::uint32_t slot, VkSampler sampler) const {
    const VkDescriptorImageInfo info{.sampler = sampler,
                                     .imageView = VK_NULL_HANDLE,
                                     .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet write = writeFor(set_, kSamplerBinding, slot, VK_DESCRIPTOR_TYPE_SAMPLER);
    write.pImageInfo = &info;
    table_->vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanBindlessLayout::writeAccelerationStructure(std::uint32_t slot, VkAccelerationStructureKHR structure) const {
    VkWriteDescriptorSetAccelerationStructureKHR named{};
    named.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    named.accelerationStructureCount = 1;
    named.pAccelerationStructures = &structure;
    VkWriteDescriptorSet write = writeFor(set_, kAccelerationStructureBinding, slot,
                                          VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    write.pNext = &named;
    table_->vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanBindlessLayout::writePartitionedTop(std::uint32_t slot, VkDeviceAddress structure) const {
    VkWriteDescriptorSetPartitionedAccelerationStructureNV named{};
    named.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE_NV;
    named.accelerationStructureCount = 1;
    named.pAccelerationStructures = &structure;
    VkWriteDescriptorSet write = writeFor(set_, kPartitionedTopBinding, slot,
                                          VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV);
    write.pNext = &named;
    table_->vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

VulkanBindlessLayout::VulkanBindlessLayout(VulkanBindlessLayout&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      setLayout_(std::exchange(other.setLayout_, VK_NULL_HANDLE)),
      pool_(std::exchange(other.pool_, VK_NULL_HANDLE)),
      set_(std::exchange(other.set_, VK_NULL_HANDLE)),
      pipelineLayout_(std::exchange(other.pipelineLayout_, VK_NULL_HANDLE)) {}

VulkanBindlessLayout& VulkanBindlessLayout::operator=(VulkanBindlessLayout&& other) noexcept {
    if (this != &other) {
        destroy();
        table_ = other.table_;
        device_ = other.device_;
        setLayout_ = std::exchange(other.setLayout_, VK_NULL_HANDLE);
        pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);
        set_ = std::exchange(other.set_, VK_NULL_HANDLE);
        pipelineLayout_ = std::exchange(other.pipelineLayout_, VK_NULL_HANDLE);
    }
    return *this;
}

void VulkanBindlessLayout::destroy() {
    if (table_ == nullptr) {
        return;
    }
    // The set goes with its pool.
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        table_->vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    }
    if (pool_ != VK_NULL_HANDLE) {
        table_->vkDestroyDescriptorPool(device_, pool_, nullptr);
    }
    if (setLayout_ != VK_NULL_HANDLE) {
        table_->vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
    }
    pipelineLayout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    set_ = VK_NULL_HANDLE;
    setLayout_ = VK_NULL_HANDLE;
}

VulkanPipeline::VulkanPipeline(const VolkDeviceTable& table, VkDevice device, VkPipeline pipeline,
                               VkPipelineBindPoint bindPoint, VkPrimitiveTopology topology, std::uint64_t stackSize)
    : table_(&table),
      device_(device),
      pipeline_(pipeline),
      bindPoint_(bindPoint),
      topology_(topology),
      stackSize_(stackSize) {}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      pipeline_(std::exchange(other.pipeline_, VK_NULL_HANDLE)),
      bindPoint_(other.bindPoint_),
      topology_(other.topology_),
      stackSize_(other.stackSize_) {}

VulkanPipeline::~VulkanPipeline() {
    if (pipeline_ != VK_NULL_HANDLE) {
        table_->vkDestroyPipeline(device_, pipeline_, nullptr);
    }
}

core::Result<VulkanPipeline> createVulkanComputePipeline(const VolkDeviceTable& table, VkDevice device,
                                                         VkPipelineLayout layout, VkPipelineCache cache,
                                                         const ComputePipelineDesc& desc) {
    StageModules modules{table, device};
    if (auto added = modules.add(VK_SHADER_STAGE_COMPUTE_BIT, desc.compute); !added) {
        return std::unexpected(std::move(added.error()));
    }
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage = modules.stages().front();
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = table.vkCreateComputePipelines(device, cache, 1, &info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateComputePipelines", .result = result});
    }
    return VulkanPipeline{table, device, pipeline, VK_PIPELINE_BIND_POINT_COMPUTE, VK_PRIMITIVE_TOPOLOGY_MAX_ENUM};
}

core::Result<VulkanPipeline> createVulkanGraphicsPipeline(const VolkDeviceTable& table, VkDevice device,
                                                          VkPipelineLayout layout, VkPipelineCache cache,
                                                          bool dynamicTopology, const GraphicsPipelineDesc& desc) {
    const std::span targets{desc.colorTargets.data(), desc.colorTargetCount};
    std::array<VkFormat, kMaxColorTargets> colorFormats{};
    std::array<VkPipelineColorBlendAttachmentState, kMaxColorTargets> blends{};
    for (std::size_t i = 0; i < targets.size(); ++i) {
        colorFormats.at(i) = vulkanFormat(targets[i].format);
        blends.at(i) = blendFor(targets[i].blend);
    }

    // Vertices, or task and mesh shaders in their place, then fragments.
    const bool meshes = !desc.mesh.code.empty();
    StageModules modules{table, device};
    const auto added = [&]() -> core::Result<void> {
        if (!meshes) {
            return modules.add(VK_SHADER_STAGE_VERTEX_BIT, desc.vertex);
        }
        if (!desc.task.code.empty()) {
            if (auto task = modules.add(VK_SHADER_STAGE_TASK_BIT_EXT, desc.task); !task) {
                return task;
            }
        }
        return modules.add(VK_SHADER_STAGE_MESH_BIT_EXT, desc.mesh);
    }();
    if (!added) {
        return std::unexpected(added.error());
    }
    if (!desc.fragment.code.empty()) {
        if (auto fragment = modules.add(VK_SHADER_STAGE_FRAGMENT_BIT, desc.fragment); !fragment) {
            return std::unexpected(std::move(fragment.error()));
        }
    }

    // Vertices are pulled, so there is no vertex input. With a mesh shader,
    // vertex input and input assembly are ignored (pipelines.adoc:2132, :2152).
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topologyFor(desc.topology);
    // Vulkan restarts on the all-ones value of the bound index type.
    inputAssembly.primitiveRestartEnable = desc.primitiveRestart != PrimitiveRestart::Off ? VK_TRUE : VK_FALSE;
    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = kViewports;
    viewport.scissorCount = kViewports;
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.depthClampEnable = desc.raster.depthClamp ? VK_TRUE : VK_FALSE;
    raster.polygonMode = desc.raster.fill == FillMode::Wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    raster.cullMode = cullFor(desc.raster.cull);
    raster.frontFace = desc.raster.frontFace == FrontFace::Clockwise ? VK_FRONT_FACE_CLOCKWISE
                                                                     : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.depthBiasEnable = desc.raster.depthBias != 0 || desc.raster.depthBiasSlope != 0 ? VK_TRUE : VK_FALSE;
    raster.depthBiasConstantFactor = desc.raster.depthBias;
    raster.depthBiasClamp = desc.raster.depthBiasClamp;
    raster.depthBiasSlopeFactor = desc.raster.depthBiasSlope;
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = static_cast<VkSampleCountFlagBits>(desc.samples);
    multisample.alphaToCoverageEnable = desc.alphaToCoverage ? VK_TRUE : VK_FALSE;
    multisample.alphaToOneEnable = desc.alphaToOne ? VK_TRUE : VK_FALSE;
    multisample.sampleShadingEnable = desc.minSampleShading > 0 ? VK_TRUE : VK_FALSE;
    multisample.minSampleShading = desc.minSampleShading;
    const std::array<VkSampleMask, 2> sampleMask{static_cast<VkSampleMask>(desc.sampleMask),
                                                 static_cast<VkSampleMask>(desc.sampleMask >> kSampleMaskBits)};
    multisample.pSampleMask = sampleMask.data();
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = desc.depthStencil.depthTest ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = desc.depthStencil.depthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = compareFor(desc.depthStencil.depthCompare);
    depthStencil.stencilTestEnable = desc.depthStencil.stencilTest ? VK_TRUE : VK_FALSE;
    depthStencil.front = stencilFaceFor(desc.depthStencil.front);
    depthStencil.back = stencilFaceFor(desc.depthStencil.back);
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = desc.colorTargetCount;
    blend.pAttachments = blends.data();
    // Topology, last, is dynamic only where the device sets it, and never with
    // a mesh shader (VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07065).
    constexpr std::array kDynamic{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                  VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    constexpr std::size_t kWithoutTopology = kDynamic.size() - 1;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicTopology && !meshes ? kDynamic.size()
                                                                                      : kWithoutTopology);
    dynamic.pDynamicStates = kDynamic.data();

    // Dynamic rendering: the formats stand in for a render pass. A depth
    // format fills only the depth slot, a stencil one only the stencil slot.
    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = desc.colorTargetCount;
    rendering.pColorAttachmentFormats = colorFormats.data();
    if (desc.depthStencilFormat != Format::Undefined) {
        const FormatBlock& block = blockOf(desc.depthStencilFormat);
        const VkFormat format = vulkanFormat(desc.depthStencilFormat);
        rendering.depthAttachmentFormat = block.depthBytes != 0 ? format : VK_FORMAT_UNDEFINED;
        rendering.stencilAttachmentFormat = block.stencilBytes != 0 ? format : VK_FORMAT_UNDEFINED;
    }

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.pNext = &rendering;
    info.stageCount = static_cast<std::uint32_t>(modules.stages().size());
    info.pStages = modules.stages().data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depthStencil;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = table.vkCreateGraphicsPipelines(device, cache, 1, &info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateGraphicsPipelines", .result = result});
    }
    // A mesh pipeline has no topology to set as it is bound.
    return VulkanPipeline{table, device, pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          meshes ? VK_PRIMITIVE_TOPOLOGY_MAX_ENUM : topologyFor(desc.topology)};
}

VulkanShader::VulkanShader(const VolkDeviceTable& table, VkDevice device, VkShaderModule module, std::string entryPoint)
    : table_(&table), device_(device), module_(module), entryPoint_(std::move(entryPoint)) {}

VulkanShader::VulkanShader(VulkanShader&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      module_(std::exchange(other.module_, VK_NULL_HANDLE)),
      entryPoint_(std::move(other.entryPoint_)) {}

VulkanShader::~VulkanShader() {
    if (module_ != VK_NULL_HANDLE) {
        table_->vkDestroyShaderModule(device_, module_, nullptr);
    }
}

core::Result<VulkanShader> createVulkanShader(const VolkDeviceTable& table, VkDevice device, const ShaderCode& code) {
    auto module = createModule(table, device, code.code);
    if (!module) {
        return std::unexpected(std::move(module.error()));
    }
    return VulkanShader{table, device, *module, std::string{code.entryPoint}};
}

VkShaderStageFlagBits rayTracingStageFor(RayTracingStage stage) {
    switch (stage) {
        case RayTracingStage::Miss:
            return VK_SHADER_STAGE_MISS_BIT_KHR;
        case RayTracingStage::ClosestHit:
            return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        case RayTracingStage::AnyHit:
            return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
        case RayTracingStage::Intersection:
            return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        case RayTracingStage::Callable:
            return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
        case RayTracingStage::RayGeneration:
        default:
            return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    }
}

namespace {

// BFR's no-shader index is Vulkan's.
static_assert(kNoShader == VK_SHADER_UNUSED_KHR);

VkRayTracingShaderGroupTypeKHR groupTypeFor(RayTracingGroupType type) {
    switch (type) {
        case RayTracingGroupType::TriangleHit:
            return VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        case RayTracingGroupType::BoxHit:
            return VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        case RayTracingGroupType::General:
        default:
            return VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    }
}

} // namespace

core::Result<VulkanPipeline> createVulkanRayTracingPipeline(const VolkDeviceTable& table, VkDevice device,
                                                            VkPipelineLayout layout, VkPipelineCache cache,
                                                            std::span<const VkPipelineShaderStageCreateInfo> stages,
                                                            std::span<const VkPipeline> libraries,
                                                            std::uint32_t replaySize,
                                                            const RayTracingPipelineDesc& desc) {
    // Pipeline creation is cold, so the groups go in a vector. A group's
    // shader indices are the desc's, in the stages' order.
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(desc.groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const RayTracingGroup& group = desc.groups[i];
        groups[i] = {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                     .pNext = nullptr,
                     .type = groupTypeFor(group.type),
                     .generalShader = group.general,
                     .closestHitShader = group.closestHit,
                     .anyHitShader = group.anyHit,
                     .intersectionShader = group.intersection,
                     .pShaderGroupCaptureReplayHandle = desc.replayIdentifiers.empty()
                                                            ? nullptr
                                                            : desc.replayIdentifiers.subspan(i * replaySize).data()};
    }
    VkRayTracingPipelineCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    if (desc.library) {
        info.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
    }
    if (desc.captureReplay) {
        info.flags |= VK_PIPELINE_CREATE_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR;
    }
    if (desc.allowOpacityMicromaps) {
        info.flags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
    }
    if (desc.allowMotion) {
        info.flags |= VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;
    }
    // Payload and attribute sizes reach Vulkan only through a library
    // interface, which a library and a pipeline linking libraries take
    // (VUID-VkRayTracingPipelineCreateInfoKHR-flags-03465, -pLibraryInfo-03590),
    // and one without the extension may not (-pLibraryInfo-03595).
    VkPipelineLibraryCreateInfoKHR linked{};
    linked.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
    linked.libraryCount = static_cast<std::uint32_t>(libraries.size());
    linked.pLibraries = libraries.data();
    VkRayTracingPipelineInterfaceCreateInfoKHR libraryInterface{};
    libraryInterface.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR;
    libraryInterface.maxPipelineRayPayloadSize = desc.maxPayloadSize;
    libraryInterface.maxPipelineRayHitAttributeSize = desc.maxAttributeSize;
    if (desc.library || !libraries.empty()) {
        info.pLibraryInfo = &linked;
        info.pLibraryInterface = &libraryInterface;
    }
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.groupCount = static_cast<std::uint32_t>(groups.size());
    info.pGroups = groups.data();
    info.maxPipelineRayRecursionDepth = desc.maxRecursionDepth;
    info.layout = layout;
    // A stack size the caller gives is set as the pipeline is bound, which
    // takes it as dynamic state (pipelines.adoc:7954-7959).
    constexpr VkDynamicState kStackSize = VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR;
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 1;
    dynamic.pDynamicStates = &kStackSize;
    if (desc.stackSize != 0) {
        info.pDynamicState = &dynamic;
    }
    VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV clusters{};
    clusters.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    clusters.allowClusterAccelerationStructure = VK_TRUE;
    if (desc.allowClusters) {
        info.pNext = &clusters;
    }
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = table.vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, cache, 1, &info, nullptr,
                                                                 &pipeline);
    if (result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateRayTracingPipelinesKHR", .result = result});
    }
    return VulkanPipeline{
        table,         device, pipeline, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, VK_PRIMITIVE_TOPOLOGY_MAX_ENUM,
        desc.stackSize};
}

// The device's pipeline calls, beside the pipelines they make.

core::Result<VulkanPipeline*> VulkanDevice::pipelineFor(PipelineHandle pipeline) {
    VulkanPipeline* found = pipelines_.find(pipeline);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    return found;
}

core::Result<PipelineHandle> VulkanDevice::createComputePipeline(const ComputePipelineDesc& desc) {
    // Cold, so pipelines the GPU is done with are freed here, never on the
    // frame path.
    collectRetired();
    const auto cache = pipelineCacheFor(desc.cache);
    if (!cache) {
        return std::unexpected(std::move(cache.error()));
    }
    auto made = createVulkanComputePipeline(table_, device_, layout_.pipelineLayout(), *cache, desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines",
                             desc_.maxPipelines);
    }
    return pipeline;
}

core::Result<PipelineHandle> VulkanDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkGraphicsPipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkMeshStages(desc, features_); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    collectRetired();
    const auto cache = pipelineCacheFor(desc.cache);
    if (!cache) {
        return std::unexpected(std::move(cache.error()));
    }
    auto made = createVulkanGraphicsPipeline(table_, device_, layout_.pipelineLayout(), *cache,
                                             features_.dynamicTopology, desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines",
                             desc_.maxPipelines);
    }
    return pipeline;
}

core::Result<void> VulkanDevice::destroyPipeline(PipelineHandle pipeline) {
    if (pipeline.isNull()) {
        return {};
    }
    if (pipelines_.find(pipeline) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    // Retired against every submit made so far, on every queue, so it lives
    // until the GPU has run all of them.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!pipelines_.retire(pipeline, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    collectRetired();
    return {};
}

core::Result<ShaderHandle> VulkanDevice::createShader(const ShaderCode& code) {
    auto made = createVulkanShader(table_, device_, code);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const ShaderHandle shader = shaders_.emplace(std::move(*made));
    if (shader.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} shaders", desc_.maxShaders);
    }
    return shader;
}

core::Result<void> VulkanDevice::destroyShader(ShaderHandle shader) {
    if (shader.isNull()) {
        return {};
    }
    // No GPU work reads a module, so it goes at once.
    if (!shaders_.erase(shader)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the shader handle is stale");
    }
    return {};
}

core::Result<PipelineHandle> VulkanDevice::createRayTracingPipeline(const RayTracingPipelineDesc& desc) {
    if (auto checked = checkRayTracingPipelines(features_.rayTracingPipeline); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkPipelineExtras(desc, features_, rayTracing_.captureReplayIdentifierSize); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    collectRetired();
    const auto cache = pipelineCacheFor(desc.cache);
    if (!cache) {
        return std::unexpected(std::move(cache.error()));
    }
    // Cold, so the stages and libraries go in vectors; the stages point into
    // the shaders, which stay while the pipeline is made.
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(desc.shaders.size());
    for (const RayTracingShader& shader : desc.shaders) {
        const VulkanShader* found = shaders_.find(shader.shader);
        if (found == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidHandle, "a shader handle is stale");
        }
        stages.push_back(stageFor(rayTracingStageFor(shader.stage), found->module(), found->entryPoint().c_str()));
    }
    std::vector<VkPipeline> libraries;
    libraries.reserve(desc.libraries.size());
    for (const PipelineHandle library : desc.libraries) {
        auto found = pipelineFor(library);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        libraries.push_back((*found)->pipeline());
    }
    auto made = createVulkanRayTracingPipeline(table_, device_, layout_.pipelineLayout(), *cache, stages, libraries,
                                               rayTracing_.captureReplayIdentifierSize, desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines",
                             desc_.maxPipelines);
    }
    return pipeline;
}

core::Result<void> VulkanDevice::shaderIdentifiers(PipelineHandle pipeline, std::uint32_t first, std::uint32_t count,
                                                   std::span<std::byte> out) {
    if (auto checked = checkRayTracingPipelines(features_.rayTracingPipeline); !checked) {
        return checked;
    }
    auto found = pipelineFor(pipeline);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (auto checked = checkIdentifiers(count, out, rayTracing_.identifierSize); !checked) {
        return checked;
    }
    const std::size_t bytes = std::size_t{count} * rayTracing_.identifierSize;
    if (const VkResult result = table_.vkGetRayTracingShaderGroupHandlesKHR(device_, (*found)->pipeline(), first, count,
                                                                            bytes, out.data());
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "vkGetRayTracingShaderGroupHandlesKHR", .result = result});
    }
    return {};
}

core::Result<void> VulkanDevice::captureReplayIdentifiers(PipelineHandle pipeline, std::uint32_t first,
                                                          std::uint32_t count, std::span<std::byte> out) {
    if (auto checked = checkRayTracingPipelines(features_.rayTracingPipeline); !checked) {
        return checked;
    }
    auto found = pipelineFor(pipeline);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (auto checked = checkIdentifiers(count, out, rayTracing_.captureReplayIdentifierSize); !checked) {
        return checked;
    }
    const std::size_t bytes = std::size_t{count} * rayTracing_.captureReplayIdentifierSize;
    if (const VkResult result = table_.vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(device_, (*found)->pipeline(),
                                                                                         first, count, bytes,
                                                                                         out.data());
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR", .result = result});
    }
    return {};
}

namespace {

VkShaderGroupShaderKHR groupShaderFor(RayTracingGroupShader shader) {
    switch (shader) {
        case RayTracingGroupShader::ClosestHit:
            return VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR;
        case RayTracingGroupShader::AnyHit:
            return VK_SHADER_GROUP_SHADER_ANY_HIT_KHR;
        case RayTracingGroupShader::Intersection:
            return VK_SHADER_GROUP_SHADER_INTERSECTION_KHR;
        case RayTracingGroupShader::General:
        default:
            return VK_SHADER_GROUP_SHADER_GENERAL_KHR;
    }
}

} // namespace

core::Result<std::uint64_t> VulkanDevice::rayTracingStackSize(PipelineHandle pipeline, std::uint32_t group,
                                                              RayTracingGroupShader shader) {
    if (auto checked = checkRayTracingPipelines(features_.rayTracingPipeline); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkKnown(shader, "ray tracing group shader"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    auto found = pipelineFor(pipeline);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return table_.vkGetRayTracingShaderGroupStackSizeKHR(device_, (*found)->pipeline(), group, groupShaderFor(shader));
}

} // namespace bfr::rhi::vulkan
