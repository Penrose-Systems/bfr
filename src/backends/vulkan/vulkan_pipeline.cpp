#include "vulkan/vulkan_pipeline.h"

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/format_blocks.h"
#include "rhi/pipeline_check.h"
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
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bfr::rhi::vulkan {

namespace {

constexpr std::uint32_t kBindlessSet = 0;
constexpr std::uint32_t kBindingCount = 4;
constexpr VkDescriptorBindingFlags kTableFlags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                                 VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

struct Table {
    std::uint32_t binding;
    VkDescriptorType type;
    std::uint32_t count;
};

// The four tables, in binding order, sized as desc asks.
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
    }};
}

// SPIR-V as the API takes it: whole words, aligned. The caller's bytes may be
// neither, so they are copied; pipeline creation is cold.
core::Result<std::vector<std::uint32_t>> wordsOf(std::span<const std::byte> code) {
    if (code.size() % sizeof(std::uint32_t) != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the shader code is {} bytes; SPIR-V is whole words",
                             code.size());
    }
    std::vector<std::uint32_t> words(code.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    return words;
}

// A shader module from code, which the caller destroys once its pipeline is
// made: a pipeline keeps what it needs of it.
core::Result<VkShaderModule> createModule(const VolkDeviceTable& table, VkDevice device,
                                          std::span<const std::byte> code) {
    auto words = wordsOf(code);
    if (!words) {
        return std::unexpected(std::move(words.error()));
    }
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = words->data();
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

VkPrimitiveTopology topologyFor(TopologyClass topology) {
    switch (topology) {
        case TopologyClass::Point:
            return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case TopologyClass::Line:
            return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case TopologyClass::Triangle:
        default:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

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

bool sameBlend(const VkPipelineColorBlendAttachmentState& a, const VkPipelineColorBlendAttachmentState& b) {
    return a.blendEnable == b.blendEnable && a.srcColorBlendFactor == b.srcColorBlendFactor &&
           a.dstColorBlendFactor == b.dstColorBlendFactor && a.colorBlendOp == b.colorBlendOp &&
           a.srcAlphaBlendFactor == b.srcAlphaBlendFactor && a.dstAlphaBlendFactor == b.dstAlphaBlendFactor &&
           a.alphaBlendOp == b.alphaBlendOp && a.colorWriteMask == b.colorWriteMask;
}

// Fails with RhiUnsupported when desc's state needs a feature the device lacks.
core::Result<void> checkFeatures(const RasterFeatures& features, const GraphicsPipelineDesc& desc,
                                 std::span<const VkPipelineColorBlendAttachmentState> blends) {
    if (desc.raster.fill == FillMode::Wireframe && !features.fillModeNonSolid) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the adapter has no wireframe fill");
    }
    if (desc.raster.depthClamp && !features.depthClamp) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the adapter has no depth clamp");
    }
    if (desc.raster.depthBiasClamp != 0 && !features.depthBiasClamp) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the adapter has no depth bias clamp");
    }
    if (!features.independentBlend) {
        for (const VkPipelineColorBlendAttachmentState& blend : blends) {
            if (!sameBlend(blend, blends.front())) {
                return core::failure(core::ErrorCode::RhiUnsupported,
                                     "the adapter blends every color target the same way");
            }
        }
    }
    return {};
}

} // namespace

core::Result<VulkanBindlessLayout> VulkanBindlessLayout::create(const VolkDeviceTable& table, VkDevice device,
                                                                const DeviceDesc& desc) {
    const std::array<Table, kBindingCount> tables = tablesFor(desc);
    std::array<VkDescriptorSetLayoutBinding, kBindingCount> bindings{};
    std::array<VkDescriptorBindingFlags, kBindingCount> flags{};
    std::array<VkDescriptorPoolSize, kBindingCount> sizes{};
    for (std::size_t i = 0; i < kBindingCount; ++i) {
        bindings.at(i) = {.binding = tables.at(i).binding,
                          .descriptorType = tables.at(i).type,
                          .descriptorCount = tables.at(i).count,
                          .stageFlags = VK_SHADER_STAGE_ALL,
                          .pImmutableSamplers = nullptr};
        flags.at(i) = kTableFlags;
        sizes.at(i) = {.type = tables.at(i).type, .descriptorCount = tables.at(i).count};
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{};
    bindingFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlags.bindingCount = kBindingCount;
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlags;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = kBindingCount;
    layoutInfo.pBindings = bindings.data();

    // The API says first whether it can hold a set this large; the per-type
    // limits were checked against the caps, but the whole set has its own.
    VkDescriptorSetLayoutSupport support{};
    support.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT;
    table.vkGetDescriptorSetLayoutSupport(device, &layoutInfo, &support);
    if (support.supported == VK_FALSE) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the adapter cannot hold bindless tables of {} textures, {} storage textures, {} "
                             "storage buffers and {} samplers together",
                             desc.maxBindlessTextures, desc.maxBindlessStorageTextures, desc.maxBindlessStorageBuffers,
                             desc.maxBindlessSamplers);
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
    poolInfo.poolSizeCount = kBindingCount;
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

    const VkPushConstantRange pushConstants{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = kPushConstantsSize};
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
                               VkPipelineBindPoint bindPoint,
                               std::array<std::uint32_t, kWorkgroupDimensions> workgroupSize)
    : table_(&table), device_(device), pipeline_(pipeline), bindPoint_(bindPoint), workgroupSize_(workgroupSize) {}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      pipeline_(std::exchange(other.pipeline_, VK_NULL_HANDLE)),
      bindPoint_(other.bindPoint_),
      workgroupSize_(other.workgroupSize_) {}

VulkanPipeline::~VulkanPipeline() {
    if (pipeline_ != VK_NULL_HANDLE) {
        table_->vkDestroyPipeline(device_, pipeline_, nullptr);
    }
}

core::Result<VulkanPipeline> createVulkanComputePipeline(const VolkDeviceTable& table, VkDevice device,
                                                         VkPipelineLayout layout, const ComputePipelineDesc& desc) {
    auto module = createModule(table, device, desc.compute.code);
    if (!module) {
        return std::unexpected(std::move(module.error()));
    }
    const std::string entryPoint{desc.compute.entryPoint};
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage = stageFor(VK_SHADER_STAGE_COMPUTE_BIT, *module, entryPoint.c_str());
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = table.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    table.vkDestroyShaderModule(device, *module, nullptr);
    if (result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateComputePipelines", .result = result});
    }
    return VulkanPipeline{table, device, pipeline, VK_PIPELINE_BIND_POINT_COMPUTE, desc.workgroupSize};
}

core::Result<VulkanPipeline> createVulkanGraphicsPipeline(const VolkDeviceTable& table, VkDevice device,
                                                          VkPipelineLayout layout, const RasterFeatures& features,
                                                          const GraphicsPipelineDesc& desc) {
    const std::span targets{desc.colorTargets.data(), desc.colorTargetCount};
    std::array<VkFormat, kMaxColorTargets> colorFormats{};
    std::array<VkPipelineColorBlendAttachmentState, kMaxColorTargets> blends{};
    for (std::size_t i = 0; i < targets.size(); ++i) {
        colorFormats.at(i) = vulkanFormat(targets[i].format);
        blends.at(i) = blendFor(targets[i].blend);
    }
    if (auto checked = checkFeatures(features, desc, std::span{blends.data(), targets.size()}); !checked) {
        return std::unexpected(std::move(checked.error()));
    }

    auto vertex = createModule(table, device, desc.vertex.code);
    if (!vertex) {
        return std::unexpected(std::move(vertex.error()));
    }
    VkShaderModule fragment = VK_NULL_HANDLE;
    if (!desc.fragment.code.empty()) {
        auto made = createModule(table, device, desc.fragment.code);
        if (!made) {
            table.vkDestroyShaderModule(device, *vertex, nullptr);
            return std::unexpected(std::move(made.error()));
        }
        fragment = *made;
    }
    const std::string vertexEntry{desc.vertex.entryPoint};
    const std::string fragmentEntry{desc.fragment.entryPoint};
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{stageFor(VK_SHADER_STAGE_VERTEX_BIT, *vertex,
                                                                   vertexEntry.c_str()),
                                                          stageFor(VK_SHADER_STAGE_FRAGMENT_BIT, fragment,
                                                                   fragmentEntry.c_str())};
    const std::uint32_t stageCount = fragment != VK_NULL_HANDLE ? 2 : 1;

    // Vertices are pulled, so there is no vertex input.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topologyFor(desc.topology);
    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
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
    constexpr std::array kDynamic{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                  VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                  VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(kDynamic.size());
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
    info.stageCount = stageCount;
    info.pStages = stages.data();
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
    const VkResult result = table.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    table.vkDestroyShaderModule(device, *vertex, nullptr);
    if (fragment != VK_NULL_HANDLE) {
        table.vkDestroyShaderModule(device, fragment, nullptr);
    }
    if (result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateGraphicsPipelines", .result = result});
    }
    return VulkanPipeline{table, device, pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS, {}};
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
    if (auto checked = checkComputePipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so pipelines the GPU is done with are freed here, never on the
    // frame path.
    collectBuffers();
    auto made = createVulkanComputePipeline(table_, device_, layout_.pipelineLayout(), desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines", kBufferLimit);
    }
    return pipeline;
}

core::Result<PipelineHandle> VulkanDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkGraphicsPipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    collectBuffers();
    auto made = createVulkanGraphicsPipeline(table_, device_, layout_.pipelineLayout(), rasterFeatures_, desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines", kBufferLimit);
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
    collectBuffers();
    return {};
}

} // namespace bfr::rhi::vulkan
