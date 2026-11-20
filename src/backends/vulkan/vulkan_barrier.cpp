#include "bfr/core/error.h"
#include "bfr/rhi/barrier.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "rhi/barrier_check.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_texture.h"

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

struct StageBit {
    bfr_bool Stages::* stage;
    VkPipelineStageFlags2 native;
};

// Each BFR stage's one Vulkan stage. acceleration_structure_copy and
// acceleration_structure_properties are the build stage: the spec lets a copy
// or a property write be synchronized with it or with ACCELERATION_STRUCTURE_COPY
// (accelstructures.adoc, vkCmdCopyAccelerationStructureKHR and its kin,
// vkCmdWriteAccelerationStructuresPropertiesKHR), and the copy stage needs
// rayTracingMaintenance1 (VUID-VkMemoryBarrier2-srcStageMask-10752), where the
// build stage needs only the acceleration structures the commands need anyway.
constexpr std::array kStageBits{
    StageBit{.stage = &Stages::all, .native = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
    StageBit{.stage = &Stages::drawIndirect, .native = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT},
    StageBit{.stage = &Stages::indexInput, .native = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT},
    StageBit{.stage = &Stages::vertexShader, .native = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT},
    StageBit{.stage = &Stages::taskShader, .native = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT},
    StageBit{.stage = &Stages::meshShader, .native = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT},
    StageBit{.stage = &Stages::fragmentShader, .native = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT},
    StageBit{.stage = &Stages::earlyFragmentTests, .native = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT},
    StageBit{.stage = &Stages::lateFragmentTests, .native = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
    StageBit{.stage = &Stages::colorOutput, .native = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
    StageBit{.stage = &Stages::computeShader, .native = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    StageBit{.stage = &Stages::copy, .native = VK_PIPELINE_STAGE_2_COPY_BIT},
    StageBit{.stage = &Stages::resolve, .native = VK_PIPELINE_STAGE_2_RESOLVE_BIT},
    StageBit{.stage = &Stages::rayTracingShader, .native = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR},
    StageBit{.stage = &Stages::accelerationStructureBuild,
             .native = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR},
    StageBit{.stage = &Stages::accelerationStructureCopy,
             .native = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR},
    StageBit{.stage = &Stages::accelerationStructureProperties,
             .native = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR},
};

struct AccessBit {
    bfr_bool Accesses::* access;
    VkAccessFlags2 native;
};

// Each BFR access's one Vulkan access. Resolves are transfer accesses, which
// the resolve stage pairs with (VUID-VkMemoryBarrier2-srcAccessMask-03914,
// -03915). A build's inputs are SHADER_READ at the build stage
// (accelstructures.adoc, "synchronized with the
// VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and an
// access type of VK_ACCESS_SHADER_READ_BIT"). Shader binding table reads are
// SHADER_READ too, which holds them (vk.xml:32913-32915) and needs none of
// the rayTracingMaintenance1 that SHADER_BINDING_TABLE_READ comes with
// (vk.xml:26355, 26370).
constexpr std::array kAccessBits{
    AccessBit{.access = &Accesses::indirectRead, .native = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT},
    AccessBit{.access = &Accesses::indexRead, .native = VK_ACCESS_2_INDEX_READ_BIT},
    AccessBit{.access = &Accesses::uniformRead, .native = VK_ACCESS_2_UNIFORM_READ_BIT},
    AccessBit{.access = &Accesses::sampledRead, .native = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
    AccessBit{.access = &Accesses::storageRead, .native = VK_ACCESS_2_SHADER_STORAGE_READ_BIT},
    AccessBit{.access = &Accesses::storageWrite, .native = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
    AccessBit{.access = &Accesses::colorTargetRead, .native = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT},
    AccessBit{.access = &Accesses::colorTargetWrite, .native = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT},
    AccessBit{.access = &Accesses::depthStencilRead, .native = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT},
    AccessBit{.access = &Accesses::depthStencilWrite, .native = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
    AccessBit{.access = &Accesses::copyRead, .native = VK_ACCESS_2_TRANSFER_READ_BIT},
    AccessBit{.access = &Accesses::copyWrite, .native = VK_ACCESS_2_TRANSFER_WRITE_BIT},
    AccessBit{.access = &Accesses::resolveRead, .native = VK_ACCESS_2_TRANSFER_READ_BIT},
    AccessBit{.access = &Accesses::resolveWrite, .native = VK_ACCESS_2_TRANSFER_WRITE_BIT},
    AccessBit{.access = &Accesses::accelerationStructureInputRead, .native = VK_ACCESS_2_SHADER_READ_BIT},
    AccessBit{.access = &Accesses::accelerationStructureRead,
              .native = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR},
    AccessBit{.access = &Accesses::accelerationStructureWrite,
              .native = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR},
    AccessBit{.access = &Accesses::shaderBindingTableRead, .native = VK_ACCESS_2_SHADER_READ_BIT},
};

VkPipelineStageFlags2 stagesFor(const Stages& stages) {
    VkPipelineStageFlags2 native = VK_PIPELINE_STAGE_2_NONE;
    for (const auto& [stage, bit] : kStageBits) {
        if (set(stages.*stage)) {
            native |= bit;
        }
    }
    return native;
}

VkAccessFlags2 accessesFor(const Accesses& accesses) {
    VkAccessFlags2 native = VK_ACCESS_2_NONE;
    for (const auto& [access, bit] : kAccessBits) {
        if (set(accesses.*access)) {
            native |= bit;
        }
    }
    return native;
}

struct NativeScope {
    VkPipelineStageFlags2 stages;
    VkAccessFlags2 accesses;
};

// scope as recorded. Where the device has opacity micromaps, the acceleration
// structure stages and accesses also cover them: micromap builds, copies and
// property writes run in MICROMAP_BUILD with MICROMAP_READ and MICROMAP_WRITE
// (micromaps.adoc, vkCmdBuildMicromapsEXT, vkCmdCopyMicromapEXT and its kin,
// vkCmdWriteMicromapsPropertiesEXT), so the AS stages add MICROMAP_BUILD, and
// the AS accesses their micromap twins on a side whose stages take them:
// MICROMAP_BUILD, or ALL_COMMANDS, "all operations performed by all commands
// supported on the queue" (synchronization.adoc), on a family that runs
// compute, where MICROMAP_BUILD runs (vk.xml:32881-32883)
// (VUID-VkMemoryBarrier2-srcAccessMask-07457, -07458). The micromap stage
// needs the micromap feature (VUID-VkMemoryBarrier2-srcStageMask-10753),
// which Features::opacityMicromap enables.
NativeScope nativeScope(const BarrierScope& scope, bool micromaps, const QueueFamily& family) {
    NativeScope native{.stages = stagesFor(scope.stages), .accesses = accessesFor(scope.accesses)};
    if (!micromaps) {
        return native;
    }
    if ((native.stages & VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR) != 0) {
        native.stages |= VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
    }
    const bool micromapStages = (native.stages & VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT) != 0 ||
                                ((native.stages & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) != 0 && family.compute);
    if (!micromapStages) {
        return native;
    }
    if ((native.accesses & VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR) != 0) {
        native.accesses |= VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT;
    }
    if ((native.accesses & VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR) != 0) {
        native.accesses |= VK_ACCESS_2_MICROMAP_READ_BIT_EXT;
    }
    return native;
}

// The VK_PIPELINE_STAGE_*_SHADER_BIT stages BFR records, which the shader
// accesses pair with.
constexpr VkPipelineStageFlags2 kShaderStages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                                VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

// Stages any access pairs with.
constexpr VkPipelineStageFlags2 kAnyAccess = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

struct AccessPairing {
    VkAccessFlags2 access;
    VkPipelineStageFlags2 stages;
    std::string_view vuid;
};

// The stages each access BFR records needs one of, as
// VUID-VkMemoryBarrier2-srcAccessMask-* and -dstAccessMask-* of the same
// number, and their VkImageMemoryBarrier2 kin, list them: of each list, the
// stages BFR records.
constexpr std::array kAccessPairings{
    AccessPairing{.access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                  .stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | kAnyAccess,
                  .vuid = "03900"},
    AccessPairing{.access = VK_ACCESS_2_INDEX_READ_BIT,
                  .stages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | kAnyAccess,
                  .vuid = "03901"},
    AccessPairing{.access = VK_ACCESS_2_UNIFORM_READ_BIT, .stages = kShaderStages | kAnyAccess, .vuid = "03904"},
    AccessPairing{.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, .stages = kShaderStages | kAnyAccess, .vuid = "03905"},
    AccessPairing{.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT, .stages = kShaderStages | kAnyAccess, .vuid = "03906"},
    AccessPairing{.access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  .stages = kShaderStages | kAnyAccess,
                  .vuid = "03907"},
    AccessPairing{.access = VK_ACCESS_2_SHADER_READ_BIT,
                  .stages = kShaderStages | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | kAnyAccess,
                  .vuid = "07454"},
    AccessPairing{.access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                  .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | kAnyAccess,
                  .vuid = "03910"},
    AccessPairing{.access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | kAnyAccess,
                  .vuid = "03911"},
    AccessPairing{.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                  .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                            kAnyAccess,
                  .vuid = "03912"},
    AccessPairing{.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                            kAnyAccess,
                  .vuid = "03913"},
    AccessPairing{.access = VK_ACCESS_2_TRANSFER_READ_BIT,
                  .stages = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .vuid = "03914"},
    AccessPairing{.access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  .stages = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .vuid = "03915"},
    AccessPairing{.access = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                  .stages = kShaderStages | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .vuid = "03927"},
    AccessPairing{.access = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                  .stages = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .vuid = "03928"},
};

struct StageNeed {
    VkPipelineStageFlags2 stage;
    bool Features::* feature;
    std::string_view vuid;
};

// The stages a feature gates, as VUID-VkMemoryBarrier2-srcStageMask-* and
// -dstStageMask-* of the same number say.
constexpr std::array kStageFeatures{
    StageNeed{.stage = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, .feature = &Features::meshShader, .vuid = "03934"},
    StageNeed{.stage = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, .feature = &Features::taskShader, .vuid = "03935"},
    StageNeed{.stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
              .feature = &Features::rayTracingPipeline,
              .vuid = "07946"},
    StageNeed{.stage = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
              .feature = &Features::accelerationStructure,
              .vuid = "10751"},
};

// The stages a queue family runs, as vk.xml's <syncsupport> lists them for
// each (vk.xml:32763-32870): graphics alone, compute alone, or either, and the
// copy and resolve stages on a family that copies. ALL_COMMANDS takes any.
constexpr VkPipelineStageFlags2 kGraphicsStages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                                                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                                  VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                                  VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
constexpr VkPipelineStageFlags2 kComputeStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                 VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
constexpr VkPipelineStageFlags2 kGraphicsOrComputeStages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
constexpr VkPipelineStageFlags2 kCopyStages = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT;

VkPipelineStageFlags2 familyStages(const QueueFamily& family) {
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    if (family.graphics) {
        stages |= kGraphicsStages;
    }
    if (family.compute) {
        stages |= kComputeStages;
    }
    if (family.graphics || family.compute) {
        stages |= kGraphicsOrComputeStages;
    }
    if (family.copy) {
        stages |= kCopyStages;
    }
    return stages;
}

} // namespace

core::Result<void> VulkanDevice::checkBarrierScope(const BarrierScope& scope, const QueueFamily& family) const {
    const VkPipelineStageFlags2 stages = stagesFor(scope.stages);
    const VkAccessFlags2 accesses = accessesFor(scope.accesses);
    for (const StageNeed& need : kStageFeatures) {
        if ((stages & need.stage) != 0 && !(features_.*need.feature)) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 "a barrier's stages need a feature the device was created without "
                                 "(VUID-VkMemoryBarrier2-srcStageMask-{})",
                                 need.vuid);
        }
    }
    if ((stages & ~familyStages(family)) != 0) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "a barrier's stages are not ones the list's queue family runs "
                             "(VUID-vkCmdPipelineBarrier2-srcStageMask-09673)");
    }
    // The acceleration structure accesses are VK_KHR_acceleration_structure's
    // (vk.xml:25397-25400). A flag is valid only where the extension that
    // added it is enabled for the device (fundamentals.adoc, "Valid Usage for
    // Enumerated Types" and "Valid Usage for Flags";
    // VUID-VkMemoryBarrier2-srcAccessMask-parameter), and BFR enables that
    // extension only with Features::accelerationStructure
    // (vulkan_features.cpp).
    constexpr VkAccessFlags2 kStructureAccesses = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    if ((accesses & kStructureAccesses) != 0 && !features_.accelerationStructure) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "a barrier's acceleration structure accesses need the accelerationStructure feature "
                             "(VUID-VkMemoryBarrier2-srcAccessMask-parameter, fundamentals.adoc \"Valid Usage "
                             "for Flags\")");
    }
    if ((accesses & VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR) != 0 && !features_.rayQuery &&
        (stages & kShaderStages & ~VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) != 0) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "a barrier's acceleration structure reads in a shader stage other than ray tracing "
                             "need the rayQuery feature (VUID-VkMemoryBarrier2-srcAccessMask-06256)");
    }
    for (const AccessPairing& pairing : kAccessPairings) {
        if ((accesses & pairing.access) != 0 && (stages & pairing.stages) == 0) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "a barrier's access has none of the stages it pairs with "
                                 "(VUID-VkMemoryBarrier2-srcAccessMask-{})",
                                 pairing.vuid);
        }
    }
    return {};
}

// Every barrier is checked, its texture looked up, before any is recorded,
// then they are recorded kBarrierChunk of each kind to a vkCmdPipelineBarrier2.
// Families are ignored: a resource shared by more than one family is
// concurrent (createVulkanTexture, createVulkanBuffer), so none changes owner.
core::Result<void> VulkanDevice::barrier(CommandListHandle commandList, const BarrierDesc& desc) {
    if (auto checked = checkBarrierLayouts(desc); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto queue = queueOf(**list);
    if (!queue) {
        return std::unexpected(std::move(queue.error()));
    }
    const QueueFamily& family = queueFamilies_[(*queue)->family];
    for (const GlobalBarrier& global : desc.globals) {
        for (const BarrierScope* scope : {&global.before, &global.after}) {
            if (auto checked = checkBarrierScope(*scope, family); !checked) {
                return checked;
            }
        }
    }
    for (const TextureBarrier& moved : desc.textures) {
        for (const BarrierScope* scope : {&moved.before, &moved.after}) {
            if (auto checked = checkBarrierScope(*scope, family); !checked) {
                return checked;
            }
        }
        if (moved.layoutAfter == TextureLayout::Undefined) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "a texture barrier moves to layout undefined, which Vulkan takes only as a layout to "
                                 "move from (VUID-VkImageMemoryBarrier2-newLayout-01198)");
        }
        auto found = textureFor(moved.texture);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        if (auto checked = checkTextureRange(moved.range, (*found)->mipLevels(), (*found)->layers()); !checked) {
            return checked;
        }
    }
    // kRemaining is Vulkan's VK_REMAINING_MIP_LEVELS and _ARRAY_LAYERS.
    static_assert(kRemaining == VK_REMAINING_MIP_LEVELS);
    static_assert(kRemaining == VK_REMAINING_ARRAY_LAYERS);
    const bool micromaps = features_.opacityMicromap;
    std::array<VkMemoryBarrier2, kBarrierChunk> globals{};
    std::array<VkImageMemoryBarrier2, kBarrierChunk> images{};
    for (std::size_t chunk = 0; chunk < barrierChunks(desc); ++chunk) {
        const std::span chunkGlobals = barrierChunk(desc.globals, chunk);
        const std::span chunkTextures = barrierChunk(desc.textures, chunk);
        for (std::size_t i = 0; i < chunkGlobals.size(); ++i) {
            const GlobalBarrier& global = chunkGlobals[i];
            const NativeScope before = nativeScope(global.before, micromaps, family);
            const NativeScope after = nativeScope(global.after, micromaps, family);
            globals.at(i) = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                             .pNext = nullptr,
                             .srcStageMask = before.stages,
                             .srcAccessMask = before.accesses,
                             .dstStageMask = after.stages,
                             .dstAccessMask = after.accesses};
        }
        for (std::size_t i = 0; i < chunkTextures.size(); ++i) {
            const TextureBarrier& moved = chunkTextures[i];
            // Looked up above, in this same call.
            const VulkanTexture& texture = **textureFor(moved.texture);
            const NativeScope before = nativeScope(moved.before, micromaps, family);
            const NativeScope after = nativeScope(moved.after, micromaps, family);
            images.at(i) = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                            .pNext = nullptr,
                            .srcStageMask = before.stages,
                            .srcAccessMask = before.accesses,
                            .dstStageMask = after.stages,
                            .dstAccessMask = after.accesses,
                            .oldLayout = layoutFor(moved.layoutBefore, texture.aspects()),
                            .newLayout = layoutFor(moved.layoutAfter, texture.aspects()),
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = texture.image(),
                            .subresourceRange = {.aspectMask = texture.aspects(),
                                                 .baseMipLevel = moved.range.mipLevel,
                                                 .levelCount = moved.range.mipCount,
                                                 .baseArrayLayer = moved.range.layer,
                                                 .layerCount = moved.range.layerCount}};
        }
        const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                          .pNext = nullptr,
                                          .dependencyFlags = 0,
                                          .memoryBarrierCount = static_cast<std::uint32_t>(chunkGlobals.size()),
                                          .pMemoryBarriers = globals.data(),
                                          .bufferMemoryBarrierCount = 0,
                                          .pBufferMemoryBarriers = nullptr,
                                          .imageMemoryBarrierCount = static_cast<std::uint32_t>(chunkTextures.size()),
                                          .pImageMemoryBarriers = images.data()};
        table_.vkCmdPipelineBarrier2((*list)->buffer, &dependency);
    }
    return {};
}

} // namespace bfr::rhi::vulkan
