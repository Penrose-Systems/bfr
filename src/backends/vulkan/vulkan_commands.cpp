#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/command_check.h"
#include "rhi/enum_check.h"
#include "rhi/ray_tracing_check.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_pipeline.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

// The device's recording commands between beginRendering and a submit: the
// state draws and dispatches take, and the draws and dispatches. Each makes
// the usage checks every backend makes, then records one call.

core::Result<void> VulkanDevice::setViewport(CommandListHandle commandList, const Viewport& viewport) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    // Vulkan's framebuffer y points down and D3D12's and Metal's clip y points
    // up; a viewport of negative height, core since 1.1, makes clip space the
    // same here as there, winding included.
    const VkViewport flipped{.x = viewport.x,
                             .y = viewport.y + viewport.height,
                             .width = viewport.width,
                             .height = -viewport.height,
                             .minDepth = viewport.minDepth,
                             .maxDepth = viewport.maxDepth};
    table_.vkCmdSetViewport((*list)->buffer, 0, kViewports, &flipped);
    return {};
}

core::Result<void> VulkanDevice::setScissor(CommandListHandle commandList, const Rect& scissor) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const VkRect2D rect{.offset = {.x = static_cast<std::int32_t>(scissor.x),
                                   .y = static_cast<std::int32_t>(scissor.y)},
                        .extent = {.width = scissor.width, .height = scissor.height}};
    table_.vkCmdSetScissor((*list)->buffer, 0, kViewports, &rect);
    return {};
}

core::Result<void> VulkanDevice::setBlendConstants(CommandListHandle commandList, const std::array<float, 4>& color) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    table_.vkCmdSetBlendConstants((*list)->buffer, color.data());
    return {};
}

core::Result<void> VulkanDevice::setStencilReference(CommandListHandle commandList, std::uint32_t reference) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    table_.vkCmdSetStencilReference((*list)->buffer, VK_STENCIL_FACE_FRONT_AND_BACK, reference);
    return {};
}

core::Result<void> VulkanDevice::bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = pipelineFor(pipeline);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdBindPipeline((*list)->buffer, (*found)->bindPoint(), (*found)->pipeline());
    if ((*found)->bindPoint() == VK_PIPELINE_BIND_POINT_GRAPHICS && features_.dynamicTopology &&
        (*found)->topology() != VK_PRIMITIVE_TOPOLOGY_MAX_ENUM) {
        // Topology is dynamic, so a pipeline's own is set as it is bound; a
        // mesh pipeline has none.
        table_.vkCmdSetPrimitiveTopology((*list)->buffer, (*found)->topology());
    }
    // A ray tracing pipeline made with a stack size takes it as each list's
    // state, before each trace (VUID-vkCmdTraceRaysKHR-None-09458).
    if ((*found)->stackSize() != 0) {
        table_.vkCmdSetRayTracingPipelineStackSizeKHR((*list)->buffer,
                                                      static_cast<std::uint32_t>((*found)->stackSize()));
    }
    return {};
}

core::Result<void> VulkanDevice::setTopology(CommandListHandle commandList, Topology topology) {
    if (auto checked = checkKnown(topology, "topology"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (!features_.dynamicTopology) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device sets no topology: it has neither Vulkan 1.3 nor extendedDynamicState");
    }
    table_.vkCmdSetPrimitiveTopology((*list)->buffer, topologyFor(topology));
    return {};
}

core::Result<void> VulkanDevice::pushConstants(CommandListHandle commandList, std::span<const std::byte> data) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    table_.vkCmdPushConstants((*list)->buffer, layout_.pipelineLayout(), VK_SHADER_STAGE_ALL, 0,
                              static_cast<std::uint32_t>(data.size()), data.data());
    return {};
}

core::Result<void> VulkanDevice::bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
                                                 std::uint64_t offset, IndexType type) {
    if (auto checked = checkKnown(type, "index type"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdBindIndexBuffer((*list)->buffer, (*found)->buffer(), offset,
                                type == IndexType::Uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    return {};
}

core::Result<void> VulkanDevice::draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                      std::uint32_t instanceCount, std::uint32_t firstVertex,
                                      std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    table_.vkCmdDraw((*list)->buffer, vertexCount, instanceCount, firstVertex, firstInstance);
    return {};
}

core::Result<void> VulkanDevice::drawIndexed(CommandListHandle commandList, std::uint32_t indexCount,
                                             std::uint32_t instanceCount, std::uint32_t firstIndex,
                                             std::int32_t vertexOffset, std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    table_.vkCmdDrawIndexed((*list)->buffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    return {};
}

core::Result<void> VulkanDevice::dispatch(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                          std::uint32_t z) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    table_.vkCmdDispatch((*list)->buffer, x, y, z);
    return {};
}

core::Result<void> VulkanDevice::drawIndirect(CommandListHandle commandList, BufferHandle buffer, std::uint64_t offset,
                                              std::uint32_t drawCount, std::uint32_t stride) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdDrawIndirect((*list)->buffer, (*found)->buffer(), offset, drawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::drawIndexedIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                     std::uint64_t offset, std::uint32_t drawCount,
                                                     std::uint32_t stride) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdDrawIndexedIndirect((*list)->buffer, (*found)->buffer(), offset, drawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::drawIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                   std::uint64_t offset, BufferHandle countBuffer,
                                                   std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                   std::uint32_t stride) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    auto count = bufferFor(countBuffer);
    if (!count) {
        return std::unexpected(std::move(count.error()));
    }
    table_.vkCmdDrawIndirectCount((*list)->buffer, (*found)->buffer(), offset, (*count)->buffer(), countOffset,
                                  maxDrawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::drawIndexedIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                          std::uint64_t offset, BufferHandle countBuffer,
                                                          std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                          std::uint32_t stride) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    auto count = bufferFor(countBuffer);
    if (!count) {
        return std::unexpected(std::move(count.error()));
    }
    table_.vkCmdDrawIndexedIndirectCount((*list)->buffer, (*found)->buffer(), offset, (*count)->buffer(), countOffset,
                                         maxDrawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::dispatchIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                  std::uint64_t offset) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdDispatchIndirect((*list)->buffer, (*found)->buffer(), offset);
    return {};
}

core::Result<void> VulkanDevice::drawMesh(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                          std::uint32_t z) {
    if (auto checked = checkMeshDraws(features_.meshShader); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    table_.vkCmdDrawMeshTasksEXT((*list)->buffer, x, y, z);
    return {};
}

core::Result<void> VulkanDevice::drawMeshIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                  std::uint64_t offset, std::uint32_t drawCount, std::uint32_t stride) {
    if (auto checked = checkMeshDraws(features_.meshShader); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdDrawMeshTasksIndirectEXT((*list)->buffer, (*found)->buffer(), offset, drawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::drawMeshIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                       std::uint64_t offset, BufferHandle countBuffer,
                                                       std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                       std::uint32_t stride) {
    if (auto checked = checkMeshDraws(features_.meshShader); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    auto count = bufferFor(countBuffer);
    if (!count) {
        return std::unexpected(std::move(count.error()));
    }
    table_.vkCmdDrawMeshTasksIndirectCountEXT((*list)->buffer, (*found)->buffer(), offset, (*count)->buffer(),
                                              countOffset, maxDrawCount, stride);
    return {};
}

core::Result<VkStridedDeviceAddressRegionKHR> VulkanDevice::regionOf(const ShaderTableRegion& region) {
    if (region.buffer.isNull()) {
        return VkStridedDeviceAddressRegionKHR{};
    }
    const auto address = addressOf(region.buffer, region.offset);
    if (!address) {
        return std::unexpected(address.error());
    }
    return VkStridedDeviceAddressRegionKHR{.deviceAddress = *address, .stride = region.stride, .size = region.size};
}

core::Result<void> VulkanDevice::traceRays(CommandListHandle commandList, const TraceRaysDesc& desc) {
    if (auto checked = checkRayTracingPipelines(features_.rayTracingPipeline); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    std::array<VkStridedDeviceAddressRegionKHR, 4> regions{};
    const std::array tables{&desc.rayGeneration, &desc.miss, &desc.hit, &desc.callable};
    for (std::size_t i = 0; i < regions.size(); ++i) {
        const auto region = regionOf(*tables.at(i));
        if (!region) {
            return std::unexpected(region.error());
        }
        regions.at(i) = *region;
    }
    table_.vkCmdTraceRaysKHR((*list)->buffer, &regions[0], &regions[1], &regions[2], &regions[3], desc.width,
                             desc.height, desc.depth);
    return {};
}

core::Result<void> VulkanDevice::traceRaysIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                   std::uint64_t offset) {
    if (auto checked = checkTraceRaysIndirect(features_.traceRaysIndirect); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const auto address = addressOf(buffer, offset);
    if (!address) {
        return std::unexpected(address.error());
    }
    // The 2 form reads the regions with the dimensions, as
    // TraceRaysIndirectArgs holds them.
    static_assert(sizeof(TraceRaysIndirectArgs) == sizeof(VkTraceRaysIndirectCommand2KHR));
    table_.vkCmdTraceRaysIndirect2KHR((*list)->buffer, *address);
    return {};
}

} // namespace bfr::rhi::vulkan
