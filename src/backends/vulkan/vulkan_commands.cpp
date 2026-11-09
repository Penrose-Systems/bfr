#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/command_check.h"
#include "rhi/enum_check.h"
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
    // VkRect2D's offset is int32_t.
    if (!std::in_range<std::int32_t>(scissor.x) || !std::in_range<std::int32_t>(scissor.y)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "scissor offset {}, {} is past what Vulkan takes",
                             scissor.x, scissor.y);
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
    if ((*found)->bindPoint() == VK_PIPELINE_BIND_POINT_GRAPHICS && features_.dynamicTopology) {
        // Topology is dynamic, so a pipeline's own is set as it is bound.
        table_.vkCmdSetPrimitiveTopology((*list)->buffer, (*found)->topology());
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
    if (auto checked = checkPushConstants(data, pushConstantsSize_); !checked) {
        return checked;
    }
    VulkanCommandPool* pool = commandPools_.find((*list)->pool);
    if (pool == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    if (auto counted = countPushConstants(pool->pushed, pool->pushConstantUpdates); !counted) {
        return counted;
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
    if (auto checked = checkDrawCount(drawCount, multiDrawIndirect_); !checked) {
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
    table_.vkCmdDrawIndirect((*list)->buffer, (*found)->buffer(), offset, drawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::drawIndexedIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                     std::uint64_t offset, std::uint32_t drawCount,
                                                     std::uint32_t stride) {
    if (auto checked = checkDrawCount(drawCount, multiDrawIndirect_); !checked) {
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
    table_.vkCmdDrawIndexedIndirect((*list)->buffer, (*found)->buffer(), offset, drawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::drawIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                   std::uint64_t offset, BufferHandle countBuffer,
                                                   std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                   std::uint32_t stride) {
    if (auto checked = checkCountDraws(drawIndirectCount_); !checked) {
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
    table_.vkCmdDrawIndirectCount((*list)->buffer, (*found)->buffer(), offset, (*count)->buffer(), countOffset,
                                  maxDrawCount, stride);
    return {};
}

core::Result<void> VulkanDevice::drawIndexedIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                          std::uint64_t offset, BufferHandle countBuffer,
                                                          std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                          std::uint32_t stride) {
    if (auto checked = checkCountDraws(drawIndirectCount_); !checked) {
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

} // namespace bfr::rhi::vulkan
