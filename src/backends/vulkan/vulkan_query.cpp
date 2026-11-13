#include "vulkan/vulkan_query.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/query.h"
#include "rhi/enum_check.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

// Each statistic's bit, in PipelineStatistic's order, which is Vulkan's.
constexpr std::array<VkQueryPipelineStatisticFlagBits, kPipelineStatistics> kStatisticBits{
    VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT,
    VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT,
    VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT,
    VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT,
    VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT,
    VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT,
    VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT,
    VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
    VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT,
    VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT,
    VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT,
    VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT,
    VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT,
};

// mask's statistics in PipelineStatistic's order.
std::array<bool, kPipelineStatistics> selected(const PipelineStatisticsMask& mask) {
    return {mask.inputVertices,       mask.inputPrimitives,
            mask.vertexInvocations,   mask.geometryInvocations,
            mask.geometryPrimitives,  mask.clipInvocations,
            mask.clipPrimitives,      mask.fragmentInvocations,
            mask.tessellationControl, mask.tessellationEvaluationInvocations,
            mask.computeInvocations,  mask.taskInvocations,
            mask.meshInvocations};
}

VkQueryType queryTypeFor(QueryType type) {
    switch (type) {
        case QueryType::PipelineStatistics:
            return VK_QUERY_TYPE_PIPELINE_STATISTICS;
        case QueryType::MeshPrimitivesGenerated:
            return VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT;
        case QueryType::Occlusion:
        default:
            return VK_QUERY_TYPE_OCCLUSION;
    }
}

constexpr std::uint32_t kValueBytes = sizeof(std::uint64_t);

// Where vkCmdCopyQueryPoolResults with VK_QUERY_RESULT_64_BIT puts a query's
// results: one 64-bit value, or one for each statistic the pool counts, in
// bit order, packed.
QueryLayout layoutFor(const QueryPoolDesc& desc) {
    QueryLayout layout{};
    layout.statistics.fill(kNoOffset);
    if (desc.type != QueryType::PipelineStatistics) {
        layout.stride = kValueBytes;
        layout.value = 0;
        return layout;
    }
    const auto chosen = selected(desc.statistics);
    std::uint32_t offset = 0;
    for (std::size_t i = 0; i < kPipelineStatistics; ++i) {
        if (chosen.at(i)) {
            layout.statistics.at(i) = offset;
            offset += kValueBytes;
        }
    }
    layout.stride = offset;
    return layout;
}

} // namespace

VulkanQueryPool::VulkanQueryPool(const VolkDeviceTable& table, VkDevice device, VkQueryPool pool,
                                 const QueryLayout& layout)
    : table_(&table), device_(device), pool_(pool), layout_(layout) {}

VulkanQueryPool::VulkanQueryPool(VulkanQueryPool&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      pool_(std::exchange(other.pool_, VK_NULL_HANDLE)),
      layout_(other.layout_) {}

VulkanQueryPool::~VulkanQueryPool() {
    if (pool_ != VK_NULL_HANDLE) {
        table_->vkDestroyQueryPool(device_, pool_, nullptr);
    }
}

// The device's query calls, beside the pools they make. Each passes its
// values to the one Vulkan call that does what it says; Vulkan's validation
// reports misuse (VUID-VkQueryPoolCreateInfo-queryType-00791, -meshShaderQueries-07068,
// VUID-vkCmdBeginQuery-queryType-00800, VUID-vkCmdResetQueryPool-renderpass).

core::Result<QueryPoolHandle> VulkanDevice::createQueryPool(const QueryPoolDesc& desc) {
    if (auto checked = checkKnown(desc.type, "query type"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = queryTypeFor(desc.type);
    info.queryCount = desc.count;
    if (desc.type == QueryType::PipelineStatistics) {
        const auto chosen = selected(desc.statistics);
        for (std::size_t i = 0; i < kPipelineStatistics; ++i) {
            if (chosen.at(i)) {
                info.pipelineStatistics |= static_cast<VkQueryPipelineStatisticFlags>(kStatisticBits.at(i));
            }
        }
    }
    VkQueryPool pool = VK_NULL_HANDLE;
    if (const VkResult result = table_.vkCreateQueryPool(device_, &info, nullptr, &pool); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateQueryPool", .result = result});
    }
    const QueryPoolHandle made = queryPools_.emplace(VulkanQueryPool{table_, device_, pool, layoutFor(desc)});
    if (made.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} query pools",
                             desc_.maxQueryPools);
    }
    return made;
}

core::Result<void> VulkanDevice::destroyQueryPool(QueryPoolHandle pool) {
    if (pool.isNull()) {
        return {};
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!queryPools_.retire(pool, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
    }
    collectRetired();
    return {};
}

core::Result<const VulkanQueryPool*> VulkanDevice::queryPoolFor(QueryPoolHandle pool) {
    const VulkanQueryPool* found = queryPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
    }
    return found;
}

core::Result<QueryLayout> VulkanDevice::queryLayout(QueryPoolHandle pool) {
    auto found = queryPoolFor(pool);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->layout();
}

core::Result<void> VulkanDevice::resetQueries(CommandListHandle commandList, QueryPoolHandle pool, std::uint32_t first,
                                              std::uint32_t count) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = queryPoolFor(pool);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdResetQueryPool((*list)->buffer, (*found)->pool(), first, count);
    return {};
}

core::Result<void> VulkanDevice::beginQuery(CommandListHandle commandList, QueryPoolHandle pool, std::uint32_t index,
                                            bool precise) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = queryPoolFor(pool);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdBeginQuery((*list)->buffer, (*found)->pool(), index,
                           precise ? VkQueryControlFlags{VK_QUERY_CONTROL_PRECISE_BIT} : 0U);
    return {};
}

core::Result<void> VulkanDevice::endQuery(CommandListHandle commandList, QueryPoolHandle pool, std::uint32_t index) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = queryPoolFor(pool);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    table_.vkCmdEndQuery((*list)->buffer, (*found)->pool(), index);
    return {};
}

core::Result<void> VulkanDevice::resolveQueries(CommandListHandle commandList, QueryPoolHandle pool,
                                                std::uint32_t first, std::uint32_t count, BufferHandle buffer,
                                                std::uint64_t offset) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = queryPoolFor(pool);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    auto target = bufferFor(buffer);
    if (!target) {
        return std::unexpected(std::move(target.error()));
    }
    // Waiting for each query's results, as resolveTimestamps does, as 64-bit
    // values a layout's stride apart.
    table_.vkCmdCopyQueryPoolResults((*list)->buffer, (*found)->pool(), first, count, (*target)->buffer(), offset,
                                     (*found)->layout().stride, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return {};
}

} // namespace bfr::rhi::vulkan
