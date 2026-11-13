#include "vulkan/vulkan_timestamp.h"

#include "bfr/core/error.h"
#include "bfr/rhi/timestamp.h"
#include "rhi/command_check.h"
#include "rhi/enum_check.h"
#include "rhi/name_check.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

VkPipelineStageFlags2 stageFor(TimestampStage stage) {
    switch (stage) {
        case TimestampStage::Start:
            return VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        case TimestampStage::Vertex:
            return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        case TimestampStage::Fragment:
            return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case TimestampStage::Compute:
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case TimestampStage::Copy:
            return VK_PIPELINE_STAGE_2_COPY_BIT;
        case TimestampStage::Task:
            return VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
        case TimestampStage::Mesh:
            return VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        case TimestampStage::RayTracing:
            return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        // Builds and copies both (synchronization.adoc:428-431).
        case TimestampStage::AccelerationStructureBuild:
            return VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        case TimestampStage::End:
        default:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
}

// The time domain clock is, and whether the device reads it.
std::pair<VkTimeDomainKHR, bool> domainFor(HostClock clock, const HostClocks& clocks) {
    switch (clock) {
        case HostClock::MonotonicRaw:
            return {VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR, clocks.monotonicRaw};
        case HostClock::PerformanceCounter:
            return {VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR, clocks.performanceCounter};
        case HostClock::Monotonic:
        default:
            return {VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR, clocks.monotonic};
    }
}

} // namespace

VulkanTimestampPool::VulkanTimestampPool(const VolkDeviceTable& table, VkDevice device, VkQueryPool pool)
    : table_(&table), device_(device), pool_(pool) {}

VulkanTimestampPool::VulkanTimestampPool(VulkanTimestampPool&& other) noexcept
    : table_(other.table_), device_(other.device_), pool_(std::exchange(other.pool_, VK_NULL_HANDLE)) {}

VulkanTimestampPool::~VulkanTimestampPool() {
    if (pool_ != VK_NULL_HANDLE) {
        table_->vkDestroyQueryPool(device_, pool_, nullptr);
    }
}

// The device's timestamp calls, beside the pools they make. Each passes its
// values to the one Vulkan call that does what it says.

core::Result<TimestampPoolHandle> VulkanDevice::createTimestampPool(const TimestampPoolDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkObjectName(features_, desc.name); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = desc.count;
    VkQueryPool pool = VK_NULL_HANDLE;
    if (const VkResult result = table_.vkCreateQueryPool(device_, &info, nullptr, &pool); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateQueryPool", .result = result});
    }
    VulkanTimestampPool held{table_, device_, pool};
    if (auto named = nameObject(VK_OBJECT_TYPE_QUERY_POOL, pool, desc.name); !named) {
        return std::unexpected(std::move(named.error()));
    }
    const TimestampPoolHandle made = timestampPools_.emplace(std::move(held));
    if (made.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} timestamp pools",
                             desc_.maxTimestampPools);
    }
    return made;
}

core::Result<void> VulkanDevice::destroyTimestampPool(TimestampPoolHandle pool) {
    if (pool.isNull()) {
        return {};
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!timestampPools_.retire(pool, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    collectRetired();
    return {};
}

core::Result<QueueType> VulkanDevice::queueTypeOf(const VulkanCommandList& list) {
    const VulkanCommandPool* commands = commandPools_.find(list.pool);
    if (commands == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    return queues_[commands->queue]->type;
}

core::Result<void> VulkanDevice::resetTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                 std::uint32_t first, std::uint32_t count) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const auto type = queueTypeOf(**list);
    if (!type) {
        return std::unexpected(std::move(type.error()));
    }
    const VulkanTimestampPool* found = timestampPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    table_.vkCmdResetQueryPool((*list)->buffer, found->pool(), first, count);
    return {};
}

core::Result<void> VulkanDevice::writeTimestamp(CommandListHandle commandList, TimestampPoolHandle pool,
                                                std::uint32_t index, TimestampStage stage,
                                                TimestampGranularity granularity) {
    if (auto checked = checkKnown(stage, "timestamp stage"); !checked) {
        return checked;
    }
    // Vulkan takes no granularity, as the caps say, so it is only checked.
    if (auto checked = checkKnown(granularity, "timestamp granularity"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    // Each kind of queue takes the stages its caps list (Caps::timestamps).
    const auto type = queueTypeOf(**list);
    if (!type) {
        return std::unexpected(std::move(type.error()));
    }
    const VulkanTimestampPool* found = timestampPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    table_.vkCmdWriteTimestamp2((*list)->buffer, stageFor(stage), found->pool(), index);
    return {};
}

core::Result<void> VulkanDevice::resolveTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                   std::uint32_t first, std::uint32_t count, BufferHandle buffer,
                                                   std::uint64_t offset) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const VulkanTimestampPool* found = timestampPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    auto target = bufferFor(buffer);
    if (!target) {
        return std::unexpected(std::move(target.error()));
    }
    // Waiting for each timestamp to be written, as D3D12's ResolveQueryData
    // and Metal's resolveCounterHeap do, and as 64-bit values, the caps'
    // entryBytes apart.
    table_.vkCmdCopyQueryPoolResults((*list)->buffer, found->pool(), first, count, (*target)->buffer(), offset,
                                     sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return {};
}

core::Result<TimestampRate> VulkanDevice::timestampRate(std::uint32_t queue) const {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // One period for the whole device.
    return TimestampRate{.periodNs = timestampPeriod_, .frequency = 0};
}

core::Result<CalibratedTimestamps> VulkanDevice::calibrateTimestamps(std::uint32_t queue, HostClock clock) const {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkKnown(clock, "host clock"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const auto [domain, read] = domainFor(clock, timestamps_.hostClocks);
    if (!read) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device doesn't read that host clock beside its own");
    }
    // The device has one clock for every queue. The KHR call, or the EXT one
    // where only that extension was enabled.
    const PFN_vkGetCalibratedTimestampsKHR calibrate = table_.vkGetCalibratedTimestampsKHR != nullptr
                                                           ? table_.vkGetCalibratedTimestampsKHR
                                                           : table_.vkGetCalibratedTimestampsEXT;
    std::array<VkCalibratedTimestampInfoKHR, 2> infos{};
    for (auto& info : infos) {
        info.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
    }
    infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;
    infos[1].timeDomain = domain;
    std::array<std::uint64_t, 2> taken{};
    std::uint64_t deviation = 0;
    if (const VkResult result = calibrate(device_, static_cast<std::uint32_t>(infos.size()), infos.data(), taken.data(),
                                          &deviation);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkGetCalibratedTimestampsKHR", .result = result});
    }
    return CalibratedTimestamps{.gpu = taken[0], .host = taken[1], .maxDeviationNs = deviation, .deviationGiven = true};
}

} // namespace bfr::rhi::vulkan
