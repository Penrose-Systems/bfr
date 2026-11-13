#include "d3d12/d3d12_timestamp.h"

#include "bfr/core/error.h"
#include "bfr/rhi/timestamp.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_result.h"
#include "rhi/enum_check.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <expected>
#include <utility>

namespace bfr::rhi::d3d12 {

using Microsoft::WRL::ComPtr;

// The device's timestamp calls. Each passes its values to the one D3D12
// call that does what it says.

core::Result<TimestampPoolHandle> D3D12Device::createTimestampPool(const TimestampPoolDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    // A copy queue writes into its own kind of heap.
    const D3D12_QUERY_HEAP_DESC heapDesc{.Type = queues_[desc.queue]->listType == D3D12_COMMAND_LIST_TYPE_COPY
                                                     ? D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP
                                                     : D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                                         .Count = desc.count,
                                         .NodeMask = 0};
    ComPtr<ID3D12QueryHeap> heap;
    if (const HRESULT result = device_->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&heap)); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "ID3D12Device::CreateQueryHeap", .result = result});
    }
    const TimestampPoolHandle made = timestampPools_.emplace(D3D12TimestampPool{.heap = std::move(heap)});
    if (made.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} timestamp pools",
                             desc_.maxTimestampPools);
    }
    return made;
}

core::Result<void> D3D12Device::destroyTimestampPool(TimestampPoolHandle pool) {
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

core::Result<void> D3D12Device::resetTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                std::uint32_t /*first*/, std::uint32_t /*count*/) {
    // D3D12 needs no reset before a timestamp is written again, so nothing is
    // recorded.
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (timestampPools_.find(pool) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    return {};
}

core::Result<void> D3D12Device::writeTimestamp(CommandListHandle commandList, TimestampPoolHandle pool,
                                               std::uint32_t index, TimestampStage stage,
                                               TimestampGranularity granularity) {
    if (auto checked = checkKnown(stage, "timestamp stage"); !checked) {
        return checked;
    }
    // D3D12 takes no granularity, as the caps say, so it is only checked.
    if (auto checked = checkKnown(granularity, "timestamp granularity"); !checked) {
        return checked;
    }
    if (stage != TimestampStage::End) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "D3D12's EndQuery has no setting for a stage but the end of the work before it");
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const D3D12TimestampPool* found = timestampPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    (*list)->list->EndQuery(found->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
    return {};
}

core::Result<void> D3D12Device::resolveTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                  std::uint32_t first, std::uint32_t count, BufferHandle buffer,
                                                  std::uint64_t offset) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const D3D12TimestampPool* found = timestampPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    auto target = bufferFor(buffer);
    if (!target) {
        return std::unexpected(std::move(target.error()));
    }
    (*list)->list->ResolveQueryData(found->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, first, count,
                                    (*target)->resource.Get(), offset);
    return {};
}

core::Result<TimestampRate> D3D12Device::timestampRate(std::uint32_t queue) const {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    UINT64 frequency = 0;
    if (const HRESULT result = queues_[queue]->queue->GetTimestampFrequency(&frequency); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12CommandQueue::GetTimestampFrequency", .result = result});
    }
    return TimestampRate{.periodNs = 0, .frequency = frequency};
}

core::Result<CalibratedTimestamps> D3D12Device::calibrateTimestamps(std::uint32_t queue, HostClock clock) const {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkKnown(clock, "host clock"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (clock != HostClock::PerformanceCounter) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "D3D12's GetClockCalibration reads only QueryPerformanceCounter");
    }
    UINT64 gpu = 0;
    UINT64 host = 0;
    if (const HRESULT result = queues_[queue]->queue->GetClockCalibration(&gpu, &host); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12CommandQueue::GetClockCalibration", .result = result});
    }
    // D3D12 says nothing of how far apart the two readings may be.
    return CalibratedTimestamps{.gpu = gpu, .host = host, .maxDeviationNs = 0, .deviationGiven = false};
}

} // namespace bfr::rhi::d3d12
