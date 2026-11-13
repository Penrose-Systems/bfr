#include "metal/metal_timestamp.h"

#include "bfr/core/error.h"
#include "bfr/rhi/timestamp.h"
#include "metal/metal_device.h"
#include "metal/metal_util.h"
#include "rhi/enum_check.h"
#include "rhi/name_check.h"

#include <Foundation/NSError.hpp>
#include <Foundation/NSRange.hpp>
#include <Metal/Metal.hpp>

#include <atomic>
#include <cstdint>
#include <expected>
#include <utility>

namespace bfr::rhi::metal {

namespace {

MTL4::TimestampGranularity granularityFor(TimestampGranularity granularity) {
    return granularity == TimestampGranularity::Precise ? MTL4::TimestampGranularityPrecise
                                                        : MTL4::TimestampGranularityRelaxed;
}

} // namespace

// The device's timestamp calls. Each passes its values to the one Metal call
// that does what it says, on the command buffer between encoders and on the
// render encoder in rendering.

core::Result<TimestampPoolHandle> MetalDevice::createTimestampPool(const TimestampPoolDesc& desc) {
    if (auto checked = checkObjectName(features_, desc.name); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    const AutoreleasePool pool;
    const Ref<MTL4::CounterHeapDescriptor> descriptor = adopt(MTL4::CounterHeapDescriptor::alloc()->init());
    descriptor->setType(MTL4::CounterHeapTypeTimestamp);
    descriptor->setCount(desc.count);
    NS::Error* error = nullptr;
    MTL4::CounterHeap* heap = device_->newCounterHeap(descriptor.get(), &error);
    if (heap == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no counter heap: {}",
                             error != nullptr ? error->localizedDescription()->utf8String() : "no reason given");
    }
    const TimestampPoolHandle made = timestampPools_.emplace(MetalTimestampPool{.heap = adopt(heap)});
    if (made.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} timestamp pools",
                             desc_.maxTimestampPools);
    }
    return made;
}

core::Result<void> MetalDevice::destroyTimestampPool(TimestampPoolHandle pool) {
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

core::Result<void> MetalDevice::resetTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                std::uint32_t /*first*/, std::uint32_t /*count*/) {
    // Metal needs no reset before a timestamp is written again, so nothing is
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

core::Result<void> MetalDevice::writeTimestamp(CommandListHandle commandList, TimestampPoolHandle pool,
                                               std::uint32_t index, TimestampStage stage,
                                               TimestampGranularity granularity) {
    if (auto checked = checkKnown(stage, "timestamp stage"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(granularity, "timestamp granularity"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const MetalTimestampPool* found = timestampPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    if (!(*list)->state.rendering) {
        if (stage != TimestampStage::End) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 "outside rendering Metal writes a timestamp only at the end of the work before it");
        }
        (*list)->buffer->writeTimestampIntoHeap(found->heap.get(), index);
        return {};
    }
    if (stage != TimestampStage::Vertex && stage != TimestampStage::Fragment) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "in rendering Metal writes a timestamp only after the vertex or fragment stage");
    }
    (*list)->encoder->writeTimestamp(granularityFor(granularity),
                                     stage == TimestampStage::Vertex ? MTL::RenderStageVertex
                                                                     : MTL::RenderStageFragment,
                                     found->heap.get(), index);
    return {};
}

core::Result<void> MetalDevice::resolveTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                  std::uint32_t first, std::uint32_t count, BufferHandle buffer,
                                                  std::uint64_t offset) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    // The command buffer resolves between encoders; a render encoder is open.
    if ((*list)->state.rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "Metal resolves timestamps only outside rendering");
    }
    const MetalTimestampPool* found = timestampPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the timestamp pool handle is stale");
    }
    auto target = bufferFor(buffer);
    if (!target) {
        return std::unexpected(std::move(target.error()));
    }
    const std::uint64_t bytes = std::uint64_t{count} * device_->sizeOfCounterHeapEntry(MTL4::CounterHeapTypeTimestamp);
    (*list)->buffer->resolveCounterHeap(found->heap.get(), NS::Range::Make(first, count),
                                        MTL4::BufferRange::Make((*target)->buffer->gpuAddress() + offset, bytes),
                                        nullptr, nullptr);
    return {};
}

core::Result<TimestampRate> MetalDevice::timestampRate(std::uint32_t queue) const {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // One frequency for the whole device.
    return TimestampRate{.periodNs = 0, .frequency = device_->queryTimestampFrequency()};
}

core::Result<CalibratedTimestamps> MetalDevice::calibrateTimestamps(std::uint32_t queue, HostClock clock) const {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkKnown(clock, "host clock"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // sampleTimestamps reads a CPU clock metal-cpp doesn't name, so none of
    // BFR's is one it reads, as the caps say.
    return core::failure(core::ErrorCode::RhiUnsupported, "Metal names no host clock it reads beside the GPU's");
}

} // namespace bfr::rhi::metal
