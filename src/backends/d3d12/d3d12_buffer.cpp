#include "d3d12/d3d12_buffer.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_result.h"
#include "rhi/buffer_check.h"

#include <d3d12.h>
#include <D3D12MemAlloc.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <utility>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

// The heap each BufferMemory lives in, and the state D3D12 requires a buffer
// in that heap to be created in.
struct Placement {
    D3D12_HEAP_TYPE heap;
    D3D12_RESOURCE_STATES state;
};

Placement placementFor(BufferMemory memory) {
    switch (memory) {
        case BufferMemory::Device:
            return {.heap = D3D12_HEAP_TYPE_DEFAULT, .state = D3D12_RESOURCE_STATE_COMMON};
        case BufferMemory::Upload:
            return {.heap = D3D12_HEAP_TYPE_UPLOAD, .state = D3D12_RESOURCE_STATE_GENERIC_READ};
        case BufferMemory::Readback:
            return {.heap = D3D12_HEAP_TYPE_READBACK, .state = D3D12_RESOURCE_STATE_COPY_DEST};
    }
    return {.heap = D3D12_HEAP_TYPE_DEFAULT, .state = D3D12_RESOURCE_STATE_COMMON};
}

} // namespace

core::Result<ComPtr<D3D12MA::Allocator>> createAllocator(ID3D12Device* device, IDXGIAdapter* adapter) {
    D3D12MA::ALLOCATOR_DESC desc{};
    desc.pDevice = device;
    desc.pAdapter = adapter;
    ComPtr<D3D12MA::Allocator> allocator;
    if (const HRESULT result = D3D12MA::CreateAllocator(&desc, &allocator); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "D3D12MA::CreateAllocator", .result = result});
    }
    return allocator;
}

core::Result<D3D12Buffer> createD3D12Buffer(D3D12MA::Allocator& allocator, const BufferDesc& desc) {
    const Placement placement = placementFor(desc.memory);
    const D3D12_RESOURCE_DESC resource{
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = desc.size,
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1, .Quality = 0},
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = desc.usage.storage ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE,
    };
    D3D12MA::ALLOCATION_DESC allocation{};
    allocation.HeapType = placement.heap;
    D3D12Buffer buffer{.allocation = {}, .resource = {}, .memory = desc.memory, .mapped = {}};
    if (const HRESULT result = allocator.CreateResource(&allocation, &resource, placement.state, nullptr,
                                                        &buffer.allocation, IID_PPV_ARGS(&buffer.resource));
        FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "D3D12MA::Allocator::CreateResource", .result = result});
    }
    if (desc.memory == BufferMemory::Device) {
        return buffer;
    }
    // The CPU reads none of an upload buffer, and all of a readback one.
    const D3D12_RANGE none{.Begin = 0, .End = 0};
    void* data = nullptr;
    if (const HRESULT result = buffer.resource->Map(0, desc.memory == BufferMemory::Upload ? &none : nullptr, &data);
        FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "ID3D12Resource::Map", .result = result});
    }
    buffer.mapped = {static_cast<std::byte*>(data), static_cast<std::size_t>(desc.size)};
    return buffer;
}

// The device's buffer calls, beside the buffers they make.

core::Result<D3D12Buffer*> D3D12Device::bufferFor(BufferHandle buffer) {
    D3D12Buffer* found = buffers_.find(buffer);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    return found;
}

void D3D12Device::collectBuffers() {
    buffers_.collect(
        submissions_.completed([this](std::uint32_t queue) { return queues_[queue]->fence->GetCompletedValue(); }));
}

core::Result<BufferHandle> D3D12Device::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so buffers the GPU is done with are freed here, never on the
    // frame path: freeing one frees the allocator's memory.
    collectBuffers();
    auto made = createD3D12Buffer(*allocator_.Get(), desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const BufferHandle buffer = buffers_.emplace(std::move(*made));
    if (buffer.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", kBufferLimit);
    }
    return buffer;
}

core::Result<void> D3D12Device::copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
                                           std::span<const BufferCopy> regions) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = bufferFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = bufferFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    for (const BufferCopy& region : regions) {
        (*list)->list->CopyBufferRegion((*to)->resource.Get(), region.dstOffset, (*from)->resource.Get(),
                                        region.srcOffset, region.size);
    }
    return {};
}

core::Result<std::span<std::byte>> D3D12Device::mapBuffer(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*found)->memory == BufferMemory::Device) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the buffer is in device memory, which the CPU does not map");
    }
    return (*found)->mapped;
}

core::Result<void> D3D12Device::destroyBuffer(BufferHandle buffer) {
    if (buffer.isNull()) {
        return {};
    }
    if (buffers_.find(buffer) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    // Retired against every submit made so far, on every queue, so it lives
    // until the GPU has run all of them.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!buffers_.retire(buffer, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    collectBuffers();
    return {};
}

} // namespace bfr::rhi::d3d12
