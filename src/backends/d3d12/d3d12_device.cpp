#include "d3d12/d3d12_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"
#include "core/fixed_vector.h"
#include "core/slot_chains.h"
#include "d3d12/d3d12_instance.h"
#include "d3d12/d3d12_pipeline.h"
#include "d3d12/d3d12_result.h"
#include "d3d12/d3d12_swapchain.h"
#include "rhi/command_check.h"
#include "rhi/format_blocks.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

core::Result<void> deviceFailed(const char* call, HRESULT result) {
    return failure(core::ErrorCode::RhiDeviceFailed, {.call = call, .result = result});
}

// A barrier that moves image between the states a swapchain image is presented
// and rendered to in.
D3D12_RESOURCE_BARRIER transition(ID3D12Resource* image, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = image;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

} // namespace

core::Result<std::unique_ptr<D3D12Device>> D3D12Device::create(D3D12Instance& rhi, ComPtr<ID3D12Device> device,
                                                               ComPtr<IDXGIAdapter3> adapter, std::size_t heapCount,
                                                               const DeviceDesc& desc) {
    auto made = std::make_unique<D3D12Device>(rhi, std::move(device), std::move(adapter), heapCount, desc);
    if (auto opened = made->open(desc); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    auto allocator = createAllocator(made->device_.Get(), made->adapter_.Get());
    if (!allocator) {
        return std::unexpected(std::move(allocator.error()));
    }
    made->allocator_ = std::move(*allocator);
    auto layout = createD3D12BindlessLayout(*made->device_.Get(), desc);
    if (!layout) {
        return std::unexpected(std::move(layout.error()));
    }
    made->layout_ = std::move(*layout);
    return made;
}

D3D12Device::D3D12Device(D3D12Instance& rhi, ComPtr<ID3D12Device> device, ComPtr<IDXGIAdapter3> adapter,
                         std::size_t heapCount, const DeviceDesc& desc)
    : instance_(&rhi),
      device_(std::move(device)),
      adapter_(std::move(adapter)),
      heapCount_(heapCount),
      desc_(desc),
      swapchains_("swapchains", desc.maxSwapchains, desc.maxSwapchains, rhi.logger()),
      commandPools_("command pools", desc.maxCommandPools, desc.maxCommandPools, rhi.logger()),
      commandLists_("command lists", desc.maxCommandLists, desc.maxCommandLists, rhi.logger()),
      listSlots_(desc.maxCommandLists),
      lists_(desc.maxCommandLists),
      renderSlots_(desc.maxCommandLists),
      renders_(desc.maxCommandLists),
      submissions_(1 + desc.compute + desc.copy),
      buffers_("buffers", kBufferCapacity, kBufferLimit, rhi.logger()),
      textures_("textures", kBufferCapacity, kBufferLimit, rhi.logger()),
      bindless_(desc),
      pipelines_("pipelines", kBufferCapacity, kBufferLimit, rhi.logger()) {}

core::Result<void> D3D12Device::open(const DeviceDesc& desc) {
    const auto open = [this, &desc](QueueType type, D3D12_COMMAND_LIST_TYPE listType) -> core::Result<void> {
        auto queue = std::make_unique<D3D12Queue>();
        queue->type = type;
        queue->listType = listType;
        // A submit's arrays, made now at the device's maximums. Every pool
        // may be destroyed before the GPU is done with it.
        queue->lists = core::FixedVector<ID3D12CommandList*>{desc.maxCommandLists};
        queue->chains = core::FixedVector<D3D12Swapchain*>{desc.maxSwapchains};
        queue->retired = core::FixedVector<D3D12RetiredPool>{desc.maxCommandPools};
        const D3D12_COMMAND_QUEUE_DESC queueDesc{.Type = listType,
                                                 .Priority = 0,
                                                 .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
                                                 .NodeMask = 0};
        if (const HRESULT result = device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue->queue));
            FAILED(result)) {
            return deviceFailed("ID3D12Device::CreateCommandQueue", result);
        }
        if (const HRESULT result = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&queue->fence));
            FAILED(result)) {
            return deviceFailed("ID3D12Device::CreateFence", result);
        }
        queue->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (queue->event == nullptr) {
            return core::failure(core::ErrorCode::RhiDeviceFailed, "CreateEventW: error {}", GetLastError());
        }
        queues_.push_back(std::move(queue));
        return {};
    };
    if (auto opened = open(QueueType::Graphics, D3D12_COMMAND_LIST_TYPE_DIRECT); !opened) {
        return opened;
    }
    for (std::uint32_t index = 0; index < desc.compute; ++index) {
        if (auto opened = open(QueueType::Compute, D3D12_COMMAND_LIST_TYPE_COMPUTE); !opened) {
            return opened;
        }
    }
    for (std::uint32_t index = 0; index < desc.copy; ++index) {
        if (auto opened = open(QueueType::Copy, D3D12_COMMAND_LIST_TYPE_COPY); !opened) {
            return opened;
        }
    }
    return {};
}

D3D12Device::~D3D12Device() {
    // A failed wait leaves nothing to do but release, and no result to report
    // it through, so it is sent as an error whatever BFR_LOGGING says.
    if (auto idle = waitIdle(); !idle) {
        core::Message message;
        message.format("{}; destroying the device anyway", idle.error().message.view());
        instance_->logger().send(core::Severity::Error, message.view());
    }
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    if (commandPools_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} command pools were left when the device went", commandPools_.size());
    }
    swapchains_.clear();
    // A list left recording is closed before it goes.
    commandLists_.forEach([this](D3D12CommandList& list) { discard(list.list); });
    commandLists_.clear();
    commandPools_.clear();
    if (buffers_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} buffers were left when the device went", buffers_.size());
    }
    if (textures_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} textures were left when the device went", textures_.size());
    }
    if (pipelines_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} pipelines were left when the device went", pipelines_.size());
    }
    // Every buffer and texture gives its memory back before the allocator goes,
    // and every pipeline goes before the layout it was made on.
    buffers_.clear();
    textures_.clear();
    pipelines_.clear();
    layout_ = {};
    allocator_.Reset();
    // The slots' vector releases every command allocator and list as it goes.
    for (const auto& queue : queues_) {
        if (queue->event != nullptr) {
            CloseHandle(queue->event);
        }
    }
}

core::Result<void> D3D12Device::waitFor(D3D12Queue& queue, std::uint64_t value) {
    if (queue.fence->GetCompletedValue() >= value) {
        return {};
    }
    const std::scoped_lock lock{queue.waitLock};
    if (const HRESULT result = queue.fence->SetEventOnCompletion(value, queue.event); FAILED(result)) {
        return deviceFailed("ID3D12Fence::SetEventOnCompletion", result);
    }
    if (WaitForSingleObject(queue.event, INFINITE) != WAIT_OBJECT_0) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "waiting for the GPU failed");
    }
    return {};
}

core::Result<void> D3D12Device::checkQueue(std::uint32_t queue) const {
    if (queue >= queues_.size()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the device has no queue {}; it has {}", queue,
                             queues_.size());
    }
    return {};
}

core::Result<void> D3D12Device::checkToken(Token token) const {
    if (auto checked = checkQueue(token.queue); !checked) {
        return checked;
    }
    const std::uint64_t submitted = queues_[token.queue]->submitted.load(std::memory_order_acquire);
    if (token.value > submitted) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "no submit has returned token {} on queue {}; the last is {}", token.value, token.queue,
                             submitted);
    }
    return {};
}

void D3D12Device::collectPools(D3D12Queue& queue) {
    if (queue.retired.empty()) {
        return;
    }
    const std::uint64_t completed = queue.fence->GetCompletedValue();
    queue.retired.eraseIf([this, completed](const D3D12RetiredPool& retired) {
        if (retired.value > completed) {
            return false;
        }
        releaseSlots(retired.first);
        poolsHeld_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    });
    queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
}

void D3D12Device::releaseSlots(std::uint32_t first) {
    for (std::uint32_t slot = first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        lists_[slot].list.Reset();
        lists_[slot].allocator.Reset();
    }
    listSlots_.give(first);
}

core::Result<void> D3D12Device::waitIdle() {
    for (const auto& queue : queues_) {
        if (auto waited = waitFor(*queue, queue->submitted.load(std::memory_order_acquire)); !waited) {
            return waited;
        }
    }
    return {};
}

void D3D12Device::discard(ID3D12GraphicsCommandList7* list) {
    if (const HRESULT result = list->Close(); FAILED(result)) {
        BFR_LOG_WARN(instance_->logger(), "ID3D12GraphicsCommandList::Close: {} (0x{:08X})",
                     D3D12Failure::nameOf(result), static_cast<std::uint32_t>(result));
    }
}

core::Result<D3D12Swapchain*> D3D12Device::swapchainFor(SwapchainHandle swapchain) {
    D3D12Swapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
    }
    return found;
}

core::Result<D3D12CommandPool*> D3D12Device::commandPoolFor(CommandPoolHandle commandPool) {
    D3D12CommandPool* found = commandPools_.find(commandPool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command pool handle is stale");
    }
    return found;
}

core::Result<D3D12CommandList*> D3D12Device::commandListFor(CommandListHandle commandList) {
    D3D12CommandList* found = commandLists_.find(commandList);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command list handle is stale");
    }
    return found;
}

core::Result<SwapchainHandle> D3D12Device::createSwapchain(const core::window::NativeWindow& window,
                                                           const SwapchainDesc& desc) {
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, ComPtr<IDXGIFactory2>{instance_->factory()},
                                                          queues_[0]->queue, window, desc, instance_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "a device holds at most {} swapchains",
                             desc_.maxSwapchains);
    }
    return swapchain;
}

core::Result<SwapchainImages> D3D12Device::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                           std::uint32_t height) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // The old images may still be on the GPU.
    if (auto idle = waitIdle(); !idle) {
        return std::unexpected(std::move(idle.error()));
    }
    return (*found)->build(width, height);
}

core::Result<void> D3D12Device::destroySwapchain(SwapchainHandle swapchain) {
    if (swapchain.isNull()) {
        return {};
    }
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // Its window is the caller's to close once this returns, so the GPU is
    // waited for now.
    if (auto idle = waitIdle(); !idle) {
        return std::unexpected(std::move(idle.error()));
    }
    if (swapchains_.erase(swapchain)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

core::Result<void> D3D12Device::acquire(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->acquire();
}

core::Result<void> D3D12Device::present(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const std::scoped_lock lock{queues_[0]->lock};
    return (*found)->present();
}

core::Result<CommandPoolHandle> D3D12Device::createCommandPool(const CommandPoolDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so pools the GPU is done with are freed first, making their slots
    // and room count.
    for (const auto& queue : queues_) {
        if (queue->retiredCount.load(std::memory_order_acquire) != 0) {
            const std::scoped_lock lock{queue->lock};
            collectPools(*queue);
        }
    }
    if (poolsHeld_.fetch_add(1, std::memory_order_relaxed) >= desc_.maxCommandPools) {
        poolsHeld_.fetch_sub(1, std::memory_order_relaxed);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command pools",
                             desc_.maxCommandPools);
    }
    const auto first = listSlots_.take(desc.listCount);
    if (!first) {
        poolsHeld_.fetch_sub(1, std::memory_order_relaxed);
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the pool asks for {} lists; {} of the device's {} are free", desc.listCount,
                             listSlots_.available(), desc_.maxCommandLists);
    }
    const auto release = [this, first] {
        releaseSlots(*first);
        poolsHeld_.fetch_sub(1, std::memory_order_relaxed);
    };
    // Every allocator and list the pool will hand out is made now, so
    // beginCommands only resets one. A list is made recording, so it is
    // closed until then.
    const D3D12_COMMAND_LIST_TYPE listType = queues_[desc.queue]->listType;
    for (std::uint32_t slot = *first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        D3D12ListSlot& made = lists_[slot];
        if (const HRESULT result = device_->CreateCommandAllocator(listType, IID_PPV_ARGS(&made.allocator));
            FAILED(result)) {
            release();
            return std::unexpected(deviceFailed("ID3D12Device::CreateCommandAllocator", result).error());
        }
        if (const HRESULT result = device_->CreateCommandList(0, listType, made.allocator.Get(), nullptr,
                                                              IID_PPV_ARGS(&made.list));
            FAILED(result)) {
            release();
            return std::unexpected(deviceFailed("ID3D12Device::CreateCommandList", result).error());
        }
        if (const HRESULT result = made.list->Close(); FAILED(result)) {
            release();
            return std::unexpected(deviceFailed("ID3D12GraphicsCommandList::Close", result).error());
        }
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(
        D3D12CommandPool{.queue = desc.queue,
                         .pushConstantUpdates = desc.pushConstantUpdates,
                         .pushed = 0,
                         .first = *first,
                         .next = *first,
                         .recording = 0,
                         .lastSubmit = 0});
    if (commandPool.isNull()) {
        release();
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command pools",
                             desc_.maxCommandPools);
    }
    return commandPool;
}

core::Result<void> D3D12Device::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if ((*pool)->recording != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} lists from the pool are still recording; submit or destroy them", (*pool)->recording);
    }
    if (queues_[(*pool)->queue]->fence->GetCompletedValue() < (*pool)->lastSubmit) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the GPU has not passed the pool's last submit, token {}; wait on it first",
                             (*pool)->lastSubmit);
    }
    // Only the lists handed out since the last reset recorded into their
    // allocators.
    for (std::uint32_t slot = (*pool)->first; slot != (*pool)->next; slot = listSlots_.next(slot)) {
        if (const HRESULT result = lists_[slot].allocator->Reset(); FAILED(result)) {
            return deviceFailed("ID3D12CommandAllocator::Reset", result);
        }
    }
    (*pool)->next = (*pool)->first;
    (*pool)->pushed = 0;
    return {};
}

core::Result<void> D3D12Device::destroyCommandPool(CommandPoolHandle commandPool) {
    if (commandPool.isNull()) {
        return {};
    }
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const D3D12CommandPool gone = **pool;
    for (std::uint32_t slot = gone.first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        if (const D3D12CommandList* list = commandLists_.find(lists_[slot].live); list != nullptr) {
            discard(list->list);
            renderSlots_.give(list->rendered);
        }
        commandLists_.erase(std::exchange(lists_[slot].live, {}));
    }
    // Out of the table before its room comes back, so a createCommandPool
    // that finds the room finds a free entry too.
    commandPools_.erase(commandPool);
    D3D12Queue& queue = *queues_[gone.queue];
    {
        const std::scoped_lock lock{queue.lock};
        // poolsHeld_ counts this pool until it is collected, so there is room.
        std::ignore = queue.retired.push({.first = gone.first, .value = gone.lastSubmit});
        queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
        collectPools(queue);
    }
    return {};
}

core::Result<CommandListHandle> D3D12Device::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const std::uint32_t slot = (*pool)->next;
    if (slot == core::SlotChains::kEnd) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "every list in the pool was handed out since its last reset");
    }
    // The allocator and list were made with the pool; this only resets one.
    const D3D12ListSlot& made = lists_[slot];
    if (const HRESULT result = made.list->Reset(made.allocator.Get(), nullptr); FAILED(result)) {
        return std::unexpected(deviceFailed("ID3D12GraphicsCommandList::Reset", result).error());
    }
    // The one layout, bound once per list where the queue takes it: the heaps
    // first, as SM 6.6 asks, then the root signature and its tables for each
    // kind of pipeline the queue runs.
    const D3D12_COMMAND_LIST_TYPE type = queues_[(*pool)->queue]->listType;
    if (type != D3D12_COMMAND_LIST_TYPE_COPY) {
        // A device with no tables of a kind has no heap of it, and binds none.
        std::array<ID3D12DescriptorHeap*, 2> heaps{};
        UINT heapCount = 0;
        for (ID3D12DescriptorHeap* heap : {layout_.resources.Get(), layout_.samplers.Get()}) {
            if (heap != nullptr) {
                heaps.at(heapCount++) = heap;
            }
        }
        made.list->SetDescriptorHeaps(heapCount, heaps.data());
        const auto bindTables = [&](auto setRootSignature, auto setTable) {
            (made.list.Get()->*setRootSignature)(layout_.rootSignature.Get());
            if (layout_.resources) {
                (made.list.Get()->*setTable)(kResourceTableParameter,
                                             layout_.resources->GetGPUDescriptorHandleForHeapStart());
            }
            if (layout_.samplers) {
                (made.list.Get()->*setTable)(kSamplerTableParameter,
                                             layout_.samplers->GetGPUDescriptorHandleForHeapStart());
            }
        };
        bindTables(&ID3D12GraphicsCommandList::SetComputeRootSignature,
                   &ID3D12GraphicsCommandList::SetComputeRootDescriptorTable);
        if (type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
            bindTables(&ID3D12GraphicsCommandList::SetGraphicsRootSignature,
                       &ID3D12GraphicsCommandList::SetGraphicsRootDescriptorTable);
        }
    }
    // A new handle for each recording, so last frame's handle never names
    // this one. The list pool was made at its maximum, so this never grows.
    const CommandListHandle commandList = commandLists_.emplace(D3D12CommandList{.list = made.list.Get(),
                                                                                 .pool = commandPool,
                                                                                 .slot = slot,
                                                                                 .state = {},
                                                                                 .presenting = {},
                                                                                 .presentStoreOp = StoreOp::Store,
                                                                                 .discards = {},
                                                                                 .discardCount = 0,
                                                                                 .rendered = core::SlotChains::kEnd});
    if (commandList.isNull()) {
        discard(made.list.Get());
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command lists",
                             desc_.maxCommandLists);
    }
    (*pool)->next = listSlots_.next(slot);
    ++(*pool)->recording;
    lists_[slot].live = commandList;
    return commandList;
}

core::Result<void> D3D12Device::beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                               const RenderingDesc& desc) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if ((*list)->state.rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is already open");
    }
    if (auto checked = checkRenderingDesc(!swapchain.isNull(), desc); !checked) {
        return checked;
    }
    if (const D3D12CommandPool* pool = commandPools_.find((*list)->pool);
        pool != nullptr && queues_[pool->queue]->type != QueueType::Graphics) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the list records for queue {}, which does not render",
                             pool->queue);
    }
    // Every handle is looked up before anything is recorded, so a bad one
    // leaves the list as it was.
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, kMaxColorTargets> colors{};
    std::array<ID3D12Resource*, kMaxColorTargets + 1> discards{};
    std::uint32_t discardCount = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    D3D12Swapchain* chain = nullptr;
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        const ColorAttachment& attachment = desc.colorAttachments.at(i);
        ID3D12Resource* resource = nullptr;
        if (i == 0 && !swapchain.isNull()) {
            auto found = swapchainFor(swapchain);
            if (!found) {
                return std::unexpected(std::move(found.error()));
            }
            chain = *found;
            if (chain->submitted()) {
                return core::failure(core::ErrorCode::RhiInvalidCall,
                                     "the acquired image was already submitted; present it first");
            }
            resource = chain->image();
            if (resource == nullptr) {
                return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
            }
            colors.at(i) = chain->view();
            const D3D12_RESOURCE_DESC image = resource->GetDesc();
            width = static_cast<std::uint32_t>(image.Width);
            height = image.Height;
        }
        else {
            auto texture = textureFor(attachment.texture);
            if (!texture) {
                return std::unexpected(std::move(texture.error()));
            }
            resource = (*texture)->resource.Get();
            colors.at(i) = (*texture)->target;
            width = (*texture)->desc.width;
            height = (*texture)->desc.height;
        }
        if (attachment.storeOp == StoreOp::DontCare) {
            discards.at(discardCount++) = resource;
        }
    }
    const D3D12Texture* depth = nullptr;
    if (!desc.depthStencil.texture.isNull()) {
        auto texture = textureFor(desc.depthStencil.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        depth = *texture;
        width = depth->desc.width;
        height = depth->desc.height;
        if (desc.depthStencil.storeOp == StoreOp::DontCare) {
            discards.at(discardCount++) = depth->resource.Get();
        }
    }
    if (chain != nullptr) {
        // Rendered to earlier in this list, the swapchain is already in its chain.
        bool renderedBefore = false;
        for (std::uint32_t slot = (*list)->rendered; slot != core::SlotChains::kEnd; slot = renderSlots_.next(slot)) {
            renderedBefore = renderedBefore || renders_[slot] == swapchain;
        }
        if (!renderedBefore) {
            const auto slot = renderSlots_.push((*list)->rendered);
            if (!slot) {
                return core::failure(core::ErrorCode::RhiLimitReached,
                                     "the device's recording lists render to at most {} swapchains at once",
                                     desc_.maxCommandLists);
            }
            renders_[*slot] = swapchain;
        }
    }

    ID3D12GraphicsCommandList7* recording = (*list)->list;
    if (chain != nullptr) {
        const D3D12_RESOURCE_BARRIER toRender = transition(chain->image(), D3D12_RESOURCE_STATE_PRESENT,
                                                           D3D12_RESOURCE_STATE_RENDER_TARGET);
        recording->ResourceBarrier(1, &toRender);
    }
    recording->OMSetRenderTargets(desc.colorAttachmentCount, colors.data(), FALSE,
                                  depth != nullptr ? &depth->target : nullptr);
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        if (desc.colorAttachments.at(i).loadOp == LoadOp::Clear) {
            recording->ClearRenderTargetView(colors.at(i), desc.colorAttachments.at(i).clearColor.data(), 0, nullptr);
        }
    }
    if (depth != nullptr && desc.depthStencil.loadOp == LoadOp::Clear) {
        const FormatBlock& block = blockOf(depth->desc.format);
        const D3D12_CLEAR_FLAGS flags = block.depthBytes != 0 && block.stencilBytes != 0
                                            ? D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL
                                        : block.depthBytes != 0 ? D3D12_CLEAR_FLAG_DEPTH
                                                                : D3D12_CLEAR_FLAG_STENCIL;
        recording->ClearDepthStencilView(depth->target, flags, desc.depthStencil.clearDepth,
                                         static_cast<UINT8>(desc.depthStencil.clearStencil), 0, nullptr);
    }
    (*list)->state.rendering = true;
    (*list)->presenting = chain != nullptr ? std::optional{swapchain} : std::nullopt;
    (*list)->presentStoreOp = chain != nullptr ? desc.colorAttachments.at(0).storeOp : StoreOp::Store;
    (*list)->discards = discards;
    (*list)->discardCount = discardCount;
    // The whole area, until the caller sets its own.
    if (auto set = setViewport(commandList, {.x = 0,
                                             .y = 0,
                                             .width = static_cast<float>(width),
                                             .height = static_cast<float>(height),
                                             .minDepth = 0,
                                             .maxDepth = 1});
        !set) {
        return set;
    }
    return setScissor(commandList, {.x = 0, .y = 0, .width = width, .height = height});
}

core::Result<void> D3D12Device::endRendering(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    ID3D12GraphicsCommandList7* recording = (*list)->list;
    // What the caller does not keep is discarded, the swapchain's image among
    // it before it moves to present.
    for (std::uint32_t i = 0; i < (*list)->discardCount; ++i) {
        recording->DiscardResource((*list)->discards.at(i), nullptr);
    }
    if (const std::optional<SwapchainHandle> open = (*list)->presenting; open.has_value()) {
        auto chain = swapchainFor(*open);
        if (!chain) {
            return std::unexpected(std::move(chain.error()));
        }
        const D3D12_RESOURCE_BARRIER toPresent = transition((*chain)->image(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                            D3D12_RESOURCE_STATE_PRESENT);
        recording->ResourceBarrier(1, &toPresent);
    }
    (*list)->state.rendering = false;
    (*list)->presenting.reset();
    (*list)->discardCount = 0;
    return {};
}

namespace {

// Every write a list of type may make, which a barrier waits on. The spec
// warns on D3D12_BARRIER_ACCESS_COMMON before a barrier and allows each queue
// type only its own accesses ("Command Queue Access Compatibility").
// Raytracing writes are left out: BFR records none, and whether a device
// without raytracing takes the bit is not documented.
D3D12_BARRIER_ACCESS writesOf(D3D12_COMMAND_LIST_TYPE type) {
    using std::to_underlying;
    auto writes = to_underlying(D3D12_BARRIER_ACCESS_COPY_DEST);
    if (type != D3D12_COMMAND_LIST_TYPE_COPY) {
        writes |= to_underlying(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
    }
    if (type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
        writes |= to_underlying(D3D12_BARRIER_ACCESS_RENDER_TARGET) |
                  to_underlying(D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE) |
                  to_underlying(D3D12_BARRIER_ACCESS_STREAM_OUTPUT) | to_underlying(D3D12_BARRIER_ACCESS_RESOLVE_DEST);
    }
    // A flag enum holds any OR of its flags, which the analyzer does not know.
    return static_cast<D3D12_BARRIER_ACCESS>(writes); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}

// Texture layouts as enhanced barriers take them. Copies use COMMON, the one
// layout every queue type takes them in, so a texture moves between queues
// without a change of layout.
D3D12_BARRIER_LAYOUT layoutFor(TextureLayout layout) {
    switch (layout) {
        case TextureLayout::Sampled:
            return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
        case TextureLayout::Storage:
            return D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
        case TextureLayout::ColorTarget:
            return D3D12_BARRIER_LAYOUT_RENDER_TARGET;
        case TextureLayout::DepthStencilTarget:
            return D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
        case TextureLayout::Undefined:
            return D3D12_BARRIER_LAYOUT_UNDEFINED;
        case TextureLayout::CopySrc:
        case TextureLayout::CopyDst:
        default:
            return D3D12_BARRIER_LAYOUT_COMMON;
    }
}

D3D12_BARRIER_ACCESS accessFor(TextureLayout layout) {
    switch (layout) {
        case TextureLayout::CopySrc:
            return D3D12_BARRIER_ACCESS_COPY_SOURCE;
        case TextureLayout::CopyDst:
            return D3D12_BARRIER_ACCESS_COPY_DEST;
        case TextureLayout::Sampled:
            return D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
        case TextureLayout::Storage:
            return D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
        case TextureLayout::ColorTarget:
            return D3D12_BARRIER_ACCESS_RENDER_TARGET;
        case TextureLayout::DepthStencilTarget:
            return D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
        case TextureLayout::Undefined:
        default:
            return D3D12_BARRIER_ACCESS_NO_ACCESS;
    }
}

} // namespace

core::Result<D3D12_GLOBAL_BARRIER> D3D12Device::fullBarrierFor(const D3D12CommandList& list) {
    const D3D12CommandPool* pool = commandPools_.find(list.pool);
    if (pool == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    return D3D12_GLOBAL_BARRIER{.SyncBefore = D3D12_BARRIER_SYNC_ALL,
                                .SyncAfter = D3D12_BARRIER_SYNC_ALL,
                                .AccessBefore = writesOf(queues_[pool->queue]->listType),
                                .AccessAfter = D3D12_BARRIER_ACCESS_COMMON};
}

core::Result<void> D3D12Device::barrier(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const auto global = fullBarrierFor(**list);
    if (!global) {
        return std::unexpected(global.error());
    }
    const D3D12_BARRIER_GROUP group{.Type = D3D12_BARRIER_TYPE_GLOBAL, .NumBarriers = 1, .pGlobalBarriers = &*global};
    (*list)->list->Barrier(1, &group);
    return {};
}

core::Result<void> D3D12Device::textureBarrier(CommandListHandle commandList, TextureHandle texture,
                                               TextureLayout before, TextureLayout after) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const auto global = fullBarrierFor(**list);
    if (!global) {
        return std::unexpected(global.error());
    }
    // A texture barrier orders only the texture, so the full one goes with it.
    // Mip 0xffffffff with no mip count names every subresource; moving from
    // UNDEFINED discards, which a placed render target or depth texture needs
    // before its first use.
    constexpr UINT kAllSubresources = 0xffffffff;
    // From an undefined layout nothing was accessed, which D3D12 pairs with
    // no sync before.
    const D3D12_TEXTURE_BARRIER moved{.SyncBefore = before == TextureLayout::Undefined ? D3D12_BARRIER_SYNC_NONE
                                                                                       : D3D12_BARRIER_SYNC_ALL,
                                      .SyncAfter = D3D12_BARRIER_SYNC_ALL,
                                      .AccessBefore = accessFor(before),
                                      .AccessAfter = accessFor(after),
                                      .LayoutBefore = layoutFor(before),
                                      .LayoutAfter = layoutFor(after),
                                      .pResource = (*found)->resource.Get(),
                                      .Subresources = {.IndexOrFirstMipLevel = kAllSubresources,
                                                       .NumMipLevels = 0,
                                                       .FirstArraySlice = 0,
                                                       .NumArraySlices = 0,
                                                       .FirstPlane = 0,
                                                       .NumPlanes = 0},
                                      .Flags = before == TextureLayout::Undefined ? D3D12_TEXTURE_BARRIER_FLAG_DISCARD
                                                                                  : D3D12_TEXTURE_BARRIER_FLAG_NONE};
    const std::array groups{
        D3D12_BARRIER_GROUP{.Type = D3D12_BARRIER_TYPE_GLOBAL, .NumBarriers = 1, .pGlobalBarriers = &*global},
        D3D12_BARRIER_GROUP{.Type = D3D12_BARRIER_TYPE_TEXTURE, .NumBarriers = 1, .pTextureBarriers = &moved}};
    (*list)->list->Barrier(static_cast<UINT32>(groups.size()), groups.data());
    return {};
}

core::Result<Token> D3D12Device::submit(const SubmitDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    for (const Token wait : desc.waits) {
        if (auto checked = checkToken(wait); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    D3D12Queue& queue = *queues_[desc.queue];
    const std::scoped_lock lock{queue.lock};
    queue.lists.clear();
    queue.chains.clear();
    // Each submit has a serial of its own, so a list named twice is caught
    // in this one pass, without searching what was gathered so far.
    const std::uint64_t serial = submitSerial_.fetch_add(1, std::memory_order_relaxed) + 1;
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
        }
        if ((*list)->queued == serial) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the list is named twice in the submit");
        }
        (*list)->queued = serial;
        if ((*list)->state.rendering) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
        }
        if (const D3D12CommandPool* pool = commandPools_.find((*list)->pool);
            pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the list was recorded for queue {}, not {}",
                                 pool->queue, desc.queue);
        }
        for (std::uint32_t slot = (*list)->rendered; slot != core::SlotChains::kEnd; slot = renderSlots_.next(slot)) {
            auto chain = swapchainFor(renders_[slot]);
            if (!chain) {
                return std::unexpected(std::move(chain.error()));
            }
            if ((*chain)->image() == nullptr) {
                return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
            }
            if ((*chain)->submitted()) {
                return core::failure(core::ErrorCode::RhiInvalidCall,
                                     "the acquired image was already submitted; present it first");
            }
            if (std::ranges::find(queue.chains, *chain) == queue.chains.end()) {
                std::ignore = queue.chains.push(*chain);
            }
        }
        if (!queue.lists.push((*list)->list)) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a submit takes at most {} lists",
                                 desc_.maxCommandLists);
        }
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    // The lists are spent whether or not the submit goes through. Their pools
    // remember it only once the GPU is sent it: a value no submit signals
    // would otherwise keep the pool from ever being reset or freed. A Signal
    // that fails after ExecuteCommandLists is the device removed, when
    // nothing runs.
    const auto finish = [this, &desc, value](bool sent) {
        for (const CommandListHandle commandList : desc.commandLists) {
            const D3D12CommandList* list = commandLists_.find(commandList);
            if (list == nullptr) {
                continue;
            }
            if (D3D12CommandPool* pool = commandPools_.find(list->pool); sent && pool != nullptr) {
                pool->lastSubmit = value;
            }
            end(commandList);
        }
    };
    core::Result<void> closed;
    for (const CommandListHandle commandList : desc.commandLists) {
        const D3D12CommandList* list = commandLists_.find(commandList);
        if (list == nullptr) {
            continue;
        }
        if (const HRESULT result = list->list->Close(); FAILED(result) && closed) {
            closed = deviceFailed("ID3D12GraphicsCommandList::Close", result);
        }
    }
    if (!closed) {
        finish(false);
        return std::unexpected(std::move(closed.error()));
    }
    for (const Token wait : desc.waits) {
        if (const HRESULT result = queue.queue->Wait(queues_[wait.queue]->fence.Get(), wait.value); FAILED(result)) {
            finish(false);
            return std::unexpected(deviceFailed("ID3D12CommandQueue::Wait", result).error());
        }
    }
    if (!queue.lists.empty()) {
        queue.queue->ExecuteCommandLists(static_cast<UINT>(queue.lists.size()), queue.lists.data());
    }
    if (const HRESULT result = queue.queue->Signal(queue.fence.Get(), value); FAILED(result)) {
        finish(false);
        return std::unexpected(deviceFailed("ID3D12CommandQueue::Signal", result).error());
    }
    queue.submitted.store(value, std::memory_order_release);
    finish(true);
    for (D3D12Swapchain* chain : queue.chains) {
        chain->setSubmitted();
    }
    collectPools(queue);
    return Token{.queue = desc.queue, .value = value};
}

core::Result<void> D3D12Device::wait(Token token) {
    if (auto checked = checkToken(token); !checked) {
        return checked;
    }
    D3D12Queue& queue = *queues_[token.queue];
    if (auto waited = waitFor(queue, token.value); !waited) {
        return waited;
    }
    if (queue.retiredCount.load(std::memory_order_acquire) != 0) {
        const std::scoped_lock lock{queue.lock};
        collectPools(queue);
    }
    return {};
}

core::Result<bool> D3D12Device::poll(Token token) {
    if (auto checked = checkToken(token); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    D3D12Queue& queue = *queues_[token.queue];
    const std::uint64_t completed = queue.fence->GetCompletedValue();
    if (queue.retiredCount.load(std::memory_order_acquire) != 0) {
        const std::scoped_lock lock{queue.lock};
        collectPools(queue);
    }
    return completed >= token.value;
}

core::Result<void> D3D12Device::destroyCommandList(CommandListHandle commandList) {
    if (commandList.isNull()) {
        return {};
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    discard((*list)->list);
    end(commandList);
    return {};
}

void D3D12Device::end(CommandListHandle commandList) {
    const D3D12CommandList* list = commandLists_.find(commandList);
    if (list == nullptr) {
        return;
    }
    if (D3D12CommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
        --pool->recording;
    }
    renderSlots_.give(list->rendered);
    lists_[list->slot].live = {};
    commandLists_.erase(commandList);
}

core::Result<MemoryBudget> D3D12Device::budget() const {
    // The first heap is the adapter's own memory, and a second is the machine's.
    constexpr std::array kGroups{DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL};
    MemoryBudget budget;
    for (const DXGI_MEMORY_SEGMENT_GROUP group : std::span{kGroups}.first(std::min(heapCount_, kGroups.size()))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (const HRESULT result = adapter_->QueryVideoMemoryInfo(0, group, &info); FAILED(result)) {
            return failure(core::ErrorCode::RhiQueryFailed,
                           {.call = "IDXGIAdapter3::QueryVideoMemoryInfo", .result = result});
        }
        budget.heaps.push_back({.budget = info.Budget, .usage = info.CurrentUsage});
    }
    return budget;
}

} // namespace bfr::rhi::d3d12
