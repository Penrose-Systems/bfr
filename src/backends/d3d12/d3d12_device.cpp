#include "d3d12/d3d12_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"
#include "d3d12/d3d12_instance.h"
#include "d3d12/d3d12_result.h"
#include "d3d12/d3d12_swapchain.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
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
                                                               const DeviceQueues& queues) {
    auto made = std::make_unique<D3D12Device>(rhi, std::move(device), std::move(adapter), heapCount);
    if (auto opened = made->open(queues); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    return made;
}

D3D12Device::D3D12Device(D3D12Instance& rhi, ComPtr<ID3D12Device> device, ComPtr<IDXGIAdapter3> adapter,
                         std::size_t heapCount)
    : instance_(&rhi),
      device_(std::move(device)),
      adapter_(std::move(adapter)),
      heapCount_(heapCount),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandPools_("command pools", kCommandPoolCapacity, kCommandPoolLimit, rhi.logger()),
      commandLists_("command lists", kCommandListCapacity, kCommandListLimit, rhi.logger()) {}

core::Result<void> D3D12Device::open(const DeviceQueues& queues) {
    const auto open = [this](QueueType type, D3D12_COMMAND_LIST_TYPE listType) -> core::Result<void> {
        auto queue = std::make_unique<D3D12Queue>();
        queue->type = type;
        queue->listType = listType;
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
            return core::failure(core::ErrorCode::RhiDeviceFailed,
                                 std::format("CreateEventW: error {}", GetLastError()));
        }
        queues_.push_back(std::move(queue));
        return {};
    };
    if (auto opened = open(QueueType::Graphics, D3D12_COMMAND_LIST_TYPE_DIRECT); !opened) {
        return opened;
    }
    for (std::uint32_t index = 0; index < queues.compute; ++index) {
        if (auto opened = open(QueueType::Compute, D3D12_COMMAND_LIST_TYPE_COMPUTE); !opened) {
            return opened;
        }
    }
    for (std::uint32_t index = 0; index < queues.copy; ++index) {
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
        instance_->logger().send(core::Severity::Error,
                                 std::format("{}; destroying the device anyway", idle.error().message));
    }
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    if (commandPools_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} command pools were left when the device went", commandPools_.size());
    }
    swapchains_.clear();
    // A list left recording is closed before it goes.
    commandLists_.forEach([this](D3D12CommandList& list) { discard(list.list.Get()); });
    commandLists_.clear();
    commandPools_.clear();
    for (const auto& queue : queues_) {
        queue->retired.clear();
        if (queue->event != nullptr) {
            CloseHandle(queue->event);
        }
    }
}

core::Result<void> D3D12Device::waitFor(D3D12Queue& queue, std::uint64_t value) {
    if (queue.fence->GetCompletedValue() >= value) {
        return {};
    }
    const std::scoped_lock lock{queue.lock};
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
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the device has no queue {}; it has {}", queue, queues_.size()));
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
                             std::format("no submit has returned token {} on queue {}; the last is {}", token.value,
                                         token.queue, submitted));
    }
    return {};
}

void D3D12Device::collectPools(D3D12Queue& queue) {
    if (queue.retired.empty()) {
        return;
    }
    const std::uint64_t completed = queue.fence->GetCompletedValue();
    std::erase_if(queue.retired, [completed](const auto& retired) { return retired.second <= completed; });
    queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
}

core::Result<void> D3D12Device::waitIdle() {
    for (const auto& queue : queues_) {
        if (auto waited = waitFor(*queue, queue->submitted.load(std::memory_order_acquire)); !waited) {
            return waited;
        }
    }
    return {};
}

void D3D12Device::discard(ID3D12GraphicsCommandList* list) {
    if (const HRESULT result = list->Close(); FAILED(result)) {
        BFR_LOG_WARN(instance_->logger(), "{}",
                     D3D12Failure{.call = "ID3D12GraphicsCommandList::Close", .result = result}.message());
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
                                                           const SwapchainRequest& request) {
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, ComPtr<IDXGIFactory2>{instance_->factory()},
                                                          queues_[0]->queue, window, request, instance_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("a device holds at most {} swapchains", kSwapchainLimit));
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

core::Result<CommandPoolHandle> D3D12Device::createCommandPool(std::uint32_t queue) {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(
        D3D12CommandPool{.queue = queue, .entries = {}, .used = 0, .recording = {}, .lastSubmit = 0});
    if (commandPool.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command pools", kCommandPoolLimit));
    }
    return commandPool;
}

core::Result<void> D3D12Device::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if (!(*pool)->recording.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("{} lists from the pool are still recording; submit or destroy them",
                                         (*pool)->recording.size()));
    }
    if (queues_[(*pool)->queue]->fence->GetCompletedValue() < (*pool)->lastSubmit) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the GPU has not passed the pool's last submit, token {}; wait on it first",
                                         (*pool)->lastSubmit));
    }
    for (const D3D12CommandPool::Entry& entry : (*pool)->entries) {
        if (const HRESULT result = entry.allocator->Reset(); FAILED(result)) {
            return deviceFailed("ID3D12CommandAllocator::Reset", result);
        }
    }
    (*pool)->used = 0;
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
    for (const CommandListHandle handle : (*pool)->recording) {
        if (D3D12CommandList* list = commandLists_.find(handle); list != nullptr) {
            discard(list->list.Get());
        }
        commandLists_.erase(handle);
    }
    D3D12Queue& queue = *queues_[(*pool)->queue];
    {
        const std::scoped_lock lock{queue.lock};
        queue.retired.emplace_back(std::move((*pool)->entries), (*pool)->lastSubmit);
        queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
        collectPools(queue);
    }
    commandPools_.erase(commandPool);
    return {};
}

core::Result<CommandListHandle> D3D12Device::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if ((*pool)->used == (*pool)->entries.size()) {
        D3D12CommandPool::Entry entry;
        const D3D12_COMMAND_LIST_TYPE listType = queues_[(*pool)->queue]->listType;
        if (const HRESULT result = device_->CreateCommandAllocator(listType, IID_PPV_ARGS(&entry.allocator));
            FAILED(result)) {
            return std::unexpected(deviceFailed("ID3D12Device::CreateCommandAllocator", result).error());
        }
        if (const HRESULT result = device_->CreateCommandList(0, listType, entry.allocator.Get(), nullptr,
                                                              IID_PPV_ARGS(&entry.list));
            FAILED(result)) {
            return std::unexpected(deviceFailed("ID3D12Device::CreateCommandList", result).error());
        }
        (*pool)->entries.push_back(std::move(entry));
    }
    else {
        const D3D12CommandPool::Entry& entry = (*pool)->entries[(*pool)->used];
        if (const HRESULT result = entry.list->Reset(entry.allocator.Get(), nullptr); FAILED(result)) {
            return std::unexpected(deviceFailed("ID3D12GraphicsCommandList::Reset", result).error());
        }
    }
    const ComPtr<ID3D12GraphicsCommandList>& list = (*pool)->entries[(*pool)->used].list;
    const CommandListHandle commandList = commandLists_.emplace(D3D12CommandList{
        .list = list, .pool = commandPool, .rendering = {}, .storeOp = StoreOp::Store, .rendered = {}});
    if (commandList.isNull()) {
        discard(list.Get());
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command lists", kCommandListLimit));
    }
    ++(*pool)->used;
    (*pool)->recording.push_back(commandList);
    return commandList;
}

core::Result<void> D3D12Device::beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                               const RenderingDesc& desc) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto chain = swapchainFor(swapchain);
    if (!chain) {
        return std::unexpected(std::move(chain.error()));
    }
    if ((*list)->rendering.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is already open");
    }
    if (const D3D12CommandPool* pool = commandPools_.find((*list)->pool);
        pool != nullptr && queues_[pool->queue]->type != QueueType::Graphics) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the list records for queue {}, which does not render", pool->queue));
    }
    if ((*chain)->submitted()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the acquired image was already submitted; present it first");
    }
    ID3D12Resource* image = (*chain)->image();
    if (image == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }

    ID3D12GraphicsCommandList* recording = (*list)->list.Get();
    const D3D12_RESOURCE_BARRIER toRender = transition(image, D3D12_RESOURCE_STATE_PRESENT,
                                                       D3D12_RESOURCE_STATE_RENDER_TARGET);
    recording->ResourceBarrier(1, &toRender);
    const D3D12_CPU_DESCRIPTOR_HANDLE view = (*chain)->view();
    recording->OMSetRenderTargets(1, &view, FALSE, nullptr);
    if (desc.loadOp == LoadOp::Clear) {
        recording->ClearRenderTargetView(view, desc.clearColor.data(), 0, nullptr);
    }
    (*list)->rendering = swapchain;
    (*list)->storeOp = desc.storeOp;
    if (std::ranges::find((*list)->rendered, swapchain) == (*list)->rendered.end()) {
        (*list)->rendered.push_back(swapchain);
    }
    return {};
}

core::Result<void> D3D12Device::endRendering(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const std::optional<SwapchainHandle> open = (*list)->rendering;
    if (!open.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no rendering is open");
    }
    auto chain = swapchainFor(*open);
    if (!chain) {
        return std::unexpected(std::move(chain.error()));
    }
    ID3D12Resource* image = (*chain)->image();
    ID3D12GraphicsCommandList* recording = (*list)->list.Get();
    if ((*list)->storeOp == StoreOp::DontCare) {
        recording->DiscardResource(image, nullptr);
    }
    const D3D12_RESOURCE_BARRIER toPresent = transition(image, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                        D3D12_RESOURCE_STATE_PRESENT);
    recording->ResourceBarrier(1, &toPresent);
    (*list)->rendering.reset();
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
    std::vector<D3D12Swapchain*> chains;
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
        }
        if ((*list)->rendering.has_value()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
        }
        if (const D3D12CommandPool* pool = commandPools_.find((*list)->pool);
            pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 std::format("the list was recorded for queue {}, not {}", pool->queue, desc.queue));
        }
        for (const SwapchainHandle swapchain : (*list)->rendered) {
            auto chain = swapchainFor(swapchain);
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
            if (std::ranges::find(chains, *chain) == chains.end()) {
                chains.push_back(*chain);
            }
        }
        queue.lists.push_back((*list)->list.Get());
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    core::Result<void> closed;
    for (const CommandListHandle commandList : desc.commandLists) {
        D3D12CommandList* list = commandLists_.find(commandList);
        if (list == nullptr) {
            continue;
        }
        if (D3D12CommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
            std::erase(pool->recording, commandList);
            pool->lastSubmit = value;
        }
        if (const HRESULT result = list->list->Close(); FAILED(result) && closed) {
            closed = deviceFailed("ID3D12GraphicsCommandList::Close", result);
        }
        commandLists_.erase(commandList);
    }
    if (!closed) {
        return std::unexpected(std::move(closed.error()));
    }
    for (const Token wait : desc.waits) {
        if (const HRESULT result = queue.queue->Wait(queues_[wait.queue]->fence.Get(), wait.value); FAILED(result)) {
            return std::unexpected(deviceFailed("ID3D12CommandQueue::Wait", result).error());
        }
    }
    if (!queue.lists.empty()) {
        queue.queue->ExecuteCommandLists(static_cast<UINT>(queue.lists.size()), queue.lists.data());
    }
    if (const HRESULT result = queue.queue->Signal(queue.fence.Get(), value); FAILED(result)) {
        return std::unexpected(deviceFailed("ID3D12CommandQueue::Signal", result).error());
    }
    queue.submitted.store(value, std::memory_order_release);
    for (D3D12Swapchain* chain : chains) {
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
    discard((*list)->list.Get());
    if (D3D12CommandPool* pool = commandPools_.find((*list)->pool); pool != nullptr) {
        std::erase(pool->recording, commandList);
    }
    commandLists_.erase(commandList);
    return {};
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
