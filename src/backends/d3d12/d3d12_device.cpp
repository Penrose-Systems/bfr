#include "d3d12/d3d12_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"
#include "d3d12/d3d12_result.h"
#include "d3d12/d3d12_rhi.h"
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

core::Result<std::unique_ptr<D3D12Device>> D3D12Device::create(D3D12Rhi& rhi, ComPtr<ID3D12Device> device,
                                                               ComPtr<IDXGIAdapter3> adapter, std::size_t heapCount) {
    auto made = std::make_unique<D3D12Device>(rhi, std::move(device), std::move(adapter), heapCount);
    if (auto opened = made->open(); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    return made;
}

D3D12Device::D3D12Device(D3D12Rhi& rhi, ComPtr<ID3D12Device> device, ComPtr<IDXGIAdapter3> adapter,
                         std::size_t heapCount)
    : rhi_(&rhi),
      device_(std::move(device)),
      adapter_(std::move(adapter)),
      heapCount_(heapCount),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandLists_("command lists", kFramesInFlight, kFramesInFlight, rhi.logger()) {}

core::Result<void> D3D12Device::open() {
    const D3D12_COMMAND_QUEUE_DESC queueDesc{.Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             .Priority = 0,
                                             .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
                                             .NodeMask = 0};
    if (const HRESULT result = device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)); FAILED(result)) {
        return deviceFailed("ID3D12Device::CreateCommandQueue", result);
    }
    if (const HRESULT result = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)); FAILED(result)) {
        return deviceFailed("ID3D12Device::CreateFence", result);
    }
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event_ == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, std::format("CreateEventW: error {}", GetLastError()));
    }
    for (Frame& frame : frames_) {
        if (const HRESULT result = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                                   IID_PPV_ARGS(&frame.allocator));
            FAILED(result)) {
            return deviceFailed("ID3D12Device::CreateCommandAllocator", result);
        }
        if (const HRESULT result = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frame.allocator.Get(),
                                                              nullptr, IID_PPV_ARGS(&frame.list));
            FAILED(result)) {
            return deviceFailed("ID3D12Device::CreateCommandList", result);
        }
        // A list is made recording; one waits closed for beginCommands.
        if (const HRESULT result = frame.list->Close(); FAILED(result)) {
            return deviceFailed("ID3D12GraphicsCommandList::Close", result);
        }
    }
    return {};
}

D3D12Device::~D3D12Device() {
    // A failed wait leaves nothing to do but release, and no result to report
    // it through, so it is sent as an error whatever BFR_LOGGING says.
    if (auto idle = waitIdle(); !idle) {
        rhi_->logger().send(core::Severity::Error,
                            std::format("{}; destroying the device anyway", idle.error().message));
    }
    // Swapchains and command lists are the device's children, so they go first.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(rhi_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    swapchains_.clear();
    // A list left recording is closed before it goes.
    for (Frame& frame : frames_) {
        if (frame.recording) {
            discard(frame);
        }
    }
    commandLists_.clear();
    if (event_ != nullptr) {
        CloseHandle(event_);
    }
}

core::Result<void> D3D12Device::waitFor(std::uint64_t value) {
    if (fence_->GetCompletedValue() >= value) {
        return {};
    }
    if (const HRESULT result = fence_->SetEventOnCompletion(value, event_); FAILED(result)) {
        return deviceFailed("ID3D12Fence::SetEventOnCompletion", result);
    }
    if (WaitForSingleObject(event_, INFINITE) != WAIT_OBJECT_0) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "waiting for the GPU failed");
    }
    return {};
}

core::Result<void> D3D12Device::waitFrame(Frame& frame) {
    if (!frame.pending) {
        return {};
    }
    if (auto waited = waitFor(frame.doneValue); !waited) {
        return waited;
    }
    frame.pending = false;
    return {};
}

core::Result<void> D3D12Device::waitIdle() {
    const std::scoped_lock lock{lock_};
    if (submitted_ == 0) {
        return {};
    }
    return waitFor(submitted_);
}

void D3D12Device::discard(Frame& frame) {
    if (const HRESULT result = frame.list->Close(); FAILED(result)) {
        BFR_LOG_WARN(rhi_->logger(), "{}",
                     D3D12Failure{.call = "ID3D12GraphicsCommandList::Close", .result = result}.message());
    }
}

D3D12Device::Frame& D3D12Device::frameAt(std::uint32_t index) {
    return std::span{frames_}[index];
}

core::Result<D3D12Swapchain*> D3D12Device::swapchainFor(SwapchainHandle swapchain) {
    D3D12Swapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
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
    const SwapchainHandle swapchain = swapchains_.emplace(device_, ComPtr<IDXGIFactory2>{rhi_->factory()}, queue_,
                                                          window, request, rhi_->logger());
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
    const std::scoped_lock lock{lock_};
    return (*found)->present();
}

core::Result<CommandListHandle> D3D12Device::beginCommands() {
    const std::scoped_lock lock{lock_};
    // The oldest frame, or the next one when that is still recording.
    std::uint32_t index = nextFrame_;
    while (frameAt(index).recording) {
        index = (index + 1) % kFramesInFlight;
        if (index == nextFrame_) {
            return core::failure(core::ErrorCode::RhiLimitReached,
                                 std::format("{} command lists are already recording; submit or destroy one",
                                             kFramesInFlight));
        }
    }
    Frame& frame = frameAt(index);
    if (auto waited = waitFrame(frame); !waited) {
        return std::unexpected(std::move(waited.error()));
    }
    if (const HRESULT result = frame.allocator->Reset(); FAILED(result)) {
        return std::unexpected(deviceFailed("ID3D12CommandAllocator::Reset", result).error());
    }
    if (const HRESULT result = frame.list->Reset(frame.allocator.Get(), nullptr); FAILED(result)) {
        return std::unexpected(deviceFailed("ID3D12GraphicsCommandList::Reset", result).error());
    }
    const CommandListHandle commandList = commandLists_.emplace(
        D3D12CommandList{.frame = index, .rendering = {}, .storeOp = StoreOp::Store, .rendered = {}});
    if (commandList.isNull()) {
        discard(frame);
        return core::failure(core::ErrorCode::RhiLimitReached, "no command list is free");
    }
    frame.recording = true;
    nextFrame_ = (index + 1) % kFramesInFlight;
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
    if ((*chain)->submitted()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the acquired image was already submitted; present it first");
    }
    ID3D12Resource* image = (*chain)->image();
    if (image == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }

    ID3D12GraphicsCommandList* recording = frameAt((*list)->frame).list.Get();
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
    ID3D12GraphicsCommandList* recording = frameAt((*list)->frame).list.Get();
    if ((*list)->storeOp == StoreOp::DontCare) {
        recording->DiscardResource(image, nullptr);
    }
    const D3D12_RESOURCE_BARRIER toPresent = transition(image, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                        D3D12_RESOURCE_STATE_PRESENT);
    recording->ResourceBarrier(1, &toPresent);
    (*list)->rendering.reset();
    return {};
}

core::Result<void> D3D12Device::submit(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if ((*list)->rendering.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
    }
    const std::scoped_lock lock{lock_};
    // Each swapchain rendered to: its image is marked as sent.
    std::vector<D3D12Swapchain*> chains;
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
        chains.push_back(*chain);
    }
    Frame& frame = frameAt((*list)->frame);
    // The list is done either way.
    frame.recording = false;
    commandLists_.erase(commandList);
    if (const HRESULT result = frame.list->Close(); FAILED(result)) {
        return std::unexpected(deviceFailed("ID3D12GraphicsCommandList::Close", result).error());
    }
    const std::array<ID3D12CommandList*, 1> lists{frame.list.Get()};
    queue_->ExecuteCommandLists(static_cast<UINT>(lists.size()), lists.data());
    frame.doneValue = ++submitted_;
    frame.pending = true;
    for (D3D12Swapchain* chain : chains) {
        chain->setSubmitted();
    }
    if (const HRESULT result = queue_->Signal(fence_.Get(), submitted_); FAILED(result)) {
        return std::unexpected(deviceFailed("ID3D12CommandQueue::Signal", result).error());
    }
    return {};
}

core::Result<void> D3D12Device::destroyCommandList(CommandListHandle commandList) {
    if (commandList.isNull()) {
        return {};
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const std::scoped_lock lock{lock_};
    Frame& frame = frameAt((*list)->frame);
    discard(frame);
    frame.recording = false;
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
