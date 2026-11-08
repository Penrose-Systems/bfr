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
#include "d3d12/d3d12_caps.h"
#include "d3d12/d3d12_instance.h"
#include "d3d12/d3d12_pipeline.h"
#include "d3d12/d3d12_result.h"
#include "d3d12/d3d12_swapchain.h"
#include "rhi/clear_color.h"
#include "rhi/command_check.h"
#include "rhi/device_limits.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"
#include "rhi/timeout.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
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

// waitIdle waits with no limit, as vkDeviceWaitIdle does.
constexpr std::uint64_t kNoTimeLimit = std::numeric_limits<std::uint64_t>::max();

core::Result<void> deviceFailed(const char* call, HRESULT result) {
    return failure(core::ErrorCode::RhiDeviceFailed, {.call = call, .result = result});
}

// A barrier that moves image between the states a swapchain image is presented,
// rendered to and resolved into in.
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

// Mip 0xffffffff with no mip count names every subresource.
constexpr UINT kAllSubresources = 0xffffffff;
constexpr D3D12_BARRIER_SUBRESOURCE_RANGE kEverySubresource{.IndexOrFirstMipLevel = kAllSubresources,
                                                            .NumMipLevels = 0,
                                                            .FirstArraySlice = 0,
                                                            .NumArraySlices = 0,
                                                            .FirstPlane = 0,
                                                            .NumPlanes = 0};

// Mip 0 of layer 0, which a rendering's resolve covers.
constexpr UINT kFirstSubresource = 0;

// How a texture is used on one side of a barrier.
struct TextureUse {
    D3D12_BARRIER_SYNC sync;
    D3D12_BARRIER_ACCESS access;
    D3D12_BARRIER_LAYOUT layout;
};

constexpr TextureUse kRenderTargetUse{.sync = D3D12_BARRIER_SYNC_RENDER_TARGET,
                                      .access = D3D12_BARRIER_ACCESS_RENDER_TARGET,
                                      .layout = D3D12_BARRIER_LAYOUT_RENDER_TARGET};
constexpr TextureUse kResolveSourceUse{.sync = D3D12_BARRIER_SYNC_RESOLVE,
                                       .access = D3D12_BARRIER_ACCESS_RESOLVE_SOURCE,
                                       .layout = D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE};
constexpr TextureUse kResolveDestUse{.sync = D3D12_BARRIER_SYNC_RESOLVE,
                                     .access = D3D12_BARRIER_ACCESS_RESOLVE_DEST,
                                     .layout = D3D12_BARRIER_LAYOUT_RESOLVE_DEST};

D3D12_TEXTURE_BARRIER barrierFor(ID3D12Resource* resource, const TextureUse& before, const TextureUse& after) {
    return {.SyncBefore = before.sync,
            .SyncAfter = after.sync,
            .AccessBefore = before.access,
            .AccessAfter = after.access,
            .LayoutBefore = before.layout,
            .LayoutAfter = after.layout,
            .pResource = resource,
            .Subresources = kEverySubresource,
            .Flags = D3D12_TEXTURE_BARRIER_FLAG_NONE};
}

// A source and a target barrier for each color attachment.
constexpr std::size_t kResolveBarriers = 2 * kMaxColorTargets;

// Records each resolve, the first into a swapchain image when intoImage. The
// textures move from RENDER_TARGET to the resolve layouts and back by
// enhanced barriers; the image, which presents in legacy states, from PRESENT
// to RESOLVE_DEST and back. Godot resolves so as a subpass ends
// (rendering_device_driver_d3d12.cpp:4740-4782) but leaves the textures in
// the resolve layouts; BFR returns them to the layout the rendering was given
// them in.
void recordResolves(ID3D12GraphicsCommandList7* list, std::span<const D3D12Resolve> resolves, bool intoImage) {
    std::array<D3D12_TEXTURE_BARRIER, kResolveBarriers> toResolve{};
    std::array<D3D12_TEXTURE_BARRIER, kResolveBarriers> back{};
    UINT32 count = 0;
    for (std::size_t i = 0; i < resolves.size(); ++i) {
        const D3D12Resolve& resolve = resolves[i];
        toResolve.at(count) = barrierFor(resolve.source, kRenderTargetUse, kResolveSourceUse);
        back.at(count++) = barrierFor(resolve.source, kResolveSourceUse, kRenderTargetUse);
        if (i != 0 || !intoImage) {
            toResolve.at(count) = barrierFor(resolve.target, kRenderTargetUse, kResolveDestUse);
            back.at(count++) = barrierFor(resolve.target, kResolveDestUse, kRenderTargetUse);
        }
    }
    const D3D12_BARRIER_GROUP toGroup{.Type = D3D12_BARRIER_TYPE_TEXTURE,
                                      .NumBarriers = count,
                                      .pTextureBarriers = toResolve.data()};
    list->Barrier(1, &toGroup);
    if (intoImage) {
        const D3D12_RESOURCE_BARRIER toDest = transition(resolves.front().target, D3D12_RESOURCE_STATE_PRESENT,
                                                         D3D12_RESOURCE_STATE_RESOLVE_DEST);
        list->ResourceBarrier(1, &toDest);
    }
    for (const D3D12Resolve& resolve : resolves) {
        list->ResolveSubresource(resolve.target, kFirstSubresource, resolve.source, kFirstSubresource, resolve.format);
    }
    const D3D12_BARRIER_GROUP backGroup{.Type = D3D12_BARRIER_TYPE_TEXTURE,
                                        .NumBarriers = count,
                                        .pTextureBarriers = back.data()};
    list->Barrier(1, &backGroup);
    if (intoImage) {
        const D3D12_RESOURCE_BARRIER toPresent = transition(resolves.front().target, D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                                            D3D12_RESOURCE_STATE_PRESENT);
        list->ResourceBarrier(1, &toPresent);
    }
}

// D3D12's priority for level; nothing for one it has none for.
std::optional<INT> d3d12Priority(QueueLevel level) {
    switch (level) {
        case QueueLevel::Normal:
            return D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        case QueueLevel::High:
            return D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
        case QueueLevel::Realtime:
            return D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME;
        case QueueLevel::Low:
            return std::nullopt;
    }
    return std::nullopt;
}

// A command signature of one argument of type, stride bytes apart. It sets no
// root arguments, so it takes no root signature, as NVRHI's and Godot's do.
core::Result<ComPtr<ID3D12CommandSignature>> createSignature(ID3D12Device& device, D3D12_INDIRECT_ARGUMENT_TYPE type,
                                                             std::uint32_t stride) {
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = type;
    const D3D12_COMMAND_SIGNATURE_DESC desc{.ByteStride = stride,
                                            .NumArgumentDescs = 1,
                                            .pArgumentDescs = &argument,
                                            .NodeMask = 0};
    ComPtr<ID3D12CommandSignature> signature;
    if (const HRESULT result = device.CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&signature));
        FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12Device::CreateCommandSignature", .result = result});
    }
    return signature;
}

// BFR's indirect structs are D3D12's argument structs field for field
// (d3d12.h:4592-4614), which ExecuteIndirect reads.
static_assert(sizeof(DrawIndirectArgs) == sizeof(D3D12_DRAW_ARGUMENTS));
static_assert(sizeof(DrawIndexedIndirectArgs) == sizeof(D3D12_DRAW_INDEXED_ARGUMENTS));
static_assert(sizeof(DispatchIndirectArgs) == sizeof(D3D12_DISPATCH_ARGUMENTS));

} // namespace

core::Result<std::unique_ptr<D3D12Device>> D3D12Device::create(D3D12Instance& rhi, ComPtr<ID3D12Device> device,
                                                               ComPtr<IDXGIAdapter3> adapter, const Features& features,
                                                               const ResolveCaps& resolve, std::size_t heapCount,
                                                               std::uint32_t viewHeapSize, const DeviceDesc& desc) {
    auto made = std::make_unique<D3D12Device>(rhi, std::move(device), std::move(adapter), heapCount, desc);
    // A runtime without it answers E_NOINTERFACE and leaves device11_ null.
    static_cast<void>(made->device_.As(&made->device11_));
    static_cast<void>(made->device_.As(&made->device1_));
    made->sparse_ = sparseOf(made->device_.Get());
    made->multiDrawIndirect_ = features.multiDrawIndirect;
    made->drawIndirectCount_ = features.drawIndirectCount;
    made->features_ = features;
    made->resolve_ = resolve;
    if (auto opened = made->open(desc); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    auto allocator = createAllocator(made->device_.Get(), made->adapter_.Get());
    if (!allocator) {
        return std::unexpected(std::move(allocator.error()));
    }
    made->allocator_ = std::move(*allocator);
    auto layout = createD3D12BindlessLayout(*made->device_.Get(), desc, viewHeapSize);
    if (!layout) {
        return std::unexpected(std::move(layout.error()));
    }
    made->layout_ = std::move(*layout);
    for (std::size_t i = 0; i < desc.indirectStrides.size(); ++i) {
        // A kind's signature only where the stride holds its arguments, so a
        // packed stride of the smaller draw arguments makes no indexed one
        // (NVRHI makes each at its own size, d3d12-device.cpp:299-307);
        // Vulkan takes no smaller stride either. That D3D12 refuses one is
        // not stated in d3d12.h.
        const std::uint32_t stride = desc.indirectStrides[i];
        D3D12DrawSignatures& signatures = made->drawSignatures_[i];
        signatures.stride = stride;
        if (stride >= sizeof(D3D12_DRAW_ARGUMENTS)) {
            auto draw = createSignature(*made->device_.Get(), D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, stride);
            if (!draw) {
                return std::unexpected(std::move(draw.error()));
            }
            signatures.draw = std::move(*draw);
        }
        if (stride >= sizeof(D3D12_DRAW_INDEXED_ARGUMENTS)) {
            auto drawIndexed = createSignature(*made->device_.Get(), D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, stride);
            if (!drawIndexed) {
                return std::unexpected(std::move(drawIndexed.error()));
            }
            signatures.drawIndexed = std::move(*drawIndexed);
        }
    }
    auto dispatch = createSignature(*made->device_.Get(), D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                                    sizeof(D3D12_DISPATCH_ARGUMENTS));
    if (!dispatch) {
        return std::unexpected(std::move(dispatch.error()));
    }
    made->dispatchSignature_ = std::move(*dispatch);
    return made;
}

D3D12Device::D3D12Device(D3D12Instance& rhi, ComPtr<ID3D12Device> device, ComPtr<IDXGIAdapter3> adapter,
                         std::size_t heapCount, const DeviceDesc& desc)
    : instance_(&rhi),
      device_(std::move(device)),
      adapter_(std::move(adapter)),
      heapCount_(heapCount),
      desc_(heldPart(desc)),
      drawSignatures_(desc.indirectStrides.size()),
      swapchains_("swapchains", desc.maxSwapchains, desc.maxSwapchains, rhi.logger()),
      commandPools_("command pools", desc.maxCommandPools, desc.maxCommandPools, rhi.logger()),
      commandLists_("command lists", desc.maxCommandLists, desc.maxCommandLists, rhi.logger()),
      listSlots_(desc.maxCommandLists),
      lists_(desc.maxCommandLists),
      renderSlots_(desc.maxCommandLists),
      renders_(desc.maxCommandLists),
      submissions_(queueCount(desc)),
      buffers_("buffers", kBufferCapacity, kBufferLimit, rhi.logger()),
      textures_("textures", kBufferCapacity, kBufferLimit, rhi.logger()),
      samplers_("samplers", desc.maxBindlessSamplers, desc.maxBindlessSamplers, rhi.logger()),
      timestampPools_("timestamp pools", desc.maxTimestampPools, desc.maxTimestampPools, rhi.logger()),
      sparseMemories_("sparse memories", desc.maxSparseMemories, desc.maxSparseMemories, rhi.logger()),
      pipelineCaches_("pipeline caches", kBufferCapacity, kBufferLimit, rhi.logger()),
      bindless_(desc),
      pipelines_("pipelines", kBufferCapacity, kBufferLimit, rhi.logger()) {}

core::Result<void> D3D12Device::open(const DeviceDesc& desc) {
    const auto open = [this, &desc](QueueType type, D3D12_COMMAND_LIST_TYPE listType,
                                    const QueueDesc& wanted) -> core::Result<void> {
        const auto priority = d3d12Priority(wanted.level);
        if (!priority) {
            return core::failure(core::ErrorCode::RhiUnsupported, "D3D12 has no queue level {}",
                                 std::to_underlying(wanted.level));
        }
        auto queue = std::make_unique<D3D12Queue>();
        queue->type = type;
        queue->listType = listType;
        // A submit's arrays, made now at the device's maximums. Every pool
        // may be destroyed before the GPU is done with it.
        queue->lists = core::FixedVector<ID3D12CommandList*>{desc.maxCommandLists};
        queue->chains = core::FixedVector<D3D12Swapchain*>{desc.maxSwapchains};
        queue->retired = core::FixedVector<D3D12RetiredPool>{desc.maxCommandPools};
        queue->textureTiles = D3D12TileMappings{desc.maxTileMappings};
        queue->bufferTiles = D3D12TileMappings{desc.maxTileMappings};
        const D3D12_COMMAND_QUEUE_DESC queueDesc{.Type = listType,
                                                 .Priority = *priority,
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
    for (const QueueDesc& queue : desc.graphics) {
        if (auto opened = open(QueueType::Graphics, D3D12_COMMAND_LIST_TYPE_DIRECT, queue); !opened) {
            return opened;
        }
    }
    for (const QueueDesc& queue : desc.compute) {
        if (auto opened = open(QueueType::Compute, D3D12_COMMAND_LIST_TYPE_COMPUTE, queue); !opened) {
            return opened;
        }
    }
    for (const QueueDesc& queue : desc.copy) {
        if (auto opened = open(QueueType::Copy, D3D12_COMMAND_LIST_TYPE_COPY, queue); !opened) {
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
    if (samplers_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} samplers were left when the device went", samplers_.size());
    }
    if (timestampPools_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} timestamp pools were left when the device went", timestampPools_.size());
    }
    if (pipelines_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} pipelines were left when the device went", pipelines_.size());
    }
    if (sparseMemories_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} sparse memories were left when the device went", sparseMemories_.size());
    }
    if (pipelineCaches_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} pipeline caches were left when the device went", pipelineCaches_.size());
    }
    // Every buffer and texture gives its memory back before the allocator goes,
    // and every pipeline goes before the layout it was made on.
    buffers_.clear();
    textures_.clear();
    sparseMemories_.clear();
    samplers_.clear();
    timestampPools_.clear();
    pipelines_.clear();
    pipelineCaches_.clear();
    layout_ = {};
    allocator_.Reset();
    // The slots' vector releases every command allocator and list as it goes.
    for (const auto& queue : queues_) {
        if (queue->event != nullptr) {
            CloseHandle(queue->event);
        }
    }
}

core::Result<void> D3D12Device::waitFor(D3D12Queue& queue, std::uint64_t value, std::uint64_t timeoutNs) {
    if (queue.fence->GetCompletedValue() >= value) {
        return {};
    }
    const std::scoped_lock lock{queue.waitLock};
    const auto start = std::chrono::steady_clock::now();
    // An earlier wait that ran out of time leaves its fence value set on the
    // event, which may wake this one early, so each wake asks the fence again.
    while (queue.fence->GetCompletedValue() < value) {
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        if (elapsed >= timeoutNs) {
            return core::failure(core::ErrorCode::RhiTimeout, "the GPU had not passed the token in time");
        }
        if (const HRESULT result = queue.fence->SetEventOnCompletion(value, queue.event); FAILED(result)) {
            return deviceFailed("ID3D12Fence::SetEventOnCompletion", result);
        }
        // WaitForSingleObject's longest count is INFINITE - 1 milliseconds;
        // INFINITE itself waits with no limit.
        const std::uint64_t remainingMs = millisecondsRoundedUp(timeoutNs - elapsed);
        const DWORD waitMs = remainingMs >= INFINITE ? INFINITE : static_cast<DWORD>(remainingMs);
        if (const DWORD waited = WaitForSingleObject(queue.event, waitMs);
            waited != WAIT_OBJECT_0 && waited != WAIT_TIMEOUT) {
            return core::failure(core::ErrorCode::RhiDeviceFailed, "waiting for the GPU failed");
        }
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
    return {};
}

core::Result<void> D3D12Device::waitOn(D3D12Queue& queue, std::span<const Token> waits) {
    for (const Token wait : waits) {
        if (const HRESULT result = queue.queue->Wait(queues_[wait.queue]->fence.Get(), wait.value); FAILED(result)) {
            return deviceFailed("ID3D12CommandQueue::Wait", result);
        }
    }
    return {};
}

core::Result<void> D3D12Device::signal(D3D12Queue& queue, std::uint64_t value) {
    if (const HRESULT result = queue.queue->Signal(queue.fence.Get(), value); FAILED(result)) {
        return deviceFailed("ID3D12CommandQueue::Signal", result);
    }
    queue.submitted.store(value, std::memory_order_release);
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
        if (auto waited = waitFor(*queue, queue->submitted.load(std::memory_order_acquire), kNoTimeLimit); !waited) {
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
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Only direct queues present (Godot's rendering_device_driver_d3d12.cpp, command_queue_family_get).
    if (queues_[desc.queue]->listType != D3D12_COMMAND_LIST_TYPE_DIRECT) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "queue {} is not a graphics queue, and only those present on D3D12", desc.queue);
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, ComPtr<IDXGIFactory2>{instance_->factory()},
                                                          queues_[desc.queue]->queue, window, desc,
                                                          instance_->logger());
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

// GetCurrentBackBufferIndex doesn't wait, so there is no time to limit.
core::Result<SwapchainStatus> D3D12Device::acquire(SwapchainHandle swapchain, std::uint64_t /*timeoutNs*/) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->acquire();
}

core::Result<SwapchainStatus> D3D12Device::present(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // Present runs on the swapchain's queue, which may be submitted to from
    // another thread at once. Rendering submitted to another queue is waited
    // for there first, as D3D12 orders only one queue's work.
    D3D12Queue& queue = *queues_[(*found)->queue()];
    const std::scoped_lock lock{queue.lock};
    if (const Token rendered = (*found)->rendered(); (*found)->submitted() && rendered.queue != (*found)->queue()) {
        if (const HRESULT result = queue.queue->Wait(queues_[rendered.queue]->fence.Get(), rendered.value);
            FAILED(result)) {
            return std::unexpected(deviceFailed("ID3D12CommandQueue::Wait", result).error());
        }
    }
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
                                                                                 .presenting = {},
                                                                                 .presentStoreOp = StoreOp::Store,
                                                                                 .resolves = {},
                                                                                 .resolveCount = 0,
                                                                                 .resolvesIntoImage = false,
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
    std::array<std::array<FLOAT, std::tuple_size_v<ClearValues>>, kMaxColorTargets> clears{};
    std::array<D3D12Resolve, kMaxColorTargets> resolves{};
    std::uint32_t resolveCount = 0;
    bool resolvesIntoImage = false;
    std::array<D3D12Target, kMaxColorTargets + 1> discards{};
    std::uint32_t discardCount = 0;
    // The attachments whose contents are not loaded, discarded as rendering
    // begins, as the store's are as it ends.
    std::array<D3D12Target, kMaxColorTargets + 1> unloaded{};
    std::uint32_t unloadedCount = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    D3D12Swapchain* chain = nullptr;
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        const ColorAttachment& attachment = desc.colorAttachments.at(i);
        D3D12Target target;
        // A swapchain's image is never a signed integer format.
        Format format = Format::Undefined;
        // The swapchain's image is attachment 0, or what a texture there
        // resolves into.
        ID3D12Resource* resolveTarget = nullptr;
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
            ID3D12Resource* image = chain->image();
            if (image == nullptr) {
                return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
            }
            if (attachment.texture.isNull()) {
                target.resource = image;
                colors.at(i) = chain->view();
                const D3D12_RESOURCE_DESC imageDesc = image->GetDesc();
                width = static_cast<std::uint32_t>(imageDesc.Width);
                height = imageDesc.Height;
            }
            else {
                resolveTarget = image;
                resolvesIntoImage = true;
            }
        }
        if (!attachment.texture.isNull()) {
            auto texture = textureFor(attachment.texture);
            if (!texture) {
                return std::unexpected(std::move(texture.error()));
            }
            target = targetOf(**texture);
            colors.at(i) = (*texture)->target;
            width = (*texture)->desc.width;
            height = (*texture)->desc.height;
            format = (*texture)->desc.format;
            if (!attachment.resolveTexture.isNull()) {
                auto resolved = textureFor(attachment.resolveTexture);
                if (!resolved) {
                    return std::unexpected(std::move(resolved.error()));
                }
                resolveTarget = (*resolved)->resource.Get();
            }
            if (resolveTarget != nullptr) {
                resolves.at(resolveCount++) = {.source = (*texture)->resource.Get(),
                                               .target = resolveTarget,
                                               .format = (*texture)->dxgi};
            }
        }
        // D3D12 clears with floats, which hold every value clearColor does
        // but not every integer.
        const ClearValues values = clearValues(attachment, format);
        for (std::size_t c = 0; c < values.size(); ++c) {
            clears.at(i).at(c) = static_cast<FLOAT>(values.at(c));
            if (attachment.clearInteger && static_cast<double>(clears.at(i).at(c)) != values.at(c)) {
                return core::failure(core::ErrorCode::RhiInvalidCall,
                                     "color attachment {} clears to {}, which D3D12's float clear does not hold", i,
                                     values.at(c));
            }
        }
        if (attachment.loadOp == LoadOp::DontCare) {
            unloaded.at(unloadedCount++) = target;
        }
        if (attachment.storeOp == StoreOp::DontCare) {
            discards.at(discardCount++) = target;
        }
    }
    const DepthStencilAttachment& depthStencil = desc.depthStencil;
    const D3D12Texture* depth = nullptr;
    if (!depthStencil.texture.isNull()) {
        auto texture = textureFor(depthStencil.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        depth = *texture;
        width = depth->desc.width;
        height = depth->desc.height;
        // ResolveCaps has no mode, so only None passes, which resolves
        // nothing.
        if (!depthStencil.resolveTexture.isNull()) {
            if (auto checked = checkDepthResolve(depthStencil, depth->desc.format, resolve_); !checked) {
                return checked;
            }
            if (auto resolved = textureFor(depthStencil.resolveTexture); !resolved) {
                return std::unexpected(std::move(resolved.error()));
            }
        }
        // The aspects an op drops, of those the format has, as one target.
        const D3D12Target notLoaded = aspectsOf(*depth, depthStencil.depthLoadOp == LoadOp::DontCare,
                                                depthStencil.stencilLoadOp == LoadOp::DontCare);
        if (notLoaded.planeCount != 0) {
            unloaded.at(unloadedCount++) = notLoaded;
        }
        const D3D12Target notStored = aspectsOf(*depth, depthStencil.depthStoreOp == StoreOp::DontCare,
                                                depthStencil.stencilStoreOp == StoreOp::DontCare);
        if (notStored.planeCount != 0) {
            discards.at(discardCount++) = notStored;
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
    // An image resolved into is moved as rendering ends.
    if (chain != nullptr && !resolvesIntoImage) {
        const D3D12_RESOURCE_BARRIER toRender = transition(chain->image(), D3D12_RESOURCE_STATE_PRESENT,
                                                           D3D12_RESOURCE_STATE_RENDER_TARGET);
        recording->ResourceBarrier(1, &toRender);
    }
    for (std::uint32_t i = 0; i < unloadedCount; ++i) {
        discardTarget(recording, unloaded.at(i));
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE* depthView = nullptr;
    if (depth != nullptr) {
        depthView = depthStencil.readOnly ? &depth->readOnlyTarget : &depth->target;
    }
    recording->OMSetRenderTargets(desc.colorAttachmentCount, colors.data(), FALSE, depthView);
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        if (desc.colorAttachments.at(i).loadOp == LoadOp::Clear) {
            recording->ClearRenderTargetView(colors.at(i), clears.at(i).data(), 0, nullptr);
        }
    }
    if (depth != nullptr) {
        const FormatBlock& block = blockOf(depth->desc.format);
        D3D12_CLEAR_FLAGS flags{};
        if (block.depthBytes != 0 && depthStencil.depthLoadOp == LoadOp::Clear) {
            flags |= D3D12_CLEAR_FLAG_DEPTH;
        }
        if (block.stencilBytes != 0 && depthStencil.stencilLoadOp == LoadOp::Clear) {
            flags |= D3D12_CLEAR_FLAG_STENCIL;
        }
        if (flags != D3D12_CLEAR_FLAGS{}) {
            recording->ClearDepthStencilView(depth->target, flags, depthStencil.clearDepth,
                                             static_cast<UINT8>(depthStencil.clearStencil), 0, nullptr);
        }
    }
    (*list)->presenting = chain != nullptr ? std::optional{swapchain} : std::nullopt;
    // An image resolved into keeps the resolve; attachment 0's store op is
    // its multisampled texture's.
    (*list)->presentStoreOp = chain != nullptr && !resolvesIntoImage ? desc.colorAttachments.at(0).storeOp
                                                                     : StoreOp::Store;
    (*list)->resolves = resolves;
    (*list)->resolveCount = resolveCount;
    (*list)->resolvesIntoImage = resolvesIntoImage;
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
    D3D12Swapchain* chain = nullptr;
    if (const std::optional<SwapchainHandle> open = (*list)->presenting; open.has_value()) {
        auto found = swapchainFor(*open);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        chain = *found;
    }
    ID3D12GraphicsCommandList7* recording = (*list)->list;
    // The resolves read what the discards then drop.
    if (const std::span resolves{(*list)->resolves.data(), (*list)->resolveCount}; !resolves.empty()) {
        recordResolves(recording, resolves, (*list)->resolvesIntoImage);
    }
    // What the caller does not keep is discarded, the swapchain's image among
    // it before it moves to present.
    for (std::uint32_t i = 0; i < (*list)->discardCount; ++i) {
        discardTarget(recording, (*list)->discards.at(i));
    }
    if (chain != nullptr && !(*list)->resolvesIntoImage) {
        const D3D12_RESOURCE_BARRIER toPresent = transition(chain->image(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                            D3D12_RESOURCE_STATE_PRESENT);
        recording->ResourceBarrier(1, &toPresent);
    }
    (*list)->presenting.reset();
    (*list)->resolveCount = 0;
    (*list)->resolvesIntoImage = false;
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
        // The one layout whose accesses take read-only depth and shader reads
        // together; DEPTH_STENCIL_READ takes depth reads alone (Enhanced
        // Barriers spec, "Layout Access Compatibility"). It is a direct
        // queue's, as rendering is.
        case TextureLayout::DepthStencilRead:
            return D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ;
        case TextureLayout::ResolveSrc:
            return D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE;
        case TextureLayout::ResolveDst:
            return D3D12_BARRIER_LAYOUT_RESOLVE_DEST;
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
        // Shaders may sample a read-only attachment while it is tested.
        case TextureLayout::DepthStencilRead:
            return D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ | D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
        case TextureLayout::ResolveSrc:
            return D3D12_BARRIER_ACCESS_RESOLVE_SOURCE;
        case TextureLayout::ResolveDst:
            return D3D12_BARRIER_ACCESS_RESOLVE_DEST;
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
    if (auto checked = checkLayouts(before, after); !checked) {
        return checked;
    }
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
    // It moves every subresource; moving from UNDEFINED discards, which a
    // placed render target or depth texture needs before its first use.
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
                                      .Subresources = kEverySubresource,
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
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
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
    if (auto waited = waitOn(queue, desc.waits); !waited) {
        finish(false);
        return std::unexpected(std::move(waited.error()));
    }
    if (!queue.lists.empty()) {
        queue.queue->ExecuteCommandLists(static_cast<UINT>(queue.lists.size()), queue.lists.data());
    }
    if (auto signalled = signal(queue, value); !signalled) {
        finish(false);
        return std::unexpected(std::move(signalled.error()));
    }
    finish(true);
    for (D3D12Swapchain* chain : queue.chains) {
        chain->setSubmitted({.queue = desc.queue, .value = value});
    }
    collectPools(queue);
    return Token{.queue = desc.queue, .value = value};
}

core::Result<void> D3D12Device::wait(Token token, std::uint64_t timeoutNs) {
    if (auto checked = checkToken(token); !checked) {
        return checked;
    }
    D3D12Queue& queue = *queues_[token.queue];
    if (auto waited = waitFor(queue, token.value, timeoutNs); !waited) {
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
