#include "metal/metal_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"
#include "core/fixed_vector.h"
#include "core/slot_chains.h"
#include "metal/metal_instance.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_util.h"

#include <Foundation/NSError.hpp>
#include <Metal/MTL4CommandAllocator.hpp>
#include <Metal/MTL4CommandEncoder.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandEncoder.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLEvent.hpp>
#include <Metal/MTLRenderPass.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace bfr::rhi::metal {

namespace {

constexpr std::uint64_t kForever = std::numeric_limits<std::uint64_t>::max();

MTL::LoadAction loadActionFor(LoadOp op) {
    switch (op) {
        case LoadOp::Load:
            return MTL::LoadActionLoad;
        case LoadOp::Clear:
            return MTL::LoadActionClear;
        case LoadOp::DontCare:
            return MTL::LoadActionDontCare;
    }
    return MTL::LoadActionDontCare;
}

MTL::StoreAction storeActionFor(StoreOp op) {
    switch (op) {
        case StoreOp::Store:
            return MTL::StoreActionStore;
        case StoreOp::DontCare:
            return MTL::StoreActionDontCare;
    }
    return MTL::StoreActionDontCare;
}

core::Result<void> deviceFailed(const char* what) {
    return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no {}", what);
}

} // namespace

core::Result<std::unique_ptr<MetalDevice>> MetalDevice::create(MetalInstance& rhi, Ref<MTL::Device> device,
                                                               const DeviceDesc& desc) {
    auto made = std::make_unique<MetalDevice>(rhi, std::move(device), desc);
    if (auto opened = made->open(desc); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    return made;
}

MetalDevice::MetalDevice(MetalInstance& rhi, Ref<MTL::Device> device, const DeviceDesc& desc)
    : instance_(&rhi),
      device_(std::move(device)),
      desc_(desc),
      swapchains_("swapchains", desc.maxSwapchains, desc.maxSwapchains, rhi.logger()),
      commandPools_("command pools", desc.maxCommandPools, desc.maxCommandPools, rhi.logger()),
      commandLists_("command lists", desc.maxCommandLists, desc.maxCommandLists, rhi.logger()),
      listSlots_(desc.maxCommandLists),
      lists_(desc.maxCommandLists),
      renderSlots_(desc.maxCommandLists),
      renders_(desc.maxCommandLists),
      submissions_(1 + desc.compute + desc.copy),
      buffers_("buffers", kBufferCapacity, kBufferLimit, rhi.logger()) {}

core::Result<void> MetalDevice::open(const DeviceDesc& desc) {
    const AutoreleasePool pool;
    const Ref<MTL::ResidencySetDescriptor> residency = adopt(MTL::ResidencySetDescriptor::alloc()->init());
    NS::Error* error = nullptr;
    MTL::ResidencySet* set = device_->newResidencySet(residency.get(), &error);
    if (set == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no residency set: {}",
                             error != nullptr ? error->localizedDescription()->utf8String() : "no reason given");
    }
    residency_ = adopt(set);
    const auto open = [this, &desc](QueueType type) -> core::Result<void> {
        auto queue = std::make_unique<MetalQueue>();
        queue->type = type;
        // A submit's arrays, made now at the device's maximums. Every pool
        // may be destroyed before the GPU is done with it.
        queue->buffers = core::FixedVector<const MTL4::CommandBuffer*>{desc.maxCommandLists};
        queue->chains = core::FixedVector<MetalSwapchain*>{desc.maxSwapchains};
        queue->retired = core::FixedVector<MetalRetiredPool>{desc.maxCommandPools};
        MTL4::CommandQueue* native = device_->newMTL4CommandQueue();
        if (native == nullptr) {
            return deviceFailed("command queue");
        }
        queue->queue = adopt(native);
        queue->queue->addResidencySet(residency_.get());
        MTL::SharedEvent* event = device_->newSharedEvent();
        if (event == nullptr) {
            return deviceFailed("event");
        }
        queue->event = adopt(event);
        queues_.push_back(std::move(queue));
        return {};
    };
    if (auto opened = open(QueueType::Graphics); !opened) {
        return opened;
    }
    for (std::uint32_t index = 0; index < desc.compute; ++index) {
        if (auto opened = open(QueueType::Compute); !opened) {
            return opened;
        }
    }
    for (std::uint32_t index = 0; index < desc.copy; ++index) {
        if (auto opened = open(QueueType::Copy); !opened) {
            return opened;
        }
    }
    return {};
}

MetalDevice::~MetalDevice() {
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
    // A list left recording has an open encoder and buffer, which Metal wants
    // ended before they go.
    commandLists_.forEach([](MetalCommandList& list) { discard(list); });
    commandLists_.clear();
    commandPools_.clear();
    if (buffers_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} buffers were left when the device went", buffers_.size());
    }
    buffers_.clear();
    // The slots' vector releases every allocator and buffer as it goes.
}

core::Result<void> MetalDevice::checkQueue(std::uint32_t queue) const {
    if (queue >= queues_.size()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the device has no queue {}; it has {}", queue,
                             queues_.size());
    }
    return {};
}

core::Result<void> MetalDevice::checkToken(Token token) const {
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

void MetalDevice::collectPools(MetalQueue& queue) {
    if (queue.retired.empty()) {
        return;
    }
    const std::uint64_t completed = queue.event->signaledValue();
    queue.retired.eraseIf([this, completed](const MetalRetiredPool& retired) {
        if (retired.value > completed) {
            return false;
        }
        releaseSlots(retired.first);
        poolsHeld_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    });
    queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
}

void MetalDevice::releaseSlots(std::uint32_t first) {
    for (std::uint32_t slot = first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        lists_[slot].buffer.reset();
        lists_[slot].allocator.reset();
    }
    listSlots_.give(first);
}

core::Result<void> MetalDevice::waitIdle() {
    for (const auto& queue : queues_) {
        const std::uint64_t submitted = queue->submitted.load(std::memory_order_acquire);
        if (submitted != 0 && !queue->event->waitUntilSignaledValue(submitted, kForever)) {
            return core::failure(core::ErrorCode::RhiDeviceFailed, "waiting for the GPU timed out");
        }
    }
    return {};
}

void MetalDevice::discard(MetalCommandList& list) {
    if (list.encoder) {
        list.encoder->endEncoding();
        list.encoder.reset();
    }
    list.buffer->endCommandBuffer();
}

core::Result<MetalSwapchain*> MetalDevice::swapchainFor(SwapchainHandle swapchain) {
    MetalSwapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
    }
    return found;
}

core::Result<MetalCommandPool*> MetalDevice::commandPoolFor(CommandPoolHandle commandPool) {
    MetalCommandPool* found = commandPools_.find(commandPool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command pool handle is stale");
    }
    return found;
}

core::Result<MetalCommandList*> MetalDevice::commandListFor(CommandListHandle commandList) {
    MetalCommandList* found = commandLists_.find(commandList);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command list handle is stale");
    }
    return found;
}

core::Result<SwapchainHandle> MetalDevice::createSwapchain(const core::window::NativeWindow& window,
                                                           const SwapchainDesc& desc) {
    auto layer = layerFor(window);
    if (!layer) {
        return std::unexpected(std::move(layer.error()));
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, *layer, desc, instance_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "a device holds at most {} swapchains",
                             desc_.maxSwapchains);
    }
    // The layer's drawables are resident for the queue that renders to them.
    if (MTL::ResidencySet* residency = (*layer)->residencySet(); residency != nullptr) {
        const std::scoped_lock lock{queues_[0]->lock};
        queues_[0]->queue->addResidencySet(residency);
    }
    return swapchain;
}

core::Result<SwapchainImages> MetalDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                           std::uint32_t height) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // The old drawables may still be on the GPU or with the presentation engine.
    if (auto idle = waitIdle(); !idle) {
        return std::unexpected(std::move(idle.error()));
    }
    return (*found)->build(width, height);
}

core::Result<void> MetalDevice::destroySwapchain(SwapchainHandle swapchain) {
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
    if (MTL::ResidencySet* residency = (*found)->layer()->residencySet(); residency != nullptr) {
        const std::scoped_lock lock{queues_[0]->lock};
        queues_[0]->queue->removeResidencySet(residency);
    }
    if (swapchains_.erase(swapchain)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

core::Result<void> MetalDevice::acquire(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->acquire();
}

core::Result<void> MetalDevice::present(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const std::scoped_lock lock{queues_[0]->lock};
    return (*found)->present();
}

core::Result<CommandPoolHandle> MetalDevice::createCommandPool(const CommandPoolDesc& desc) {
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
    // Every allocator and buffer the pool will hand out is made now, so
    // beginCommands only begins one.
    const AutoreleasePool autorelease;
    for (std::uint32_t slot = *first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        MTL4::CommandAllocator* allocator = device_->newCommandAllocator();
        if (allocator == nullptr) {
            release();
            return std::unexpected(deviceFailed("command allocator").error());
        }
        lists_[slot].allocator = adopt(allocator);
        MTL4::CommandBuffer* buffer = device_->newCommandBuffer();
        if (buffer == nullptr) {
            release();
            return std::unexpected(deviceFailed("command buffer").error());
        }
        lists_[slot].buffer = adopt(buffer);
        MTL4::RenderPassDescriptor* pass = MTL4::RenderPassDescriptor::alloc()->init();
        if (pass == nullptr) {
            release();
            return std::unexpected(deviceFailed("render pass descriptor").error());
        }
        lists_[slot].pass = adopt(pass);
        // Asked for once here, so the attachment exists before the first pass.
        std::ignore = pass->colorAttachments()->object(0);
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(
        MetalCommandPool{.queue = desc.queue, .first = *first, .next = *first, .recording = 0, .lastSubmit = 0});
    if (commandPool.isNull()) {
        release();
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command pools",
                             desc_.maxCommandPools);
    }
    return commandPool;
}

core::Result<void> MetalDevice::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if ((*pool)->recording != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} lists from the pool are still recording; submit or destroy them", (*pool)->recording);
    }
    if (queues_[(*pool)->queue]->event->signaledValue() < (*pool)->lastSubmit) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the GPU has not passed the pool's last submit, token {}; wait on it first",
                             (*pool)->lastSubmit);
    }
    // Only the lists handed out since the last reset recorded into their
    // allocators.
    for (std::uint32_t slot = (*pool)->first; slot != (*pool)->next; slot = listSlots_.next(slot)) {
        lists_[slot].allocator->reset();
    }
    (*pool)->next = (*pool)->first;
    return {};
}

core::Result<void> MetalDevice::destroyCommandPool(CommandPoolHandle commandPool) {
    if (commandPool.isNull()) {
        return {};
    }
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const MetalCommandPool gone = **pool;
    for (std::uint32_t slot = gone.first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        if (MetalCommandList* list = commandLists_.find(lists_[slot].live); list != nullptr) {
            discard(*list);
            renderSlots_.give(list->rendered);
        }
        commandLists_.erase(std::exchange(lists_[slot].live, {}));
    }
    // Out of the table before its room comes back, so a createCommandPool
    // that finds the room finds a free entry too.
    commandPools_.erase(commandPool);
    MetalQueue& queue = *queues_[gone.queue];
    {
        const std::scoped_lock lock{queue.lock};
        // poolsHeld_ counts this pool until it is collected, so there is room.
        std::ignore = queue.retired.push({.first = gone.first, .value = gone.lastSubmit});
        queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
        collectPools(queue);
    }
    return {};
}

core::Result<CommandListHandle> MetalDevice::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const std::uint32_t slot = (*pool)->next;
    if (slot == core::SlotChains::kEnd) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "every list in the pool was handed out since its last reset");
    }
    // The allocator and buffer were made with the pool; this only begins it.
    const AutoreleasePool autorelease;
    MetalListSlot& made = lists_[slot];
    made.buffer->beginCommandBuffer(made.allocator.get());
    // A new handle for each recording, so last frame's handle never names
    // this one. The list pool was made at its maximum, so this never grows.
    const CommandListHandle commandList = commandLists_.emplace(MetalCommandList{.buffer = made.buffer.get(),
                                                                                 .pool = commandPool,
                                                                                 .slot = slot,
                                                                                 .encoder = {},
                                                                                 .rendering = {},
                                                                                 .rendered = core::SlotChains::kEnd});
    if (commandList.isNull()) {
        made.buffer->endCommandBuffer();
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command lists",
                             desc_.maxCommandLists);
    }
    (*pool)->next = listSlots_.next(slot);
    ++(*pool)->recording;
    made.live = commandList;
    return commandList;
}

core::Result<void> MetalDevice::beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
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
    if (const MetalCommandPool* pool = commandPools_.find((*list)->pool);
        pool != nullptr && queues_[pool->queue]->type != QueueType::Graphics) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the list records for queue {}, which does not render",
                             pool->queue);
    }
    if ((*chain)->submitted()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the acquired image was already submitted; present it first");
    }
    CA::MetalDrawable* drawable = (*chain)->drawable();
    if (drawable == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }

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

    // The slot's descriptor is refilled; the pool drains the autoreleased
    // encoder reference, and pushing one is a pointer onto the thread's page.
    const AutoreleasePool pool;
    MTL4::RenderPassDescriptor* pass = lists_[(*list)->slot].pass.get();
    MTL::RenderPassColorAttachmentDescriptor* color = pass->colorAttachments()->object(0);
    color->setTexture(drawable->texture());
    color->setLoadAction(loadActionFor(desc.loadOp));
    color->setStoreAction(storeActionFor(desc.storeOp));
    color->setClearColor(
        MTL::ClearColor::Make(static_cast<double>(desc.clearColor[0]), static_cast<double>(desc.clearColor[1]),
                              static_cast<double>(desc.clearColor[2]), static_cast<double>(desc.clearColor[3])));
    MTL4::RenderCommandEncoder* encoder = (*list)->buffer->renderCommandEncoder(pass);
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no render command encoder");
    }
    (*list)->encoder = share(encoder);
    (*list)->rendering = swapchain;
    return {};
}

core::Result<void> MetalDevice::endRendering(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (!(*list)->rendering.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no rendering is open");
    }
    (*list)->encoder->endEncoding();
    (*list)->encoder.reset();
    (*list)->rendering.reset();
    return {};
}

// Metal 4 tracks no hazards. A consumer barrier blocks every stage of its own
// pass and all later passes on the queue until every stage of the earlier
// passes is done, so a pass of its own leaves nothing pending.
core::Result<void> MetalDevice::barrier(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    // The pool drains the autoreleased encoder reference.
    const AutoreleasePool pool;
    MTL4::ComputeCommandEncoder* encoder = (*list)->buffer->computeCommandEncoder();
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compute command encoder");
    }
    encoder->barrierAfterQueueStages(MTL::StageAll, MTL::StageAll, MTL4::VisibilityOptionDevice);
    encoder->endEncoding();
    return {};
}

core::Result<Token> MetalDevice::submit(const SubmitDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    for (const Token wait : desc.waits) {
        if (auto checked = checkToken(wait); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    MetalQueue& queue = *queues_[desc.queue];
    const std::scoped_lock lock{queue.lock};
    queue.buffers.clear();
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
        if ((*list)->rendering.has_value()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
        }
        if (const MetalCommandPool* pool = commandPools_.find((*list)->pool);
            pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the list was recorded for queue {}, not {}",
                                 pool->queue, desc.queue);
        }
        for (std::uint32_t slot = (*list)->rendered; slot != core::SlotChains::kEnd; slot = renderSlots_.next(slot)) {
            auto chain = swapchainFor(renders_[slot]);
            if (!chain) {
                return std::unexpected(std::move(chain.error()));
            }
            if ((*chain)->drawable() == nullptr) {
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
        if (!queue.buffers.push((*list)->buffer)) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a submit takes at most {} lists",
                                 desc_.maxCommandLists);
        }
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    for (const CommandListHandle commandList : desc.commandLists) {
        MetalCommandList* list = commandLists_.find(commandList);
        if (list == nullptr) {
            continue;
        }
        if (MetalCommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
            pool->lastSubmit = value;
        }
        list->buffer->endCommandBuffer();
        end(commandList);
    }
    for (const Token wait : desc.waits) {
        queue.queue->wait(queues_[wait.queue]->event.get(), wait.value);
    }
    for (const MetalSwapchain* chain : queue.chains) {
        queue.queue->wait(chain->drawable());
    }
    if (!queue.buffers.empty()) {
        queue.queue->commit(queue.buffers.data(), queue.buffers.size());
    }
    for (const MetalSwapchain* chain : queue.chains) {
        queue.queue->signalDrawable(chain->drawable());
    }
    queue.queue->signalEvent(queue.event.get(), value);
    queue.submitted.store(value, std::memory_order_release);
    for (MetalSwapchain* chain : queue.chains) {
        chain->setSubmitted();
    }
    collectPools(queue);
    return Token{.queue = desc.queue, .value = value};
}

core::Result<void> MetalDevice::wait(Token token) {
    if (auto checked = checkToken(token); !checked) {
        return checked;
    }
    MetalQueue& queue = *queues_[token.queue];
    if (!queue.event->waitUntilSignaledValue(token.value, kForever)) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "waiting for the GPU timed out");
    }
    if (queue.retiredCount.load(std::memory_order_acquire) != 0) {
        const std::scoped_lock lock{queue.lock};
        collectPools(queue);
    }
    return {};
}

core::Result<bool> MetalDevice::poll(Token token) {
    if (auto checked = checkToken(token); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    MetalQueue& queue = *queues_[token.queue];
    const std::uint64_t completed = queue.event->signaledValue();
    if (queue.retiredCount.load(std::memory_order_acquire) != 0) {
        const std::scoped_lock lock{queue.lock};
        collectPools(queue);
    }
    return completed >= token.value;
}

core::Result<void> MetalDevice::destroyCommandList(CommandListHandle commandList) {
    if (commandList.isNull()) {
        return {};
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    discard(**list);
    end(commandList);
    return {};
}

void MetalDevice::end(CommandListHandle commandList) {
    const MetalCommandList* list = commandLists_.find(commandList);
    if (list == nullptr) {
        return;
    }
    if (MetalCommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
        --pool->recording;
    }
    renderSlots_.give(list->rendered);
    lists_[list->slot].live = {};
    commandLists_.erase(commandList);
}

core::Result<MemoryBudget> MetalDevice::budget() const {
    return MemoryBudget{
        .heaps = {{.budget = device_->recommendedMaxWorkingSetSize(), .usage = device_->currentAllocatedSize()}}};
}

} // namespace bfr::rhi::metal
