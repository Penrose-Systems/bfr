#include "metal/metal_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"
#include "metal/metal_instance.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_util.h"

#include <Metal/MTL4CommandAllocator.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLEvent.hpp>
#include <Metal/MTLRenderPass.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
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
    return core::failure(core::ErrorCode::RhiDeviceFailed, std::format("Metal gave no {}", what));
}

} // namespace

core::Result<std::unique_ptr<MetalDevice>> MetalDevice::create(MetalInstance& rhi, Ref<MTL::Device> device,
                                                               const DeviceQueues& queues) {
    auto made = std::make_unique<MetalDevice>(rhi, std::move(device));
    if (auto opened = made->open(queues); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    return made;
}

MetalDevice::MetalDevice(MetalInstance& rhi, Ref<MTL::Device> device)
    : instance_(&rhi),
      device_(std::move(device)),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandPools_("command pools", kCommandPoolCapacity, kCommandPoolLimit, rhi.logger()),
      commandLists_("command lists", kCommandListCapacity, kCommandListLimit, rhi.logger()) {}

core::Result<void> MetalDevice::open(const DeviceQueues& queues) {
    const AutoreleasePool pool;
    const auto open = [this](QueueType type) -> core::Result<void> {
        auto queue = std::make_unique<MetalQueue>();
        queue->type = type;
        MTL4::CommandQueue* native = device_->newMTL4CommandQueue();
        if (native == nullptr) {
            return deviceFailed("command queue");
        }
        queue->queue = adopt(native);
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
    for (std::uint32_t index = 0; index < queues.compute; ++index) {
        if (auto opened = open(QueueType::Compute); !opened) {
            return opened;
        }
    }
    for (std::uint32_t index = 0; index < queues.copy; ++index) {
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
    // A list left recording has an open encoder and buffer, which Metal wants
    // ended before they go.
    commandLists_.forEach([](MetalCommandList& list) { discard(list); });
    commandLists_.clear();
    commandPools_.clear();
    for (const auto& queue : queues_) {
        queue->retired.clear();
    }
}

core::Result<void> MetalDevice::checkQueue(std::uint32_t queue) const {
    if (queue >= queues_.size()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the device has no queue {}; it has {}", queue, queues_.size()));
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
                             std::format("no submit has returned token {} on queue {}; the last is {}", token.value,
                                         token.queue, submitted));
    }
    return {};
}

void MetalDevice::collectPools(MetalQueue& queue) {
    if (queue.retired.empty()) {
        return;
    }
    const std::uint64_t completed = queue.event->signaledValue();
    std::erase_if(queue.retired, [completed](const auto& retired) { return retired.second <= completed; });
    queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
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
                                                           const SwapchainRequest& request) {
    auto layer = layerFor(window);
    if (!layer) {
        return std::unexpected(std::move(layer.error()));
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, *layer, request, instance_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("a device holds at most {} swapchains", kSwapchainLimit));
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

core::Result<CommandPoolHandle> MetalDevice::createCommandPool(std::uint32_t queue) {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(
        MetalCommandPool{.queue = queue, .entries = {}, .used = 0, .recording = {}, .lastSubmit = 0});
    if (commandPool.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command pools", kCommandPoolLimit));
    }
    return commandPool;
}

core::Result<void> MetalDevice::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if (!(*pool)->recording.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("{} lists from the pool are still recording; submit or destroy them",
                                         (*pool)->recording.size()));
    }
    if (queues_[(*pool)->queue]->event->signaledValue() < (*pool)->lastSubmit) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the GPU has not passed the pool's last submit, token {}; wait on it first",
                                         (*pool)->lastSubmit));
    }
    for (const MetalCommandPool::Entry& entry : (*pool)->entries) {
        entry.allocator->reset();
    }
    (*pool)->used = 0;
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
    for (const CommandListHandle handle : (*pool)->recording) {
        if (MetalCommandList* list = commandLists_.find(handle); list != nullptr) {
            discard(*list);
        }
        commandLists_.erase(handle);
    }
    MetalQueue& queue = *queues_[(*pool)->queue];
    {
        const std::scoped_lock lock{queue.lock};
        queue.retired.emplace_back(std::move((*pool)->entries), (*pool)->lastSubmit);
        queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
        collectPools(queue);
    }
    commandPools_.erase(commandPool);
    return {};
}

core::Result<CommandListHandle> MetalDevice::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const AutoreleasePool autorelease;
    if ((*pool)->used == (*pool)->entries.size()) {
        MTL4::CommandAllocator* allocator = device_->newCommandAllocator();
        if (allocator == nullptr) {
            return std::unexpected(deviceFailed("command allocator").error());
        }
        MetalCommandPool::Entry entry{.allocator = adopt(allocator), .buffer = {}};
        MTL4::CommandBuffer* buffer = device_->newCommandBuffer();
        if (buffer == nullptr) {
            return std::unexpected(deviceFailed("command buffer").error());
        }
        entry.buffer = adopt(buffer);
        (*pool)->entries.push_back(std::move(entry));
    }
    const MetalCommandPool::Entry& entry = (*pool)->entries[(*pool)->used];
    entry.buffer->beginCommandBuffer(entry.allocator.get());
    const CommandListHandle commandList = commandLists_.emplace(
        MetalCommandList{.buffer = entry.buffer, .pool = commandPool, .encoder = {}, .rendering = {}, .rendered = {}});
    if (commandList.isNull()) {
        entry.buffer->endCommandBuffer();
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command lists", kCommandListLimit));
    }
    ++(*pool)->used;
    (*pool)->recording.push_back(commandList);
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
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the list records for queue {}, which does not render", pool->queue));
    }
    if ((*chain)->submitted()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the acquired image was already submitted; present it first");
    }
    CA::MetalDrawable* drawable = (*chain)->drawable();
    if (drawable == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }

    const AutoreleasePool pool;
    const auto pass = adopt(MTL4::RenderPassDescriptor::alloc()->init());
    MTL::RenderPassColorAttachmentDescriptor* color = pass->colorAttachments()->object(0);
    color->setTexture(drawable->texture());
    color->setLoadAction(loadActionFor(desc.loadOp));
    color->setStoreAction(storeActionFor(desc.storeOp));
    color->setClearColor(
        MTL::ClearColor::Make(static_cast<double>(desc.clearColor[0]), static_cast<double>(desc.clearColor[1]),
                              static_cast<double>(desc.clearColor[2]), static_cast<double>(desc.clearColor[3])));
    MTL4::RenderCommandEncoder* encoder = (*list)->buffer->renderCommandEncoder(pass.get());
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no render command encoder");
    }
    (*list)->encoder = share(encoder);
    (*list)->rendering = swapchain;
    if (std::ranges::find((*list)->rendered, swapchain) == (*list)->rendered.end()) {
        (*list)->rendered.push_back(swapchain);
    }
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
    std::vector<MetalSwapchain*> chains;
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
        }
        if ((*list)->rendering.has_value()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
        }
        if (const MetalCommandPool* pool = commandPools_.find((*list)->pool);
            pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 std::format("the list was recorded for queue {}, not {}", pool->queue, desc.queue));
        }
        for (const SwapchainHandle swapchain : (*list)->rendered) {
            auto chain = swapchainFor(swapchain);
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
            if (std::ranges::find(chains, *chain) == chains.end()) {
                chains.push_back(*chain);
            }
        }
        queue.buffers.push_back((*list)->buffer.get());
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    std::vector<Ref<MTL4::CommandBuffer>> held;
    held.reserve(desc.commandLists.size());
    for (const CommandListHandle commandList : desc.commandLists) {
        MetalCommandList* list = commandLists_.find(commandList);
        if (list == nullptr) {
            continue;
        }
        if (MetalCommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
            std::erase(pool->recording, commandList);
            pool->lastSubmit = value;
        }
        list->buffer->endCommandBuffer();
        held.push_back(list->buffer);
        commandLists_.erase(commandList);
    }
    for (const Token wait : desc.waits) {
        queue.queue->wait(queues_[wait.queue]->event.get(), wait.value);
    }
    for (const MetalSwapchain* chain : chains) {
        queue.queue->wait(chain->drawable());
    }
    if (!queue.buffers.empty()) {
        queue.queue->commit(queue.buffers.data(), queue.buffers.size());
    }
    for (const MetalSwapchain* chain : chains) {
        queue.queue->signalDrawable(chain->drawable());
    }
    queue.queue->signalEvent(queue.event.get(), value);
    queue.submitted.store(value, std::memory_order_release);
    for (MetalSwapchain* chain : chains) {
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
    if (MetalCommandPool* pool = commandPools_.find((*list)->pool); pool != nullptr) {
        std::erase(pool->recording, commandList);
    }
    commandLists_.erase(commandList);
    return {};
}

core::Result<MemoryBudget> MetalDevice::budget() const {
    return MemoryBudget{
        .heaps = {{.budget = device_->recommendedMaxWorkingSetSize(), .usage = device_->currentAllocatedSize()}}};
}

} // namespace bfr::rhi::metal
