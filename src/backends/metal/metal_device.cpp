#include "metal/metal_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/swapchain.h"
#include "core/fixed_vector.h"
#include "core/slot_chains.h"
#include "metal/metal_instance.h"
#include "metal/metal_pipeline.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_util.h"
#include "rhi/clear_color.h"
#include "rhi/command_check.h"
#include "rhi/device_limits.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"
#include "rhi/timeout.h"

#include <Foundation/NSError.hpp>
#include <Metal/MTL4ArgumentTable.hpp>
#include <Metal/MTL4CommandAllocator.hpp>
#include <Metal/MTL4CommandEncoder.hpp>
#include <Metal/MTL4Compiler.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTL4RenderCommandEncoder.hpp>
#include <Metal/MTL4RenderPass.hpp>
#include <Metal/MTLArgument.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandEncoder.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLEvent.hpp>
#include <Metal/MTLRenderCommandEncoder.hpp>
#include <Metal/MTLRenderPass.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <Metal/MTLResource.hpp>
#include <Metal/MTLTexture.hpp>
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
#include <optional>
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
      desc_(heldPart(desc)),
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
      bindless_(desc),
      pipelines_("pipelines", kBufferCapacity, kBufferLimit, rhi.logger()) {}

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
    for (std::size_t index = 0; index < desc.graphics.size(); ++index) {
        if (auto opened = open(QueueType::Graphics); !opened) {
            return opened;
        }
    }
    for (std::size_t index = 0; index < desc.compute.size(); ++index) {
        if (auto opened = open(QueueType::Compute); !opened) {
            return opened;
        }
    }
    for (std::size_t index = 0; index < desc.copy.size(); ++index) {
        if (auto opened = open(QueueType::Copy); !opened) {
            return opened;
        }
    }
    const Ref<MTL4::CompilerDescriptor> compilerDesc = adopt(MTL4::CompilerDescriptor::alloc()->init());
    MTL4::Compiler* compiler = device_->newCompiler(compilerDesc.get(), &error);
    if (compiler == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compiler: {}",
                             error != nullptr ? error->localizedDescription()->utf8String() : "no reason given");
    }
    compiler_ = adopt(compiler);
    {
        const std::scoped_lock lock{residencyLock_};
        auto tables = createMetalBindlessTables(*device_.get(), desc, residency_);
        if (!tables) {
            return std::unexpected(std::move(tables.error()));
        }
        tables_.emplace(std::move(*tables));
        residency_->commit();
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
    if (textures_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} textures were left when the device went", textures_.size());
    }
    textures_.clear();
    if (samplers_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} samplers were left when the device went", samplers_.size());
    }
    samplers_.clear();
    if (pipelines_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} pipelines were left when the device went", pipelines_.size());
    }
    pipelines_.clear();
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
        lists_[slot].arguments.reset();
        // The ring leaves the residency set, which is not thread-safe.
        const std::scoped_lock lock{residencyLock_};
        lists_[slot].pushResident.reset();
        lists_[slot].pushRing.reset();
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
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    auto layer = layerFor(window);
    if (!layer) {
        return std::unexpected(std::move(layer.error()));
    }
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, *layer, desc, instance_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "a device holds at most {} swapchains",
                             desc_.maxSwapchains);
    }
    // The layer's drawables are resident for every queue, as buffers and
    // textures are: any queue may render to them. The layer's residency set
    // is the one for its device, so the device is set first, as build sets it.
    (*layer)->setDevice(device_.get());
    if (MTL::ResidencySet* residency = (*layer)->residencySet(); residency != nullptr) {
        for (const auto& queue : queues_) {
            const std::scoped_lock lock{queue->lock};
            queue->queue->addResidencySet(residency);
        }
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
        for (const auto& queue : queues_) {
            const std::scoped_lock lock{queue->lock};
            queue->queue->removeResidencySet(residency);
        }
    }
    if (swapchains_.erase(swapchain)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

core::Result<SwapchainStatus> MetalDevice::acquire(SwapchainHandle swapchain, std::uint64_t timeoutNs) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->acquire(timeoutNs);
}

core::Result<SwapchainStatus> MetalDevice::present(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // The swapchain's queue's submits read the drawable present gives back.
    const std::scoped_lock lock{queues_[(*found)->queue()]->lock};
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
        // Asked for once here, so every attachment exists before the first pass.
        for (std::uint32_t i = 0; i < kMaxColorTargets; ++i) {
            std::ignore = pass->colorAttachments()->object(i);
        }
        auto arguments = createArgumentTable(*device_.get());
        if (!arguments) {
            release();
            return std::unexpected(std::move(arguments.error()));
        }
        lists_[slot].arguments = std::move(*arguments);
    }
    // The pool's push constant ring, in its first slot: shared, so the CPU
    // copies into it, and resident, so the GPU reads it. None for a pool that
    // records no push constants, or has no list to record them.
    if (desc.pushConstantUpdates != 0 && *first != core::SlotChains::kEnd) {
        const std::uint64_t ringBytes = std::uint64_t{desc.pushConstantUpdates} * kPushConstantsSize;
        MTL::Buffer* ring = device_->newBuffer(ringBytes, MTL::ResourceStorageModeShared);
        if (ring == nullptr) {
            release();
            return std::unexpected(deviceFailed("push constant ring").error());
        }
        const std::scoped_lock lock{residencyLock_};
        lists_[*first].pushRing = adopt(ring);
        lists_[*first].pushResident.emplace(residency_, ring);
        residency_->commit();
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(
        MetalCommandPool{.queue = desc.queue,
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
    (*pool)->pushed = 0;
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
    // The tables are bound through the slot's argument table for the whole
    // list; the push constants join them as they are set.
    if (tables().buffer) {
        made.arguments->setAddress(tables().buffer->gpuAddress(), kTablesBufferIndex);
    }
    const CommandListHandle commandList = commandLists_.emplace(MetalCommandList{.buffer = made.buffer.get(),
                                                                                 .pool = commandPool,
                                                                                 .slot = slot,
                                                                                 .state = {},
                                                                                 .graphics = {},
                                                                                 .compute = {},
                                                                                 .encoderGraphics = {},
                                                                                 .indexType = MTL::IndexTypeUInt32,
                                                                                 .indexBuffer = 0,
                                                                                 .indexBytes = 0,
                                                                                 .encoder = {},
                                                                                 .presenting = {},
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
    if (auto checked = checkRenderingDesc(!swapchain.isNull(), desc); !checked) {
        return checked;
    }
    // Every handle is looked up before anything is recorded, so a bad one
    // leaves the list as it was.
    std::array<MTL::Texture*, kMaxColorTargets> colors{};
    // A swapchain's image is never a signed integer format.
    std::array<Format, kMaxColorTargets> formats{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    MetalSwapchain* chain = nullptr;
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
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
            CA::MetalDrawable* drawable = chain->drawable();
            if (drawable == nullptr) {
                return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
            }
            colors.at(i) = drawable->texture();
            width = static_cast<std::uint32_t>(colors.at(i)->width());
            height = static_cast<std::uint32_t>(colors.at(i)->height());
            continue;
        }
        auto texture = textureFor(desc.colorAttachments.at(i).texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        colors.at(i) = (*texture)->texture.get();
        formats.at(i) = (*texture)->format;
        width = (*texture)->width;
        height = (*texture)->height;
    }
    const MetalTexture* depth = nullptr;
    if (!desc.depthStencil.texture.isNull()) {
        auto texture = textureFor(desc.depthStencil.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        depth = *texture;
        width = depth->width;
        height = depth->height;
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

    // The slot's descriptor is refilled, every attachment of it, so none of a
    // past rendering's is left; the pool drains the autoreleased encoder
    // reference, and pushing one is a pointer onto the thread's page.
    const AutoreleasePool pool;
    MTL4::RenderPassDescriptor* pass = lists_[(*list)->slot].pass.get();
    for (std::uint32_t i = 0; i < kMaxColorTargets; ++i) {
        MTL::RenderPassColorAttachmentDescriptor* color = pass->colorAttachments()->object(i);
        if (i >= desc.colorAttachmentCount) {
            color->setTexture(nullptr);
            continue;
        }
        const ColorAttachment& attachment = desc.colorAttachments.at(i);
        color->setTexture(colors.at(i));
        color->setLoadAction(loadActionFor(attachment.loadOp));
        color->setStoreAction(storeActionFor(attachment.storeOp));
        const auto [red, green, blue, alpha] = clearValues(attachment, formats.at(i));
        color->setClearColor(MTL::ClearColor::Make(red, green, blue, alpha));
    }
    MTL::RenderPassDepthAttachmentDescriptor* depthAttachment = pass->depthAttachment();
    MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment = pass->stencilAttachment();
    depthAttachment->setTexture(nullptr);
    stencilAttachment->setTexture(nullptr);
    if (depth != nullptr) {
        // Each aspect the format has gets its attachment, loaded and stored alike.
        const FormatBlock& block = blockOf(depth->format);
        if (block.depthBytes != 0) {
            depthAttachment->setTexture(depth->texture.get());
            depthAttachment->setLoadAction(loadActionFor(desc.depthStencil.loadOp));
            depthAttachment->setStoreAction(storeActionFor(desc.depthStencil.storeOp));
            depthAttachment->setClearDepth(static_cast<double>(desc.depthStencil.clearDepth));
        }
        if (block.stencilBytes != 0) {
            stencilAttachment->setTexture(depth->texture.get());
            stencilAttachment->setLoadAction(loadActionFor(desc.depthStencil.loadOp));
            stencilAttachment->setStoreAction(storeActionFor(desc.depthStencil.storeOp));
            stencilAttachment->setClearStencil(desc.depthStencil.clearStencil);
        }
    }
    MTL4::RenderCommandEncoder* encoder = (*list)->buffer->renderCommandEncoder(pass);
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no render command encoder");
    }
    encoder->setArgumentTable(lists_[(*list)->slot].arguments.get(), MTL::RenderStageVertex | MTL::RenderStageFragment);
    (*list)->encoder = share(encoder);
    (*list)->encoderGraphics = {};
    (*list)->state.rendering = true;
    (*list)->presenting = chain != nullptr ? std::optional{swapchain} : std::nullopt;
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

core::Result<void> MetalDevice::endRendering(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    (*list)->encoder->endEncoding();
    (*list)->encoder.reset();
    (*list)->state.rendering = false;
    (*list)->presenting.reset();
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
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
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

core::Result<void> MetalDevice::wait(Token token, std::uint64_t timeoutNs) {
    if (auto checked = checkToken(token); !checked) {
        return checked;
    }
    MetalQueue& queue = *queues_[token.queue];
    if (!queue.event->waitUntilSignaledValue(token.value, millisecondsRoundedUp(timeoutNs))) {
        return core::failure(core::ErrorCode::RhiTimeout, "the GPU had not passed the token in time");
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
