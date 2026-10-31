#include "metal/metal_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"
#include "metal/metal_rhi.h"
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

core::Result<std::unique_ptr<MetalDevice>> MetalDevice::create(MetalRhi& rhi, Ref<MTL::Device> device) {
    auto made = std::make_unique<MetalDevice>(rhi, std::move(device));
    if (auto opened = made->open(); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    return made;
}

MetalDevice::MetalDevice(MetalRhi& rhi, Ref<MTL::Device> device)
    : rhi_(&rhi),
      device_(std::move(device)),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandLists_("command lists", kFramesInFlight, kFramesInFlight, rhi.logger()) {}

core::Result<void> MetalDevice::open() {
    const AutoreleasePool pool;
    MTL4::CommandQueue* queue = device_->newMTL4CommandQueue();
    if (queue == nullptr) {
        return deviceFailed("command queue");
    }
    queue_ = adopt(queue);
    MTL::SharedEvent* event = device_->newSharedEvent();
    if (event == nullptr) {
        return deviceFailed("event");
    }
    event_ = adopt(event);
    for (Frame& frame : frames_) {
        MTL4::CommandAllocator* allocator = device_->newCommandAllocator();
        if (allocator == nullptr) {
            return deviceFailed("command allocator");
        }
        frame.allocator = adopt(allocator);
        MTL4::CommandBuffer* buffer = device_->newCommandBuffer();
        if (buffer == nullptr) {
            return deviceFailed("command buffer");
        }
        frame.buffer = adopt(buffer);
    }
    return {};
}

MetalDevice::~MetalDevice() {
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
    // A list left recording has an open encoder and buffer, which Metal wants
    // ended before they go.
    for (Frame& frame : frames_) {
        if (frame.recording) {
            discard(frame);
        }
    }
    commandLists_.clear();
}

core::Result<void> MetalDevice::waitFrame(Frame& frame) {
    if (!frame.pending) {
        return {};
    }
    if (!event_->waitUntilSignaledValue(frame.doneValue, kForever)) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "waiting for the GPU timed out");
    }
    frame.pending = false;
    return {};
}

core::Result<void> MetalDevice::waitIdle() {
    const std::scoped_lock lock{lock_};
    if (submitted_ == 0) {
        return {};
    }
    if (!event_->waitUntilSignaledValue(submitted_, kForever)) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "waiting for the GPU timed out");
    }
    return {};
}

void MetalDevice::discard(Frame& frame) {
    if (frame.encoder) {
        frame.encoder->endEncoding();
        frame.encoder.reset();
    }
    frame.buffer->endCommandBuffer();
}

MetalDevice::Frame& MetalDevice::frameAt(std::uint32_t index) {
    return std::span{frames_}[index];
}

core::Result<MetalSwapchain*> MetalDevice::swapchainFor(SwapchainHandle swapchain) {
    MetalSwapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
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
    const SwapchainHandle swapchain = swapchains_.emplace(device_, *layer, request, rhi_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("a device holds at most {} swapchains", kSwapchainLimit));
    }
    // The layer's drawables are resident for the queue that renders to them.
    if (MTL::ResidencySet* residency = (*layer)->residencySet(); residency != nullptr) {
        const std::scoped_lock lock{lock_};
        queue_->addResidencySet(residency);
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
        const std::scoped_lock lock{lock_};
        queue_->removeResidencySet(residency);
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
    const std::scoped_lock lock{lock_};
    return (*found)->present();
}

core::Result<CommandListHandle> MetalDevice::beginCommands() {
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
    frame.allocator->reset();
    frame.buffer->beginCommandBuffer(frame.allocator.get());
    const CommandListHandle commandList = commandLists_.emplace(
        MetalCommandList{.frame = index, .rendering = {}, .rendered = {}});
    if (commandList.isNull()) {
        frame.buffer->endCommandBuffer();
        return core::failure(core::ErrorCode::RhiLimitReached, "no command list is free");
    }
    frame.recording = true;
    nextFrame_ = (index + 1) % kFramesInFlight;
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
    Frame& frame = frameAt((*list)->frame);
    MTL4::RenderCommandEncoder* encoder = frame.buffer->renderCommandEncoder(pass.get());
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no render command encoder");
    }
    frame.encoder = share(encoder);
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
    Frame& frame = frameAt((*list)->frame);
    frame.encoder->endEncoding();
    frame.encoder.reset();
    (*list)->rendering.reset();
    return {};
}

core::Result<void> MetalDevice::submit(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if ((*list)->rendering.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
    }
    const std::scoped_lock lock{lock_};
    // Each swapchain rendered to: its drawable is waited on before the work
    // starts and signaled once it is queued.
    std::vector<MetalSwapchain*> chains;
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
        chains.push_back(*chain);
    }
    Frame& frame = frameAt((*list)->frame);
    // The list is done either way.
    frame.recording = false;
    commandLists_.erase(commandList);
    frame.buffer->endCommandBuffer();
    for (const MetalSwapchain* chain : chains) {
        queue_->wait(chain->drawable());
    }
    const std::array<const MTL4::CommandBuffer*, 1> buffers{frame.buffer.get()};
    queue_->commit(buffers.data(), buffers.size());
    for (const MetalSwapchain* chain : chains) {
        queue_->signalDrawable(chain->drawable());
    }
    queue_->signalEvent(event_.get(), ++submitted_);
    frame.doneValue = submitted_;
    frame.pending = true;
    for (MetalSwapchain* chain : chains) {
        chain->setSubmitted();
    }
    return {};
}

core::Result<void> MetalDevice::destroyCommandList(CommandListHandle commandList) {
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

core::Result<MemoryBudget> MetalDevice::budget() const {
    return MemoryBudget{
        .heaps = {{.budget = device_->recommendedMaxWorkingSetSize(), .usage = device_->currentAllocatedSize()}}};
}

} // namespace bfr::rhi::metal
