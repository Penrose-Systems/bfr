#include "vulkan/vulkan_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"
#include "vulkan/queue_families.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_result.h"
#include "vulkan/vulkan_rhi.h"
#include "vulkan/vulkan_swapchain.h"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace bfr::rhi::vulkan {

namespace {

VkAttachmentLoadOp loadOpFor(LoadOp op) {
    switch (op) {
        case LoadOp::Load:
            return VK_ATTACHMENT_LOAD_OP_LOAD;
        case LoadOp::Clear:
            return VK_ATTACHMENT_LOAD_OP_CLEAR;
        case LoadOp::DontCare:
            return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
}

VkAttachmentStoreOp storeOpFor(StoreOp op) {
    switch (op) {
        case StoreOp::Store:
            return VK_ATTACHMENT_STORE_OP_STORE;
        case StoreOp::DontCare:
            return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }
    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

// A barrier moving a swapchain image between layouts around rendering to it.
VkImageMemoryBarrier2 layoutBarrier(VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (to == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    else {
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    }
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
    return barrier;
}

void recordBarrier(const VolkDeviceTable& table, VkCommandBuffer buffer, const VkImageMemoryBarrier2& barrier) {
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    table.vkCmdPipelineBarrier2(buffer, &dependency);
}

} // namespace

core::Result<std::unique_ptr<VulkanDevice>> VulkanDevice::create(VulkanRhi& rhi, VkPhysicalDevice adapter,
                                                                 VkDevice device, const QueueFamilies& families,
                                                                 bool hasBudget) {
    auto made = std::make_unique<VulkanDevice>(rhi, adapter, device, families, hasBudget);
    if (auto opened = made->openFrames(); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    return made;
}

VulkanDevice::VulkanDevice(VulkanRhi& rhi, VkPhysicalDevice adapter, VkDevice device, const QueueFamilies& families,
                           bool hasBudget)
    : rhi_(&rhi),
      adapter_(adapter),
      device_(device),
      graphicsFamily_(families.graphics),
      presentFamily_(families.present.value_or(families.graphics)),
      hasBudget_(hasBudget),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandLists_("command lists", kFramesInFlight, kFramesInFlight, rhi.logger()) {
    volkLoadDeviceTable(&table_, device_);
    table_.vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
    table_.vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
}

core::Result<void> VulkanDevice::openFrames() {
    for (Frame& frame : frames_) {
        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.queueFamilyIndex = graphicsFamily_;
        if (const VkResult result = table_.vkCreateCommandPool(device_, &pool, nullptr, &frame.pool);
            result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateCommandPool", .result = result});
        }
        VkCommandBufferAllocateInfo buffer{};
        buffer.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        buffer.commandPool = frame.pool;
        buffer.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        buffer.commandBufferCount = 1;
        if (const VkResult result = table_.vkAllocateCommandBuffers(device_, &buffer, &frame.buffer);
            result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkAllocateCommandBuffers", .result = result});
        }
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (const VkResult result = table_.vkCreateFence(device_, &fence, nullptr, &frame.fence);
            result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateFence", .result = result});
        }
    }
    return {};
}

core::Result<void> VulkanDevice::waitFrame(Frame& frame) {
    if (!frame.pending) {
        return {};
    }
    if (const VkResult result = table_.vkWaitForFences(device_, 1, &frame.fence, VK_TRUE,
                                                       std::numeric_limits<std::uint64_t>::max());
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkWaitForFences", .result = result});
    }
    if (const VkResult result = table_.vkResetFences(device_, 1, &frame.fence); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkResetFences", .result = result});
    }
    frame.pending = false;
    return {};
}

core::Result<void> VulkanDevice::waitIdle() {
    const std::scoped_lock lock{queues_};
    if (const VkResult result = table_.vkDeviceWaitIdle(device_); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkDeviceWaitIdle", .result = result});
    }
    return {};
}

core::Result<VulkanSwapchain*> VulkanDevice::swapchainFor(SwapchainHandle swapchain) {
    VulkanSwapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
    }
    return found;
}

core::Result<VulkanCommandList*> VulkanDevice::commandListFor(CommandListHandle commandList) {
    VulkanCommandList* found = commandLists_.find(commandList);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command list handle is stale");
    }
    return found;
}

core::Result<SwapchainHandle> VulkanDevice::createSwapchain(const core::window::NativeWindow& window,
                                                            const SwapchainRequest& request) {
    auto surface = rhi_->surfaceFor(window);
    if (!surface) {
        return std::unexpected(std::move(surface.error()));
    }
    // Fixed for the surface, so asked once.
    const auto presents = rhi_->presents(adapter_, presentFamily_, *surface);
    if (!presents) {
        return std::unexpected(presents.error());
    }
    if (!*presents) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "the device's present queue doesn't present to the window");
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, table_, rhi_->instanceTable(), std::move(*surface),
                                                          request, rhi_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("a device holds at most {} swapchains", kSwapchainLimit));
    }
    return swapchain;
}

core::Result<SwapchainImages> VulkanDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                            std::uint32_t height) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // The old images may still be on the GPU or with the presentation engine.
    if (auto idle = waitIdle(); !idle) {
        return std::unexpected(std::move(idle.error()));
    }
    return (*found)->build(adapter_, {.width = width, .height = height});
}

core::Result<void> VulkanDevice::destroySwapchain(SwapchainHandle swapchain) {
    if (swapchain.isNull()) {
        return {};
    }
    if (auto found = swapchainFor(swapchain); !found) {
        return std::unexpected(std::move(found.error()));
    }
    // Its window is the caller's to close once this returns, so the GPU is
    // waited for now, not retired against.
    if (auto idle = waitIdle(); !idle) {
        return std::unexpected(std::move(idle.error()));
    }
    if (swapchains_.erase(swapchain)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

core::Result<void> VulkanDevice::acquire(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->acquire();
}

core::Result<void> VulkanDevice::present(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const std::scoped_lock lock{queues_};
    return (*found)->present(presentQueue_);
}

core::Result<CommandListHandle> VulkanDevice::beginCommands() {
    const std::scoped_lock lock{queues_};
    // The oldest frame, or the next one when that is still recording.
    std::uint32_t index = nextFrame_;
    while (frames_[index].recording) {
        index = (index + 1) % kFramesInFlight;
        if (index == nextFrame_) {
            return core::failure(core::ErrorCode::RhiLimitReached,
                                 std::format("{} command lists are already recording; submit or destroy one",
                                             kFramesInFlight));
        }
    }
    Frame& frame = frames_[index];
    if (auto waited = waitFrame(frame); !waited) {
        return std::unexpected(std::move(waited.error()));
    }
    if (const VkResult result = table_.vkResetCommandPool(device_, frame.pool, 0); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkResetCommandPool", .result = result});
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (const VkResult result = table_.vkBeginCommandBuffer(frame.buffer, &begin); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkBeginCommandBuffer", .result = result});
    }
    const CommandListHandle commandList = commandLists_.emplace(
        VulkanCommandList{.buffer = frame.buffer, .frame = index, .rendering = {}, .rendered = {}});
    if (commandList.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "no command list is free");
    }
    frame.recording = true;
    nextFrame_ = (index + 1) % kFramesInFlight;
    return commandList;
}

core::Result<void> VulkanDevice::beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
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
    const auto image = (*chain)->acquired();
    if (!image) {
        return std::unexpected(image.error());
    }
    // Rendered to earlier in this list, the image is where that left it; else
    // where the last submit did.
    const auto rendered = std::ranges::find((*list)->rendered, swapchain, &RenderedSwapchain::swapchain);
    const VkImageLayout before = rendered == (*list)->rendered.end() ? image->layout : rendered->layout;
    VkCommandBuffer buffer = (*list)->buffer;
    recordBarrier(table_, buffer, layoutBarrier(image->image, before, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));

    VkRenderingAttachmentInfo color{};
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = image->view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = loadOpFor(desc.loadOp);
    color.storeOp = storeOpFor(desc.storeOp);
    std::ranges::copy(desc.clearColor, color.clearValue.color.float32);
    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea = {.offset = {.x = 0, .y = 0}, .extent = image->extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color;
    table_.vkCmdBeginRendering(buffer, &rendering);

    (*list)->rendering = swapchain;
    if (rendered == (*list)->rendered.end()) {
        (*list)->rendered.push_back({.swapchain = swapchain, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }
    else {
        rendered->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    return {};
}

core::Result<void> VulkanDevice::endRendering(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (!(*list)->rendering.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no rendering is open");
    }
    auto chain = swapchainFor(*(*list)->rendering);
    if (!chain) {
        return std::unexpected(std::move(chain.error()));
    }
    const auto image = (*chain)->acquired();
    if (!image) {
        return std::unexpected(image.error());
    }
    VkCommandBuffer buffer = (*list)->buffer;
    table_.vkCmdEndRendering(buffer);
    // Rendering is open on it, so beginRendering listed it.
    const auto rendered = std::ranges::find((*list)->rendered, *(*list)->rendering, &RenderedSwapchain::swapchain);
    recordBarrier(table_, buffer, layoutBarrier(image->image, rendered->layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR));
    rendered->layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    (*list)->rendering.reset();
    return {};
}

core::Result<void> VulkanDevice::submit(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if ((*list)->rendering.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
    }
    const std::scoped_lock lock{queues_};
    // Each swapchain rendered to: its image's acquire is waited on and its
    // present semaphore signaled.
    std::vector<VkSemaphoreSubmitInfo> waits;
    std::vector<VkSemaphoreSubmitInfo> signals;
    std::vector<std::pair<VulkanSwapchain*, VkImageLayout>> chains;
    for (const RenderedSwapchain& rendered : (*list)->rendered) {
        auto chain = swapchainFor(rendered.swapchain);
        if (!chain) {
            return std::unexpected(std::move(chain.error()));
        }
        const auto image = (*chain)->acquired();
        if (!image) {
            return std::unexpected(image.error());
        }
        if ((*chain)->submitted()) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "the acquired image was already submitted; present it first");
        }
        VkSemaphoreSubmitInfo wait{};
        wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        wait.semaphore = image->acquireSemaphore;
        wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        waits.push_back(wait);
        VkSemaphoreSubmitInfo signal{};
        signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signal.semaphore = image->presentSemaphore;
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        signals.push_back(signal);
        chains.emplace_back(*chain, rendered.layout);
    }
    Frame& frame = frames_[(*list)->frame];
    // The list is done either way; the pool reset before its next recording
    // clears a buffer that failed here.
    frame.recording = false;
    commandLists_.erase(commandList);
    if (const VkResult result = table_.vkEndCommandBuffer(frame.buffer); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkEndCommandBuffer", .result = result});
    }
    VkCommandBufferSubmitInfo buffer{};
    buffer.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    buffer.commandBuffer = frame.buffer;
    VkSubmitInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.waitSemaphoreInfoCount = static_cast<std::uint32_t>(waits.size());
    info.pWaitSemaphoreInfos = waits.data();
    info.commandBufferInfoCount = 1;
    info.pCommandBufferInfos = &buffer;
    info.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signals.size());
    info.pSignalSemaphoreInfos = signals.data();
    if (const VkResult result = table_.vkQueueSubmit2(graphicsQueue_, 1, &info, frame.fence); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkQueueSubmit2", .result = result});
    }
    frame.pending = true;
    // The GPU now has the transitions, so the images are in these layouts.
    for (const auto& [chain, layout] : chains) {
        chain->setLayout(layout);
        chain->setSubmitted();
    }
    return {};
}

core::Result<void> VulkanDevice::destroyCommandList(CommandListHandle commandList) {
    if (commandList.isNull()) {
        return {};
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    // The buffer stays begun; the pool reset before its next recording clears it.
    const std::scoped_lock lock{queues_};
    frames_[(*list)->frame].recording = false;
    commandLists_.erase(commandList);
    return {};
}

core::Result<MemoryBudget> VulkanDevice::budget() const {
    if (!hasBudget_) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the driver lacks VK_EXT_memory_budget");
    }
    return readBudget(rhi_->instanceTable(), adapter_);
}

VulkanDevice::~VulkanDevice() {
    // A failed wait leaves nothing to do but release, and no result to report
    // it through, so it is sent as an error whatever BFR_LOGGING says.
    if (const VkResult result = table_.vkDeviceWaitIdle(device_); result != VK_SUCCESS) {
        rhi_->logger().send(core::Severity::Error,
                            std::format("vkDeviceWaitIdle: {}; destroying the device anyway", string_VkResult(result)));
    }
    // Swapchains and command lists are the device's children, so they go first.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(rhi_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    commandLists_.clear();
    swapchains_.clear();
    for (Frame& frame : frames_) {
        if (frame.fence != VK_NULL_HANDLE) {
            table_.vkDestroyFence(device_, frame.fence, nullptr);
        }
        if (frame.pool != VK_NULL_HANDLE) {
            table_.vkDestroyCommandPool(device_, frame.pool, nullptr);
        }
    }
    table_.vkDestroyDevice(device_, nullptr);
    BFR_LOG_DEBUG(rhi_->logger(), "Vulkan device destroyed");
}

} // namespace bfr::rhi::vulkan
