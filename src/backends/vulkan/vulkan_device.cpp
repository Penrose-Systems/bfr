#include "vulkan/vulkan_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"
#include "vulkan/queue_families.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_instance.h"
#include "vulkan/vulkan_result.h"
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

core::Result<std::unique_ptr<VulkanDevice>> VulkanDevice::create(VulkanInstance& rhi, VkPhysicalDevice adapter,
                                                                 VkDevice device, const QueueFamilies& families,
                                                                 const DeviceQueues& queues, bool hasBudget) {
    auto made = std::make_unique<VulkanDevice>(rhi, adapter, device, families, hasBudget);
    if (auto opened = made->openQueues(families, queues); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    return made;
}

VulkanCommandPool::VulkanCommandPool(VkDevice device, const VolkDeviceTable& table, VkCommandPool pool,
                                     std::uint32_t queue)
    : pool(pool), queue(queue), device_(device), table_(&table) {}

VulkanCommandPool::~VulkanCommandPool() {
    if (pool != VK_NULL_HANDLE) {
        table_->vkDestroyCommandPool(device_, pool, nullptr);
    }
}

VkCommandPool VulkanCommandPool::release() {
    return std::exchange(pool, VK_NULL_HANDLE);
}

VulkanDevice::VulkanDevice(VulkanInstance& rhi, VkPhysicalDevice adapter, VkDevice device,
                           const QueueFamilies& families, bool hasBudget)
    : instance_(&rhi),
      adapter_(adapter),
      device_(device),
      graphicsFamily_(families.graphics),
      presentFamily_(families.present.value_or(families.graphics)),
      hasBudget_(hasBudget),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandPools_("command pools", kCommandPoolCapacity, kCommandPoolLimit, rhi.logger()),
      commandLists_("command lists", kCommandListCapacity, kCommandListLimit, rhi.logger()) {
    volkLoadDeviceTable(&table_, device_);
    table_.vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
}

core::Result<void> VulkanDevice::openQueues(const QueueFamilies& families, const DeviceQueues& queues) {
    const auto open = [this](QueueType type, std::uint32_t family, std::uint32_t index) -> core::Result<void> {
        auto queue = std::make_unique<VulkanQueue>();
        queue->type = type;
        queue->family = family;
        table_.vkGetDeviceQueue(device_, family, index, &queue->queue);
        VkSemaphoreTypeCreateInfo timeline{};
        timeline.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline.initialValue = 0;
        VkSemaphoreCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        info.pNext = &timeline;
        if (const VkResult result = table_.vkCreateSemaphore(device_, &info, nullptr, &queue->timeline);
            result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateSemaphore", .result = result});
        }
        queues_.push_back(std::move(queue));
        return {};
    };
    if (auto opened = open(QueueType::Graphics, families.graphics, 0); !opened) {
        return opened;
    }
    for (std::uint32_t index = 0; index < queues.compute; ++index) {
        if (auto opened = open(QueueType::Compute, families.compute, index); !opened) {
            return opened;
        }
    }
    for (std::uint32_t index = 0; index < queues.copy; ++index) {
        if (auto opened = open(QueueType::Copy, families.transfer, index); !opened) {
            return opened;
        }
    }
    for (const auto& queue : queues_) {
        if (queue->queue == presentQueue_) {
            presentOn_ = queue.get();
        }
    }
    return {};
}

core::Result<std::uint64_t> VulkanDevice::completedValue(const VulkanQueue& queue) const {
    std::uint64_t value = 0;
    if (const VkResult result = table_.vkGetSemaphoreCounterValue(device_, queue.timeline, &value);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkGetSemaphoreCounterValue", .result = result});
    }
    return value;
}

core::Result<void> VulkanDevice::checkQueue(std::uint32_t queue) const {
    if (queue >= queues_.size()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the device has no queue {}; it has {}", queue, queues_.size()));
    }
    return {};
}

core::Result<void> VulkanDevice::checkToken(Token token) const {
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

void VulkanDevice::collectPools(VulkanQueue& queue) {
    if (queue.retired.empty()) {
        return;
    }
    const auto completed = completedValue(queue);
    if (!completed) {
        return;
    }
    std::erase_if(queue.retired, [this, completed](const std::pair<VkCommandPool, std::uint64_t>& retired) {
        if (retired.second > *completed) {
            return false;
        }
        table_.vkDestroyCommandPool(device_, retired.first, nullptr);
        return true;
    });
    queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
}

core::Result<void> VulkanDevice::waitIdle() {
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(queues_.size());
    for (const auto& queue : queues_) {
        locks.emplace_back(queue->lock);
    }
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

core::Result<VulkanCommandPool*> VulkanDevice::commandPoolFor(CommandPoolHandle commandPool) {
    VulkanCommandPool* found = commandPools_.find(commandPool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command pool handle is stale");
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
    auto surface = instance_->surfaceFor(window);
    if (!surface) {
        return std::unexpected(std::move(surface.error()));
    }
    // Fixed for the surface, so asked once.
    const auto presents = instance_->presents(adapter_, presentFamily_, *surface);
    if (!presents) {
        return std::unexpected(presents.error());
    }
    if (!*presents) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "the device's present queue doesn't present to the window");
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, table_, instance_->instanceTable(),
                                                          std::move(*surface), request, instance_->logger());
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
    if (presentOn_ != nullptr) {
        const std::scoped_lock lock{presentOn_->lock};
        return (*found)->present(presentQueue_);
    }
    return (*found)->present(presentQueue_);
}

core::Result<CommandPoolHandle> VulkanDevice::createCommandPool(std::uint32_t queue) {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    info.queueFamilyIndex = queues_[queue]->family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (const VkResult result = table_.vkCreateCommandPool(device_, &info, nullptr, &pool); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateCommandPool", .result = result});
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(device_, table_, pool, queue);
    if (commandPool.isNull()) {
        table_.vkDestroyCommandPool(device_, pool, nullptr);
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command pools", kCommandPoolLimit));
    }
    return commandPool;
}

core::Result<void> VulkanDevice::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if (!(*pool)->recording.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("{} lists from the pool are still recording; submit or destroy them",
                                         (*pool)->recording.size()));
    }
    const auto completed = completedValue(*queues_[(*pool)->queue]);
    if (!completed) {
        return std::unexpected(completed.error());
    }
    if (*completed < (*pool)->lastSubmit) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the GPU has not passed the pool's last submit, token {}; wait on it first",
                                         (*pool)->lastSubmit));
    }
    if (const VkResult result = table_.vkResetCommandPool(device_, (*pool)->pool, 0); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkResetCommandPool", .result = result});
    }
    (*pool)->used = 0;
    return {};
}

core::Result<void> VulkanDevice::destroyCommandPool(CommandPoolHandle commandPool) {
    if (commandPool.isNull()) {
        return {};
    }
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    for (const CommandListHandle list : (*pool)->recording) {
        commandLists_.erase(list);
    }
    VulkanQueue& queue = *queues_[(*pool)->queue];
    {
        const std::scoped_lock lock{queue.lock};
        queue.retired.emplace_back((*pool)->release(), (*pool)->lastSubmit);
        queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
        collectPools(queue);
    }
    commandPools_.erase(commandPool);
    return {};
}

core::Result<CommandListHandle> VulkanDevice::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if ((*pool)->used == (*pool)->buffers.size()) {
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = (*pool)->pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        if (const VkResult result = table_.vkAllocateCommandBuffers(device_, &info, &buffer); result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkAllocateCommandBuffers", .result = result});
        }
        (*pool)->buffers.push_back(buffer);
    }
    VkCommandBuffer buffer = (*pool)->buffers[(*pool)->used];
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (const VkResult result = table_.vkBeginCommandBuffer(buffer, &begin); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkBeginCommandBuffer", .result = result});
    }
    const CommandListHandle commandList = commandLists_.emplace(
        VulkanCommandList{.buffer = buffer, .pool = commandPool, .rendering = {}, .rendered = {}});
    if (commandList.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command lists", kCommandListLimit));
    }
    ++(*pool)->used;
    (*pool)->recording.push_back(commandList);
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
    if (const VulkanCommandPool* pool = commandPools_.find((*list)->pool);
        pool != nullptr && queues_[pool->queue]->type != QueueType::Graphics) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the list records for queue {}, which does not render", pool->queue));
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

core::Result<Token> VulkanDevice::submit(const SubmitDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    for (const Token wait : desc.waits) {
        if (auto checked = checkToken(wait); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    VulkanQueue& queue = *queues_[desc.queue];
    const std::scoped_lock lock{queue.lock};
    queue.buffers.clear();
    queue.waits.clear();
    queue.signals.clear();
    std::vector<std::pair<VulkanSwapchain*, VkImageLayout>> chains;
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
        }
        if ((*list)->rendering.has_value()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
        }
        if (const VulkanCommandPool* pool = commandPools_.find((*list)->pool);
            pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 std::format("the list was recorded for queue {}, not {}", pool->queue, desc.queue));
        }
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
            if (auto seen = std::ranges::find(chains, *chain, &std::pair<VulkanSwapchain*, VkImageLayout>::first);
                seen != chains.end()) {
                seen->second = rendered.layout;
                continue;
            }
            VkSemaphoreSubmitInfo wait{};
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            wait.semaphore = image->acquireSemaphore;
            wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            queue.waits.push_back(wait);
            VkSemaphoreSubmitInfo signal{};
            signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signal.semaphore = image->presentSemaphore;
            signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            queue.signals.push_back(signal);
            chains.emplace_back(*chain, rendered.layout);
        }
        VkCommandBufferSubmitInfo buffer{};
        buffer.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        buffer.commandBuffer = (*list)->buffer;
        queue.buffers.push_back(buffer);
    }
    for (const Token wait : desc.waits) {
        VkSemaphoreSubmitInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        info.semaphore = queues_[wait.queue]->timeline;
        info.value = wait.value;
        info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        queue.waits.push_back(info);
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    core::Result<void> ended;
    for (const CommandListHandle commandList : desc.commandLists) {
        VulkanCommandList* list = commandLists_.find(commandList);
        if (list == nullptr) {
            continue;
        }
        if (VulkanCommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
            std::erase(pool->recording, commandList);
            pool->lastSubmit = value;
        }
        if (const VkResult result = table_.vkEndCommandBuffer(list->buffer); result != VK_SUCCESS && ended) {
            ended = failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkEndCommandBuffer", .result = result});
        }
        commandLists_.erase(commandList);
    }
    if (!ended) {
        return std::unexpected(std::move(ended.error()));
    }
    VkSemaphoreSubmitInfo done{};
    done.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    done.semaphore = queue.timeline;
    done.value = value;
    done.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    queue.signals.push_back(done);
    VkSubmitInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.waitSemaphoreInfoCount = static_cast<std::uint32_t>(queue.waits.size());
    info.pWaitSemaphoreInfos = queue.waits.data();
    info.commandBufferInfoCount = static_cast<std::uint32_t>(queue.buffers.size());
    info.pCommandBufferInfos = queue.buffers.data();
    info.signalSemaphoreInfoCount = static_cast<std::uint32_t>(queue.signals.size());
    info.pSignalSemaphoreInfos = queue.signals.data();
    if (const VkResult result = table_.vkQueueSubmit2(queue.queue, 1, &info, VK_NULL_HANDLE); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkQueueSubmit2", .result = result});
    }
    queue.submitted.store(value, std::memory_order_release);
    for (const auto& [chain, layout] : chains) {
        chain->setLayout(layout);
        chain->setSubmitted();
    }
    collectPools(queue);
    return Token{.queue = desc.queue, .value = value};
}

core::Result<void> VulkanDevice::wait(Token token) {
    if (auto checked = checkToken(token); !checked) {
        return checked;
    }
    VulkanQueue& queue = *queues_[token.queue];
    VkSemaphoreWaitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.semaphoreCount = 1;
    info.pSemaphores = &queue.timeline;
    info.pValues = &token.value;
    if (const VkResult result = table_.vkWaitSemaphores(device_, &info, std::numeric_limits<std::uint64_t>::max());
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkWaitSemaphores", .result = result});
    }
    if (queue.retiredCount.load(std::memory_order_acquire) != 0) {
        const std::scoped_lock lock{queue.lock};
        collectPools(queue);
    }
    return {};
}

core::Result<bool> VulkanDevice::poll(Token token) {
    if (auto checked = checkToken(token); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    VulkanQueue& queue = *queues_[token.queue];
    const auto completed = completedValue(queue);
    if (!completed) {
        return std::unexpected(completed.error());
    }
    if (queue.retiredCount.load(std::memory_order_acquire) != 0) {
        const std::scoped_lock lock{queue.lock};
        collectPools(queue);
    }
    return *completed >= token.value;
}

core::Result<void> VulkanDevice::destroyCommandList(CommandListHandle commandList) {
    if (commandList.isNull()) {
        return {};
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (VulkanCommandPool* pool = commandPools_.find((*list)->pool); pool != nullptr) {
        std::erase(pool->recording, commandList);
    }
    commandLists_.erase(commandList);
    return {};
}

core::Result<MemoryBudget> VulkanDevice::budget() const {
    if (!hasBudget_) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the driver lacks VK_EXT_memory_budget");
    }
    return readBudget(instance_->instanceTable(), adapter_);
}

VulkanDevice::~VulkanDevice() {
    // A failed wait leaves nothing to do but release, and no result to report
    // it through, so it is sent as an error whatever BFR_LOGGING says.
    if (const VkResult result = table_.vkDeviceWaitIdle(device_); result != VK_SUCCESS) {
        instance_->logger().send(core::Severity::Error,
                                 std::format("vkDeviceWaitIdle: {}; destroying the device anyway",
                                             string_VkResult(result)));
    }
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    if (commandPools_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} command pools were left when the device went", commandPools_.size());
    }
    commandLists_.clear();
    commandPools_.clear();
    swapchains_.clear();
    for (const auto& queue : queues_) {
        for (const auto& [pool, value] : queue->retired) {
            table_.vkDestroyCommandPool(device_, pool, nullptr);
        }
        if (queue->timeline != VK_NULL_HANDLE) {
            table_.vkDestroySemaphore(device_, queue->timeline, nullptr);
        }
    }
    table_.vkDestroyDevice(device_, nullptr);
    BFR_LOG_DEBUG(instance_->logger(), "Vulkan device destroyed");
}

} // namespace bfr::rhi::vulkan
