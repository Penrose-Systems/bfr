#include "vulkan/vulkan_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/swapchain.h"
#include "core/fixed_vector.h"
#include "core/slot_chains.h"
#include "rhi/clear_color.h"
#include "rhi/command_check.h"
#include "rhi/device_limits.h"
#include "rhi/enum_check.h"
#include "vulkan/queue_families.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_buffer.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_instance.h"
#include "vulkan/vulkan_pipeline.h"
#include "vulkan/vulkan_result.h"
#include "vulkan/vulkan_swapchain.h"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
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

namespace bfr::rhi::vulkan {

namespace {

// The newest Vulkan VMA takes.
constexpr std::uint32_t kNewestForVma = VK_API_VERSION_1_4;

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

VkResolveModeFlagBits resolveModeFor(ResolveMode mode) {
    switch (mode) {
        case ResolveMode::SampleZero:
            return VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        case ResolveMode::Average:
            return VK_RESOLVE_MODE_AVERAGE_BIT;
        case ResolveMode::Min:
            return VK_RESOLVE_MODE_MIN_BIT;
        case ResolveMode::Max:
            return VK_RESOLVE_MODE_MAX_BIT;
        case ResolveMode::None:
        default:
            return VK_RESOLVE_MODE_NONE;
    }
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

// Every write recorded before it, made visible to the accesses after it.
void recordMemoryBarrier(const VolkDeviceTable& table, VkCommandBuffer buffer, VkPipelineStageFlags2 after,
                         VkAccessFlags2 access) {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = after;
    barrier.dstAccessMask = access;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    table.vkCmdPipelineBarrier2(buffer, &dependency);
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
                                                                 const DeviceDesc& desc,
                                                                 const VulkanDeviceSetup& setup) {
    auto made = std::make_unique<VulkanDevice>(rhi, adapter, device, desc, setup);
    if (auto opened = made->openQueues(families, desc); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    // VMA takes Vulkan 1.0 to 1.4 (vk_mem_alloc.h:1142); a later device
    // still runs at its own version.
    auto allocator = createAllocator(rhi.handle(), adapter, device, std::min(setup.apiVersion, kNewestForVma));
    if (!allocator) {
        return std::unexpected(std::move(allocator.error()));
    }
    made->allocator_ = *allocator;
    auto layout = VulkanBindlessLayout::create(made->table_, device, desc, setup.pushConstantsSize);
    if (!layout) {
        return std::unexpected(std::move(layout.error()));
    }
    made->layout_ = std::move(*layout);
    return made;
}

VulkanDevice::VulkanDevice(VulkanInstance& rhi, VkPhysicalDevice adapter, VkDevice device, const DeviceDesc& desc,
                           const VulkanDeviceSetup& setup)
    : instance_(&rhi),
      adapter_(adapter),
      device_(device),
      hasBudget_(setup.hasBudget),
      hasSwapchain_(setup.hasSwapchain),
      pushConstantsSize_(setup.pushConstantsSize),
      timestampPeriod_(setup.timestampPeriod),
      timestamps_(setup.timestamps),
      multiDrawIndirect_(setup.multiDrawIndirect),
      drawIndirectCount_(setup.drawIndirectCount),
      features_(setup.features),
      resolve_(setup.resolve),
      formats13_(setup.formats13),
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
      timestampPools_("timestamp pools", desc.maxTimestampPools, desc.maxTimestampPools, rhi.logger()),
      sparseMemories_("sparse memories", desc.maxSparseMemories, desc.maxSparseMemories, rhi.logger()),
      pipelineCaches_("pipeline caches", kBufferCapacity, kBufferLimit, rhi.logger()),
      bindless_(desc),
      pipelines_("pipelines", kBufferCapacity, kBufferLimit, rhi.logger()) {
    volkLoadDeviceTable(&table_, device_);
    // A core command above the version the device uses must not be called
    // (initialization.adoc:146-148), so below 1.3 the core entries take the
    // functions of the extensions 1.3 made core, which the device enabled.
    // vkCmdSetPrimitiveTopologyEXT is null without extendedDynamicState, and
    // then never called.
    if (!hasCore13(setup.apiVersion)) {
        table_.vkCmdBeginRendering = table_.vkCmdBeginRenderingKHR;
        table_.vkCmdEndRendering = table_.vkCmdEndRenderingKHR;
        table_.vkCmdPipelineBarrier2 = table_.vkCmdPipelineBarrier2KHR;
        table_.vkQueueSubmit2 = table_.vkQueueSubmit2KHR;
        table_.vkCmdWriteTimestamp2 = table_.vkCmdWriteTimestamp2KHR;
        table_.vkCmdSetPrimitiveTopology = table_.vkCmdSetPrimitiveTopologyEXT;
    }
}

core::Result<void> VulkanDevice::openQueues(const QueueFamilies& families, const DeviceDesc& desc) {
    const InstanceTable& vk = instance_->instanceTable();
    std::uint32_t familyCount = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(adapter_, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> properties(familyCount);
    vk.vkGetPhysicalDeviceQueueFamilyProperties(adapter_, &familyCount, properties.data());
    const auto open = [this, &properties](QueueType type, std::uint32_t family,
                                          std::uint32_t index) -> core::Result<void> {
        auto queue = std::make_unique<VulkanQueue>();
        queue->type = type;
        queue->family = family;
        queue->sparse = (properties[family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
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
    for (std::uint32_t index = 0; index < desc.graphics.size(); ++index) {
        if (auto opened = open(QueueType::Graphics, *families.graphics, index); !opened) {
            return opened;
        }
    }
    for (std::uint32_t index = 0; index < desc.compute.size(); ++index) {
        if (auto opened = open(QueueType::Compute, *families.compute, index); !opened) {
            return opened;
        }
    }
    for (std::uint32_t index = 0; index < desc.copy.size(); ++index) {
        if (auto opened = open(QueueType::Copy, *families.transfer, index); !opened) {
            return opened;
        }
    }
    // A submit's arrays, made now at the device's maximums: every list; a
    // wait for each swapchain and queue; a signal for each swapchain and the
    // timeline. Every pool may be destroyed before the GPU is done with it.
    for (const auto& queue : queues_) {
        queue->buffers = core::FixedVector<VkCommandBufferSubmitInfo>{desc.maxCommandLists};
        queue->waits = core::FixedVector<VkSemaphoreSubmitInfo>{std::size_t{desc.maxSwapchains} + queues_.size()};
        queue->signals = core::FixedVector<VkSemaphoreSubmitInfo>{std::size_t{desc.maxSwapchains} + 1};
        queue->chains = core::FixedVector<VulkanSwapchain*>{desc.maxSwapchains};
        queue->waitValues.resize(queues_.size());
        queue->retired = core::FixedVector<VulkanRetiredPool>{desc.maxCommandPools};
        queue->imageBinds = core::FixedVector<VkSparseImageMemoryBind>{desc.maxTileMappings};
        queue->tailBinds = core::FixedVector<VkSparseMemoryBind>{desc.maxTileMappings};
        queue->bufferBinds = core::FixedVector<VkSparseMemoryBind>{desc.maxTileMappings};
        queue->waitSemaphores = core::FixedVector<VkSemaphore>{queues_.size()};
        queue->waitSemaphoreValues = core::FixedVector<std::uint64_t>{queues_.size()};
        if (std::ranges::find(families_, queue->family) == families_.end()) {
            families_.push_back(queue->family);
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
        return core::failure(core::ErrorCode::RhiInvalidCall, "the device has no queue {}; it has {}", queue,
                             queues_.size());
    }
    return {};
}

core::Result<void> VulkanDevice::checkToken(Token token) const {
    if (auto checked = checkQueue(token.queue); !checked) {
        return checked;
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
    queue.retired.eraseIf([this, completed](const VulkanRetiredPool& retired) {
        if (retired.value > *completed) {
            return false;
        }
        // Destroying the pool frees its buffers; the slots held them.
        table_.vkDestroyCommandPool(device_, retired.pool, nullptr);
        listSlots_.give(retired.first);
        poolsHeld_.fetch_sub(1, std::memory_order_relaxed);
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
                                                            const SwapchainDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Without VK_KHR_swapchain there is no vkCreateSwapchainKHR to call.
    if (!hasSwapchain_) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "the device has no VK_KHR_swapchain, so it can't present");
    }
    auto surface = instance_->surfaceFor(window);
    if (!surface) {
        return std::unexpected(std::move(surface.error()));
    }
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Fixed for the surface, so asked once.
    const auto presents = instance_->presents(adapter_, queues_[desc.queue]->family, *surface);
    if (!presents) {
        return std::unexpected(presents.error());
    }
    if (!*presents) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported, "queue {} doesn't present to the window",
                             desc.queue);
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, table_, instance_->instanceTable(),
                                                          std::move(*surface), desc, families_, instance_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "a device holds at most {} swapchains",
                             desc_.maxSwapchains);
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

core::Result<SwapchainStatus> VulkanDevice::acquire(SwapchainHandle swapchain, std::uint64_t timeoutNs) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->acquire(timeoutNs);
}

core::Result<SwapchainStatus> VulkanDevice::present(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // The queue is the swapchain's, which may be submitted to from another thread at once.
    VulkanQueue& queue = *queues_[(*found)->queue()];
    const std::scoped_lock lock{queue.lock};
    return (*found)->present(queue.queue);
}

core::Result<CommandPoolHandle> VulkanDevice::createCommandPool(const CommandPoolDesc& desc) {
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
    const auto release = [this, first](VkCommandPool pool) {
        if (pool != VK_NULL_HANDLE) {
            table_.vkDestroyCommandPool(device_, pool, nullptr);
        }
        listSlots_.give(*first);
        poolsHeld_.fetch_sub(1, std::memory_order_relaxed);
    };
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    info.queueFamilyIndex = queues_[desc.queue]->family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (const VkResult result = table_.vkCreateCommandPool(device_, &info, nullptr, &pool); result != VK_SUCCESS) {
        release(VK_NULL_HANDLE);
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateCommandPool", .result = result});
    }
    // Every buffer the pool will hand out is made now, so beginCommands only
    // begins one.
    for (std::uint32_t slot = *first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        VkCommandBufferAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (const VkResult result = table_.vkAllocateCommandBuffers(device_, &allocate, &lists_[slot].buffer);
            result != VK_SUCCESS) {
            release(pool);
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkAllocateCommandBuffers", .result = result});
        }
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(
        VulkanCommandPool{.pool = pool,
                          .queue = desc.queue,
                          .pushConstantUpdates = desc.pushConstantUpdates,
                          .pushed = 0,
                          .first = *first,
                          .next = *first,
                          .recording = 0,
                          .lastSubmit = 0});
    if (commandPool.isNull()) {
        release(pool);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command pools",
                             desc_.maxCommandPools);
    }
    return commandPool;
}

core::Result<void> VulkanDevice::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const auto completed = completedValue(*queues_[(*pool)->queue]);
    if (!completed) {
        return std::unexpected(completed.error());
    }
    if (*completed < (*pool)->lastSubmit) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the GPU has not passed the pool's last submit, token {}; wait on it first",
                             (*pool)->lastSubmit);
    }
    if (const VkResult result = table_.vkResetCommandPool(device_, (*pool)->pool, 0); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkResetCommandPool", .result = result});
    }
    // Vulkan's reset returns a buffer still recording to its initial state,
    // so the list recording into it ends with it.
    for (std::uint32_t slot = (*pool)->first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        end(lists_[slot].live);
    }
    (*pool)->next = (*pool)->first;
    (*pool)->pushed = 0;
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
    const VulkanCommandPool gone = **pool;
    for (std::uint32_t slot = gone.first; slot != core::SlotChains::kEnd; slot = listSlots_.next(slot)) {
        if (const VulkanCommandList* list = commandLists_.find(lists_[slot].live); list != nullptr) {
            renderSlots_.give(list->rendered);
        }
        commandLists_.erase(std::exchange(lists_[slot].live, {}));
    }
    // Out of the table before its room comes back, so a createCommandPool
    // that finds the room finds a free entry too.
    commandPools_.erase(commandPool);
    VulkanQueue& queue = *queues_[gone.queue];
    {
        const std::scoped_lock lock{queue.lock};
        // poolsHeld_ counts this pool until it is collected, so there is room.
        std::ignore = queue.retired.push({.pool = gone.pool, .first = gone.first, .value = gone.lastSubmit});
        queue.retiredCount.store(static_cast<std::uint32_t>(queue.retired.size()), std::memory_order_release);
        collectPools(queue);
    }
    return {};
}

core::Result<CommandListHandle> VulkanDevice::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const std::uint32_t slot = (*pool)->next;
    if (slot == core::SlotChains::kEnd) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "every list in the pool was handed out since its last reset");
    }
    // The buffer was made with the pool; this only begins it.
    VkCommandBuffer buffer = lists_[slot].buffer;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (const VkResult result = table_.vkBeginCommandBuffer(buffer, &begin); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkBeginCommandBuffer", .result = result});
    }
    // The one descriptor set, bound once per list where the queue can take
    // it, so every pipeline bound later finds its tables.
    const QueueType type = queues_[(*pool)->queue]->type;
    VkDescriptorSet set = layout_.set();
    if (type == QueueType::Graphics) {
        table_.vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_.pipelineLayout(), 0, 1, &set, 0,
                                       nullptr);
    }
    if (type != QueueType::Copy) {
        table_.vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_.pipelineLayout(), 0, 1, &set, 0,
                                       nullptr);
    }
    // A new handle for each recording, so last frame's handle never names
    // this one. The list pool was made at its maximum, so this never grows.
    const CommandListHandle commandList = commandLists_.emplace(VulkanCommandList{
        .buffer = buffer, .pool = commandPool, .slot = slot, .presenting = {}, .rendered = core::SlotChains::kEnd});
    if (commandList.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command lists",
                             desc_.maxCommandLists);
    }
    (*pool)->next = listSlots_.next(slot);
    ++(*pool)->recording;
    lists_[slot].live = commandList;
    return commandList;
}

core::Result<void> VulkanDevice::beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                const RenderingDesc& desc) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRenderingDesc(!swapchain.isNull(), desc); !checked) {
        return checked;
    }
    if (const VulkanCommandPool* pool = commandPools_.find((*list)->pool);
        pool != nullptr && queues_[pool->queue]->type != QueueType::Graphics) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the list records for queue {}, which does not render",
                             pool->queue);
    }
    // Every handle is looked up before anything is recorded, so a bad one
    // leaves the list as it was.
    std::array<VkRenderingAttachmentInfo, kMaxColorTargets> colors{};
    VkRenderingAttachmentInfo depth{};
    VkRenderingAttachmentInfo stencil{};
    VkExtent2D extent{};
    std::optional<AcquiredImage> image;
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        const ColorAttachment& attachment = desc.colorAttachments.at(i);
        VkRenderingAttachmentInfo& color = colors.at(i);
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = loadOpFor(attachment.loadOp);
        color.storeOp = storeOpFor(attachment.storeOp);
        if (attachment.clearInteger) {
            std::ranges::copy(attachment.clearColorInteger, color.clearValue.color.uint32);
        }
        else {
            std::ranges::copy(attachment.clearColor, color.clearValue.color.float32);
        }
        // The swapchain's image is attachment 0, or what a texture there
        // resolves into.
        VkImageView resolveView = VK_NULL_HANDLE;
        if (i == 0 && !swapchain.isNull()) {
            auto chain = swapchainFor(swapchain);
            if (!chain) {
                return std::unexpected(std::move(chain.error()));
            }
            if ((*chain)->submitted()) {
                return core::failure(core::ErrorCode::RhiInvalidCall,
                                     "the acquired image was already submitted; present it first");
            }
            auto acquired = (*chain)->acquired();
            if (!acquired) {
                return std::unexpected(acquired.error());
            }
            image = *acquired;
            extent = image->extent;
            if (attachment.texture.isNull()) {
                color.imageView = image->view;
                continue;
            }
            resolveView = image->view;
        }
        auto texture = textureFor(attachment.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        color.imageView = (*texture)->view();
        extent = (*texture)->extent();
        if (!attachment.resolveTexture.isNull()) {
            auto target = textureFor(attachment.resolveTexture);
            if (!target) {
                return std::unexpected(std::move(target.error()));
            }
            resolveView = (*target)->view();
        }
        if (resolveView != VK_NULL_HANDLE) {
            // The only modes a color resolve takes (VUID-VkRenderingAttachmentInfo-imageView-06129, -06130).
            color.resolveMode = integerFormat((*texture)->format()) ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
                                                                    : VK_RESOLVE_MODE_AVERAGE_BIT;
            color.resolveImageView = resolveView;
            color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    }
    const DepthStencilAttachment& depthStencil = desc.depthStencil;
    if (!depthStencil.texture.isNull()) {
        auto texture = textureFor(depthStencil.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        // Each aspect the format has gets its slot, with its own ops. A
        // read-only attachment keeps what it holds without writing it.
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = (*texture)->view();
        depth.imageLayout = depthStencil.readOnly ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                  : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth.clearValue.depthStencil = {.depth = depthStencil.clearDepth, .stencil = depthStencil.clearStencil};
        stencil = depth;
        depth.loadOp = loadOpFor(depthStencil.depthLoadOp);
        depth.storeOp = depthStencil.readOnly ? VK_ATTACHMENT_STORE_OP_NONE : storeOpFor(depthStencil.depthStoreOp);
        stencil.loadOp = loadOpFor(depthStencil.stencilLoadOp);
        stencil.storeOp = depthStencil.readOnly ? VK_ATTACHMENT_STORE_OP_NONE : storeOpFor(depthStencil.stencilStoreOp);
        if (!depthStencil.resolveTexture.isNull()) {
            if (auto checked = checkDepthResolve(depthStencil, (*texture)->format(), resolve_); !checked) {
                return checked;
            }
            auto target = textureFor(depthStencil.resolveTexture);
            if (!target) {
                return std::unexpected(std::move(target.error()));
            }
            // Both aspects resolve into one view (VUID-VkRenderingInfo-pDepthAttachment-06086).
            for (VkRenderingAttachmentInfo* aspect : {&depth, &stencil}) {
                aspect->resolveImageView = (*target)->view();
                aspect->resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
            depth.resolveMode = resolveModeFor(depthStencil.depthResolve);
            stencil.resolveMode = resolveModeFor(depthStencil.stencilResolve);
        }
        extent = (*texture)->extent();
    }
    if (image.has_value()) {
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

    VkCommandBuffer buffer = (*list)->buffer;
    if (image.has_value()) {
        recordBarrier(table_, buffer,
                      layoutBarrier(image->image, image->layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
    }
    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea = {.offset = {.x = 0, .y = 0}, .extent = extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = desc.colorAttachmentCount;
    rendering.pColorAttachments = colors.data();
    if (depth.imageView != VK_NULL_HANDLE) {
        const VkImageAspectFlags aspects = (*textureFor(desc.depthStencil.texture))->aspects();
        rendering.pDepthAttachment = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 ? &depth : nullptr;
        rendering.pStencilAttachment = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 ? &stencil : nullptr;
    }
    table_.vkCmdBeginRendering(buffer, &rendering);
    (*list)->presenting = image.has_value() ? std::optional{swapchain} : std::nullopt;
    // The whole area, until the caller sets its own.
    const auto whole = static_cast<float>(extent.width);
    const auto height = static_cast<float>(extent.height);
    if (auto set = setViewport(commandList,
                               {.x = 0, .y = 0, .width = whole, .height = height, .minDepth = 0, .maxDepth = 1});
        !set) {
        return set;
    }
    return setScissor(commandList, {.x = 0, .y = 0, .width = extent.width, .height = extent.height});
}

core::Result<void> VulkanDevice::endRendering(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    VkCommandBuffer buffer = (*list)->buffer;
    table_.vkCmdEndRendering(buffer);
    if (const std::optional<SwapchainHandle> open = (*list)->presenting; open.has_value()) {
        auto chain = swapchainFor(*open);
        if (!chain) {
            return std::unexpected(std::move(chain.error()));
        }
        const auto image = (*chain)->acquired();
        if (!image) {
            return std::unexpected(image.error());
        }
        recordBarrier(table_, buffer,
                      layoutBarrier(image->image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR));
        // Recorded now, not at submit, so the next rendering recorded on this
        // image, in this list or another, starts from where this one leaves it.
        (*chain)->setLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }
    (*list)->presenting.reset();
    return {};
}

core::Result<void> VulkanDevice::barrier(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    recordMemoryBarrier(table_, (*list)->buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
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
    queue.chains.clear();
    const auto full = [this] {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a submit takes at most {} lists", desc_.maxCommandLists);
    };
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
        }
        if (const VulkanCommandPool* pool = commandPools_.find((*list)->pool);
            pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the list was recorded for queue {}, not {}",
                                 pool->queue, desc.queue);
        }
        for (std::uint32_t slot = (*list)->rendered; slot != core::SlotChains::kEnd; slot = renderSlots_.next(slot)) {
            auto chain = swapchainFor(renders_[slot]);
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
            if (std::ranges::find(queue.chains, *chain) != queue.chains.end()) {
                continue;
            }
            VkSemaphoreSubmitInfo wait{};
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            wait.semaphore = image->acquireSemaphore;
            wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSemaphoreSubmitInfo signal{};
            signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signal.semaphore = image->presentSemaphore;
            signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            // Each swapchain is listed once and has a wait and a signal, so
            // these have room.
            std::ignore = queue.waits.push(wait);
            std::ignore = queue.signals.push(signal);
            std::ignore = queue.chains.push(*chain);
        }
        VkCommandBufferSubmitInfo buffer{};
        buffer.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        buffer.commandBuffer = (*list)->buffer;
        if (!queue.buffers.push(buffer)) {
            return full();
        }
    }
    // A timeline wait passes once the value is reached, so waiting for the
    // largest value asked of each queue waits for all of them.
    std::ranges::fill(queue.waitValues, 0);
    for (const Token wait : desc.waits) {
        queue.waitValues[wait.queue] = std::max(queue.waitValues[wait.queue], wait.value);
    }
    for (std::size_t index = 0; index < queues_.size(); ++index) {
        if (queue.waitValues[index] == 0) {
            continue;
        }
        VkSemaphoreSubmitInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        info.semaphore = queues_[index]->timeline;
        info.value = queue.waitValues[index];
        info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        std::ignore = queue.waits.push(info);
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    // The lists are spent whether or not the submit goes through. Their pools
    // remember it only once the GPU is sent it: a value no submit signals
    // would otherwise keep the pool from ever being reset or freed.
    const auto finish = [this, &desc, value](bool sent) {
        for (const CommandListHandle commandList : desc.commandLists) {
            const VulkanCommandList* list = commandLists_.find(commandList);
            if (list == nullptr) {
                continue;
            }
            if (VulkanCommandPool* pool = commandPools_.find(list->pool); sent && pool != nullptr) {
                pool->lastSubmit = value;
            }
            end(commandList);
        }
    };
    core::Result<void> ended;
    for (const CommandListHandle commandList : desc.commandLists) {
        const VulkanCommandList* list = commandLists_.find(commandList);
        if (list == nullptr) {
            continue;
        }
        // A wait on the token alone does not make the list's writes visible
        // to the CPU; this does, as mapBuffer promises.
        recordMemoryBarrier(table_, list->buffer, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        if (const VkResult result = table_.vkEndCommandBuffer(list->buffer); result != VK_SUCCESS && ended) {
            ended = failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkEndCommandBuffer", .result = result});
        }
    }
    if (!ended) {
        finish(false);
        return std::unexpected(std::move(ended.error()));
    }
    VkSemaphoreSubmitInfo done{};
    done.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    done.semaphore = queue.timeline;
    done.value = value;
    done.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    std::ignore = queue.signals.push(done);
    VkSubmitInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.waitSemaphoreInfoCount = static_cast<std::uint32_t>(queue.waits.size());
    info.pWaitSemaphoreInfos = queue.waits.data();
    info.commandBufferInfoCount = static_cast<std::uint32_t>(queue.buffers.size());
    info.pCommandBufferInfos = queue.buffers.data();
    info.signalSemaphoreInfoCount = static_cast<std::uint32_t>(queue.signals.size());
    info.pSignalSemaphoreInfos = queue.signals.data();
    if (const VkResult result = table_.vkQueueSubmit2(queue.queue, 1, &info, VK_NULL_HANDLE); result != VK_SUCCESS) {
        finish(false);
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkQueueSubmit2", .result = result});
    }
    queue.submitted.store(value, std::memory_order_release);
    finish(true);
    for (VulkanSwapchain* chain : queue.chains) {
        chain->setSubmitted();
    }
    collectPools(queue);
    return Token{.queue = desc.queue, .value = value};
}

core::Result<void> VulkanDevice::wait(Token token, std::uint64_t timeoutNs) {
    if (auto checked = checkToken(token); !checked) {
        return checked;
    }
    VulkanQueue& queue = *queues_[token.queue];
    VkSemaphoreWaitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.semaphoreCount = 1;
    info.pSemaphores = &queue.timeline;
    info.pValues = &token.value;
    if (const VkResult result = table_.vkWaitSemaphores(device_, &info, timeoutNs); result == VK_TIMEOUT) {
        return core::failure(core::ErrorCode::RhiTimeout, "the GPU had not passed the token in time");
    }
    else if (result != VK_SUCCESS) {
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
    // The buffer stays as it is; the pool's reset readies it again.
    end(commandList);
    return {};
}

void VulkanDevice::end(CommandListHandle commandList) {
    const VulkanCommandList* list = commandLists_.find(commandList);
    if (list == nullptr) {
        return;
    }
    if (VulkanCommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
        --pool->recording;
    }
    renderSlots_.give(list->rendered);
    lists_[list->slot].live = {};
    commandLists_.erase(commandList);
}

core::Result<MemoryBudget> VulkanDevice::budget() const {
    if (!hasBudget_) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the driver lacks VK_EXT_memory_budget");
    }
    return readBudget(instance_->instanceTable(), adapter_);
}

void VulkanDevice::warnLeftovers() const {
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    if (commandPools_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} command pools were left when the device went", commandPools_.size());
    }
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
}

VulkanDevice::~VulkanDevice() {
    // A failed wait leaves nothing to do but release, and no result to report
    // it through, so it is sent as an error whatever BFR_LOGGING says.
    if (const VkResult result = table_.vkDeviceWaitIdle(device_); result != VK_SUCCESS) {
        core::Message message;
        message.format("vkDeviceWaitIdle: {}; destroying the device anyway", string_VkResult(result));
        instance_->logger().send(core::Severity::Error, message.view());
    }
    warnLeftovers();
    commandLists_.clear();
    commandPools_.forEach(
        [this](const VulkanCommandPool& pool) { table_.vkDestroyCommandPool(device_, pool.pool, nullptr); });
    commandPools_.clear();
    swapchains_.clear();
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
    vmaDestroyAllocator(allocator_);
    for (const auto& queue : queues_) {
        for (const VulkanRetiredPool& retired : queue->retired) {
            table_.vkDestroyCommandPool(device_, retired.pool, nullptr);
        }
        if (queue->timeline != VK_NULL_HANDLE) {
            table_.vkDestroySemaphore(device_, queue->timeline, nullptr);
        }
    }
    table_.vkDestroyDevice(device_, nullptr);
    BFR_LOG_DEBUG(instance_->logger(), "Vulkan device destroyed");
}

} // namespace bfr::rhi::vulkan
