#include "vulkan/vulkan_sparse.h"

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/sparse.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_buffer.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"
#include "vulkan/vulkan_texture.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <mutex>
#include <utility>

namespace bfr::rhi::vulkan {

VulkanSparseMemory::VulkanSparseMemory(VmaAllocator allocator, VmaAllocation allocation, VkDeviceMemory memory,
                                       VkDeviceSize offset, std::uint32_t type)
    : allocator_(allocator), allocation_(allocation), memory_(memory), offset_(offset), type_(type) {}

VulkanSparseMemory::VulkanSparseMemory(VulkanSparseMemory&& other) noexcept
    : allocator_(other.allocator_),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      memory_(other.memory_),
      offset_(other.offset_),
      type_(other.type_) {}

VulkanSparseMemory::~VulkanSparseMemory() {
    if (allocation_ != VK_NULL_HANDLE) {
        vmaFreeMemory(allocator_, allocation_);
    }
}

// The device's sparse memory and tile mapping calls. Each passes its values
// to the Vulkan call that does what it says; the texture and buffer binds are
// built beside the resources they bind.

core::Result<VulkanSparseMemory*> VulkanDevice::sparseMemoryFor(SparseMemoryHandle memory) {
    VulkanSparseMemory* found = sparseMemories_.find(memory);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sparse memory handle is stale");
    }
    return found;
}

core::Result<SparseMemoryHandle> VulkanDevice::createSparseMemory(const SparseMemoryDesc& desc) {
    if (desc.size == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "sparse memory of 0 bytes");
    }
    if (desc.texture.isNull() && desc.buffer.isNull()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "Vulkan takes sparse memory's type from a texture or buffer; neither was given");
    }
    if (desc.tileBytes != 0) {
        return core::failure(core::ErrorCode::RhiUnsupported, "Vulkan sets tile sizes itself; tileBytes is 0");
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectBuffers();
    // The memory types and alignment every resource named takes.
    VkMemoryRequirements requirements{.size = desc.size, .alignment = 1, .memoryTypeBits = ~0U};
    if (!desc.texture.isNull()) {
        auto texture = textureFor(desc.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        if (!(*texture)->sparse()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the texture is not sparse");
        }
        requirements.memoryTypeBits &= (*texture)->tiling().memoryTypes;
        requirements.alignment = std::max(requirements.alignment, (*texture)->tiling().tileBytes);
    }
    if (!desc.buffer.isNull()) {
        auto buffer = bufferFor(desc.buffer);
        if (!buffer) {
            return std::unexpected(std::move(buffer.error()));
        }
        if (!(*buffer)->sparse()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer is not sparse");
        }
        requirements.memoryTypeBits &= (*buffer)->memoryTypes();
        requirements.alignment = std::max(requirements.alignment, (*buffer)->tileBytes());
    }
    if (requirements.memoryTypeBits == 0) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the texture and buffer share no memory type, so no one memory holds both");
    }
    VmaAllocationCreateInfo create{};
    create.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo placed{};
    if (const VkResult result = vmaAllocateMemory(allocator_, &requirements, &create, &allocation, &placed);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaAllocateMemory", .result = result});
    }
    const SparseMemoryHandle memory = sparseMemories_.emplace(
        VulkanSparseMemory{allocator_, allocation, placed.deviceMemory, placed.offset, placed.memoryType});
    if (memory.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} sparse memories",
                             desc_.maxSparseMemories);
    }
    return memory;
}

core::Result<void> VulkanDevice::destroySparseMemory(SparseMemoryHandle memory) {
    if (memory.isNull()) {
        return {};
    }
    if (sparseMemories_.find(memory) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sparse memory handle is stale");
    }
    // Retired against every submit and mapping made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!sparseMemories_.retire(memory, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sparse memory handle is stale");
    }
    collectBuffers();
    return {};
}

core::Result<Token> VulkanDevice::updateTileMappings(const TileMappingDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    for (const Token wait : desc.waits) {
        if (auto checked = checkToken(wait); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    if (desc.textureRegions.size() > desc_.maxTileMappings || desc.bufferRegions.size() > desc_.maxTileMappings) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a mapping takes at most {} regions of each kind",
                             desc_.maxTileMappings);
    }
    VulkanQueue& queue = *queues_[desc.queue];
    if (!queue.sparse) {
        return core::failure(core::ErrorCode::RhiUnsupported, "queue {}'s family does not bind sparse memory",
                             desc.queue);
    }
    const VulkanSparseMemory* memory = nullptr;
    if (!desc.memory.isNull()) {
        auto found = sparseMemoryFor(desc.memory);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        memory = *found;
    }
    const std::scoped_lock lock{queue.lock};
    queue.imageBinds.clear();
    queue.tailBinds.clear();
    queue.bufferBinds.clear();
    queue.waitSemaphores.clear();
    queue.waitSemaphoreValues.clear();
    if (auto added = addTextureBinds(queue, desc, memory); !added) {
        return std::unexpected(std::move(added.error()));
    }
    if (auto added = addBufferBinds(queue, desc, memory); !added) {
        return std::unexpected(std::move(added.error()));
    }
    // As submit waits: for the largest value asked of each queue.
    std::ranges::fill(queue.waitValues, 0);
    for (const Token wait : desc.waits) {
        queue.waitValues[wait.queue] = std::max(queue.waitValues[wait.queue], wait.value);
    }
    for (std::size_t index = 0; index < queues_.size(); ++index) {
        if (queue.waitValues[index] != 0) {
            // One per queue, which the arrays were made for.
            std::ignore = queue.waitSemaphores.push(queues_[index]->timeline);
            std::ignore = queue.waitSemaphoreValues.push(queue.waitValues[index]);
        }
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    VkTimelineSemaphoreSubmitInfo timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.waitSemaphoreValueCount = static_cast<std::uint32_t>(queue.waitSemaphoreValues.size());
    timeline.pWaitSemaphoreValues = queue.waitSemaphoreValues.data();
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &value;
    // The binds' resources, which addTextureBinds and addBufferBinds found.
    const VulkanTexture* texture = textures_.find(desc.texture);
    const VulkanBuffer* buffer = buffers_.find(desc.buffer);
    const VkSparseImageMemoryBindInfo imageInfo{.image = texture != nullptr ? texture->image() : VK_NULL_HANDLE,
                                                .bindCount = static_cast<std::uint32_t>(queue.imageBinds.size()),
                                                .pBinds = queue.imageBinds.data()};
    const VkSparseImageOpaqueMemoryBindInfo tailInfo{.image = imageInfo.image,
                                                     .bindCount = static_cast<std::uint32_t>(queue.tailBinds.size()),
                                                     .pBinds = queue.tailBinds.data()};
    const VkSparseBufferMemoryBindInfo bufferInfo{.buffer = buffer != nullptr ? buffer->buffer() : VK_NULL_HANDLE,
                                                  .bindCount = static_cast<std::uint32_t>(queue.bufferBinds.size()),
                                                  .pBinds = queue.bufferBinds.data()};
    VkBindSparseInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    info.pNext = &timeline;
    info.waitSemaphoreCount = static_cast<std::uint32_t>(queue.waitSemaphores.size());
    info.pWaitSemaphores = queue.waitSemaphores.data();
    info.bufferBindCount = bufferInfo.bindCount != 0 ? 1 : 0;
    info.pBufferBinds = &bufferInfo;
    info.imageOpaqueBindCount = tailInfo.bindCount != 0 ? 1 : 0;
    info.pImageOpaqueBinds = &tailInfo;
    info.imageBindCount = imageInfo.bindCount != 0 ? 1 : 0;
    info.pImageBinds = &imageInfo;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &queue.timeline;
    if (const VkResult result = table_.vkQueueBindSparse(queue.queue, 1, &info, VK_NULL_HANDLE); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkQueueBindSparse", .result = result});
    }
    queue.submitted.store(value, std::memory_order_release);
    return Token{.queue = desc.queue, .value = value};
}

} // namespace bfr::rhi::vulkan
