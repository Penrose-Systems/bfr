#include "vulkan/vulkan_buffer.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "rhi/buffer_check.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"

#include <vulkan/vulkan_core.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

VkBufferUsageFlags usageFor(const BufferUsage& usage) {
    const auto flag = [](bool set, VkBufferUsageFlagBits bit) {
        return set ? VkBufferUsageFlags{bit} : 0U;
    };
    return flag(usage.copySrc, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) |
           flag(usage.copyDst, VK_BUFFER_USAGE_TRANSFER_DST_BIT) |
           flag(usage.uniform, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) |
           flag(usage.storage, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
           flag(usage.index, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) |
           flag(usage.indirect, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
}

// The memory properties each BufferMemory requires, and the ones it would
// like. The CPU never flushes, so mapped memory is coherent. Readback is read
// by the CPU, so cached memory is asked for but not required: every device has
// coherent host memory, and not every one has it cached.
struct MemoryProperties {
    VkMemoryPropertyFlags required = 0;
    VkMemoryPropertyFlags preferred = 0;
};

MemoryProperties propertiesFor(BufferMemory memory) {
    switch (memory) {
        case BufferMemory::Device:
            return {.required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .preferred = 0};
        case BufferMemory::Upload:
            return {.required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    .preferred = 0};
        case BufferMemory::Readback:
            return {.required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    .preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
    }
    return {};
}

} // namespace

core::Result<VmaAllocator> createAllocator(VkInstance instance, VkPhysicalDevice adapter, VkDevice device,
                                           std::uint32_t apiVersion) {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo info{};
    info.physicalDevice = adapter;
    info.device = device;
    info.instance = instance;
    info.vulkanApiVersion = apiVersion;
    info.pVulkanFunctions = &functions;
    VmaAllocator allocator = VK_NULL_HANDLE;
    if (const VkResult result = vmaCreateAllocator(&info, &allocator); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaCreateAllocator", .result = result});
    }
    return allocator;
}

VulkanBuffer::VulkanBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, BufferMemory memory,
                           std::span<std::byte> mapped)
    : allocator_(allocator), buffer_(buffer), allocation_(allocation), memory_(memory), mapped_(mapped) {}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : allocator_(other.allocator_),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      memory_(other.memory_),
      mapped_(std::exchange(other.mapped_, {})) {}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
}

core::Result<VulkanBuffer> createVulkanBuffer(VmaAllocator allocator, const BufferDesc& desc,
                                              std::span<const std::uint32_t> families) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = desc.size;
    info.usage = usageFor(desc.usage);
    if (families.size() > 1) {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
        info.pQueueFamilyIndices = families.data();
    }
    else {
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    VmaAllocationCreateInfo allocation{};
    const MemoryProperties properties = propertiesFor(desc.memory);
    allocation.requiredFlags = properties.required;
    allocation.preferredFlags = properties.preferred;
    if (desc.memory != BufferMemory::Device) {
        allocation.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation made = VK_NULL_HANDLE;
    VmaAllocationInfo placed{};
    if (const VkResult result = vmaCreateBuffer(allocator, &info, &allocation, &buffer, &made, &placed);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaCreateBuffer", .result = result});
    }
    std::span<std::byte> mapped;
    if (placed.pMappedData != nullptr) {
        mapped = {static_cast<std::byte*>(placed.pMappedData), static_cast<std::size_t>(desc.size)};
    }
    return VulkanBuffer{allocator, buffer, made, desc.memory, mapped};
}

// The device's buffer calls, beside the buffers they make.

core::Result<VulkanBuffer*> VulkanDevice::bufferFor(BufferHandle buffer) {
    VulkanBuffer* found = buffers_.find(buffer);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    return found;
}

void VulkanDevice::collectBuffers() {
    const std::uint64_t completed = submissions_.completed([this](std::uint32_t queue) {
        // A queue that can't say what it passed has passed nothing yet.
        const auto passed = completedValue(*queues_[queue]);
        return passed ? *passed : 0;
    });
    buffers_.collect(completed);
    textures_.collect(completed);
}

core::Result<BufferHandle> VulkanDevice::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so buffers the GPU is done with are freed here, never on the
    // frame path: freeing one frees the allocator's memory.
    collectBuffers();
    auto made = createVulkanBuffer(allocator_, desc, families_);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const BufferHandle buffer = buffers_.emplace(std::move(*made));
    if (buffer.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", kBufferLimit);
    }
    return buffer;
}

core::Result<std::span<std::byte>> VulkanDevice::mapBuffer(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*found)->memory() == BufferMemory::Device) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the buffer is in device memory, which the CPU does not map");
    }
    return (*found)->mapped();
}

core::Result<void> VulkanDevice::copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
                                            std::span<const BufferCopy> regions) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = bufferFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = bufferFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    for (const BufferCopy& region : regions) {
        const VkBufferCopy copy{.srcOffset = region.srcOffset, .dstOffset = region.dstOffset, .size = region.size};
        table_.vkCmdCopyBuffer((*list)->buffer, (*from)->buffer(), (*to)->buffer(), 1, &copy);
    }
    return {};
}

core::Result<void> VulkanDevice::destroyBuffer(BufferHandle buffer) {
    if (buffer.isNull()) {
        return {};
    }
    if (buffers_.find(buffer) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    // Retired against every submit made so far, on every queue, so it lives
    // until the GPU has run all of them.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!buffers_.retire(buffer, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    collectBuffers();
    return {};
}

} // namespace bfr::rhi::vulkan
