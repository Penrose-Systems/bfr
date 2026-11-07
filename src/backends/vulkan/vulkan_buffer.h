#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "vulkan/vma.h"

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi::vulkan {

/// The allocator a device's buffers take their memory from, over device,
/// which instance made on adapter at apiVersion. Destroy it with
/// vmaDestroyAllocator once every buffer is gone. Fails with RhiDeviceFailed
/// when VMA refuses.
[[nodiscard]] core::Result<VmaAllocator> createAllocator(VkInstance instance, VkPhysicalDevice adapter, VkDevice device,
                                                         std::uint32_t apiVersion);

/// A VkBuffer and the memory VMA placed it in, which it frees when it goes:
/// before the allocator does. mapped is where the CPU reaches upload and
/// readback memory, empty for device memory. slot is its place in the storage
/// buffer table, kEnd without storage usage; the device gives it back when
/// the buffer is collected. A sparse one has no allocation; tileBytes and
/// memoryTypes are its memory requirements' alignment and memoryTypeBits.
class VulkanBuffer {
public:
    VulkanBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, BufferMemory memory,
                 std::span<std::byte> mapped, std::uint32_t slot);
    VulkanBuffer(VmaAllocator allocator, VkBuffer buffer, std::uint32_t slot, const VkMemoryRequirements& sparse);
    ~VulkanBuffer();
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&&) = delete;

    [[nodiscard]] VkBuffer buffer() const { return buffer_; }
    [[nodiscard]] BufferMemory memory() const { return memory_; }
    [[nodiscard]] std::span<std::byte> mapped() const { return mapped_; }
    [[nodiscard]] std::uint32_t slot() const { return slot_; }
    [[nodiscard]] bool sparse() const { return sparse_; }
    [[nodiscard]] VkDeviceSize tileBytes() const { return tileBytes_; }
    [[nodiscard]] std::uint32_t memoryTypes() const { return memoryTypes_; }

private:
    VmaAllocator allocator_;
    VkBuffer buffer_;
    VmaAllocation allocation_;
    BufferMemory memory_;
    std::span<std::byte> mapped_;
    std::uint32_t slot_;
    bool sparse_ = false;
    VkDeviceSize tileBytes_ = 0;
    std::uint32_t memoryTypes_ = 0;
};

/// A buffer as desc says, with no memory when it is sparse, shared by the
/// queue families in families when there is more than one, so any of the
/// device's queues may use it without an ownership transfer, holding slot.
/// table is device's. Fails with RhiInvalidCall for a sparse buffer in host
/// memory, RhiUnsupported for a tileBytes, which Vulkan sets itself,
/// RhiOutOfMemory when memory runs out and RhiDeviceFailed when VMA or the
/// API refuses.
[[nodiscard]] core::Result<VulkanBuffer> createVulkanBuffer(const VolkDeviceTable& table, VkDevice device,
                                                            VmaAllocator allocator, const BufferDesc& desc,
                                                            std::span<const std::uint32_t> families,
                                                            std::uint32_t slot);

} // namespace bfr::rhi::vulkan
