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
/// which instance made on adapter at apiVersion, giving buffers device
/// addresses where deviceAddresses (the bufferDeviceAddress feature is on).
/// Destroy it with vmaDestroyAllocator once every buffer is gone. Fails with
/// RhiDeviceFailed when VMA refuses.
[[nodiscard]] core::Result<VmaAllocator> createAllocator(VkInstance instance, VkPhysicalDevice adapter, VkDevice device,
                                                         std::uint32_t apiVersion, bool deviceAddresses);

/// How a device's buffers are addressed: whether an indirect buffer gets a
/// device address (the device has bufferDeviceAddress), and the address
/// alignment a buffer of each usage starts at, beyond what its memory
/// requirements give; 0 where the device has none of that usage.
struct BufferAddressing {
    bool indirectAddresses = false;
    /// RayTracingLimits::scratchAlignment.
    VkDeviceSize scratch = 0;
    /// The largest a build input's address keeps.
    VkDeviceSize input = 0;
    /// RayTracingLimits::tableAlignment.
    VkDeviceSize table = 0;
    /// A serialized structure's.
    VkDeviceSize data = 0;
    /// Build inputs are opacity micromaps' too.
    bool micromapInputs = false;
};

/// A VkBuffer and the memory VMA placed it in, which it frees when it goes:
/// before the allocator does. mapped is where the CPU reaches upload and
/// readback memory, empty for device memory. slot is its place in the storage
/// buffer table, kEnd without storage usage; the device gives it back when
/// the buffer is collected. address is its device address, 0 for a buffer
/// made without one. A sparse one has no allocation; tileBytes and
/// memoryTypes are its memory requirements' alignment and memoryTypeBits.
class VulkanBuffer {
public:
    VulkanBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, BufferMemory memory,
                 std::span<std::byte> mapped, std::uint32_t slot, VkDeviceAddress address);
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
    [[nodiscard]] VkDeviceAddress address() const { return address_; }

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
    VkDeviceAddress address_ = 0;
};

/// A buffer of size bytes, of usage, in memory, shared by families as
/// createVulkanBuffer's are, at an address that is a multiple of alignment,
/// holding slot, with its device address read where usage has one. Fails with
/// RhiDeviceFailed when VMA or the API refuses, and RhiOutOfMemory, as
/// codeFor says, when memory runs out.
[[nodiscard]] core::Result<VulkanBuffer> placeVulkanBuffer(const VolkDeviceTable& table, VkDevice device,
                                                           VmaAllocator allocator, VkDeviceSize size,
                                                           VkBufferUsageFlags usage, BufferMemory memory,
                                                           std::span<const std::uint32_t> families,
                                                           VkDeviceSize alignment, std::uint32_t slot);

/// A buffer as desc says, addressed and aligned as its usage takes, with
/// no memory when it is sparse, shared by the
/// queue families in families when there is more than one, so any of the
/// device's queues may use it without an ownership transfer, holding slot.
/// table is device's. Fails with RhiInvalidCall for a sparse buffer in host
/// memory, RhiUnsupported for a tileBytes, which Vulkan sets itself,
/// RhiOutOfMemory when memory runs out and RhiDeviceFailed when VMA or the
/// API refuses.
[[nodiscard]] core::Result<VulkanBuffer> createVulkanBuffer(const VolkDeviceTable& table, VkDevice device,
                                                            VmaAllocator allocator, const BufferDesc& desc,
                                                            std::span<const std::uint32_t> families,
                                                            const BufferAddressing& addressing, std::uint32_t slot);

} // namespace bfr::rhi::vulkan
