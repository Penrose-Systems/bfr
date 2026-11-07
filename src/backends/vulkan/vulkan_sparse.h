#pragma once

#include "vulkan/vma.h"

#include <volk.h>

#include <cstdint>

namespace bfr::rhi::vulkan {

/// Device memory VMA placed for sparse resources' tiles, which it frees when
/// it goes: before the allocator does. It is offset bytes into memory, of
/// memory type type.
class VulkanSparseMemory {
public:
    VulkanSparseMemory(VmaAllocator allocator, VmaAllocation allocation, VkDeviceMemory memory, VkDeviceSize offset,
                       std::uint32_t type);
    ~VulkanSparseMemory();
    VulkanSparseMemory(const VulkanSparseMemory&) = delete;
    VulkanSparseMemory& operator=(const VulkanSparseMemory&) = delete;
    VulkanSparseMemory(VulkanSparseMemory&& other) noexcept;
    VulkanSparseMemory& operator=(VulkanSparseMemory&&) = delete;

    [[nodiscard]] VkDeviceMemory memory() const { return memory_; }
    [[nodiscard]] VkDeviceSize offset() const { return offset_; }
    [[nodiscard]] std::uint32_t type() const { return type_; }

private:
    VmaAllocator allocator_;
    VmaAllocation allocation_;
    VkDeviceMemory memory_;
    VkDeviceSize offset_;
    std::uint32_t type_;
};

} // namespace bfr::rhi::vulkan
