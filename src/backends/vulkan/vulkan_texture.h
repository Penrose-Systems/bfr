#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/texture.h"
#include "vulkan/vma.h"

#include <cstdint>
#include <span>

namespace bfr::rhi::vulkan {

/// A VkImage and the memory VMA placed it in, which it frees when it goes:
/// before the allocator does. aspects are the format's: color, or its depth
/// and stencil, which a barrier moves together.
class VulkanTexture {
public:
    VulkanTexture(VmaAllocator allocator, VkImage image, VmaAllocation allocation, Format format,
                  VkImageAspectFlags aspects);
    ~VulkanTexture();
    VulkanTexture(const VulkanTexture&) = delete;
    VulkanTexture& operator=(const VulkanTexture&) = delete;
    VulkanTexture(VulkanTexture&& other) noexcept;
    VulkanTexture& operator=(VulkanTexture&&) = delete;

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] Format format() const { return format_; }
    [[nodiscard]] VkImageAspectFlags aspects() const { return aspects_; }

private:
    VmaAllocator allocator_;
    VkImage image_;
    VmaAllocation allocation_;
    Format format_;
    VkImageAspectFlags aspects_;
};

/// A texture as desc says, in device memory, shared by the queue families in
/// families when there is more than one, as buffers are. Fails with
/// RhiOutOfMemory when memory runs out and RhiDeviceFailed when VMA or the API
/// refuses.
[[nodiscard]] core::Result<VulkanTexture> createVulkanTexture(VmaAllocator allocator, const TextureDesc& desc,
                                                              std::span<const std::uint32_t> families);

} // namespace bfr::rhi::vulkan
