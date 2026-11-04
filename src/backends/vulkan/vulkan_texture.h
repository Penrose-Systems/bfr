#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/texture.h"
#include "rhi/bindless_slots.h"
#include "vulkan/vma.h"

#include <volk.h>

#include <cstdint>
#include <span>

namespace bfr::rhi::vulkan {

/// A VkImage and the memory VMA placed it in, which it frees when it goes:
/// before the allocator does, with a view of the whole of it when its usage
/// has an attachment or the tables read through one, and a view of its depth
/// alone for the tables when the format has stencil too, which a descriptor
/// may not see both of. aspects are the format's: color, or its depth and
/// stencil, which a barrier moves together. slots are its places in the
/// tables, which the device gives back when it is collected.
class VulkanTexture {
public:
    VulkanTexture(const VolkDeviceTable& table, VkDevice device, VmaAllocator allocator, VkImage image,
                  VmaAllocation allocation, VkImageView view, VkImageView shaderView, const TextureDesc& desc,
                  VkImageAspectFlags aspects, TextureSlotsHeld slots);
    ~VulkanTexture();
    VulkanTexture(const VulkanTexture&) = delete;
    VulkanTexture& operator=(const VulkanTexture&) = delete;
    VulkanTexture(VulkanTexture&& other) noexcept;
    VulkanTexture& operator=(VulkanTexture&&) = delete;

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView view() const { return view_; }
    [[nodiscard]] VkImageView shaderView() const { return shaderView_; }
    [[nodiscard]] Format format() const { return format_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] VkImageAspectFlags aspects() const { return aspects_; }
    [[nodiscard]] const TextureSlotsHeld& slots() const { return slots_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VmaAllocator allocator_;
    VkImage image_;
    VmaAllocation allocation_;
    VkImageView view_;
    VkImageView shaderView_;
    Format format_;
    VkExtent2D extent_;
    VkImageAspectFlags aspects_;
    TextureSlotsHeld slots_;
};

/// A texture as desc says, in device memory, shared by the queue families in
/// families when there is more than one, as buffers are, holding slots. table
/// is device's and outlives it. Fails with RhiOutOfMemory when memory runs
/// out and RhiDeviceFailed when VMA or the API refuses.
[[nodiscard]] core::Result<VulkanTexture> createVulkanTexture(const VolkDeviceTable& table, VkDevice device,
                                                              VmaAllocator allocator, const TextureDesc& desc,
                                                              std::span<const std::uint32_t> families,
                                                              const TextureSlotsHeld& slots);

} // namespace bfr::rhi::vulkan
