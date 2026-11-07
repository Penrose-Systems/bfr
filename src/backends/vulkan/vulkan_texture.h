#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/texture.h"
#include "rhi/bindless_slots.h"
#include "vulkan/vma.h"

#include <volk.h>

#include <array>
#include <cstdint>
#include <span>

namespace bfr::rhi::vulkan {

/// The most VkSparseImageMemoryRequirements an image has: one per aspect,
/// color or depth and stencil, and one for metadata.
inline constexpr std::uint32_t kMaxSparseAspects = 4;

/// What a sparse image's tiles are, read when it is made, so mapping reads
/// nothing back: its memory requirements' alignment, which is a tile's bytes,
/// and memoryTypeBits, and its sparse requirements, count of them.
struct VulkanSparseTiling {
    VkDeviceSize tileBytes = 0;
    std::uint32_t memoryTypes = 0;
    std::uint32_t count = 0;
    std::array<VkSparseImageMemoryRequirements, kMaxSparseAspects> aspects{};
};

/// A VkImage and the memory VMA placed it in, which it frees when it goes:
/// before the allocator does, with a view of the whole of it when its usage
/// has an attachment or the tables read through one, and a view of its depth
/// alone for the tables when the format has stencil too, which a descriptor
/// may not see both of. aspects are the format's: color, or its depth and
/// stencil, which a barrier moves together. slots are its places in the
/// tables, which the device gives back when it is collected. A sparse one
/// has no allocation; sparse says what its tiles are.
class VulkanTexture {
public:
    VulkanTexture(const VolkDeviceTable& table, VkDevice device, VmaAllocator allocator, VkImage image,
                  VmaAllocation allocation, VkImageView view, VkImageView shaderView, const TextureDesc& desc,
                  VkImageAspectFlags aspects, TextureSlotsHeld slots, const VulkanSparseTiling& sparse);
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
    [[nodiscard]] bool sparse() const { return sparse_; }
    /// The whole texture's size in texels, its depth or 1.
    [[nodiscard]] VkExtent3D size() const { return size_; }
    [[nodiscard]] const VulkanSparseTiling& tiling() const { return tiling_; }

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
    bool sparse_;
    VkExtent3D size_;
    VulkanSparseTiling tiling_;
};

/// A texture as desc says, in device memory, or with none when it is sparse,
/// shared by the queue families in families when there is more than one, as
/// buffers are, holding slots. table is device's and outlives it. Fails with
/// RhiUnsupported for a tileBytes, which Vulkan sets itself, RhiOutOfMemory
/// when memory runs out and RhiDeviceFailed when VMA or the API refuses.
[[nodiscard]] core::Result<VulkanTexture> createVulkanTexture(const VolkDeviceTable& table, VkDevice device,
                                                              VmaAllocator allocator, const TextureDesc& desc,
                                                              std::span<const std::uint32_t> families,
                                                              const TextureSlotsHeld& slots);

} // namespace bfr::rhi::vulkan
