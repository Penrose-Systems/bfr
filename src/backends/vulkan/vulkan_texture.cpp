#include "vulkan/vulkan_texture.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/texture.h"
#include "core/slot_chains.h"
#include "rhi/bindless_slots.h"
#include "rhi/copy_check.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"
#include "rhi/tile_extent.h"
#include "vulkan/instance_table.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_buffer.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_instance.h"
#include "vulkan/vulkan_result.h"
#include "vulkan/vulkan_sparse.h"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

VkImageUsageFlags usageFor(const TextureUsage& usage) {
    const auto flag = [](bool set, VkImageUsageFlagBits bit) {
        return set ? VkImageUsageFlags{bit} : 0U;
    };
    return flag(usage.copySrc, VK_IMAGE_USAGE_TRANSFER_SRC_BIT) | flag(usage.copyDst, VK_IMAGE_USAGE_TRANSFER_DST_BIT) |
           flag(usage.sampled, VK_IMAGE_USAGE_SAMPLED_BIT) | flag(usage.storage, VK_IMAGE_USAGE_STORAGE_BIT) |
           flag(usage.colorAttachment, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
           flag(usage.depthStencil, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkImageType typeFor(TextureDimension dimension) {
    switch (dimension) {
        case TextureDimension::D1:
            return VK_IMAGE_TYPE_1D;
        case TextureDimension::D3:
            return VK_IMAGE_TYPE_3D;
        case TextureDimension::D2:
        case TextureDimension::Cube:
        default:
            return VK_IMAGE_TYPE_2D;
    }
}

// The layout shaders sample a texture of aspects in. A depth or stencil one
// is sampled in the layout a read-only attachment is in, which allows both,
// since a descriptor matches only the layout it names ("Image Layout Matching
// Rules"), and rendering may test it read-only while shaders sample it.
VkImageLayout sampledLayoutFor(VkImageAspectFlags aspects) {
    return (aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0
               ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
               : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

VkImageLayout layoutFor(TextureLayout layout, VkImageAspectFlags aspects) {
    switch (layout) {
        case TextureLayout::CopySrc:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case TextureLayout::CopyDst:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case TextureLayout::Sampled:
            return sampledLayoutFor(aspects);
        case TextureLayout::Storage:
            return VK_IMAGE_LAYOUT_GENERAL;
        case TextureLayout::ColorTarget:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        case TextureLayout::DepthStencilTarget:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case TextureLayout::DepthStencilRead:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        // vkCmdResolveImage takes the transfer layouts.
        case TextureLayout::ResolveSrc:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case TextureLayout::ResolveDst:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case TextureLayout::Undefined:
        default:
            return VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

// The view type of the whole texture: an array type when it has layers, so
// the shader declares what the desc made.
VkImageViewType viewTypeFor(const TextureDesc& desc) {
    switch (desc.dimension) {
        case TextureDimension::D1:
            return desc.layers > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        case TextureDimension::D3:
            return VK_IMAGE_VIEW_TYPE_3D;
        case TextureDimension::Cube:
            return desc.layers > kCubeFaces ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        case TextureDimension::D2:
        default:
            return desc.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}

// The aspects of a format: its color, or its depth and stencil.
VkImageAspectFlags aspectsOf(Format format) {
    const FormatBlock& block = blockOf(format);
    if (block.bytes != 0) {
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
    return (block.depthBytes != 0 ? VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT} : 0U) |
           (block.stencilBytes != 0 ? VkImageAspectFlags{VK_IMAGE_ASPECT_STENCIL_BIT} : 0U);
}

VkImageAspectFlags aspectFor(TextureAspect aspect) {
    switch (aspect) {
        case TextureAspect::Depth:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case TextureAspect::Stencil:
            return VK_IMAGE_ASPECT_STENCIL_BIT;
        case TextureAspect::Color:
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// region as Vulkan takes it: the buffer's rows in texels, which Vulkan counts
// in and BFR states in bytes, and its offsets as Vulkan's int32_t.
VkBufferImageCopy copyFor(const BufferTextureCopy& region, Format format) {
    const FormatBlock& block = blockOf(format);
    const std::uint32_t blockBytes = bytesOf(block, region.aspect);
    const std::uint64_t rowTexels = blockBytes == 0 ? 0 : region.bytesPerRow / blockBytes * block.width;
    return VkBufferImageCopy{.bufferOffset = region.bufferOffset,
                             .bufferRowLength = static_cast<std::uint32_t>(rowTexels),
                             .bufferImageHeight = region.rowsPerImage,
                             .imageSubresource = {.aspectMask = aspectFor(region.aspect),
                                                  .mipLevel = region.mipLevel,
                                                  .baseArrayLayer = region.layer,
                                                  .layerCount = 1},
                             .imageOffset = {.x = static_cast<std::int32_t>(region.x),
                                             .y = static_cast<std::int32_t>(region.y),
                                             .z = static_cast<std::int32_t>(region.z)},
                             .imageExtent = {.width = region.width, .height = region.height, .depth = region.depth}};
}

// The create flags of a sparse resource: bound tile by tile, partly
// resident, and aliased where asked.
VkImageCreateFlags sparseFlags(const TextureDesc& desc) {
    if (!desc.sparse) {
        return 0;
    }
    return VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
           (desc.sparseAliased ? VkImageCreateFlags{VK_IMAGE_CREATE_SPARSE_ALIASED_BIT} : 0U);
}

// What Vulkan says of a sparse image's tiles, each aspect's requirements up to
// the most an image has.
VulkanSparseTiling sparseTilingOf(const VolkDeviceTable& table, VkDevice device, VkImage image) {
    VkMemoryRequirements requirements{};
    table.vkGetImageMemoryRequirements(device, image, &requirements);
    VulkanSparseTiling tiling{.tileBytes = requirements.alignment, .memoryTypes = requirements.memoryTypeBits};
    std::uint32_t count = 0;
    table.vkGetImageSparseMemoryRequirements(device, image, &count, nullptr);
    tiling.count = std::min(count, kMaxSparseAspects);
    table.vkGetImageSparseMemoryRequirements(device, image, &tiling.count, tiling.aspects.data());
    return tiling;
}

} // namespace

VulkanTexture::VulkanTexture(const VolkDeviceTable& table, VkDevice device, VmaAllocator allocator, VkImage image,
                             VmaAllocation allocation, VkImageView view, VkImageView shaderView,
                             const TextureDesc& desc, VkImageAspectFlags aspects, TextureSlotsHeld slots,
                             const VulkanSparseTiling& sparse)
    : table_(&table),
      device_(device),
      allocator_(allocator),
      image_(image),
      allocation_(allocation),
      view_(view),
      shaderView_(shaderView),
      format_(desc.format),
      extent_{.width = desc.width, .height = desc.height},
      aspects_(aspects),
      slots_(slots),
      sparse_(desc.sparse),
      size_{.width = desc.width, .height = desc.height, .depth = desc.depth},
      tiling_(sparse),
      mipLevels_(desc.mipLevels),
      layers_(desc.layers) {}

VulkanTexture::VulkanTexture(VulkanTexture&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      allocator_(other.allocator_),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      view_(std::exchange(other.view_, VK_NULL_HANDLE)),
      shaderView_(std::exchange(other.shaderView_, VK_NULL_HANDLE)),
      format_(other.format_),
      extent_(other.extent_),
      aspects_(other.aspects_),
      slots_(other.slots_),
      sparse_(other.sparse_),
      size_(other.size_),
      tiling_(other.tiling_),
      attachmentViews_(std::move(other.attachmentViews_)),
      mipLevels_(other.mipLevels_),
      layers_(other.layers_) {}

VulkanTexture::~VulkanTexture() {
    for (VkImageView attachment : attachmentViews_) {
        if (attachment != VK_NULL_HANDLE) {
            table_->vkDestroyImageView(device_, attachment, nullptr);
        }
    }
    if (shaderView_ != VK_NULL_HANDLE && shaderView_ != view_) {
        table_->vkDestroyImageView(device_, shaderView_, nullptr);
    }
    if (view_ != VK_NULL_HANDLE) {
        table_->vkDestroyImageView(device_, view_, nullptr);
    }
    if (image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
    }
}

VkImageView VulkanTexture::attachmentView(std::uint32_t mipLevel, std::uint32_t layer) const {
    if (attachmentViews_.empty() || mipLevel >= mipLevels_ || layer >= layers_) {
        return VK_NULL_HANDLE;
    }
    return attachmentViews_[(std::size_t{mipLevel} * layers_) + layer];
}

core::Result<void> VulkanTexture::makeAttachmentViews(const TextureDesc& desc, VkFormat format, bool slices) {
    const bool volume = desc.dimension == TextureDimension::D3;
    if ((!desc.usage.colorAttachment && !desc.usage.depthStencil) || (volume && !slices)) {
        return {};
    }
    // A 3D texture's views are of its depth slices, fewer at each level; the
    // rest of a level's run stays VK_NULL_HANDLE.
    if (volume) {
        layers_ = desc.depth;
    }
    attachmentViews_.assign(std::size_t{mipLevels_} * layers_, VK_NULL_HANDLE);
    for (std::uint32_t mip = 0; mip < mipLevels_; ++mip) {
        const std::uint32_t layers = volume ? std::max(desc.depth >> mip, 1U) : layers_;
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            VkImageViewCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.image = image_;
            info.viewType = desc.dimension == TextureDimension::D1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY
                                                                   : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            info.format = format;
            info.subresourceRange = {.aspectMask = aspects_,
                                     .baseMipLevel = mip,
                                     .levelCount = 1,
                                     .baseArrayLayer = layer,
                                     .layerCount = layers - layer};
            VkImageView& view = attachmentViews_[(std::size_t{mip} * layers_) + layer];
            if (const VkResult result = table_->vkCreateImageView(device_, &info, nullptr, &view);
                result != VK_SUCCESS) {
                view = VK_NULL_HANDLE;
                return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateImageView", .result = result});
            }
        }
    }
    return {};
}

core::Result<VulkanTexture> createVulkanTexture(const VolkDeviceTable& table, VkDevice device, VmaAllocator allocator,
                                                const TextureDesc& desc, std::span<const std::uint32_t> families,
                                                const TextureSlotsHeld& slots, bool volumeColorTargets) {
    if (desc.sparse && desc.tileBytes != 0) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "Vulkan sets a sparse texture's tile size itself; tileBytes is 0");
    }
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    // A 3D color target is drawn to through 2D views of its slices, which
    // need the image made for them (VUID-VkImageViewCreateInfo-image-06723),
    // and sparse images may not be (VUID-VkImageCreateInfo-flags-09403).
    const bool slices = desc.dimension == TextureDimension::D3 && desc.usage.colorAttachment && !desc.sparse &&
                        volumeColorTargets;
    info.flags = (desc.dimension == TextureDimension::Cube ? VkImageCreateFlags{VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT}
                                                           : 0U) |
                 (slices ? VkImageCreateFlags{VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT} : 0U) | sparseFlags(desc);
    info.imageType = typeFor(desc.dimension);
    info.format = vulkanFormat(desc.format);
    info.extent = {.width = desc.width, .height = desc.height, .depth = desc.depth};
    info.mipLevels = desc.mipLevels;
    info.arrayLayers = desc.layers;
    info.samples = static_cast<VkSampleCountFlagBits>(desc.samples);
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usageFor(desc.usage);
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (families.size() > 1) {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
        info.pQueueFamilyIndices = families.data();
    }
    else {
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation made = VK_NULL_HANDLE;
    VulkanSparseTiling sparse;
    if (desc.sparse) {
        // Memory is mapped into it tile by tile, so none is placed now.
        if (const VkResult result = table.vkCreateImage(device, &info, nullptr, &image); result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateImage", .result = result});
        }
        sparse = sparseTilingOf(table, device, image);
    }
    else {
        VmaAllocationCreateInfo allocation{};
        allocation.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (const VkResult result = vmaCreateImage(allocator, &info, &allocation, &image, &made, nullptr);
            result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaCreateImage", .result = result});
        }
    }
    const VkImageAspectFlags aspects = aspectsOf(desc.format);
    // A view only where a use reads through one: the tables and attachments.
    // A texture only copied has none, as the API asks.
    VkImageView view = VK_NULL_HANDLE;
    VkImageView shaderView = VK_NULL_HANDLE;
    if (!desc.usage.sampled && !desc.usage.storage && !desc.usage.colorAttachment && !desc.usage.depthStencil) {
        return VulkanTexture{table, device, allocator, image, made, view, shaderView, desc, aspects, slots, sparse};
    }
    const auto createView = [&](VkImageAspectFlags viewAspects, VkImageView& out) -> core::Result<void> {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = viewTypeFor(desc);
        viewInfo.format = info.format;
        viewInfo.subresourceRange = {.aspectMask = viewAspects,
                                     .baseMipLevel = 0,
                                     .levelCount = VK_REMAINING_MIP_LEVELS,
                                     .baseArrayLayer = 0,
                                     .layerCount = VK_REMAINING_ARRAY_LAYERS};
        if (const VkResult result = table.vkCreateImageView(device, &viewInfo, nullptr, &out); result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateImageView", .result = result});
        }
        return {};
    };
    if (auto created = createView(aspects, view); !created) {
        vmaDestroyImage(allocator, image, made);
        return std::unexpected(std::move(created.error()));
    }
    // A descriptor sees one aspect of a depth-stencil image: its depth.
    shaderView = view;
    const bool depthStencil = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 &&
                              (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    if (depthStencil && (desc.usage.sampled || desc.usage.storage)) {
        if (auto created = createView(VK_IMAGE_ASPECT_DEPTH_BIT, shaderView); !created) {
            table.vkDestroyImageView(device, view, nullptr);
            vmaDestroyImage(allocator, image, made);
            return std::unexpected(std::move(created.error()));
        }
    }
    VulkanTexture texture{table, device, allocator, image, made, view, shaderView, desc, aspects, slots, sparse};
    if (auto views = texture.makeAttachmentViews(desc, info.format, slices); !views) {
        return std::unexpected(std::move(views.error()));
    }
    return texture;
}

// The device's texture calls, beside the textures they make.

core::Result<VulkanTexture*> VulkanDevice::textureFor(TextureHandle texture) {
    VulkanTexture* found = textures_.find(texture);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the texture handle is stale");
    }
    return found;
}

core::Result<TextureHandle> VulkanDevice::createTexture(const TextureDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    auto slots = bindless_.takeTexture(desc);
    if (!slots) {
        return std::unexpected(std::move(slots.error()));
    }
    auto made = createVulkanTexture(table_, device_, allocator_, desc, families_, *slots, features_.volumeColorTargets);
    if (!made) {
        bindless_.giveTexture(*slots);
        return std::unexpected(std::move(made.error()));
    }
    VkImageView view = made->shaderView();
    const VkImageLayout sampled = sampledLayoutFor(made->aspects());
    const TextureHandle texture = textures_.emplace(std::move(*made));
    if (texture.isNull()) {
        bindless_.giveTexture(*slots);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} textures",
                             desc_.maxTextures);
    }
    if (slots->sampled != core::SlotChains::kEnd) {
        layout_.writeSampledImage(slots->sampled, view, sampled);
    }
    if (slots->storage != core::SlotChains::kEnd) {
        layout_.writeStorageImage(slots->storage, view);
    }
    return texture;
}

core::Result<TextureSlots> VulkanDevice::textureSlots(TextureHandle texture) {
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reported((*found)->slots());
}

core::Result<void> VulkanDevice::destroyTexture(TextureHandle texture) {
    if (texture.isNull()) {
        return {};
    }
    if (textures_.find(texture) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the texture handle is stale");
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!textures_.retire(texture, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the texture handle is stale");
    }
    collectRetired();
    return {};
}

core::Result<void> VulkanDevice::textureBarrier(CommandListHandle commandList, TextureHandle texture,
                                                TextureLayout before, TextureLayout after, const TextureRange& range) {
    if (auto checked = checkLayouts(before, after); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.oldLayout = layoutFor(before, (*found)->aspects());
    barrier.newLayout = layoutFor(after, (*found)->aspects());
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = (*found)->image();
    // kRemaining is Vulkan's VK_REMAINING_MIP_LEVELS and _ARRAY_LAYERS.
    static_assert(kRemaining == VK_REMAINING_MIP_LEVELS);
    static_assert(kRemaining == VK_REMAINING_ARRAY_LAYERS);
    barrier.subresourceRange = {.aspectMask = (*found)->aspects(),
                                .baseMipLevel = range.mipLevel,
                                .levelCount = range.mipCount,
                                .baseArrayLayer = range.layer,
                                .layerCount = range.layerCount};
    // An image barrier orders only the image, so the memory barrier barrier()
    // records goes with it.
    VkMemoryBarrier2 memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memory.srcStageMask = barrier.srcStageMask;
    memory.srcAccessMask = barrier.srcAccessMask;
    memory.dstStageMask = barrier.dstStageMask;
    memory.dstAccessMask = barrier.dstAccessMask;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    table_.vkCmdPipelineBarrier2((*list)->buffer, &dependency);
    return {};
}

core::Result<void> VulkanDevice::copyBufferToTexture(CommandListHandle commandList, BufferHandle buffer,
                                                     TextureHandle texture,
                                                     std::span<const BufferTextureCopy> regions) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = bufferFor(buffer);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = textureFor(texture);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    if (auto checked = checkAspects(regions); !checked) {
        return checked;
    }
    for (const BufferTextureCopy& region : regions) {
        const VkBufferImageCopy copy = copyFor(region, (*to)->format());
        table_.vkCmdCopyBufferToImage((*list)->buffer, (*from)->buffer(), (*to)->image(),
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    }
    return {};
}

core::Result<void> VulkanDevice::copyTextureToBuffer(CommandListHandle commandList, TextureHandle texture,
                                                     BufferHandle buffer, std::span<const BufferTextureCopy> regions) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = textureFor(texture);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = bufferFor(buffer);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    if (auto checked = checkAspects(regions); !checked) {
        return checked;
    }
    for (const BufferTextureCopy& region : regions) {
        const VkBufferImageCopy copy = copyFor(region, (*from)->format());
        table_.vkCmdCopyImageToBuffer((*list)->buffer, (*from)->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      (*to)->buffer(), 1, &copy);
    }
    return {};
}

core::Result<void> VulkanDevice::copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
                                             std::span<const TextureCopy> regions) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = textureFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = textureFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    for (const TextureCopy& region : regions) {
        const VkImageAspectFlags aspect = (*from)->aspects();
        const VkImageCopy copy{.srcSubresource = {.aspectMask = aspect,
                                                  .mipLevel = region.srcMipLevel,
                                                  .baseArrayLayer = region.srcLayer,
                                                  .layerCount = 1},
                               .srcOffset = {.x = static_cast<std::int32_t>(region.srcX),
                                             .y = static_cast<std::int32_t>(region.srcY),
                                             .z = static_cast<std::int32_t>(region.srcZ)},
                               .dstSubresource = {.aspectMask = aspect,
                                                  .mipLevel = region.dstMipLevel,
                                                  .baseArrayLayer = region.dstLayer,
                                                  .layerCount = 1},
                               .dstOffset = {.x = static_cast<std::int32_t>(region.dstX),
                                             .y = static_cast<std::int32_t>(region.dstY),
                                             .z = static_cast<std::int32_t>(region.dstZ)},
                               .extent = {.width = region.width, .height = region.height, .depth = region.depth}};
        table_.vkCmdCopyImage((*list)->buffer, (*from)->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, (*to)->image(),
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    }
    return {};
}

core::Result<void> VulkanDevice::resolveTexture(CommandListHandle commandList, TextureHandle src,
                                                std::uint32_t srcMipLevel, std::uint32_t srcLayer, TextureHandle dst,
                                                std::uint32_t dstMipLevel, std::uint32_t dstLayer) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = textureFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = textureFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    const VkExtent3D size = (*from)->size();
    const VkImageResolve resolve{.srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .mipLevel = srcMipLevel,
                                                    .baseArrayLayer = srcLayer,
                                                    .layerCount = 1},
                                 .srcOffset = {},
                                 .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .mipLevel = dstMipLevel,
                                                    .baseArrayLayer = dstLayer,
                                                    .layerCount = 1},
                                 .dstOffset = {},
                                 .extent = {.width = levelSize(size.width, srcMipLevel),
                                            .height = levelSize(size.height, srcMipLevel),
                                            .depth = levelSize(size.depth, srcMipLevel)}};
    table_.vkCmdResolveImage((*list)->buffer, (*from)->image(),
                             layoutFor(TextureLayout::ResolveSrc, VK_IMAGE_ASPECT_COLOR_BIT), (*to)->image(),
                             layoutFor(TextureLayout::ResolveDst, VK_IMAGE_ASPECT_COLOR_BIT), 1, &resolve);
    return {};
}

namespace {

// The sparse requirements of tiling's aspect, null where it has none.
const VkSparseImageMemoryRequirements* requirementsOf(const VulkanSparseTiling& tiling, TextureAspect aspect) {
    const VkImageAspectFlags wanted = aspectFor(aspect);
    const auto aspects = std::span{tiling.aspects}.first(tiling.count);
    const auto found = std::ranges::find_if(aspects, [wanted](const VkSparseImageMemoryRequirements& entry) {
        return (entry.formatProperties.aspectMask & wanted) != 0;
    });
    return found != aspects.end() ? std::to_address(found) : nullptr;
}

} // namespace

core::Result<TextureTiling> VulkanDevice::textureTiling(TextureHandle texture, TextureAspect aspect) {
    if (auto checked = checkKnown(aspect, "texture aspect"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (!(*found)->sparse()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the texture is not sparse");
    }
    const VulkanSparseTiling& tiling = (*found)->tiling();
    const VkSparseImageMemoryRequirements* requirements = requirementsOf(tiling, aspect);
    if (requirements == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the texture has no sparse aspect {}",
                             std::to_underlying(aspect));
    }
    const VkExtent3D& shape = requirements->formatProperties.imageGranularity;
    return TextureTiling{
        .tileBytes = tiling.tileBytes,
        .shape = {.width = shape.width, .height = shape.height, .depth = shape.depth},
        .firstTailMip = requirements->imageMipTailFirstLod,
        .tailBytes = requirements->imageMipTailSize,
        .tailStride = requirements->imageMipTailStride,
        .singleTail = (requirements->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0,
    };
}

core::Result<SparseFormat> VulkanDevice::sparseFormat(const SparseFormatDesc& desc) const {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (desc.tileBytes != 0) {
        return core::failure(core::ErrorCode::RhiUnsupported, "Vulkan sets tile sizes itself; tileBytes is 0");
    }
    // A format the device may not be asked about is none it makes sparse.
    if (!hasFormat(desc.format, formats13_)) {
        return SparseFormat{};
    }
    VkPhysicalDeviceSparseImageFormatInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2;
    info.format = vulkanFormat(desc.format);
    info.type = typeFor(desc.dimension);
    info.samples = static_cast<VkSampleCountFlagBits>(desc.samples);
    info.usage = usageFor(desc.usage);
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    std::array<VkSparseImageFormatProperties2, kMaxSparseAspects> properties{};
    for (VkSparseImageFormatProperties2& entry : properties) {
        entry.sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2;
    }
    const InstanceTable& vk = instance_->instanceTable();
    std::uint32_t count = 0;
    vk.vkGetPhysicalDeviceSparseImageFormatProperties2(adapter_, &info, &count, nullptr);
    count = std::min(count, kMaxSparseAspects);
    vk.vkGetPhysicalDeviceSparseImageFormatProperties2(adapter_, &info, &count, properties.data());
    const VkImageAspectFlags wanted = aspectFor(desc.aspect);
    const auto listed = std::span{properties}.first(count);
    const auto found = std::ranges::find_if(listed, [wanted](const VkSparseImageFormatProperties2& entry) {
        return (entry.properties.aspectMask & wanted) != 0;
    });
    // Vulkan lists nothing for a texture it can't make sparse.
    if (found == listed.end()) {
        return SparseFormat{};
    }
    const VkExtent3D& shape = found->properties.imageGranularity;
    return SparseFormat{
        .supported = true,
        .shape = {.width = shape.width, .height = shape.height, .depth = shape.depth},
        .singleTail = (found->properties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0,
        .alignedMipSize = (found->properties.flags & VK_SPARSE_IMAGE_FORMAT_ALIGNED_MIP_SIZE_BIT) != 0,
    };
}

core::Result<void> VulkanDevice::addTextureBinds(VulkanQueue& queue, const TileMappingDesc& desc,
                                                 const VulkanSparseMemory* memory) {
    if (desc.texture.isNull()) {
        if (!desc.textureRegions.empty()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "texture regions were given with no texture");
        }
        return {};
    }
    auto found = textureFor(desc.texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const VulkanTexture& texture = **found;
    const VulkanSparseTiling& tiling = texture.tiling();
    for (const TextureTileRegion& region : desc.textureRegions) {
        if (auto checked = checkKnown(region.mapping, "tile mapping"); !checked) {
            return checked;
        }
        if (auto checked = checkKnown(region.aspect, "texture aspect"); !checked) {
            return checked;
        }
        const bool maps = region.mapping == TileMapping::Map;
        if (maps && memory == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a region maps, but no memory was given");
        }
        const VkSparseImageMemoryRequirements* requirements = requirementsOf(tiling, region.aspect);
        if (requirements == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the texture has no sparse aspect {}",
                                 std::to_underlying(region.aspect));
        }
        const VkDeviceMemory bound = maps ? memory->memory() : VK_NULL_HANDLE;
        const VkDeviceSize boundOffset = maps ? memory->offset() + region.memoryTile * tiling.tileBytes : 0;
        if (region.mipLevel >= requirements->imageMipTailFirstLod) {
            const bool single = (requirements->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;
            const VkDeviceSize layerTail = single ? 0 : region.layer * requirements->imageMipTailStride;
            // The texture's regions are at most maxTileMappings, which the array holds.
            std::ignore = queue.tailBinds.push(VkSparseMemoryBind{.resourceOffset = requirements->imageMipTailOffset +
                                                                                    layerTail +
                                                                                    region.tailTile * tiling.tileBytes,
                                                                  .size = region.tailTiles * tiling.tileBytes,
                                                                  .memory = bound,
                                                                  .memoryOffset = boundOffset,
                                                                  .flags = 0});
            continue;
        }
        // Vulkan binds in texels: whole tiles, or to the mip level's edge.
        const VkExtent3D& shape = requirements->formatProperties.imageGranularity;
        const VkExtent3D size = texture.size();
        const std::uint32_t levelWidth = levelSize(size.width, region.mipLevel);
        const std::uint32_t levelHeight = levelSize(size.height, region.mipLevel);
        const std::uint32_t levelDepth = levelSize(size.depth, region.mipLevel);
        const std::uint32_t x = region.x * shape.width;
        const std::uint32_t y = region.y * shape.height;
        const std::uint32_t z = region.z * shape.depth;
        std::ignore = queue.imageBinds.push(
            VkSparseImageMemoryBind{.subresource = {.aspectMask = aspectFor(region.aspect),
                                                    .mipLevel = region.mipLevel,
                                                    .arrayLayer = region.layer},
                                    .offset = {.x = static_cast<std::int32_t>(x),
                                               .y = static_cast<std::int32_t>(y),
                                               .z = static_cast<std::int32_t>(z)},
                                    .extent = {.width = extentWithin(region.width * shape.width, x, levelWidth),
                                               .height = extentWithin(region.height * shape.height, y, levelHeight),
                                               .depth = extentWithin(region.depth * shape.depth, z, levelDepth)},
                                    .memory = bound,
                                    .memoryOffset = boundOffset,
                                    .flags = 0});
    }
    return {};
}

} // namespace bfr::rhi::vulkan
