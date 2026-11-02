#include "vulkan/vulkan_texture.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/texture.h"
#include "rhi/format_blocks.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_buffer.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"

#include <vulkan/vulkan_core.h>

#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
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

VkImageLayout layoutFor(TextureLayout layout) {
    switch (layout) {
        case TextureLayout::CopySrc:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case TextureLayout::CopyDst:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case TextureLayout::Undefined:
        default:
            return VK_IMAGE_LAYOUT_UNDEFINED;
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
// in and BFR states in bytes.
VkBufferImageCopy copyFor(const BufferTextureCopy& region, Format format) {
    const FormatBlock& block = blockOf(format);
    const std::uint32_t blockBytes = bytesOf(block, region.aspect);
    return {.bufferOffset = region.bufferOffset,
            .bufferRowLength = blockBytes == 0
                                   ? 0
                                   : static_cast<std::uint32_t>(region.bytesPerRow / blockBytes) * block.width,
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

} // namespace

VulkanTexture::VulkanTexture(VmaAllocator allocator, VkImage image, VmaAllocation allocation, Format format,
                             VkImageAspectFlags aspects)
    : allocator_(allocator), image_(image), allocation_(allocation), format_(format), aspects_(aspects) {}

VulkanTexture::VulkanTexture(VulkanTexture&& other) noexcept
    : allocator_(other.allocator_),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      format_(other.format_),
      aspects_(other.aspects_) {}

VulkanTexture::~VulkanTexture() {
    if (image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
    }
}

core::Result<VulkanTexture> createVulkanTexture(VmaAllocator allocator, const TextureDesc& desc,
                                                std::span<const std::uint32_t> families) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = desc.dimension == TextureDimension::Cube ? VkImageCreateFlags{VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT}
                                                          : 0U;
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
    VmaAllocationCreateInfo allocation{};
    allocation.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation made = VK_NULL_HANDLE;
    if (const VkResult result = vmaCreateImage(allocator, &info, &allocation, &image, &made, nullptr);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaCreateImage", .result = result});
    }
    return VulkanTexture{allocator, image, made, desc.format, aspectsOf(desc.format)};
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
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectBuffers();
    auto made = createVulkanTexture(allocator_, desc, families_);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const TextureHandle texture = textures_.emplace(std::move(*made));
    if (texture.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} textures", kBufferLimit);
    }
    return texture;
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
    collectBuffers();
    return {};
}

core::Result<void> VulkanDevice::textureBarrier(CommandListHandle commandList, TextureHandle texture,
                                                TextureLayout before, TextureLayout after) {
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
    barrier.oldLayout = layoutFor(before);
    barrier.newLayout = layoutFor(after);
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = (*found)->image();
    barrier.subresourceRange = {.aspectMask = (*found)->aspects(),
                                .baseMipLevel = 0,
                                .levelCount = VK_REMAINING_MIP_LEVELS,
                                .baseArrayLayer = 0,
                                .layerCount = VK_REMAINING_ARRAY_LAYERS};
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

} // namespace bfr::rhi::vulkan
