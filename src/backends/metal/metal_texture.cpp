#include "metal/metal_texture.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/texture.h"
#include "metal/metal_buffer.h"
#include "metal/metal_device.h"
#include "metal/metal_formats.h"
#include "metal/metal_util.h"
#include "rhi/format_blocks.h"

#include <Foundation/NSTypes.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTLBlitCommandEncoder.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <Metal/MTLResource.hpp>
#include <Metal/MTLTexture.hpp>
#include <Metal/MTLTypes.hpp>

#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <mutex>
#include <span>
#include <utility>

namespace bfr::rhi::metal {

namespace {

// Six faces make one cube, which Metal counts in arrayLength.
constexpr std::uint32_t kCubeFaces = 6;

MTL::TextureType typeFor(const TextureDesc& desc) {
    switch (desc.dimension) {
        case TextureDimension::D1:
            return desc.layers > 1 ? MTL::TextureType1DArray : MTL::TextureType1D;
        case TextureDimension::D3:
            return MTL::TextureType3D;
        case TextureDimension::Cube:
            return desc.layers > kCubeFaces ? MTL::TextureTypeCubeArray : MTL::TextureTypeCube;
        case TextureDimension::D2:
        default:
            if (desc.samples > 1) {
                return desc.layers > 1 ? MTL::TextureType2DMultisampleArray : MTL::TextureType2DMultisample;
            }
            return desc.layers > 1 ? MTL::TextureType2DArray : MTL::TextureType2D;
    }
}

MTL::TextureUsage usageFor(const TextureUsage& usage) {
    MTL::TextureUsage flags = MTL::TextureUsageUnknown;
    if (usage.sampled || usage.storage) {
        flags |= MTL::TextureUsageShaderRead;
    }
    if (usage.storage) {
        flags |= MTL::TextureUsageShaderWrite;
    }
    if (usage.colorAttachment || usage.depthStencil) {
        flags |= MTL::TextureUsageRenderTarget;
    }
    return flags;
}

// Which part of a depth-stencil format a buffer copy moves; none for a format
// with one aspect.
MTL::BlitOption optionFor(const FormatBlock& block, TextureAspect aspect) {
    if (block.depthBytes == 0 || block.stencilBytes == 0) {
        return MTL::BlitOptionNone;
    }
    return aspect == TextureAspect::Stencil ? MTL::BlitOptionStencilFromDepthStencil
                                            : MTL::BlitOptionDepthFromDepthStencil;
}

// Bytes between depth slices in the buffer; Metal takes 0 for one slice.
NS::UInteger bytesPerImage(const FormatBlock& block, const BufferTextureCopy& region) {
    return region.depth > 1 ? region.bytesPerRow * blocksOver(region.rowsPerImage, block.height) : 0;
}

} // namespace

core::Result<MetalTexture> createMetalTexture(MTL::Device& device, const TextureDesc& desc,
                                              Ref<MTL::ResidencySet> residency) {
    const Ref<MTL::TextureDescriptor> descriptor = adopt(MTL::TextureDescriptor::alloc()->init());
    descriptor->setTextureType(typeFor(desc));
    descriptor->setPixelFormat(pixelFormatFor(desc.format));
    descriptor->setWidth(desc.width);
    descriptor->setHeight(desc.height);
    descriptor->setDepth(desc.depth);
    descriptor->setArrayLength(desc.dimension == TextureDimension::Cube ? desc.layers / kCubeFaces : desc.layers);
    descriptor->setMipmapLevelCount(desc.mipLevels);
    descriptor->setSampleCount(desc.samples);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(usageFor(desc.usage));
    Ref<MTL::Texture> texture = adopt(device.newTexture(descriptor.get()));
    if (!texture) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no texture");
    }
    Resident resident{std::move(residency), texture.get()};
    return MetalTexture{.texture = std::move(texture), .format = desc.format, .resident = std::move(resident)};
}

// The device's texture calls, beside the textures they make.

core::Result<MetalTexture*> MetalDevice::textureFor(TextureHandle texture) {
    MetalTexture* found = textures_.find(texture);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the texture handle is stale");
    }
    return found;
}

core::Result<TextureHandle> MetalDevice::createTexture(const TextureDesc& desc) {
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectBuffers();
    // Held until made is gone, which takes it out of the set if it is not kept.
    const std::scoped_lock lock{residencyLock_};
    auto made = createMetalTexture(*device_.get(), desc, residency_);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    residency_->commit();
    const TextureHandle texture = textures_.emplace(std::move(*made));
    if (texture.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} textures", kBufferLimit);
    }
    return texture;
}

core::Result<void> MetalDevice::destroyTexture(TextureHandle texture) {
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

// Metal 4 has no texture layouts, so moving one is the full barrier alone.
core::Result<void> MetalDevice::textureBarrier(CommandListHandle commandList, TextureHandle texture, TextureLayout,
                                               TextureLayout) {
    if (auto found = textureFor(texture); !found) {
        return std::unexpected(std::move(found.error()));
    }
    return barrier(commandList);
}

core::Result<void> MetalDevice::copyBufferToTexture(CommandListHandle commandList, BufferHandle buffer,
                                                    TextureHandle texture, std::span<const BufferTextureCopy> regions) {
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
    // Metal 4 copies on a compute encoder. The pool drains the autoreleased
    // encoder reference.
    const AutoreleasePool pool;
    MTL4::ComputeCommandEncoder* encoder = (*list)->buffer->computeCommandEncoder();
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compute command encoder");
    }
    const FormatBlock& block = blockOf((*to)->format);
    for (const BufferTextureCopy& region : regions) {
        encoder->copyFromBuffer((*from)->buffer.get(), region.bufferOffset, region.bytesPerRow,
                                bytesPerImage(block, region),
                                MTL::Size::Make(region.width, region.height, region.depth), (*to)->texture.get(),
                                region.layer, region.mipLevel, MTL::Origin::Make(region.x, region.y, region.z),
                                optionFor(block, region.aspect));
    }
    encoder->endEncoding();
    return {};
}

core::Result<void> MetalDevice::copyTextureToBuffer(CommandListHandle commandList, TextureHandle texture,
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
    const AutoreleasePool pool;
    MTL4::ComputeCommandEncoder* encoder = (*list)->buffer->computeCommandEncoder();
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compute command encoder");
    }
    const FormatBlock& block = blockOf((*from)->format);
    for (const BufferTextureCopy& region : regions) {
        encoder->copyFromTexture((*from)->texture.get(), region.layer, region.mipLevel,
                                 MTL::Origin::Make(region.x, region.y, region.z),
                                 MTL::Size::Make(region.width, region.height, region.depth), (*to)->buffer.get(),
                                 region.bufferOffset, region.bytesPerRow, bytesPerImage(block, region),
                                 optionFor(block, region.aspect));
    }
    encoder->endEncoding();
    return {};
}

core::Result<void> MetalDevice::copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
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
    const AutoreleasePool pool;
    MTL4::ComputeCommandEncoder* encoder = (*list)->buffer->computeCommandEncoder();
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compute command encoder");
    }
    for (const TextureCopy& region : regions) {
        encoder->copyFromTexture((*from)->texture.get(), region.srcLayer, region.srcMipLevel,
                                 MTL::Origin::Make(region.srcX, region.srcY, region.srcZ),
                                 MTL::Size::Make(region.width, region.height, region.depth), (*to)->texture.get(),
                                 region.dstLayer, region.dstMipLevel,
                                 MTL::Origin::Make(region.dstX, region.dstY, region.dstZ));
    }
    encoder->endEncoding();
    return {};
}

} // namespace bfr::rhi::metal
