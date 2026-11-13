#include "metal/metal_texture.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/texture.h"
#include "core/slot_chains.h"
#include "metal/metal_buffer.h"
#include "metal/metal_device.h"
#include "metal/metal_formats.h"
#include "metal/metal_sparse.h"
#include "metal/metal_util.h"
#include "rhi/bindless_slots.h"
#include "rhi/copy_check.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"
#include "rhi/tile_extent.h"

#include <Foundation/NSTypes.hpp>
#include <Metal/MTL4CommandBuffer.hpp>
#include <Metal/MTL4CommandQueue.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTL4RenderCommandEncoder.hpp>
#include <Metal/MTL4RenderPass.hpp>
#include <Metal/MTLBlitCommandEncoder.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLRenderPass.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <Metal/MTLResource.hpp>
#include <Metal/MTLResourceStateCommandEncoder.hpp>
#include <Metal/MTLTexture.hpp>
#include <Metal/MTLTypes.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <utility>

namespace bfr::rhi::metal {

namespace {

// Six faces make one cube, which Metal counts in arrayLength.
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

// What Metal says of a sparse texture's tiles at page: the page's bytes and
// the tile's shape from the device (MTLDevice.hpp:537-539), and its tail from
// the texture (MTLTexture.hpp:233, 289). Metal gives no tail stride and says
// nothing of layers sharing one tail. Fails with RhiUnsupported for a shape or
// first tail mip past what TextureTiling holds. That the shape is in texels
// is unverified: the header names no unit.
core::Result<TextureTiling> tilingOf(MTL::Device& device, const MTL::Texture& texture, MTL::SparsePageSize page) {
    const MTL::Size shape = device.sparseTileSize(texture.textureType(), texture.pixelFormat(), texture.sampleCount(),
                                                  page);
    const NS::UInteger firstTail = texture.firstMipmapInTail();
    if (!std::in_range<std::uint32_t>(shape.width) || !std::in_range<std::uint32_t>(shape.height) ||
        !std::in_range<std::uint32_t>(shape.depth) || !std::in_range<std::uint32_t>(firstTail)) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "Metal's tile shape or first tail mip is past what BFR holds, 32 bits");
    }
    return TextureTiling{.tileBytes = device.sparseTileSizeInBytes(page),
                         .shape = {.width = static_cast<std::uint32_t>(shape.width),
                                   .height = static_cast<std::uint32_t>(shape.height),
                                   .depth = static_cast<std::uint32_t>(shape.depth)},
                         .firstTailMip = static_cast<std::uint32_t>(firstTail),
                         .tailBytes = texture.tailSizeInBytes(),
                         .tailStride = 0,
                         .singleTail = false};
}

} // namespace

core::Result<MetalTexture> createMetalTexture(MTL::Device& device, const TextureDesc& desc,
                                              Ref<MTL::ResidencySet> residency) {
    // Metal counts a cube's layers in cubes, so a remainder has nowhere to go.
    if (desc.dimension == TextureDimension::Cube && desc.layers % kCubeFaces != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a cube texture's {} layers are not whole cubes of {}",
                             desc.layers, kCubeFaces);
    }
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
    std::optional<MTL::SparsePageSize> page;
    if (desc.sparse) {
        // MTLTexture.hpp has no aliased sparse texture.
        if (desc.sparseAliased) {
            return core::failure(core::ErrorCode::RhiUnsupported, "Metal has no aliased sparse texture");
        }
        auto found = pageSizeFor(device, desc.tileBytes);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        page = *found;
        // Made at its page size, with no memory of its own (MTLTexture.hpp:158).
        descriptor->setPlacementSparsePageSize(*page);
    }
    Ref<MTL::Texture> texture = adopt(device.newTexture(descriptor.get()));
    if (!texture) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no texture");
    }
    TextureTiling tiling;
    if (page) {
        auto read = tilingOf(device, *texture.get(), *page);
        if (!read) {
            return std::unexpected(std::move(read.error()));
        }
        tiling = *read;
    }
    Resident resident{std::move(residency), texture.get()};
    return MetalTexture{.texture = std::move(texture),
                        .format = desc.format,
                        .width = desc.width,
                        .height = desc.height,
                        .depth = desc.depth,
                        .page = page,
                        .tiling = tiling,
                        .slots = {},
                        .resident = std::move(resident)};
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
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    // Held until made is gone, which takes it out of the set if it is not kept.
    const std::scoped_lock lock{residencyLock_};
    auto slots = bindless_.takeTexture(desc);
    if (!slots) {
        return std::unexpected(std::move(slots.error()));
    }
    auto made = createMetalTexture(*device_.get(), desc, residency_);
    if (!made) {
        bindless_.giveTexture(*slots);
        return std::unexpected(std::move(made.error()));
    }
    made->slots = *slots;
    residency_->commit();
    const MTL::ResourceID id = made->texture->gpuResourceID();
    const TextureHandle texture = textures_.emplace(std::move(*made));
    if (texture.isNull()) {
        bindless_.giveTexture(*slots);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} textures",
                             desc_.maxTextures);
    }
    if (slots->sampled != core::SlotChains::kEnd) {
        tables().writeSampledTexture(slots->sampled, id);
    }
    if (slots->storage != core::SlotChains::kEnd) {
        tables().writeStorageTexture(slots->storage, id);
    }
    return texture;
}

core::Result<TextureSlots> MetalDevice::textureSlots(TextureHandle texture) {
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reported((*found)->slots);
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
    collectRetired();
    return {};
}

// Metal 4 has no texture layouts, so moving one is the full barrier alone.
core::Result<void> MetalDevice::textureBarrier(CommandListHandle commandList, TextureHandle texture,
                                               TextureLayout before, TextureLayout after,
                                               const TextureRange& /*range*/) {
    // Metal 4 has no layouts, but a layout BFR doesn't have fails here as on
    // every backend.
    if (auto checked = checkLayouts(before, after); !checked) {
        return checked;
    }
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
    if (auto checked = checkAspects(regions); !checked) {
        return checked;
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
    if (auto checked = checkAspects(regions); !checked) {
        return checked;
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

// A render pass with no draws, which loads src and stores it as it resolves
// into dst, so src keeps its samples as vkCmdResolveImage and
// ResolveSubresource leave them. Godot resolves so on Metal 3
// (metal3_objects.cpp:479-492); that an MTL4 render encoder
// (MTL4CommandBuffer.hpp:81) resolves the same way is an inference, with no
// Metal 4 reference. The slot's own descriptor is refilled, so a resolve
// allocates none.
core::Result<void> MetalDevice::resolveTexture(CommandListHandle commandList, TextureHandle src,
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
    // The pool drains the autoreleased encoder reference.
    const AutoreleasePool pool;
    MTL4::RenderPassDescriptor* pass = lists_[(*list)->slot].resolvePass.get();
    MTL::RenderPassColorAttachmentDescriptor* color = pass->colorAttachments()->object(0);
    color->setTexture((*from)->texture.get());
    color->setLevel(srcMipLevel);
    color->setSlice(srcLayer);
    color->setLoadAction(MTL::LoadActionLoad);
    color->setStoreAction(MTL::StoreActionStoreAndMultisampleResolve);
    color->setResolveTexture((*to)->texture.get());
    color->setResolveLevel(dstMipLevel);
    color->setResolveSlice(dstLayer);
    MTL4::RenderCommandEncoder* encoder = (*list)->buffer->renderCommandEncoder(pass);
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no render command encoder");
    }
    encoder->endEncoding();
    return {};
}

namespace {

// Metal's tiling takes no aspect (sparseTileSize, MTLDevice.hpp:537), so BFR
// names it by the texture's color or depth aspect; it gives none of a stencil
// aspect alone.
core::Result<void> checkSparseAspect(Format format, TextureAspect aspect) {
    if (auto checked = checkAspect(format, aspect); !checked) {
        return checked;
    }
    if (aspect == TextureAspect::Stencil) {
        return core::failure(core::ErrorCode::RhiUnsupported, "Metal gives no tiling of a stencil aspect");
    }
    return {};
}

} // namespace

core::Result<TextureTiling> MetalDevice::textureTiling(TextureHandle texture, TextureAspect aspect) {
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (!(*found)->page) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the texture is not sparse");
    }
    if (auto checked = checkSparseAspect((*found)->format, aspect); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return (*found)->tiling;
}

// Metal has no query of whether a texture can be made sparse, and
// sparseTileSize (MTLDevice.hpp:537) wants a texture type SparseFormatDesc
// can't name, having no layers to tell an array from one texture, and has no
// way to fail; so nothing is given.
core::Result<SparseFormat> MetalDevice::sparseFormat(const SparseFormatDesc& desc) const {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto page = pageSizeFor(*device_.get(), desc.tileBytes); !page) {
        return std::unexpected(std::move(page.error()));
    }
    return SparseFormat{};
}

// A region in texels: its tiles times the tile's shape, clamped to the mip
// level's edge, at the level and slice it names. The heap offset is in bytes,
// tiles times the texture's tile bytes. The header (MTL4CommandQueue.hpp:63-70)
// names neither unit, and Metal has both pixel and tile regions
// (MTLDevice.hpp:333-335), so texels and bytes are both unverified. The tail is refused: Metal 4's op
// takes a region, with no offset into the tail.
core::Result<const MTL::Texture*> MetalDevice::addTextureMappings(MetalQueue& queue, const TileMappingDesc& desc,
                                                                  const MetalSparseMemory* memory) {
    if (desc.texture.isNull()) {
        if (!desc.textureRegions.empty()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "texture regions were given with no texture");
        }
        return nullptr;
    }
    auto found = textureFor(desc.texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const MetalTexture& texture = **found;
    if (memory != nullptr) {
        if (auto fits = checkPageFits(*texture.page, memory->page); !fits) {
            return std::unexpected(std::move(fits.error()));
        }
    }
    const TextureTiling& tiling = texture.tiling;
    for (const TextureTileRegion& region : desc.textureRegions) {
        if (auto checked = checkKnown(region.mapping, "tile mapping"); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
        if (auto checked = checkSparseAspect(texture.format, region.aspect); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
        const bool maps = region.mapping == TileMapping::Map;
        if (maps && memory == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a region maps, but no memory was given");
        }
        if (region.mipLevel >= tiling.firstTailMip) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 "mip level {} is in the tail, which Metal 4's mapping op has no offset into",
                                 region.mipLevel);
        }
        const NS::UInteger x = NS::UInteger{region.x} * tiling.shape.width;
        const NS::UInteger y = NS::UInteger{region.y} * tiling.shape.height;
        const NS::UInteger z = NS::UInteger{region.z} * tiling.shape.depth;
        const MTL::Region pixels =
            MTL::Region::Make3D(x, y, z,
                                extentWithin(NS::UInteger{region.width} * tiling.shape.width, x,
                                             NS::UInteger{levelSize(texture.width, region.mipLevel)}),
                                extentWithin(NS::UInteger{region.height} * tiling.shape.height, y,
                                             NS::UInteger{levelSize(texture.height, region.mipLevel)}),
                                extentWithin(NS::UInteger{region.depth} * tiling.shape.depth, z,
                                             NS::UInteger{levelSize(texture.depth, region.mipLevel)}));
        // The texture's regions are at most maxTileMappings, which the array holds.
        std::ignore = queue.textureMappings.push(
            MTL4::UpdateSparseTextureMappingOperation{.mode = maps ? MTL::SparseTextureMappingModeMap
                                                                   : MTL::SparseTextureMappingModeUnmap,
                                                      .textureRegion = pixels,
                                                      .textureLevel = region.mipLevel,
                                                      .textureSlice = region.layer,
                                                      .heapOffset = maps ? region.memoryTile * tiling.tileBytes : 0});
    }
    return texture.texture.get();
}

} // namespace bfr::rhi::metal
