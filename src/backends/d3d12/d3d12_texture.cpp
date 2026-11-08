#include "d3d12/d3d12_texture.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/texture.h"
#include "core/slot_chains.h"
#include "d3d12/d3d12_buffer.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_formats.h"
#include "d3d12/d3d12_result.h"
#include "d3d12/d3d12_sparse.h"
#include "rhi/bindless_slots.h"
#include "rhi/copy_check.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"

#include <d3d12.h>
#include <D3D12MemAlloc.h>
#include <dxgiformat.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <utility>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

D3D12_RESOURCE_DIMENSION dimensionFor(TextureDimension dimension) {
    switch (dimension) {
        case TextureDimension::D1:
            return D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        case TextureDimension::D3:
            return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        case TextureDimension::D2:
        case TextureDimension::Cube:
        default:
            return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    }
}

D3D12_RESOURCE_FLAGS flagsFor(const TextureUsage& usage) {
    auto flags = D3D12_RESOURCE_FLAG_NONE;
    if (usage.storage) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if (usage.colorAttachment) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if (usage.depthStencil) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    }
    return flags;
}

// The one kind of resource a ResourceHeapTier 1 heap holds that a texture of
// usage is: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES (d3d12.h) denies resources
// made with ALLOW_RENDER_TARGET or ALLOW_DEPTH_STENCIL, which flagsFor sets
// for these usages.
D3D12_HEAP_FLAGS tileHeapOf(const TextureUsage& usage) {
    return usage.colorAttachment || usage.depthStencil ? D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES
                                                       : D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
}

// Depth-stencil formats keep depth and stencil in planes 0 and 1.
bool planar(const FormatBlock& block) {
    return block.depthBytes != 0 && block.stencilBytes != 0;
}

std::uint32_t planeOf(const FormatBlock& block, TextureAspect aspect) {
    return aspect == TextureAspect::Stencil && planar(block) ? 1 : 0;
}

UINT subresourceOf(const D3D12Texture& texture, std::uint32_t mip, std::uint32_t layer, std::uint32_t plane) {
    return mip + (layer * texture.mipLevels) + (plane * texture.mipLevels * texture.layers);
}

} // namespace

D3D12Target targetOf(const D3D12Texture& texture) {
    constexpr std::uint32_t kDepthAndStencil = 2;
    const std::uint32_t planes = planar(blockOf(texture.desc.format)) ? kDepthAndStencil : 1;
    return {.resource = texture.resource.Get(),
            .mipLevels = texture.mipLevels,
            .layers = texture.layers,
            .planes = planes,
            .firstPlane = 0,
            .planeCount = planes};
}

D3D12Target aspectsOf(const D3D12Texture& texture, bool depth, bool stencil) {
    const FormatBlock& block = blockOf(texture.desc.format);
    const bool depthPlane = depth && block.depthBytes != 0;
    const bool stencilPlane = stencil && block.stencilBytes != 0;
    D3D12Target target = targetOf(texture);
    target.firstPlane = planeOf(block, depthPlane ? TextureAspect::Depth : TextureAspect::Stencil);
    target.planeCount = (depthPlane ? 1U : 0U) + (stencilPlane ? 1U : 0U);
    return target;
}

void discardTarget(ID3D12GraphicsCommandList* list, const D3D12Target& target) {
    // With one mip and every plane, every subresource is covered.
    if (target.mipLevels == 1 && target.planeCount == target.planes) {
        list->DiscardResource(target.resource, nullptr);
        return;
    }
    for (std::uint32_t plane = target.firstPlane; plane < target.firstPlane + target.planeCount; ++plane) {
        for (std::uint32_t layer = 0; layer < target.layers; ++layer) {
            const D3D12_DISCARD_REGION region{.NumRects = 0,
                                              .pRects = nullptr,
                                              .FirstSubresource = (layer * target.mipLevels) +
                                                                  (plane * target.mipLevels * target.layers),
                                              .NumSubresources = 1};
            list->DiscardResource(target.resource, &region);
        }
    }
}

namespace {

// What a buffer holds of aspect: the texture's own format, or the plane's
// format for a planar depth-stencil one (the d3dx12 format table's planes).
DXGI_FORMAT footprintFormat(const D3D12Texture& texture, TextureAspect aspect) {
    if (!planar(blockOf(texture.desc.format))) {
        return texture.dxgi;
    }
    return aspect == TextureAspect::Stencil ? DXGI_FORMAT_R8_TYPELESS : DXGI_FORMAT_R32_TYPELESS;
}

std::uint32_t roundToBlock(std::uint32_t texels, std::uint32_t size) {
    return blocksOver(texels, size) * size;
}

// D3D12 copies a depth or stencil plane, and a multisampled texture, only
// whole, with no box.
bool whole(const D3D12Texture& texture, TextureAspect aspect) {
    return aspect != TextureAspect::Color || texture.samples > 1;
}

// The buffer side of region: where its bytes start and how its rows lie. A
// whole copy's footprint is the mip level itself, whatever rows_per_image is.
D3D12_TEXTURE_COPY_LOCATION bufferSide(ID3D12Resource* buffer, const D3D12Texture& texture,
                                       const BufferTextureCopy& region) {
    const FormatBlock& block = blockOf(texture.desc.format);
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = buffer;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint = {
        .Offset = region.bufferOffset,
        .Footprint = {.Format = footprintFormat(texture, region.aspect),
                      .Width = roundToBlock(region.width, block.width),
                      .Height = roundToBlock(whole(texture, region.aspect) ? region.height : region.rowsPerImage,
                                             block.height),
                      .Depth = region.depth,
                      .RowPitch = static_cast<UINT>(region.bytesPerRow)}};
    return location;
}

// Fails with RhiInvalidCall when a region's bytes per row has no room in the
// footprint's UINT RowPitch, before any region is recorded.
core::Result<void> checkRowPitches(std::span<const BufferTextureCopy> regions) {
    for (const BufferTextureCopy& region : regions) {
        if (!std::in_range<UINT>(region.bytesPerRow)) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes per row is more than D3D12 takes",
                                 region.bytesPerRow);
        }
    }
    return {};
}

D3D12_TEXTURE_COPY_LOCATION textureSide(const D3D12Texture& texture, UINT subresource) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = texture.resource.Get();
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = subresource;
    return location;
}

D3D12_BOX boxOf(const FormatBlock& block, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t width,
                std::uint32_t height, std::uint32_t depth) {
    return {.left = x,
            .top = y,
            .front = z,
            .right = x + roundToBlock(width, block.width),
            .bottom = y + roundToBlock(height, block.height),
            .back = z + depth};
}

} // namespace

namespace {

// A color texture's heap holds its render target view; a depth texture's its
// writable depth-stencil view and its read-only one.
constexpr UINT kColorTargetViews = 1;
constexpr UINT kDepthTargetViews = 2;

// The heap of count descriptors a texture's render target or depth-stencil
// views live in: made with the texture, so a rendering allocates nothing.
core::Result<ComPtr<ID3D12DescriptorHeap>> createTargetHeap(ID3D12Device& device, D3D12_DESCRIPTOR_HEAP_TYPE type,
                                                            UINT count) {
    const D3D12_DESCRIPTOR_HEAP_DESC heapDesc{.Type = type,
                                              .NumDescriptors = count,
                                              .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                                              .NodeMask = 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (const HRESULT result = device.CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12Device::CreateDescriptorHeap", .result = result});
    }
    return heap;
}

// The whole first mip level of a depth texture as a depth-stencil view of
// format, every layer, with flags; a resource made typeless needs the format
// named.
D3D12_DEPTH_STENCIL_VIEW_DESC depthViewFor(const D3D12Texture& texture, DXGI_FORMAT format, D3D12_DSV_FLAGS flags) {
    D3D12_DEPTH_STENCIL_VIEW_DESC view{};
    view.Format = format;
    view.Flags = flags;
    const bool array = texture.layers > 1;
    if (texture.samples > 1) {
        if (array) {
            view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
            view.Texture2DMSArray = {.FirstArraySlice = 0, .ArraySize = texture.layers};
        }
        else {
            view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
        }
    }
    else if (array) {
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        view.Texture2DArray = {.MipSlice = 0, .FirstArraySlice = 0, .ArraySize = texture.layers};
    }
    else {
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        view.Texture2D = {.MipSlice = 0};
    }
    return view;
}

// The flags of a depth-stencil view that writes none of the aspects format has.
D3D12_DSV_FLAGS readOnlyFlagsFor(Format format) {
    const FormatBlock& block = blockOf(format);
    auto flags = D3D12_DSV_FLAG_NONE;
    if (block.depthBytes != 0) {
        flags |= D3D12_DSV_FLAG_READ_ONLY_DEPTH;
    }
    if (block.stencilBytes != 0) {
        flags |= D3D12_DSV_FLAG_READ_ONLY_STENCIL;
    }
    return flags;
}

} // namespace

core::Result<D3D12Texture> createD3D12Texture(D3D12MA::Allocator& allocator, ID3D12Device& device,
                                              const TextureDesc& desc, const TextureSlotsHeld& slots) {
    if (desc.sparse && desc.tileBytes != D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES) {
        return core::failure(core::ErrorCode::RhiUnsupported, "D3D12's tiles are {} bytes; tileBytes is {}",
                             D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, desc.tileBytes);
    }
    const DxgiFormats formats = dxgiFormatsFor(desc.format);
    const bool slices = desc.dimension == TextureDimension::D3;
    const std::uint32_t depthOrLayers = slices ? desc.depth : desc.layers;
    if (!std::in_range<UINT16>(depthOrLayers) || !std::in_range<UINT16>(desc.mipLevels)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} {} and {} mip levels are more than D3D12 takes",
                             depthOrLayers, slices ? "slices" : "layers", desc.mipLevels);
    }
    // A depth texture shaders also read is made typeless, so its depth view
    // and its shader view are both views of it; its views name their formats.
    const bool depth = formats.depth != DXGI_FORMAT_UNKNOWN;
    const bool readToo = desc.usage.sampled || desc.usage.storage;
    D3D12Texture texture{.allocation = {},
                         .resource = {},
                         .targetHeap = {},
                         .target = {},
                         .readOnlyTarget = {},
                         .desc = desc,
                         .dxgi = depth ? (readToo ? formats.typeless : formats.depth) : formats.texture,
                         .mipLevels = desc.mipLevels,
                         .layers = slices ? 1 : desc.layers,
                         .samples = desc.samples,
                         .slots = slots,
                         .sparse = desc.sparse,
                         .tileShape = {},
                         .packedMips = {},
                         .tileHeap = tileHeapOf(desc.usage)};
    const D3D12_RESOURCE_DESC resource{
        .Dimension = dimensionFor(desc.dimension),
        .Alignment = 0,
        .Width = desc.width,
        .Height = desc.height,
        .DepthOrArraySize = static_cast<UINT16>(depthOrLayers),
        .MipLevels = static_cast<UINT16>(desc.mipLevels),
        .Format = texture.dxgi,
        .SampleDesc = {.Count = desc.samples, .Quality = 0},
        // A reserved resource's layout is the one its tiles are mapped in.
        .Layout = desc.sparse ? D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE : D3D12_TEXTURE_LAYOUT_UNKNOWN,
        .Flags = flagsFor(desc.usage),
    };
    if (desc.sparse) {
        // Heaps' tiles are mapped into it, so no memory is placed now.
        // sparseAliased needs no D3D12 flag: any reserved resource's tiles may
        // be mapped to heap tiles others are mapped to.
        if (const HRESULT result = device.CreateReservedResource(&resource, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&texture.resource));
            FAILED(result)) {
            return failure(core::ErrorCode::RhiDeviceFailed,
                           {.call = "ID3D12Device::CreateReservedResource", .result = result});
        }
        UINT tiles = 0;
        UINT subresourceTilings = 0;
        device.GetResourceTiling(texture.resource.Get(), &tiles, &texture.packedMips, &texture.tileShape,
                                 &subresourceTilings, 0, nullptr);
    }
    else {
        D3D12MA::ALLOCATION_DESC allocation{};
        allocation.HeapType = D3D12_HEAP_TYPE_DEFAULT;
        if (const HRESULT result = allocator.CreateResource(&allocation, &resource, D3D12_RESOURCE_STATE_COMMON,
                                                            nullptr, &texture.allocation,
                                                            IID_PPV_ARGS(&texture.resource));
            FAILED(result)) {
            return failure(core::ErrorCode::RhiDeviceFailed,
                           {.call = "D3D12MA::Allocator::CreateResource", .result = result});
        }
    }
    // MipLevels 0 asks D3D12 for the whole chain, so its subresources are
    // numbered by the levels it made.
    texture.mipLevels = texture.resource->GetDesc().MipLevels;
    // The whole texture as a target, with the API's default view of it.
    if (desc.usage.colorAttachment || desc.usage.depthStencil) {
        auto heap = desc.usage.depthStencil
                        ? createTargetHeap(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, kDepthTargetViews)
                        : createTargetHeap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, kColorTargetViews);
        if (!heap) {
            return std::unexpected(std::move(heap.error()));
        }
        texture.targetHeap = std::move(*heap);
        texture.target = texture.targetHeap->GetCPUDescriptorHandleForHeapStart();
        if (desc.usage.depthStencil) {
            const D3D12_DEPTH_STENCIL_VIEW_DESC view = depthViewFor(texture, formats.depth, D3D12_DSV_FLAG_NONE);
            device.CreateDepthStencilView(texture.resource.Get(), &view, texture.target);
            texture.readOnlyTarget = {.ptr = texture.target.ptr +
                                             device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV)};
            const D3D12_DEPTH_STENCIL_VIEW_DESC readOnly = depthViewFor(texture, formats.depth,
                                                                        readOnlyFlagsFor(desc.format));
            device.CreateDepthStencilView(texture.resource.Get(), &readOnly, texture.readOnlyTarget);
        }
        else {
            device.CreateRenderTargetView(texture.resource.Get(), nullptr, texture.target);
        }
    }
    return texture;
}

namespace {

// Views of the whole texture, every mip level and layer, as the shader
// declares it: an array type when it has layers, multisampled when it has
// samples.
D3D12_SHADER_RESOURCE_VIEW_DESC sampledViewFor(const D3D12Texture& texture) {
    constexpr UINT kAllMips = std::numeric_limits<UINT>::max();
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = dxgiFormatsFor(texture.desc.format).texture;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    const bool array = texture.layers > 1;
    switch (texture.desc.dimension) {
        case TextureDimension::D1:
            if (array) {
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                view.Texture1DArray = {.MostDetailedMip = 0,
                                       .MipLevels = kAllMips,
                                       .FirstArraySlice = 0,
                                       .ArraySize = texture.layers,
                                       .ResourceMinLODClamp = 0};
            }
            else {
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                view.Texture1D = {.MostDetailedMip = 0, .MipLevels = kAllMips, .ResourceMinLODClamp = 0};
            }
            break;
        case TextureDimension::D3:
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            view.Texture3D = {.MostDetailedMip = 0, .MipLevels = kAllMips, .ResourceMinLODClamp = 0};
            break;
        case TextureDimension::Cube:
            if (texture.layers > kCubeFaces) {
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                view.TextureCubeArray = {.MostDetailedMip = 0,
                                         .MipLevels = kAllMips,
                                         .First2DArrayFace = 0,
                                         .NumCubes = texture.layers / kCubeFaces,
                                         .ResourceMinLODClamp = 0};
            }
            else {
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                view.TextureCube = {.MostDetailedMip = 0, .MipLevels = kAllMips, .ResourceMinLODClamp = 0};
            }
            break;
        case TextureDimension::D2:
        default:
            if (texture.samples > 1) {
                if (array) {
                    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                    view.Texture2DMSArray = {.FirstArraySlice = 0, .ArraySize = texture.layers};
                }
                else {
                    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
                }
            }
            else if (array) {
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                view.Texture2DArray = {.MostDetailedMip = 0,
                                       .MipLevels = kAllMips,
                                       .FirstArraySlice = 0,
                                       .ArraySize = texture.layers,
                                       .PlaneSlice = 0,
                                       .ResourceMinLODClamp = 0};
            }
            else {
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                view.Texture2D = {.MostDetailedMip = 0,
                                  .MipLevels = kAllMips,
                                  .PlaneSlice = 0,
                                  .ResourceMinLODClamp = 0};
            }
            break;
    }
    return view;
}

// The first mip level of the whole texture, every layer; a cube is its faces.
D3D12_UNORDERED_ACCESS_VIEW_DESC storageViewFor(const D3D12Texture& texture) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = dxgiFormatsFor(texture.desc.format).texture;
    const bool array = texture.layers > 1;
    switch (texture.desc.dimension) {
        case TextureDimension::D1:
            if (array) {
                view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                view.Texture1DArray = {.MipSlice = 0, .FirstArraySlice = 0, .ArraySize = texture.layers};
            }
            else {
                view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                view.Texture1D = {.MipSlice = 0};
            }
            break;
        case TextureDimension::D3:
            view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            view.Texture3D = {.MipSlice = 0, .FirstWSlice = 0, .WSize = texture.desc.depth};
            break;
        case TextureDimension::Cube:
        case TextureDimension::D2:
        default:
            if (array || texture.desc.dimension == TextureDimension::Cube) {
                view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                view.Texture2DArray = {.MipSlice = 0,
                                       .FirstArraySlice = 0,
                                       .ArraySize = texture.layers,
                                       .PlaneSlice = 0};
            }
            else {
                view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                view.Texture2D = {.MipSlice = 0, .PlaneSlice = 0};
            }
            break;
    }
    return view;
}

} // namespace

// The device's texture calls, beside the textures they make.

core::Result<D3D12Texture*> D3D12Device::textureFor(TextureHandle texture) {
    D3D12Texture* found = textures_.find(texture);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the texture handle is stale");
    }
    return found;
}

core::Result<TextureHandle> D3D12Device::createTexture(const TextureDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectBuffers();
    auto slots = bindless_.takeTexture(desc);
    if (!slots) {
        return std::unexpected(std::move(slots.error()));
    }
    auto made = createD3D12Texture(*allocator_.Get(), *device_.Get(), desc, *slots);
    if (!made) {
        bindless_.giveTexture(*slots);
        return std::unexpected(std::move(made.error()));
    }
    if (slots->sampled != core::SlotChains::kEnd) {
        const D3D12_SHADER_RESOURCE_VIEW_DESC view = sampledViewFor(*made);
        device_->CreateShaderResourceView(made->resource.Get(), &view, layout_.resourceAt(slots->sampled));
    }
    if (slots->storage != core::SlotChains::kEnd) {
        const D3D12_UNORDERED_ACCESS_VIEW_DESC view = storageViewFor(*made);
        device_->CreateUnorderedAccessView(made->resource.Get(), nullptr, &view,
                                           layout_.resourceAt(layout_.storageTextures + slots->storage));
    }
    const TextureHandle texture = textures_.emplace(std::move(*made));
    if (texture.isNull()) {
        bindless_.giveTexture(*slots);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} textures", kBufferLimit);
    }
    return texture;
}

core::Result<TextureSlots> D3D12Device::textureSlots(TextureHandle texture) {
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reported((*found)->slots);
}

core::Result<void> D3D12Device::destroyTexture(TextureHandle texture) {
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

core::Result<void> D3D12Device::copyBufferToTexture(CommandListHandle commandList, BufferHandle buffer,
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
    if (auto checked = checkAspects((*to)->desc.format, regions); !checked) {
        return checked;
    }
    if (auto checked = checkRowPitches(regions); !checked) {
        return checked;
    }
    const FormatBlock& block = blockOf((*to)->desc.format);
    for (const BufferTextureCopy& region : regions) {
        const D3D12_TEXTURE_COPY_LOCATION source = bufferSide((*from)->resource.Get(), **to, region);
        const D3D12_TEXTURE_COPY_LOCATION target = textureSide(**to, subresourceOf(**to, region.mipLevel, region.layer,
                                                                                   planeOf(block, region.aspect)));
        if (whole(**to, region.aspect)) {
            (*list)->list->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
            continue;
        }
        const D3D12_BOX box = boxOf(block, 0, 0, 0, region.width, region.height, region.depth);
        (*list)->list->CopyTextureRegion(&target, region.x, region.y, region.z, &source, &box);
    }
    return {};
}

core::Result<void> D3D12Device::copyTextureToBuffer(CommandListHandle commandList, TextureHandle texture,
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
    if (auto checked = checkAspects((*from)->desc.format, regions); !checked) {
        return checked;
    }
    if (auto checked = checkRowPitches(regions); !checked) {
        return checked;
    }
    const FormatBlock& block = blockOf((*from)->desc.format);
    for (const BufferTextureCopy& region : regions) {
        const D3D12_TEXTURE_COPY_LOCATION source = textureSide(**from,
                                                               subresourceOf(**from, region.mipLevel, region.layer,
                                                                             planeOf(block, region.aspect)));
        const D3D12_TEXTURE_COPY_LOCATION target = bufferSide((*to)->resource.Get(), **from, region);
        if (whole(**from, region.aspect)) {
            (*list)->list->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
            continue;
        }
        const D3D12_BOX box = boxOf(block, region.x, region.y, region.z, region.width, region.height, region.depth);
        (*list)->list->CopyTextureRegion(&target, 0, 0, 0, &source, &box);
    }
    return {};
}

core::Result<void> D3D12Device::copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
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
    const FormatBlock& block = blockOf((*from)->desc.format);
    // Every aspect: a depth-stencil format's two planes, one at a time.
    const std::uint32_t planes = planar(block) ? 2 : 1;
    const bool depthStencil = block.bytes == 0;
    for (const TextureCopy& region : regions) {
        for (std::uint32_t plane = 0; plane < planes; ++plane) {
            const D3D12_TEXTURE_COPY_LOCATION source = textureSide(**from, subresourceOf(**from, region.srcMipLevel,
                                                                                         region.srcLayer, plane));
            const D3D12_TEXTURE_COPY_LOCATION target = textureSide(**to, subresourceOf(**to, region.dstMipLevel,
                                                                                       region.dstLayer, plane));
            if (depthStencil || (*from)->samples > 1) {
                (*list)->list->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
                continue;
            }
            const D3D12_BOX box = boxOf(block, region.srcX, region.srcY, region.srcZ, region.width, region.height,
                                        region.depth);
            (*list)->list->CopyTextureRegion(&target, region.dstX, region.dstY, region.dstZ, &source, &box);
        }
    }
    return {};
}

core::Result<void> D3D12Device::resolveTexture(CommandListHandle commandList, TextureHandle src,
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
    const std::uint32_t plane = planeOf(blockOf((*from)->desc.format), TextureAspect::Color);
    (*list)->list->ResolveSubresource((*to)->resource.Get(), subresourceOf(**to, dstMipLevel, dstLayer, plane),
                                      (*from)->resource.Get(), subresourceOf(**from, srcMipLevel, srcLayer, plane),
                                      (*from)->dxgi);
    return {};
}

namespace {

// Fails with RhiInvalidCall when format has no aspect, and RhiUnsupported for
// its stencil: GetResourceTiling gives one tile shape and one packed mip count
// for the whole resource, and Diligent maps a tiled texture's plane 0 alone
// (D3DTileMappingHelper.hpp, CalcSubresource with plane 0), so D3D12 is not
// shown to tile a stencil plane apart.
core::Result<void> checkTileAspect(Format format, TextureAspect aspect) {
    if (auto checked = checkAspect(format, aspect); !checked) {
        return checked;
    }
    if (aspect == TextureAspect::Stencil) {
        return core::failure(core::ErrorCode::RhiUnsupported, "D3D12 gives no tiling of a stencil plane alone");
    }
    return {};
}

} // namespace

core::Result<TextureTiling> D3D12Device::textureTiling(TextureHandle texture, TextureAspect aspect) {
    if (auto checked = checkKnown(aspect, "texture aspect"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    auto found = textureFor(texture);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const D3D12Texture& tiled = **found;
    if (!tiled.sparse) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the texture is not sparse");
    }
    if (auto checked = checkTileAspect(tiled.desc.format, aspect); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // GetResourceTiling's answer when the texture was made: its packed mips
    // (D3D12_PACKED_MIP_INFO) are its tail. Each layer has a tail of its own
    // (Diligent's TextureD3D12Impl.cpp, InitSparseProperties), mapped by its
    // subresource; D3D12 reports no stride between them, which Diligent
    // derives itself (:743).
    return TextureTiling{
        .tileBytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES,
        .shape = {.width = tiled.tileShape.WidthInTexels,
                  .height = tiled.tileShape.HeightInTexels,
                  .depth = tiled.tileShape.DepthInTexels},
        .firstTailMip = tiled.packedMips.NumStandardMips,
        .tailBytes = std::uint64_t{tiled.packedMips.NumTilesForPackedMips} * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES,
        .tailStride = 0,
        .singleTail = false,
    };
}

core::Result<SparseFormat> D3D12Device::sparseFormat(const SparseFormatDesc& desc) const {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (desc.tileBytes != D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES) {
        return core::failure(core::ErrorCode::RhiUnsupported, "D3D12's tiles are {} bytes; tileBytes is {}",
                             D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, desc.tileBytes);
    }
    // What D3D12 tiles: the aspect textureTiling gives, of a 2D or cube
    // texture, or a 3D one where the tier has them (sparseOf). It tiles no 1D
    // texture: Sparse has no such cap, and Diligent refuses one and takes a
    // cube as 2D (RenderDeviceBase.cpp, CheckSparseTextureFormatSupport).
    const FormatBlock& block = blockOf(desc.format);
    const bool aspect = desc.aspect == TextureAspect::Color
                            ? block.bytes != 0
                            : desc.aspect == TextureAspect::Depth && block.depthBytes != 0;
    const bool dimension = desc.dimension == TextureDimension::D3
                               ? sparse_.textures3D
                               : desc.dimension != TextureDimension::D1 && sparse_.textures2D;
    // A depth texture is asked of as its depth format, as formatCapsFor asks.
    const DxgiFormats formats = dxgiFormatsFor(desc.format);
    const DXGI_FORMAT dxgi = formats.depth != DXGI_FORMAT_UNKNOWN ? formats.depth : formats.texture;
    if (!aspect || !dimension || dxgi == DXGI_FORMAT_UNKNOWN) {
        return SparseFormat{};
    }
    auto tiled = tiledFor(device_.Get(), dxgi, desc.samples);
    if (!tiled) {
        return std::unexpected(std::move(tiled.error()));
    }
    if (!*tiled) {
        return SparseFormat{};
    }
    // Each layer has a tail of its own, as textureTiling says, and D3D12 has
    // no aligned mip size to report.
    return SparseFormat{.supported = true,
                        .shape = standardTileShape(desc.format, desc.dimension, desc.samples),
                        .singleTail = false,
                        .alignedMipSize = false};
}

core::Result<ID3D12Resource*> D3D12Device::addTextureTiles(D3D12TileMappings& tiles, const TileMappingDesc& desc,
                                                           const D3D12SparseMemory* memory) {
    tiles.clear();
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
    const D3D12Texture& texture = **found;
    if (!texture.sparse) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the texture is not sparse");
    }
    if (memory != nullptr && !memory->takes(texture.tileHeap)) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "on ResourceHeapTier 1 a heap holds one kind of resource, and the memory's is not the "
                             "texture's");
    }
    const FormatBlock& block = blockOf(texture.desc.format);
    const std::uint32_t firstTail = texture.packedMips.NumStandardMips;
    for (const TextureTileRegion& region : desc.textureRegions) {
        if (auto checked = checkKnown(region.mapping, "tile mapping"); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
        if (auto checked = checkTileAspect(texture.desc.format, region.aspect); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
        const bool maps = region.mapping == TileMapping::Map;
        if (maps && memory == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a region maps, but no memory was given");
        }
        // D3D12 numbers subresources in one run, so a mip or layer past the
        // texture's would name another of its subresources.
        if (region.mipLevel >= texture.mipLevels || region.layer >= texture.layers) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "mip level {} of layer {} is past the texture's {} mip levels and {} layers",
                                 region.mipLevel, region.layer, texture.mipLevels, texture.layers);
        }
        const std::uint32_t plane = planeOf(block, region.aspect);
        D3D12_TILED_RESOURCE_COORDINATE coordinate{};
        D3D12_TILE_REGION_SIZE size{};
        if (region.mipLevel >= firstTail) {
            // The layer's first packed mip names its tail, and X is a tile in
            // it, as Diligent maps one (D3DTileMappingHelper.hpp:90, 111-123).
            if (!std::in_range<UINT>(region.tailTile) || !std::in_range<UINT>(region.tailTiles)) {
                return core::failure(core::ErrorCode::RhiInvalidCall,
                                     "{} tail tiles from tile {} are past what D3D12 takes", region.tailTiles,
                                     region.tailTile);
            }
            coordinate = {.X = static_cast<UINT>(region.tailTile),
                          .Y = 0,
                          .Z = 0,
                          .Subresource = subresourceOf(texture, firstTail, region.layer, plane)};
            size = {
                .NumTiles = static_cast<UINT>(region.tailTiles), .UseBox = FALSE, .Width = 0, .Height = 0, .Depth = 0};
        }
        else {
            // A box of tiles in the mip level, whose Height and Depth are UINT16s.
            const std::uint64_t count = std::uint64_t{region.width} * region.height * region.depth;
            if (!std::in_range<UINT16>(region.height) || !std::in_range<UINT16>(region.depth) ||
                !std::in_range<UINT>(count)) {
                return core::failure(core::ErrorCode::RhiInvalidCall,
                                     "a box of {} by {} by {} tiles is past what D3D12 takes", region.width,
                                     region.height, region.depth);
            }
            coordinate = {.X = region.x,
                          .Y = region.y,
                          .Z = region.z,
                          .Subresource = subresourceOf(texture, region.mipLevel, region.layer, plane)};
            size = {.NumTiles = static_cast<UINT>(count),
                    .UseBox = TRUE,
                    .Width = region.width,
                    .Height = static_cast<UINT16>(region.height),
                    .Depth = static_cast<UINT16>(region.depth)};
        }
        if (auto added = tiles.add(coordinate, size, maps, region.memoryTile); !added) {
            return std::unexpected(std::move(added.error()));
        }
    }
    return texture.resource.Get();
}

} // namespace bfr::rhi::d3d12
