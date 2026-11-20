#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/texture.h"
#include "rhi/bindless_slots.h"

#include <d3d12.h>
#include <D3D12MemAlloc.h>
#include <wrl/client.h>

#include <cstdint>

namespace bfr::rhi::d3d12 {

/// A texture resource and the memory D3D12MA placed it in, both released when
/// it goes: before the allocator is. dxgi is the resource's DXGI format, and
/// targetFormat the one its render target or depth-stencil views name, and
/// the rest what its subresources are numbered by and its views are made as.
/// The slots are its places in the tables, kEnd in a table its usage does not
/// set; the device gives them back when the texture is collected. A sparse
/// one is a reserved resource with no allocation: tileShape and packedMips
/// are what GetResourceTiling said when it was made, so mapping reads nothing
/// back, and tileHeap is the D3D12_HEAP_FLAG_ALLOW_ONLY_* kind of heap
/// ResourceHeapTier 1 maps it from.
struct D3D12Texture {
    // The allocation is declared first so that the resource is released first.
    Microsoft::WRL::ComPtr<D3D12MA::Allocation> allocation;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    TextureDesc desc;
    DXGI_FORMAT dxgi = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT targetFormat = DXGI_FORMAT_UNKNOWN;
    std::uint32_t mipLevels = 0;
    std::uint32_t layers = 0;
    std::uint32_t samples = 0;
    TextureSlotsHeld slots;
    bool sparse = false;
    D3D12_TILE_SHAPE tileShape{};
    D3D12_PACKED_MIP_INFO packedMips{};
    D3D12_HEAP_FLAGS tileHeap = D3D12_HEAP_FLAG_NONE;
};

/// What a rendering's attachment covers of a resource: mipLevel of
/// layerCount of its layers from firstLayer, in planeCount of its planes from
/// firstPlane. A swapchain's image has one of each.
struct D3D12Target {
    ID3D12Resource* resource = nullptr;
    std::uint32_t mipLevels = 1;
    std::uint32_t layers = 1;
    std::uint32_t planes = 1;
    std::uint32_t firstPlane = 0;
    std::uint32_t planeCount = 1;
    std::uint32_t mipLevel = 0;
    std::uint32_t firstLayer = 0;
    std::uint32_t layerCount = 1;
};

/// The layers a texture's mipLevel has targets for: a 3D texture's slices at
/// that level, any other's layers.
[[nodiscard]] std::uint32_t targetLayersAt(const D3D12Texture& texture, std::uint32_t mipLevel);

/// What an attachment of texture at mipLevel covers, from layer, layerCount
/// layers.
[[nodiscard]] D3D12Target targetOf(const D3D12Texture& texture, std::uint32_t mipLevel, std::uint32_t layer,
                                   std::uint32_t layerCount);

/// What target covers of texture's depth plane, when depth, and its stencil
/// plane, when stencil: an op of a depth-stencil attachment on those aspects.
/// planeCount is 0 when the format has neither.
[[nodiscard]] D3D12Target aspectsOf(const D3D12Texture& texture, const D3D12Target& target, bool depth, bool stencil);

/// Writes at view, a render target descriptor, a view of texture's mipLevel
/// over exactly layerCount layers from layer, a 3D texture's depth slices.
/// Fails with RhiInvalidCall when its usage has no colorAttachment, it has no
/// mipLevel, or checkAttachmentLayers fails; view is then left as it was.
[[nodiscard]] core::Result<void> writeColorTarget(ID3D12Device& device, const D3D12Texture& texture,
                                                  std::uint32_t mipLevel, std::uint32_t layer, std::uint32_t layerCount,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE view);

/// As writeColorTarget, a depth-stencil view, at a depth-stencil descriptor,
/// writing neither aspect when readOnly, of a texture whose usage has
/// depthStencil.
[[nodiscard]] core::Result<void> writeDepthTarget(ID3D12Device& device, const D3D12Texture& texture,
                                                  std::uint32_t mipLevel, std::uint32_t layer, std::uint32_t layerCount,
                                                  bool readOnly, D3D12_CPU_DESCRIPTOR_HANDLE view);

/// Records on list the discard of what target covers, and nothing else.
void discardTarget(ID3D12GraphicsCommandList* list, const D3D12Target& target);

/// A texture as desc says, in the default heap, or a reserved resource with
/// no memory when it is sparse, made in the legacy COMMON state, which
/// enhanced barriers read as layout COMMON, holding the slots given. Fails with
/// RhiUnsupported for a sparse tileBytes other than D3D12's tile size,
/// RhiOutOfMemory when memory runs out and RhiDeviceFailed when D3D12MA or
/// D3D12 refuses.
[[nodiscard]] core::Result<D3D12Texture> createD3D12Texture(D3D12MA::Allocator& allocator, ID3D12Device& device,
                                                            const TextureDesc& desc, const TextureSlotsHeld& slots);

} // namespace bfr::rhi::d3d12
