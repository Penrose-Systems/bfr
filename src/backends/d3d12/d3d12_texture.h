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
/// the rest what its subresources are numbered by and its views are made as.
/// target is its render target or depth-stencil view, in a heap of its own,
/// when its usage has rendering write it. The slots are its places in the
/// tables, kEnd in a table its usage does not set; the device gives them back
/// when the texture is collected.
struct D3D12Texture {
    // The allocation is declared first so that the resource is released first.
    Microsoft::WRL::ComPtr<D3D12MA::Allocation> allocation;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> targetHeap;
    D3D12_CPU_DESCRIPTOR_HANDLE target{};
    TextureDesc desc;
    DXGI_FORMAT dxgi = DXGI_FORMAT_UNKNOWN;
    std::uint32_t mipLevels = 0;
    std::uint32_t layers = 0;
    std::uint32_t samples = 0;
    TextureSlotsHeld slots;
};

/// What a rendering's attachment covers of a resource: mip 0 of each of its
/// layers and planes. A swapchain's image has one of each.
struct D3D12Target {
    ID3D12Resource* resource = nullptr;
    std::uint32_t mipLevels = 1;
    std::uint32_t layers = 1;
    std::uint32_t planes = 1;
};

/// What an attachment of texture covers.
[[nodiscard]] D3D12Target targetOf(const D3D12Texture& texture);

/// Records on list the discard of what target covers, and nothing of its other mips.
void discardTarget(ID3D12GraphicsCommandList* list, const D3D12Target& target);

/// A texture as desc says, in the default heap, made in the legacy COMMON
/// state, which enhanced barriers read as layout COMMON, with its target view
/// on device when rendering writes it, holding the slots given. Fails with
/// RhiOutOfMemory when memory runs out and RhiDeviceFailed when D3D12MA or
/// D3D12 refuses.
[[nodiscard]] core::Result<D3D12Texture> createD3D12Texture(D3D12MA::Allocator& allocator, ID3D12Device& device,
                                                            const TextureDesc& desc, const TextureSlotsHeld& slots);

} // namespace bfr::rhi::d3d12
