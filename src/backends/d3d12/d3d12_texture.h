#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/texture.h"

#include <d3d12.h>
#include <D3D12MemAlloc.h>
#include <wrl/client.h>

#include <cstdint>

namespace bfr::rhi::d3d12 {

/// A texture resource and the memory D3D12MA placed it in, both released when
/// it goes: before the allocator is. format is the resource's DXGI format, and
/// the rest what its subresources are numbered by.
struct D3D12Texture {
    // The allocation is declared first so that the resource is released first.
    Microsoft::WRL::ComPtr<D3D12MA::Allocation> allocation;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    Format format = Format::R8Unorm;
    DXGI_FORMAT dxgi = DXGI_FORMAT_UNKNOWN;
    std::uint32_t mipLevels = 0;
    std::uint32_t layers = 0;
    std::uint32_t samples = 0;
};

/// A texture as desc says, in the default heap, made in the legacy COMMON
/// state, which enhanced barriers read as layout COMMON. Fails with
/// RhiOutOfMemory when memory runs out and RhiDeviceFailed when D3D12MA or
/// D3D12 refuses.
[[nodiscard]] core::Result<D3D12Texture> createD3D12Texture(D3D12MA::Allocator& allocator, const TextureDesc& desc);

} // namespace bfr::rhi::d3d12
