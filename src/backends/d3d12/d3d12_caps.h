#pragma once

#include "bfr/rhi/caps.h"
#include "bfr/rhi/instance.h"

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <string>

namespace bfr::rhi::d3d12 {

/// What device's samplers may be, as its options answer, for the caps. A
/// question the device won't answer leaves false.
struct SamplerFeatures {
    /// OPTIONS19 AnisoFilterWithPointMipSupported.
    bool anisotropyNearestMip = false;
    /// ID3D12Device11, whose CreateSampler2 takes integer border colours and
    /// unnormalized coordinates.
    bool device11 = false;
    /// OPTIONS17 NonNormalizedCoordinateSamplersSupported, with device11.
    bool unnormalizedCoordinates = false;
    /// OPTIONS14 AdvancedTextureOpsSupported, with device11: the DirectX
    /// spec puts integer border colours under it.
    bool integerBorderColor = false;
    /// OPTIONS19 MaxSamplerDescriptorHeapSize; 0 when the device won't say.
    std::uint32_t maxSamplerHeapSize = 0;
    /// OPTIONS19 MaxViewDescriptorHeapSize, the shader-visible resource heap;
    /// 0 when the device won't say.
    std::uint32_t maxViewHeapSize = 0;
};

/// device's sampler features.
[[nodiscard]] SamplerFeatures samplerFeaturesOf(ID3D12Device* device);

/// What device's sparse resources are, from OPTIONS' TiledResourcesTier and
/// ResourceHeapTier and D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT. D3D12 has
/// no query for each sparse feature, so the tiers mean what Microsoft's
/// D3D12_TILED_RESOURCES_TIER documentation says each holds:
/// - Tier 1: tiled buffers and 2D textures, aliased tiles, the standard 64KB
///   tile shapes for 2D and 4x multisampled 2D textures.
/// - Tier 2 adds reads of unmapped tiles giving 0 with writes dropped, and
///   shader residency feedback.
/// - Tier 3 adds tiled 3D textures, with their standard shape.
/// ResourceHeapTier 2 lets one heap hold buffers and every texture. A tail
/// mapping at a layer other than 0 is not claimed: Diligent maps one only
/// through NVAPI (EngineFactoryD3D12.cpp), and O3DE gives tails to arrays of
/// 1 layer only (DX12 Image.cpp); updateTileMappings passes one to D3D12 all
/// the same. Tiles are D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES.
/// D3D12 counts the address space in bits, so addressSpaceBytes stays 0.
/// Everything is false or 0, tileBytes empty, where the device has no tiled
/// resources or won't say.
[[nodiscard]] Sparse sparseOf(ID3D12Device* device);

/// Why device is below BFR's minimum, empty when it meets it. BFR needs
/// bindless resource binding, shader model 6.6 and enhanced barriers.
[[nodiscard]] std::string belowMinimum(ID3D12Device* device);

/// What kind of adapter desc describes, device being the D3D12 device on it.
[[nodiscard]] AdapterType typeOf(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device);

/// The adapter's name, in UTF-8.
[[nodiscard]] std::string nameOf(const DXGI_ADAPTER_DESC1& desc);

/// Everything fixed about the adapter desc describes, as device, made on it,
/// answers. A question the device won't answer leaves its fields at 0 or false.
[[nodiscard]] Caps readCaps(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device);

} // namespace bfr::rhi::d3d12
