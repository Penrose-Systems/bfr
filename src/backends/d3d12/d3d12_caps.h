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

/// Why device is below the floor, empty when it meets it. The floor is
/// bindless resource binding, shader model 6.6 and enhanced barriers.
[[nodiscard]] std::string belowFloor(ID3D12Device* device);

/// What kind of adapter desc describes, device being the D3D12 device on it.
[[nodiscard]] AdapterType typeOf(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device);

/// The adapter's name, in UTF-8.
[[nodiscard]] std::string nameOf(const DXGI_ADAPTER_DESC1& desc);

/// Everything fixed about the adapter desc describes, as device, made on it,
/// answers. A question the device won't answer leaves its fields at 0 or false.
[[nodiscard]] Caps readCaps(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device);

} // namespace bfr::rhi::d3d12
