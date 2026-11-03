#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>

namespace bfr::rhi::d3d12 {

/// The register spaces of the tables, which api/bfr.yaml states under
/// bfr_pipeline and every shader declares. Each table's range starts at
/// register 0 of its space and at the start of its descriptor table, so the
/// three resource tables index one heap; the sampler table has its own.
inline constexpr std::uint32_t kSampledTextureSpace = 0;
inline constexpr std::uint32_t kStorageTextureSpace = 1;
inline constexpr std::uint32_t kStorageBufferSpace = 2;
inline constexpr std::uint32_t kSamplerSpace = 3;
inline constexpr std::uint32_t kPushConstantsSpace = 0;

/// The root parameters, in order: the push constants, the resource table and
/// the sampler table.
inline constexpr std::uint32_t kPushConstantsParameter = 0;
inline constexpr std::uint32_t kResourceTableParameter = 1;
inline constexpr std::uint32_t kSamplerTableParameter = 2;

/// The one resource layout every pipeline shares: the root signature, and the
/// two shader-visible heaps holding the tables, sized as the DeviceDesc asked.
/// The three resource ranges all start at the table's start, so an index of
/// any kind names one descriptor of the resource heap; BFR gives each kind its
/// own region of it, the sampled textures first, then the storage textures,
/// then the storage buffers, so an index handed out for one kind never names
/// another's descriptor. Made with the device and released with it, after
/// every pipeline.
struct D3D12BindlessLayout {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> resources;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> samplers;
};

/// The layout on device as desc sizes it. Fails with RhiLimitReached when the
/// three resource tables together are more than a shader-visible heap holds,
/// and RhiDeviceFailed when D3D12 refuses the root signature or a heap.
[[nodiscard]] core::Result<D3D12BindlessLayout> createD3D12BindlessLayout(ID3D12Device& device, const DeviceDesc& desc);

/// A pipeline state object, released when this goes, with the topology class
/// its draws take and, for compute, the workgroup size its desc gave.
struct D3D12Pipeline {
    Microsoft::WRL::ComPtr<ID3D12PipelineState> state;
    bool compute = false;
    TopologyClass topology = TopologyClass::Triangle;
    std::array<std::uint32_t, kWorkgroupDimensions> workgroupSize{};
};

/// A compute pipeline on rootSignature as desc says, whose code is DXIL. Fails
/// with RhiDeviceFailed when D3D12 refuses it.
[[nodiscard]] core::Result<D3D12Pipeline> createD3D12ComputePipeline(ID3D12Device& device,
                                                                     ID3D12RootSignature& rootSignature,
                                                                     const ComputePipelineDesc& desc);

/// A graphics pipeline on rootSignature as desc says, whose code is DXIL, with
/// no input layout: vertices are pulled. Fails with RhiUnsupported when the
/// two stencil faces mask differently, which D3D12 does not take, and
/// RhiDeviceFailed when D3D12 refuses it.
[[nodiscard]] core::Result<D3D12Pipeline> createD3D12GraphicsPipeline(ID3D12Device& device,
                                                                      ID3D12RootSignature& rootSignature,
                                                                      const GraphicsPipelineDesc& desc);

} // namespace bfr::rhi::d3d12
