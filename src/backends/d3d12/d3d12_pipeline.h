#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"

#include <d3d12.h>
#include <wrl/client.h>

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
inline constexpr std::uint32_t kRootParameterCount = kSamplerTableParameter + 1;

/// The push constants take the root signature's room: D3D12_MAX_ROOT_COST
/// DWORDs, less the one each descriptor table, every other parameter, costs.
inline constexpr std::uint32_t kDescriptorTableCount = kRootParameterCount - 1;
inline constexpr std::uint32_t kPushConstantValues = D3D12_MAX_ROOT_COST - kDescriptorTableCount;
inline constexpr std::uint32_t kPushConstantsSize = kPushConstantValues * sizeof(std::uint32_t);

/// The viewports, and scissors, a draw has: setViewport and setScissor each
/// set one.
inline constexpr std::uint32_t kViewportCount = 1;

/// op as D3D12's, for depth and stencil tests and compare samplers.
[[nodiscard]] D3D12_COMPARISON_FUNC compareFor(CompareOp op);

/// The one resource layout every pipeline shares: the root signature, and the
/// two shader-visible heaps holding the tables, sized as the DeviceDesc asked.
/// The three resource tables are regions of the resource heap, the sampled
/// textures first, then the storage textures, then the storage buffers; each
/// range's offset from the table's start is its region's, so a shader's index
/// in any table is the slot BFR handed out, as on every backend. Made with the
/// device and released with it, after every pipeline.
struct D3D12BindlessLayout {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> resources;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> samplers;
    /// Where each region starts in the resource heap, in descriptors; the
    /// sampled textures start at 0.
    std::uint32_t storageTextures = 0;
    std::uint32_t storageBuffers = 0;
    /// Bytes between two descriptors of the resource heap and of the sampler
    /// heap, as the device says.
    std::uint32_t descriptorBytes = 0;
    std::uint32_t samplerBytes = 0;

    /// The CPU handle of descriptor index of the resource heap, which a view
    /// is created at.
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE resourceAt(std::uint32_t index) const {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = resources->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T{index} * descriptorBytes;
        return handle;
    }

    /// The CPU handle of descriptor index of the sampler heap, which a sampler
    /// is created at.
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE samplerAt(std::uint32_t index) const {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = samplers->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T{index} * samplerBytes;
        return handle;
    }
};

/// The layout on device as desc sizes it. Fails with RhiLimitReached when the
/// three resource tables together are more than viewHeapSize, the device's
/// shader-visible heap, holds, and RhiDeviceFailed when D3D12 refuses the root
/// signature or a heap.
[[nodiscard]] core::Result<D3D12BindlessLayout> createD3D12BindlessLayout(ID3D12Device& device, const DeviceDesc& desc,
                                                                          std::uint32_t viewHeapSize);

/// A pipeline state object, released when this goes, with the topology class
/// its draws take.
struct D3D12Pipeline {
    Microsoft::WRL::ComPtr<ID3D12PipelineState> state;
    bool compute = false;
    TopologyClass topology = TopologyClass::Triangle;
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
