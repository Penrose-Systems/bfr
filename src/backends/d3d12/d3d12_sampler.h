#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/sampler.h"
#include "d3d12/d3d12_caps.h"

#include <d3d12.h>

#include <cstdint>

namespace bfr::rhi::d3d12 {

/// A sampler: a descriptor in the sampler heap at its slot, which D3D12 keeps
/// no object for. The device gives the slot back when it is collected.
struct D3D12Sampler {
    std::uint32_t slot = 0;
};

/// Writes a sampler as desc says at destination, through device11 when the
/// device has it, so integer border colours and unnormalized coordinates
/// reach D3D12. Fails with RhiUnsupported when desc needs what features lack,
/// or a setting D3D12 has none for.
[[nodiscard]] core::Result<void> writeD3D12Sampler(ID3D12Device& device, ID3D12Device11* device11,
                                                   const SamplerFeatures& features, const SamplerDesc& desc,
                                                   D3D12_CPU_DESCRIPTOR_HANDLE destination);

} // namespace bfr::rhi::d3d12
