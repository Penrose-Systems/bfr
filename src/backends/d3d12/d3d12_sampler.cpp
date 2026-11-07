#include "d3d12/d3d12_sampler.h"

#include "bfr/core/error.h"
#include "bfr/rhi/sampler.h"
#include "d3d12/d3d12_caps.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_pipeline.h"
#include "rhi/bindless_slots.h"
#include "rhi/enum_check.h"

#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <utility>

namespace bfr::rhi::d3d12 {

namespace {

D3D12_FILTER_TYPE filterFor(Filter filter) {
    return filter == Filter::Linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
}

D3D12_FILTER_REDUCTION_TYPE reductionFor(SamplerReduction reduction) {
    switch (reduction) {
        case SamplerReduction::Compare:
            return D3D12_FILTER_REDUCTION_TYPE_COMPARISON;
        case SamplerReduction::Minimum:
            return D3D12_FILTER_REDUCTION_TYPE_MINIMUM;
        case SamplerReduction::Maximum:
            return D3D12_FILTER_REDUCTION_TYPE_MAXIMUM;
        case SamplerReduction::WeightedAverage:
        default:
            return D3D12_FILTER_REDUCTION_TYPE_STANDARD;
    }
}

D3D12_TEXTURE_ADDRESS_MODE addressFor(AddressMode mode) {
    switch (mode) {
        case AddressMode::MirrorRepeat:
            return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        case AddressMode::ClampToEdge:
            return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        case AddressMode::ClampToBorder:
            return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        case AddressMode::MirrorClampToEdge:
            return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
        case AddressMode::Repeat:
        default:
            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    }
}

// desc's filters as one D3D12_FILTER, built as d3d12.h's encoding macros
// build theirs: the anisotropic bit when anisotropy is asked for, on the min,
// mag and mip filters and the reduction asked for. A mix d3d12.h names no
// filter for reaches D3D12 as it is.
D3D12_FILTER encodedFilterFor(const SamplerDesc& desc) {
    const auto basic = static_cast<UINT>(D3D12_ENCODE_BASIC_FILTER(filterFor(desc.minFilter), filterFor(desc.magFilter),
                                                                   filterFor(desc.mipFilter),
                                                                   reductionFor(desc.reduction)));
    const UINT anisotropic = desc.maxAnisotropy != 0 ? UINT{D3D12_ANISOTROPIC_FILTERING_BIT} : 0U;
    return static_cast<D3D12_FILTER>(basic | anisotropic);
}

} // namespace

core::Result<void> writeD3D12Sampler(ID3D12Device& device, ID3D12Device11* device11, const SamplerDesc& desc,
                                     D3D12_CPU_DESCRIPTOR_HANDLE destination) {
    if (desc.lodAverage) {
        return core::failure(core::ErrorCode::RhiUnsupported, "D3D12 has no LOD averaging");
    }
    if ((desc.borderInteger || desc.unnormalizedCoordinates) && device11 == nullptr) {
        return core::failure(
            core::ErrorCode::RhiUnsupported,
            "the device has no ID3D12Device11, whose sampler desc alone holds integer border colours and unnormalized "
            "coordinates");
    }
    D3D12_SAMPLER_DESC2 sampler{};
    sampler.Filter = encodedFilterFor(desc);
    sampler.AddressU = addressFor(desc.addressU);
    sampler.AddressV = addressFor(desc.addressV);
    sampler.AddressW = addressFor(desc.addressW);
    sampler.MipLODBias = desc.lodBias;
    sampler.MaxAnisotropy = desc.maxAnisotropy;
    sampler.ComparisonFunc = compareFor(desc.compareOp);
    sampler.Flags = D3D12_SAMPLER_FLAG_NONE;
    if (desc.borderInteger) {
        std::ranges::copy(desc.borderColorInteger, std::begin(sampler.UintBorderColor));
        sampler.Flags |= D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
    }
    else {
        std::ranges::copy(desc.borderColor, std::begin(sampler.FloatBorderColor));
    }
    if (desc.unnormalizedCoordinates) {
        sampler.Flags |= D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES;
    }
    sampler.MinLOD = desc.minLod;
    sampler.MaxLOD = desc.maxLod;
    if (device11 != nullptr) {
        device11->CreateSampler2(&sampler, destination);
        return {};
    }
    // Without ID3D12Device11 no flag is set, as the checks above refused
    // every member that needs one, so the older desc holds all of it.
    D3D12_SAMPLER_DESC older{.Filter = sampler.Filter,
                             .AddressU = sampler.AddressU,
                             .AddressV = sampler.AddressV,
                             .AddressW = sampler.AddressW,
                             .MipLODBias = sampler.MipLODBias,
                             .MaxAnisotropy = sampler.MaxAnisotropy,
                             .ComparisonFunc = sampler.ComparisonFunc,
                             .BorderColor = {},
                             .MinLOD = sampler.MinLOD,
                             .MaxLOD = sampler.MaxLOD};
    std::ranges::copy(sampler.FloatBorderColor, std::begin(older.BorderColor));
    device.CreateSampler(&older, destination);
    return {};
}

// The device's sampler calls, beside the samplers they write.

core::Result<SamplerHandle> D3D12Device::createSampler(const SamplerDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectBuffers();
    auto slot = bindless_.takeSampler();
    if (!slot) {
        return std::unexpected(std::move(slot.error()));
    }
    const SamplerHandle sampler = samplers_.emplace(D3D12Sampler{.slot = *slot});
    if (sampler.isNull()) {
        bindless_.giveSampler(*slot);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} samplers",
                             desc_.maxBindlessSamplers);
    }
    if (auto written = writeD3D12Sampler(*device_.Get(), device11_.Get(), desc, layout_.samplerAt(*slot)); !written) {
        samplers_.erase(sampler);
        bindless_.giveSampler(*slot);
        return std::unexpected(std::move(written.error()));
    }
    return sampler;
}

core::Result<std::uint32_t> D3D12Device::samplerSlot(SamplerHandle sampler) {
    const D3D12Sampler* found = samplers_.find(sampler);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    return found->slot;
}

core::Result<void> D3D12Device::destroySampler(SamplerHandle sampler) {
    if (sampler.isNull()) {
        return {};
    }
    if (samplers_.find(sampler) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    // Retired against every submit made so far, as a buffer is: the slot's
    // descriptor stays until the GPU is done with it.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!samplers_.retire(sampler, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    collectBuffers();
    return {};
}

} // namespace bfr::rhi::d3d12
