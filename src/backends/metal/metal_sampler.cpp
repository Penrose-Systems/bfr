#include "metal/metal_sampler.h"

#include "bfr/core/error.h"
#include "bfr/rhi/sampler.h"
#include "metal/metal_device.h"
#include "metal/metal_pipeline.h"
#include "metal/metal_util.h"
#include "rhi/bindless_slots.h"
#include "rhi/enum_check.h"
#include "rhi/sampler_border.h"

#include <Metal/Metal.hpp>

#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <utility>

namespace bfr::rhi::metal {

namespace {

MTL::SamplerMinMagFilter filterFor(Filter filter) {
    return filter == Filter::Linear ? MTL::SamplerMinMagFilterLinear : MTL::SamplerMinMagFilterNearest;
}

MTL::SamplerMipFilter mipFilterFor(Filter filter) {
    return filter == Filter::Linear ? MTL::SamplerMipFilterLinear : MTL::SamplerMipFilterNearest;
}

MTL::SamplerAddressMode addressFor(AddressMode mode) {
    switch (mode) {
        case AddressMode::MirrorRepeat:
            return MTL::SamplerAddressModeMirrorRepeat;
        case AddressMode::ClampToEdge:
            return MTL::SamplerAddressModeClampToEdge;
        case AddressMode::ClampToBorder:
            return MTL::SamplerAddressModeClampToBorderColor;
        case AddressMode::MirrorClampToEdge:
            return MTL::SamplerAddressModeMirrorClampToEdge;
        case AddressMode::Repeat:
        default:
            return MTL::SamplerAddressModeRepeat;
    }
}

MTL::SamplerReductionMode reductionFor(SamplerReduction reduction) {
    switch (reduction) {
        case SamplerReduction::Minimum:
            return MTL::SamplerReductionModeMinimum;
        case SamplerReduction::Maximum:
            return MTL::SamplerReductionModeMaximum;
        case SamplerReduction::WeightedAverage:
        case SamplerReduction::Compare:
        default:
            return MTL::SamplerReductionModeWeightedAverage;
    }
}

MTL::SamplerBorderColor borderFor(BorderPreset preset) {
    switch (preset) {
        case BorderPreset::OpaqueBlack:
            return MTL::SamplerBorderColorOpaqueBlack;
        case BorderPreset::OpaqueWhite:
            return MTL::SamplerBorderColorOpaqueWhite;
        case BorderPreset::TransparentBlack:
        case BorderPreset::Custom:
        default:
            return MTL::SamplerBorderColorTransparentBlack;
    }
}

} // namespace

core::Result<Ref<MTL::SamplerState>> createMetalSampler(MTL::Device& device, const SamplerDesc& desc) {
    const BorderPreset preset = borderPreset(desc);
    if (preset == BorderPreset::Custom) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "Metal has only transparent black, opaque black and opaque white border colours");
    }
    const Ref<MTL::SamplerDescriptor> descriptor = adopt(MTL::SamplerDescriptor::alloc()->init());
    descriptor->setMagFilter(filterFor(desc.magFilter));
    descriptor->setMinFilter(filterFor(desc.minFilter));
    descriptor->setMipFilter(mipFilterFor(desc.mipFilter));
    descriptor->setSAddressMode(addressFor(desc.addressU));
    descriptor->setTAddressMode(addressFor(desc.addressV));
    descriptor->setRAddressMode(addressFor(desc.addressW));
    // Every OS Metal 4 runs on has lodBias (macOS and iOS 26), so no
    // availability guard; GPUs before Apple10 ignore it, as the caps say.
    descriptor->setLodBias(desc.lodBias);
    descriptor->setLodMinClamp(desc.minLod);
    descriptor->setLodMaxClamp(desc.maxLod);
    // Metal's 1 is no anisotropy, BFR's 0.
    if (desc.maxAnisotropy != 0) {
        descriptor->setMaxAnisotropy(desc.maxAnisotropy);
    }
    descriptor->setReductionMode(reductionFor(desc.reduction));
    if (desc.reduction == SamplerReduction::Compare) {
        descriptor->setCompareFunction(compareFor(desc.compareOp));
    }
    descriptor->setBorderColor(borderFor(preset));
    descriptor->setNormalizedCoordinates(!desc.unnormalizedCoordinates);
    descriptor->setLodAverage(desc.lodAverage);
    // The tables hold its resource ID, which needs this.
    descriptor->setSupportArgumentBuffers(true);
    MTL::SamplerState* sampler = device.newSamplerState(descriptor.get());
    if (sampler == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no sampler");
    }
    return adopt(sampler);
}

// The device's sampler calls, beside the samplers they make.

core::Result<SamplerHandle> MetalDevice::createSampler(const SamplerDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    auto slot = bindless_.takeSampler();
    if (!slot) {
        return std::unexpected(std::move(slot.error()));
    }
    auto made = createMetalSampler(*device_.get(), desc);
    if (!made) {
        bindless_.giveSampler(*slot);
        return std::unexpected(std::move(made.error()));
    }
    const MTL::ResourceID id = (*made)->gpuResourceID();
    const SamplerHandle sampler = samplers_.emplace(MetalSampler{.sampler = std::move(*made), .slot = *slot});
    if (sampler.isNull()) {
        bindless_.giveSampler(*slot);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} samplers",
                             desc_.maxBindlessSamplers);
    }
    tables().writeSampler(*slot, id);
    return sampler;
}

core::Result<std::uint32_t> MetalDevice::samplerSlot(SamplerHandle sampler) {
    const MetalSampler* found = samplers_.find(sampler);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    return found->slot;
}

core::Result<void> MetalDevice::destroySampler(SamplerHandle sampler) {
    if (sampler.isNull()) {
        return {};
    }
    if (samplers_.find(sampler) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!samplers_.retire(sampler, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    collectRetired();
    return {};
}

} // namespace bfr::rhi::metal
