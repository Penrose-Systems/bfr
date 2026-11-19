#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/sampler.h"
#include "rhi/sampler_border.h"

#include <array>

namespace bfr::rhi {

/// Fails with RhiUnsupported for anything in desc the device's features say
/// it lacks, naming the feature. Each backend's createSampler calls it after
/// checkEnums, so a sampler is never made with a setting the caps report the
/// backend can't give.
[[nodiscard]] inline core::Result<void> checkSamplerFeatures(const SamplerDesc& desc, const Features& features) {
    const auto lacks = [](const char* what) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device has no {}", what);
    };
    const std::array addresses{desc.addressU, desc.addressV, desc.addressW};
    if (desc.maxAnisotropy != 0 && !features.samplerAnisotropy) {
        return lacks("anisotropic filtering (Features::samplerAnisotropy)");
    }
    if ((desc.reduction == SamplerReduction::Minimum || desc.reduction == SamplerReduction::Maximum) &&
        !features.samplerMinmax) {
        return lacks("min or max reduction (Features::samplerMinmax)");
    }
    if (std::ranges::contains(addresses, AddressMode::MirrorClampToEdge) && !features.samplerMirrorClampToEdge) {
        return lacks("mirror clamp to edge (Features::samplerMirrorClampToEdge)");
    }
    if (borderPreset(desc) == BorderPreset::Custom && !features.samplerCustomBorderColor) {
        return lacks("custom border colors (Features::samplerCustomBorderColor)");
    }
    if (desc.borderInteger && !features.samplerIntegerBorderColor) {
        return lacks("integer border colors (Features::samplerIntegerBorderColor)");
    }
    if (desc.unnormalizedCoordinates && !features.samplerUnnormalizedCoordinates) {
        return lacks("unnormalized coordinates (Features::samplerUnnormalizedCoordinates)");
    }
    if (desc.lodBias != 0 && !features.samplerLodBias) {
        return lacks("LOD bias (Features::samplerLodBias)");
    }
    if (desc.lodAverage && !features.samplerLodAverage) {
        return lacks("LOD averaging (Features::samplerLodAverage)");
    }
    return {};
}

} // namespace bfr::rhi
