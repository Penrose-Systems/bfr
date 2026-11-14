#pragma once

#include "bfr/bfr_enums.hpp" // IWYU pragma: export
#include "bfr/core/handle.h"
#include "bfr/rhi/pipeline.h"

#include <array>
#include <cstdint>

namespace bfr::rhi {

/// How shaders read a texture through the sampler table. Only named, by a
/// SamplerHandle.
struct Sampler;

using SamplerHandle = core::Handle<Sampler>;

/// Components of a border colour: red, green, blue and alpha.
inline constexpr std::uint32_t kBorderColorComponents = 4;

/// How to create a sampler. Every member reaches the backend as it is, and
/// the backend's rules for them hold, as api/bfr.yaml says under
/// bfr_sampler_desc. borderColorInteger applies instead of borderColor when
/// borderInteger is set; a signed format reads its bits as int32.
struct SamplerDesc {
    Filter magFilter = Filter::Nearest;
    Filter minFilter = Filter::Nearest;
    Filter mipFilter = Filter::Nearest;
    AddressMode addressU = AddressMode::Repeat;
    AddressMode addressV = AddressMode::Repeat;
    AddressMode addressW = AddressMode::Repeat;
    float lodBias = 0;
    float minLod = 0;
    float maxLod = 0;
    /// 0 is none.
    std::uint32_t maxAnisotropy = 0;
    SamplerReduction reduction = SamplerReduction::WeightedAverage;
    CompareOp compareOp = CompareOp::Always;
    std::array<float, kBorderColorComponents> borderColor{};
    bool borderInteger = false;
    std::array<std::uint32_t, kBorderColorComponents> borderColorInteger{};
    bool unnormalizedCoordinates = false;
    bool lodAverage = false;
};

} // namespace bfr::rhi
