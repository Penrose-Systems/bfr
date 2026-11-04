#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/sampler.h"
#include "metal/metal_util.h"

#include <Metal/Metal.hpp>

#include <cstdint>

namespace bfr::rhi::metal {

/// A Metal sampler state, released when the last Ref to it goes, made for the
/// argument buffer, and its slot in the sampler table, which the device gives
/// back when it is collected. A sampler is not a resource, so it needs no
/// residency.
struct MetalSampler {
    Ref<MTL::SamplerState> sampler;
    std::uint32_t slot = 0;
};

/// A sampler state on device as desc says. Fails with RhiUnsupported for a
/// border colour other than Metal's three, and RhiDeviceFailed when Metal
/// gives no sampler.
[[nodiscard]] core::Result<Ref<MTL::SamplerState>> createMetalSampler(MTL::Device& device, const SamplerDesc& desc);

} // namespace bfr::rhi::metal
