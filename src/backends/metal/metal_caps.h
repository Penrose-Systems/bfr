#pragma once

#include "bfr/rhi/caps.h"
#include "metal/metal_util.h"

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <objc/message.h>
#include <objc/runtime.h>
#include <QuartzCore/QuartzCore.hpp>
#include <TargetConditionals.h>

#include <cstdint>
#include <string>

namespace bfr::rhi::metal {

/// Metal 4's render encoder takes one indirect argument address per draw and
/// no draw count (MTL4RenderCommandEncoder.hpp:69, 79), so it has neither
/// Features::multiDrawIndirect nor Features::drawIndirectCount.
inline constexpr bool kMultiDrawIndirect = false;
inline constexpr bool kDrawIndirectCount = false;

/// Why device is below the floor, empty when it meets it: Metal 4, by
/// Metal's own answer.
[[nodiscard]] std::string belowFloor(MTL::Device* device);

/// Whether device has 64-bit integer atomics, on buffers and on RG32Uint and
/// RG32Sint textures. Metal has no call for it; Godot and The Forge give it
/// to Apple9, and to Apple8 on a Mac.
[[nodiscard]] bool has64BitAtomics(MTL::Device* device);

/// Each sample count device renders at, as bits whose values are the counts.
[[nodiscard]] std::uint32_t sampleCountsOf(MTL::Device* device);

/// What device is and does, from what Metal answers about it. Where Metal has
/// no call, a field is what BFR builds on it (queues, one viewport, render
/// state) or what Metal's documentation gives; any other is 0 or false.
[[nodiscard]] Caps readCaps(MTL::Device* device);

} // namespace bfr::rhi::metal
