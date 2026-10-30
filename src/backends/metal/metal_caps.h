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

#include <string>

namespace bfr::rhi::metal {

/// Why device is below the floor, empty when it meets it: Apple GPU family 7
/// or later, with Metal 4.
[[nodiscard]] std::string belowFloor(MTL::Device* device);

/// What device is and does, from what Metal answers about it. A field Metal
/// has no call for is 0 or false.
[[nodiscard]] Caps readCaps(MTL::Device* device);

} // namespace bfr::rhi::metal
