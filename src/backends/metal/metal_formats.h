#pragma once

// What Metal says about a CAMetalLayer as a surface.

#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
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
#include <span>

namespace bfr::rhi::metal {

/// What a swapchain on a CAMetalLayer may be: nothing, since Metal has no call
/// that says.
[[nodiscard]] SwapchainCaps surfaceCapsFor();

/// The formats and color spaces a CAMetalLayer presents in: none, since Metal
/// has no call that lists them.
[[nodiscard]] std::span<const SwapchainFormat> swapchainFormats();

/// How a layer is set up to present format in colorSpace.
struct LayerFormat {
    MTL::PixelFormat pixelFormat = MTL::PixelFormatInvalid;
    /// A CGColorSpace name constant; the caller creates and releases the space.
    CFStringRef colorSpaceName = nullptr;
    /// Whether the layer wants extended dynamic range content.
    bool extendedRange = false;
};

/// How a layer is set up to present format in colorSpace; unset for a pair it
/// has no setting for.
[[nodiscard]] LayerFormat layerFormatFor(Format format, ColorSpace colorSpace);

} // namespace bfr::rhi::metal
