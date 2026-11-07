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

/// format's Metal pixel format; Invalid for Undefined, and where Metal has
/// none, which Metal then refuses. format is Undefined or one of kFormats.
[[nodiscard]] MTL::PixelFormat pixelFormatFor(Format format);

/// The format pixelFormat is; Undefined for Invalid and for a pixel format no
/// format maps to.
[[nodiscard]] Format formatFor(MTL::PixelFormat pixelFormat);

/// What format may do on device. Metal has no call for it, so it is Godot's
/// record of Apple's Metal feature set tables, changed as Godot changes it
/// for the device. format is one of kFormats.
[[nodiscard]] FormatCaps formatCapsFor(MTL::Device* device, Format format);

/// The formats and color spaces a swapchain on a CAMetalLayer is built in:
/// those layerFormatFor has a setting for. Metal has no call that lists them.
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
