#pragma once

// What Metal says about formats and about a CAMetalLayer as a surface.

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

/// The pixel format Metal has for format, MTL::PixelFormatInvalid when it has
/// none.
[[nodiscard]] MTL::PixelFormat pixelFormatFor(Format format);

/// What device does with format: every field false and no sample counts for a
/// format Metal lacks or the device doesn't support.
[[nodiscard]] FormatCaps formatCapsFor(MTL::Device* device, Format format);

/// What a swapchain on a CAMetalLayer may be on a device whose largest 2D
/// texture is maxTexture2d pixels.
[[nodiscard]] SurfaceCaps surfaceCapsFor(std::uint32_t maxTexture2d);

/// The formats and color spaces a CAMetalLayer presents in, BFR's names only.
[[nodiscard]] std::span<const SurfaceFormat> surfaceFormats();

/// How a layer is set up to present format in colorSpace.
struct LayerFormat {
    MTL::PixelFormat pixelFormat = MTL::PixelFormatInvalid;
    /// A CGColorSpace name constant; the caller creates and releases the space.
    CFStringRef colorSpaceName = nullptr;
    /// Whether the layer wants extended dynamic range content.
    bool extendedRange = false;
};

/// The layer setup for a pair surfaceFormats() lists, the two Godot's
/// drivers/metal sets a layer up with; unset for any other.
[[nodiscard]] LayerFormat layerFormatFor(Format format, ColorSpace colorSpace);

} // namespace bfr::rhi::metal
