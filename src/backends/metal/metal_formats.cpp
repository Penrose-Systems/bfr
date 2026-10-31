#include "metal/metal_formats.h"

#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"

#include <CoreGraphics/CGColorSpace.h>
#include <Metal/MTLPixelFormat.hpp>

#include <span>

namespace bfr::rhi::metal {

SwapchainCaps surfaceCapsFor() {
    return {};
}

std::span<const SwapchainFormat> swapchainFormats() {
    return {};
}

LayerFormat layerFormatFor(Format format, ColorSpace colorSpace) {
    if (colorSpace == ColorSpace::Srgb && format == Format::Bgra8Unorm) {
        return {.pixelFormat = MTL::PixelFormatBGRA8Unorm, .colorSpaceName = kCGColorSpaceSRGB, .extendedRange = false};
    }
    if (colorSpace == ColorSpace::SrgbExtendedLinear && format == Format::Rgba16Float) {
        return {.pixelFormat = MTL::PixelFormatRGBA16Float,
                .colorSpaceName = kCGColorSpaceExtendedLinearSRGB,
                .extendedRange = true};
    }
    return {};
}

} // namespace bfr::rhi::metal
