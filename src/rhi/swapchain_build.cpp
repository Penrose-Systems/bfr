#include "rhi/swapchain_build.h"

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/swapchain.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>

namespace bfr::rhi {

namespace {

core::Result<Extent> extentFor(const SurfaceCaps& caps, Extent size) {
    if (caps.width != kSizeFromSwapchain) {
        return Extent{.width = caps.width, .height = caps.height};
    }
    if (size.width == 0 || size.height == 0) {
        return Extent{};
    }
    if (size.width < caps.minWidth || size.height < caps.minHeight || size.width > caps.maxWidth ||
        size.height > caps.maxHeight) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             std::format("the window allows {}x{} to {}x{} pixels; {}x{} were asked", caps.minWidth,
                                         caps.minHeight, caps.maxWidth, caps.maxHeight, size.width, size.height));
    }
    return size;
}

core::Result<std::uint32_t> imageCountFor(const SurfaceCaps& caps, std::uint32_t images) {
    if (images == 0) {
        return caps.minImageCount;
    }
    // A maximum of 0 means there is none.
    if (images >= caps.minImageCount && (caps.maxImageCount == 0 || images <= caps.maxImageCount)) {
        return images;
    }
    const std::string allowed = caps.maxImageCount == 0
                                    ? std::format("{} or more", caps.minImageCount)
                                    : std::format("{} to {}", caps.minImageCount, caps.maxImageCount);
    return core::failure(core::ErrorCode::RhiUnsupported,
                         std::format("the window allows {} images; {} were asked", allowed, images));
}

bool allows(const SurfaceCaps& caps, PresentMode mode) {
    switch (mode) {
        case PresentMode::Fifo:
            return caps.fifo;
        case PresentMode::FifoRelaxed:
            return caps.fifoRelaxed;
        case PresentMode::Mailbox:
            return caps.mailbox;
        case PresentMode::Immediate:
            return caps.immediate;
    }
    return false;
}

bool allows(const SurfaceCaps& caps, AlphaMode mode) {
    switch (mode) {
        case AlphaMode::Opaque:
            return caps.alphaOpaque;
        case AlphaMode::Premultiplied:
            return caps.alphaPremultiplied;
        case AlphaMode::Postmultiplied:
            return caps.alphaPostmultiplied;
        case AlphaMode::Inherit:
            return caps.alphaInherit;
    }
    return false;
}

} // namespace

core::Result<SwapchainBuild> swapchainBuildFor(const SurfaceCaps& caps, std::span<const SurfaceFormat> formats,
                                               Extent size, const SwapchainRequest& request) {
    const auto imageCount = imageCountFor(caps, request.imageCount);
    if (!imageCount) {
        return std::unexpected(imageCount.error());
    }
    const bool listed = std::ranges::any_of(formats, [&request](const SurfaceFormat& format) {
        return format.format == request.format && format.colorSpace == request.colorSpace;
    });
    if (!listed) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the window does not list the format with the color space that were asked");
    }
    if (!allows(caps, request.presentMode)) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             std::format("the window does not allow {} presenting", name(request.presentMode)));
    }
    if (!allows(caps, request.alphaMode)) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             std::format("the window does not allow {} alpha", name(request.alphaMode)));
    }
    const auto extent = extentFor(caps, size);
    if (!extent) {
        return std::unexpected(extent.error());
    }
    return SwapchainBuild{.imageCount = *imageCount, .extent = *extent};
}

} // namespace bfr::rhi
