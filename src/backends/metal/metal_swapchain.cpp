#include "metal/metal_swapchain.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/swapchain.h"
#include "metal/metal_formats.h"
#include "metal/metal_util.h"

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <Metal/MTLDevice.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <TargetConditionals.h>

#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <utility>
#include <variant>

namespace bfr::rhi::metal {

namespace {

// Each window system a swapchain can be made on, or why not.
struct Checker {
    core::Result<void> operator()(std::monostate /*none*/) const {
        return core::failure(core::ErrorCode::RhiWindowFailed, "there is no window");
    }

    core::Result<void> operator()(const core::window::Metal& window) const {
        if (window.layer == nullptr) {
            return core::failure(core::ErrorCode::RhiWindowFailed, "the Metal window has no layer");
        }
        return {};
    }

    core::Result<void> operator()(const core::window::Headless& /*window*/) const { return {}; }

    // A window system this build has no surface for.
    template <class Handles>
    core::Result<void> operator()(const Handles& /*window*/) const {
        return core::failure(core::ErrorCode::RhiWindowFailed, "this build has no surface for the window's system");
    }
};

// Whether the layer waits for the display to present, as mode says. Fails with
// RhiUnsupported for a mode the layer has no setting for.
core::Result<bool> displaySyncFor(PresentMode mode) {
    switch (mode) {
        case PresentMode::Fifo:
            return true;
        case PresentMode::Immediate:
#if TARGET_OS_OSX
            return false;
#else
            // Display sync is only a Mac layer's to switch; others always wait.
            break;
#endif
        case PresentMode::FifoRelaxed:
        case PresentMode::Mailbox:
            break;
    }
    return core::failure(core::ErrorCode::RhiUnsupported, "the layer has no setting for {} presenting", name(mode));
}

// Fails with RhiUnsupported for an alpha mode other than Opaque and
// Premultiplied. CALayer has only opaque or not; Godot makes a transparent
// window's layer non-opaque (rendering_context_driver_metal.cpp). Unverified:
// Core Animation is taken to blend a non-opaque layer as premultiplied.
core::Result<void> checkAlpha(AlphaMode mode) {
    if (mode != AlphaMode::Opaque && mode != AlphaMode::Premultiplied) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the layer has no setting for {} alpha", name(mode));
    }
    return {};
}

// CAMetalLayer's maximumDrawableCount documentation: 2 or 3. Metal has no call
// for it.
constexpr std::uint32_t kMinDrawables = 2;
constexpr std::uint32_t kMaxDrawables = 3;

} // namespace

SwapchainCaps surfaceCapsFor() {
    return {.minImageCount = kMinDrawables,
            .maxImageCount = kMaxDrawables,
            .fifo = displaySyncFor(PresentMode::Fifo).has_value(),
            .fifoRelaxed = displaySyncFor(PresentMode::FifoRelaxed).has_value(),
            .mailbox = displaySyncFor(PresentMode::Mailbox).has_value(),
            .immediate = displaySyncFor(PresentMode::Immediate).has_value(),
            .alphaOpaque = checkAlpha(AlphaMode::Opaque).has_value(),
            .alphaPremultiplied = checkAlpha(AlphaMode::Premultiplied).has_value(),
            .alphaPostmultiplied = checkAlpha(AlphaMode::Postmultiplied).has_value(),
            .alphaInherit = checkAlpha(AlphaMode::Inherit).has_value()};
}

core::Result<void> checkWindow(const core::window::NativeWindow& window) {
    return std::visit(Checker{}, window);
}

core::Result<Ref<CA::MetalLayer>> layerFor(const core::window::NativeWindow& window) {
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (const auto* metal = std::get_if<core::window::Metal>(&window)) {
        return share(static_cast<CA::MetalLayer*>(metal->layer));
    }
    // A headless window: a layer of its own that nothing shows.
    const AutoreleasePool pool;
    return share(CA::MetalLayer::layer());
}

MetalSwapchain::MetalSwapchain(Ref<MTL::Device> device, Ref<CA::MetalLayer> layer, const SwapchainDesc& desc,
                               const core::Logger& logger)
    : device_(std::move(device)), layer_(std::move(layer)), desc_(desc), logger_(&logger) {}

core::Result<SwapchainImages> MetalSwapchain::build(std::uint32_t width, std::uint32_t height) {
    drawable_.reset();
    submitted_ = false;
    count_ = 0;
    if (width == 0 || height == 0) {
        // The layer is not set up, so there is nothing of it to read.
        BFR_LOG_DEBUG(*logger_, "swapchain left empty: the window has no area");
        return SwapchainImages{.width = 0,
                               .height = 0,
                               .count = 0,
                               .format = desc_.format,
                               .colorSpace = desc_.colorSpace,
                               .presentMode = desc_.presentMode,
                               .alphaMode = desc_.alphaMode};
    }

    const LayerFormat format = layerFormatFor(desc_.format, desc_.colorSpace);
    if (format.pixelFormat == MTL::PixelFormatInvalid) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the layer has no setting for the format with the color space that were asked");
    }
    const auto displaySync = displaySyncFor(desc_.presentMode);
    if (!displaySync) {
        return std::unexpected(displaySync.error());
    }
    if (auto checked = checkAlpha(desc_.alphaMode); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    layer_->setDevice(device_.get());
    layer_->setPixelFormat(format.pixelFormat);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(format.colorSpaceName);
    layer_->setColorspace(colorSpace);
    CGColorSpaceRelease(colorSpace);
    setLayerFlag(layer_.get(), "setWantsExtendedDynamicRangeContent:", format.extendedRange);
    layer_->setFramebufferOnly(true);
    setLayerFlag(layer_.get(), "setOpaque:", desc_.alphaMode == AlphaMode::Opaque);
    layer_->setDrawableSize(CGSize{.width = static_cast<CGFloat>(width), .height = static_cast<CGFloat>(height)});
    layer_->setMaximumDrawableCount(desc_.imageCount);
#if TARGET_OS_OSX
    layer_->setDisplaySyncEnabled(*displaySync);
#endif
    count_ = static_cast<std::uint32_t>(layer_->maximumDrawableCount());

    // What the layer now has. Two stay as asked: its color space comes back as
    // a CGColorSpace, which only CoreGraphics' CGColorSpaceCopyName names, and
    // metal-cpp has no getter for CALayer's opaque.
    const CGSize size = layer_->drawableSize();
#if TARGET_OS_OSX
    const PresentMode presentMode = layer_->displaySyncEnabled() ? PresentMode::Fifo : PresentMode::Immediate;
#else
    // Only a Mac layer has display sync; others always wait.
    const PresentMode presentMode = PresentMode::Fifo;
#endif
    const SwapchainImages images{.width = static_cast<std::uint32_t>(size.width),
                                 .height = static_cast<std::uint32_t>(size.height),
                                 .count = count_,
                                 .format = formatFor(layer_->pixelFormat()),
                                 .colorSpace = desc_.colorSpace,
                                 .presentMode = presentMode,
                                 .alphaMode = desc_.alphaMode};
    BFR_LOG_INFO(*logger_, "swapchain {}x{}, {} images, {}", images.width, images.height, images.count,
                 name(images.presentMode));
    return images;
}

core::Result<SwapchainStatus> MetalSwapchain::acquire(std::uint64_t timeoutNs) {
    if (count_ == 0) {
        return core::failure(core::ErrorCode::RhiOutOfDate, "the swapchain has no images; resize it");
    }
    if (drawable_) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "an image is already acquired; present it first");
    }
    // nextDrawable takes no time, only whether to give up after Metal's own.
    const bool timed = timeoutNs != std::numeric_limits<std::uint64_t>::max();
    layer_->setAllowsNextDrawableTimeout(timed);
    const AutoreleasePool pool;
    CA::MetalDrawable* next = layer_->nextDrawable();
    if (next == nullptr && timed) {
        return core::failure(core::ErrorCode::RhiTimeout, "nextDrawable gave no drawable in Metal's time");
    }
    if (next == nullptr) {
        return core::failure(core::ErrorCode::RhiSwapchainFailed, "nextDrawable gave no drawable");
    }
    drawable_ = share(next);
    submitted_ = false;
    return SwapchainStatus::Matches;
}

core::Result<SwapchainStatus> MetalSwapchain::present() {
    if (!drawable_) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }
    drawable_->present();
    drawable_.reset();
    submitted_ = false;
    return SwapchainStatus::Matches;
}

} // namespace bfr::rhi::metal
