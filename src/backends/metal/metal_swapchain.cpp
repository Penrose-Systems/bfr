#include "metal/metal_swapchain.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/swapchain.h"
#include "metal/metal_formats.h"
#include "metal/metal_util.h"
#include "rhi/swapchain_build.h"

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <Metal/MTLDevice.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <TargetConditionals.h>

#include <cstdint>
#include <expected>
#include <utility>
#include <variant>

namespace bfr::rhi::metal {

namespace {

// Each window system a swapchain can be made on, or why not.
struct Checker {
    core::Result<void> operator()(std::monostate /*none*/) const {
        return core::failure(core::ErrorCode::RhiSurfaceFailed, "there is no window");
    }

    core::Result<void> operator()(const core::window::Metal& window) const {
        if (window.layer == nullptr) {
            return core::failure(core::ErrorCode::RhiSurfaceFailed, "the Metal window has no layer");
        }
        return {};
    }

    core::Result<void> operator()(const core::window::Headless& /*window*/) const { return {}; }

    // A window system this build has no surface for.
    template <class Handles>
    core::Result<void> operator()(const Handles& /*window*/) const {
        return core::failure(core::ErrorCode::RhiSurfaceFailed, "this build has no surface for the window's system");
    }
};

} // namespace

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

MetalSwapchain::MetalSwapchain(Ref<MTL::Device> device, Ref<CA::MetalLayer> layer, const SwapchainRequest& request,
                               const core::Logger& logger)
    : device_(std::move(device)), layer_(std::move(layer)), request_(request), logger_(&logger) {}

core::Result<void> MetalSwapchain::check() const {
    const auto build = swapchainBuildFor(surfaceCapsFor(), surfaceFormats(), Extent{}, request_);
    if (!build) {
        return std::unexpected(build.error());
    }
    return {};
}

core::Result<SwapchainImages> MetalSwapchain::build(Extent size) {
    drawable_.reset();
    submitted_ = false;
    count_ = 0;
    const auto settings = swapchainBuildFor(surfaceCapsFor(), surfaceFormats(), size, request_);
    if (!settings) {
        return std::unexpected(settings.error());
    }
    const auto imagesOf = [this](std::uint32_t width, std::uint32_t height, std::uint32_t count) {
        return SwapchainImages{.width = width,
                               .height = height,
                               .count = count,
                               .format = request_.format,
                               .colorSpace = request_.colorSpace,
                               .presentMode = request_.presentMode,
                               .alphaMode = request_.alphaMode};
    };
    if (settings->extent.width == 0 || settings->extent.height == 0) {
        BFR_LOG_DEBUG(*logger_, "swapchain left empty: the window has no area");
        return imagesOf(0, 0, 0);
    }

    const LayerFormat format = layerFormatFor(request_.format, request_.colorSpace);
    layer_->setDevice(device_.get());
    layer_->setPixelFormat(format.pixelFormat);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(format.colorSpaceName);
    layer_->setColorspace(colorSpace);
    CGColorSpaceRelease(colorSpace);
    setLayerFlag(layer_.get(), "setWantsExtendedDynamicRangeContent:", format.extendedRange);
    layer_->setFramebufferOnly(true);
    setLayerFlag(layer_.get(), "setOpaque:", request_.alphaMode == AlphaMode::Opaque);
    layer_->setDrawableSize(CGSize{.width = static_cast<CGFloat>(settings->extent.width),
                                   .height = static_cast<CGFloat>(settings->extent.height)});
    layer_->setMaximumDrawableCount(settings->imageCount);
#if TARGET_OS_OSX
    // Display sync is only a Mac layer's to switch; others always wait.
    layer_->setDisplaySyncEnabled(request_.presentMode != PresentMode::Immediate);
#endif
    layer_->setAllowsNextDrawableTimeout(true);
    count_ = settings->imageCount;
    BFR_LOG_INFO(*logger_, "swapchain {}x{}, {} images, {}", settings->extent.width, settings->extent.height, count_,
                 name(request_.presentMode));
    return imagesOf(settings->extent.width, settings->extent.height, count_);
}

core::Result<void> MetalSwapchain::acquire() {
    if (count_ == 0) {
        return core::failure(core::ErrorCode::RhiOutOfDate, "the swapchain has no images; resize it");
    }
    if (drawable_) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "an image is already acquired; present it first");
    }
    const AutoreleasePool pool;
    CA::MetalDrawable* next = layer_->nextDrawable();
    if (next == nullptr) {
        return core::failure(core::ErrorCode::RhiSwapchainFailed, "nextDrawable gave no drawable");
    }
    drawable_ = share(next);
    submitted_ = false;
    return {};
}

core::Result<void> MetalSwapchain::present() {
    if (!drawable_) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }
    drawable_->present();
    drawable_.reset();
    submitted_ = false;
    return {};
}

} // namespace bfr::rhi::metal
