#pragma once

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/swapchain.h"
#include "metal/metal_util.h"
#include "rhi/swapchain_build.h"

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <objc/message.h>
#include <objc/runtime.h>
#include <QuartzCore/QuartzCore.hpp>
#include <TargetConditionals.h>

#include <cstdint>

namespace bfr::rhi::metal {

/// Whether a swapchain can be made on window: a CAMetalLayer or no window at
/// all. Fails with RhiSurfaceFailed as Device::createSwapchain describes.
[[nodiscard]] core::Result<void> checkWindow(const core::window::NativeWindow& window);

/// The layer a swapchain on window presents to: the caller's CAMetalLayer, or a
/// layer of its own, shown nowhere, for a headless window. Fails as
/// checkWindow does.
[[nodiscard]] core::Result<Ref<CA::MetalLayer>> layerFor(const core::window::NativeWindow& window);

/// One swapchain on a CAMetalLayer, and what it was last built as; empty until
/// built.
class MetalSwapchain {
public:
    /// A swapchain on layer, built as request says each time. It reports to
    /// logger, which outlives it.
    MetalSwapchain(Ref<MTL::Device> device, Ref<CA::MetalLayer> layer, const SwapchainRequest& request,
                   const core::Logger& logger);
    ~MetalSwapchain() = default;
    MetalSwapchain(const MetalSwapchain&) = delete;
    MetalSwapchain& operator=(const MetalSwapchain&) = delete;
    MetalSwapchain(MetalSwapchain&&) = delete;
    MetalSwapchain& operator=(MetalSwapchain&&) = delete;

    /// Whether the layer allows the request, or why not: RhiUnsupported, saying
    /// what it allows.
    [[nodiscard]] core::Result<void> check() const;

    /// Sets the layer up as the request says, at size in pixels, dropping any
    /// image acquired; a window with no area leaves it empty. Fails with
    /// RhiUnsupported when the layer no longer allows the request.
    [[nodiscard]] core::Result<SwapchainImages> build(Extent size);

    /// Takes the layer's next drawable. Fails with RhiOutOfDate when the
    /// swapchain has no images, RhiInvalidCall when an image is already
    /// acquired, and RhiSwapchainFailed when Metal gives no drawable.
    [[nodiscard]] core::Result<void> acquire();

    /// The acquired drawable, null when none is.
    [[nodiscard]] CA::MetalDrawable* drawable() const { return drawable_.get(); }

    /// Whether a submit already renders to the acquired drawable.
    [[nodiscard]] bool submitted() const { return submitted_; }

    /// Records that a submit renders to the acquired drawable.
    void setSubmitted() { submitted_ = true; }

    /// Shows the acquired drawable once the work submitted for it is done, or
    /// at once when none was; no image is acquired afterwards. Fails with
    /// RhiInvalidCall when none is acquired.
    [[nodiscard]] core::Result<void> present();

    [[nodiscard]] CA::MetalLayer* layer() const { return layer_.get(); }

private:
    Ref<MTL::Device> device_;
    Ref<CA::MetalLayer> layer_;
    SwapchainRequest request_;
    const core::Logger* logger_;
    // How many images the layer was last set up with; 0 while empty.
    std::uint32_t count_ = 0;
    Ref<CA::MetalDrawable> drawable_;
    bool submitted_ = false;
};

} // namespace bfr::rhi::metal
