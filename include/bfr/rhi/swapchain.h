#pragma once

#include "bfr/bfr_enums.hpp" // IWYU pragma: export
#include "bfr/core/handle.h"
#include "bfr/rhi/format.h"

#include <cstdint>
#include <string_view>

namespace bfr::rhi {

/// The images a device presents to one window in turn. Only named, by a
/// SwapchainHandle.
struct Swapchain;

using SwapchainHandle = core::Handle<Swapchain>;

/// What a swapchain is asked to be. Each value is passed to the backend as
/// given; the backend accepts or refuses it.
struct SwapchainDesc {
    /// The device's queue that presents the images. createSwapchain fails
    /// with RhiPresentUnsupported when its family doesn't present to the
    /// window, and with RhiInvalidCall when the device has no such queue.
    std::uint32_t queue = 0;
    /// The caller sets it; it is passed to the backend as given.
    std::uint32_t imageCount = 0;
    PresentMode presentMode = PresentMode::Fifo;
    Format format = Format::Undefined;
    ColorSpace colorSpace{};
    AlphaMode alphaMode = AlphaMode::Opaque;
};

/// A swapchain's images: their size in pixels and how many there are, both
/// zero while the swapchain is empty, and the rest of what it was built as.
struct SwapchainImages {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t count = 0;
    Format format = Format::Undefined;
    ColorSpace colorSpace{};
    PresentMode presentMode = PresentMode::Fifo;
    AlphaMode alphaMode = AlphaMode::Opaque;
};

} // namespace bfr::rhi
