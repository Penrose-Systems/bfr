#pragma once

#include "bfr/core/handle.h"
#include "bfr/rhi/format.h"

#include <cstdint>
#include <string_view>

namespace bfr::rhi {

/// The images a device presents to one window in turn. Only named, by a
/// SwapchainHandle.
struct Swapchain;

using SwapchainHandle = core::Handle<Swapchain>;

/// How a swapchain presents. Fifo is on every window; the others are on those
/// whose swapchain caps list them.
enum class PresentMode : std::uint8_t {
    /// Presents wait for vertical blank and queue up.
    Fifo,
    /// As Fifo, but a late frame presents at once, and may tear.
    FifoRelaxed,
    /// Presents wait for vertical blank, and a newer frame replaces a waiting one.
    Mailbox,
    /// Presents at once, and may tear.
    Immediate,
};

/// The name of mode, as bfr.yaml spells it.
[[nodiscard]] constexpr std::string_view name(PresentMode mode) {
    switch (mode) {
        case PresentMode::Fifo:
            return "fifo";
        case PresentMode::FifoRelaxed:
            return "fifo_relaxed";
        case PresentMode::Mailbox:
            return "mailbox";
        case PresentMode::Immediate:
            return "immediate";
    }
    return "unknown";
}

/// How a window blends a swapchain's alpha.
enum class AlphaMode : std::uint8_t {
    /// The window ignores alpha.
    Opaque,
    /// Colors are already multiplied by alpha.
    Premultiplied,
    /// The window multiplies colors by alpha.
    Postmultiplied,
    /// The window system decides how alpha blends.
    Inherit,
};

/// The name of mode, as bfr.yaml spells it.
[[nodiscard]] constexpr std::string_view name(AlphaMode mode) {
    switch (mode) {
        case AlphaMode::Opaque:
            return "opaque";
        case AlphaMode::Premultiplied:
            return "premultiplied";
        case AlphaMode::Postmultiplied:
            return "postmultiplied";
        case AlphaMode::Inherit:
            return "inherit";
    }
    return "unknown";
}

/// What a swapchain is asked to be. Each value is passed to the backend as
/// given; the backend accepts or refuses it.
struct SwapchainDesc {
    /// The caller sets it; it is passed to the backend as given.
    std::uint32_t imageCount = 0;
    PresentMode presentMode = PresentMode::Fifo;
    Format format = Format::Bgra8Unorm;
    ColorSpace colorSpace = ColorSpace::Srgb;
    AlphaMode alphaMode = AlphaMode::Opaque;
};

/// A swapchain's images: their size in pixels and how many there are, both
/// zero while the swapchain is empty, and the rest of what it was built as.
struct SwapchainImages {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t count = 0;
    Format format = Format::Bgra8Unorm;
    ColorSpace colorSpace = ColorSpace::Srgb;
    PresentMode presentMode = PresentMode::Fifo;
    AlphaMode alphaMode = AlphaMode::Opaque;
};

} // namespace bfr::rhi
