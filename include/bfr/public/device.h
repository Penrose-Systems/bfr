#pragma once

#include "error.h"
#include "swapchain.h"
#include "window.h"

#include <cstdint>

namespace bfr::rhi {

/// A logical GPU on one of the Rhi's adapters.
class Device {
public:
    /// Waits for the GPU to finish its work, then releases the device and the
    /// swapchains left on it.
    virtual ~Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    /// An empty swapchain on window, which must stay open until the swapchain
    /// is destroyed; resizeSwapchain builds its images. Fails with
    /// RhiSurfaceFailed when there is no window, when its window system's
    /// surface extension is unavailable, or when the API refuses the window.
    [[nodiscard]] virtual core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                        std::uint32_t imageCount) = 0;

    /// Builds the swapchain's images at width by height pixels, replacing any
    /// it has, which nothing may still be using. The surface may set the size
    /// itself. A window with no area leaves the swapchain empty, which is not a
    /// failure. Fails with RhiInvalidHandle when swapchain is destroyed,
    /// RhiPresentUnsupported when the device can't present to its surface, and
    /// RhiSwapchainFailed when the API refuses; a failure leaves it empty.
    [[nodiscard]] virtual core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                        std::uint32_t height) = 0;

    /// Destroys the swapchain, whose images nothing may still be using, before
    /// returning: its window is then free to close. A null handle does
    /// nothing. Fails with RhiInvalidHandle when the swapchain is already
    /// destroyed.
    [[nodiscard]] virtual core::Result<void> destroySwapchain(SwapchainHandle swapchain) = 0;

protected:
    Device() = default;
};

} // namespace bfr::rhi
