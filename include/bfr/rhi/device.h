#pragma once

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/swapchain.h"

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
    /// surface extension is unavailable, or when the API refuses the window,
    /// and RhiPresentUnsupported when the device can't present to it.
    [[nodiscard]] virtual core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                        const SwapchainRequest& request) = 0;

    /// Builds the swapchain's images at width by height pixels, replacing any
    /// it has once the GPU is done with them. The surface may set the size
    /// itself. A window with no area leaves the swapchain empty, which is not a
    /// failure. Builds as the request says. Fails with RhiInvalidHandle when
    /// swapchain is destroyed, RhiUnsupported when the backend has no setting
    /// for a value of the request, and RhiSwapchainFailed when the API refuses;
    /// a failure leaves it empty.
    [[nodiscard]] virtual core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                        std::uint32_t height) = 0;

    /// Destroys the swapchain once the GPU is done with its images, before
    /// returning: its window is then free to close. A null handle does
    /// nothing. Fails with RhiInvalidHandle when the swapchain is already
    /// destroyed.
    [[nodiscard]] virtual core::Result<void> destroySwapchain(SwapchainHandle swapchain) = 0;

    /// Takes the swapchain's next image, for beginRendering. Fails with
    /// RhiOutOfDate when the swapchain has no images or no longer matches its
    /// window, which resizeSwapchain mends, RhiInvalidCall when an image is
    /// already acquired, and RhiInvalidHandle when the swapchain is destroyed.
    [[nodiscard]] virtual core::Result<void> acquire(SwapchainHandle swapchain) = 0;

    /// Shows the acquired image once what submit sent for it is done. Fails
    /// with RhiOutOfDate when the swapchain no longer matches its window, the
    /// image still presented when it could be, and RhiInvalidCall when no image
    /// is acquired.
    [[nodiscard]] virtual core::Result<void> present(SwapchainHandle swapchain) = 0;

    /// Starts recording commands. The device holds a few command lists at
    /// once and waits for the GPU to finish the oldest before reusing it. Fails
    /// with RhiLimitReached while every one is still recording.
    [[nodiscard]] virtual core::Result<CommandListHandle> beginCommands() = 0;

    /// Begins rendering to the image acquire took from swapchain, as desc
    /// says; ends with endRendering. Fails with RhiInvalidCall when no image is
    /// acquired, when it was already submitted, or when rendering is open.
    [[nodiscard]] virtual core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                            const RenderingDesc& desc) = 0;

    /// Ends the rendering beginRendering began; the image is then ready to
    /// present. Fails with RhiInvalidCall when none is open.
    [[nodiscard]] virtual core::Result<void> endRendering(CommandListHandle commandList) = 0;

    /// Sends the recorded commands to the GPU, which runs them in the order
    /// submitted, and ends the command list. Fails with RhiInvalidCall while
    /// rendering is open.
    [[nodiscard]] virtual core::Result<void> submit(CommandListHandle commandList) = 0;

    /// Discards a command list that was not submitted. A null handle does
    /// nothing. Fails with RhiInvalidHandle when the list is already gone.
    [[nodiscard]] virtual core::Result<void> destroyCommandList(CommandListHandle commandList) = 0;

    /// How much of each heap the process may use and uses now. Fails with
    /// RhiUnsupported when the driver can't say.
    [[nodiscard]] virtual core::Result<MemoryBudget> budget() const = 0;

protected:
    Device() = default;
};

} // namespace bfr::rhi
