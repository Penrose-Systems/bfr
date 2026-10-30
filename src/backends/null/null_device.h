#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"

#include <cstdint>

namespace bfr::rhi::null {

class NullRhi;

/// A swapchain that presents nowhere: what it was asked to be, the size it
/// was last built at, and whether an image is acquired.
struct NullSwapchain {
    SwapchainRequest request;
    SwapchainImages images;
    bool acquired = false;
};

/// Commands that go nowhere: only whether rendering is open.
struct NullCommandList {
    bool rendering = false;
};

class NullDevice final : public Device {
public:
    /// rhi outlives this.
    explicit NullDevice(NullRhi& rhi);
    ~NullDevice() override;
    NullDevice(const NullDevice&) = delete;
    NullDevice& operator=(const NullDevice&) = delete;
    NullDevice(NullDevice&&) = delete;
    NullDevice& operator=(NullDevice&&) = delete;

    [[nodiscard]] core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                const SwapchainRequest& request) override;
    [[nodiscard]] core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                std::uint32_t height) override;
    [[nodiscard]] core::Result<void> destroySwapchain(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> acquire(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> present(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<CommandListHandle> beginCommands() override;
    [[nodiscard]] core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                    const RenderingDesc& desc) override;
    [[nodiscard]] core::Result<void> endRendering(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> submit(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> destroyCommandList(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    // Fails with RhiInvalidHandle for a swapchain or command list this device
    // no longer has.
    [[nodiscard]] core::Result<NullSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<NullCommandList*> commandListFor(CommandListHandle commandList);

    NullRhi* rhi_;
    // The same room and limits as every backend's device.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    static constexpr std::uint32_t kSwapchainLimit = 256;
    static constexpr std::uint32_t kCommandListLimit = 2;
    core::Pool<NullSwapchain, Swapchain> swapchains_;
    core::Pool<NullCommandList, CommandList> commandLists_;
};

} // namespace bfr::rhi::null
