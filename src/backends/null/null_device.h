#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"

#include <cstdint>
#include <vector>

namespace bfr::rhi::null {

class NullRhi;

/// A swapchain that presents nowhere: what it was asked to be, the size it
/// was last built at, and whether an image is acquired.
struct NullSwapchain {
    SwapchainRequest request;
    SwapchainImages images;
    bool acquired = false;
};

/// A pool whose lists go nowhere: the queue it records for, and the lists
/// still recording from it.
struct NullCommandPool {
    std::uint32_t queue = 0;
    std::vector<CommandListHandle> recording;
};

/// Commands that go nowhere: the pool they came from and whether rendering is
/// open.
struct NullCommandList {
    CommandPoolHandle pool;
    bool rendering = false;
};

class NullDevice final : public Device {
public:
    /// rhi outlives this. The device has a graphics queue and then the queues
    /// asked for, which run nothing.
    NullDevice(NullRhi& rhi, const DeviceQueues& queues);
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
    [[nodiscard]] core::Result<CommandPoolHandle> createCommandPool(std::uint32_t queue) override;
    [[nodiscard]] core::Result<void> resetCommandPool(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<void> destroyCommandPool(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<CommandListHandle> beginCommands(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                    const RenderingDesc& desc) override;
    [[nodiscard]] core::Result<void> endRendering(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<Token> submit(const SubmitDesc& desc) override;
    [[nodiscard]] core::Result<void> wait(Token token) override;
    [[nodiscard]] core::Result<bool> poll(Token token) override;
    [[nodiscard]] core::Result<void> destroyCommandList(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    [[nodiscard]] core::Result<NullSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<NullCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<NullCommandList*> commandListFor(CommandListHandle commandList);
    [[nodiscard]] core::Result<void> checkToken(Token token) const;
    [[nodiscard]] core::Result<void> checkQueue(std::uint32_t queue) const;

    NullRhi* rhi_;
    // The same room and limits as every backend's device.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    static constexpr std::uint32_t kSwapchainLimit = 256;
    static constexpr std::uint32_t kCommandPoolCapacity = 4;
    static constexpr std::uint32_t kCommandPoolLimit = 256;
    static constexpr std::uint32_t kCommandListCapacity = 8;
    static constexpr std::uint32_t kCommandListLimit = 4096;
    core::Pool<NullSwapchain, Swapchain> swapchains_;
    core::Pool<NullCommandPool, CommandPool> commandPools_;
    core::Pool<NullCommandList, CommandList> commandLists_;
    std::vector<QueueType> queues_;
    std::vector<std::uint64_t> submitted_;
};

} // namespace bfr::rhi::null
