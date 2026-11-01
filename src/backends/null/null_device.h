#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "core/slot_chains.h"

#include <cstdint>
#include <vector>

namespace bfr::rhi::null {

class NullInstance;

/// A swapchain that presents nowhere: what it was asked to be, the size it
/// was last built at, and whether an image is acquired.
struct NullSwapchain {
    SwapchainDesc desc;
    SwapchainImages images;
    bool acquired = false;
};

/// A pool whose lists go nowhere, with the same bookkeeping as every
/// backend's: the queue it records for, its chain of list slots, the next one
/// beginCommands hands out, and how many it has handed out that still record.
struct NullCommandPool {
    std::uint32_t queue = 0;
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint32_t next = core::SlotChains::kEnd;
    std::uint32_t recording = 0;
};

/// Commands that go nowhere: the pool and slot they came from and whether
/// rendering is open.
struct NullCommandList {
    CommandPoolHandle pool;
    std::uint32_t slot = 0;
    bool rendering = false;
};

class NullDevice final : public Device {
public:
    /// rhi outlives this. The device has a graphics queue and then the queues
    /// asked for, which run nothing.
    NullDevice(NullInstance& rhi, const DeviceDesc& desc);
    ~NullDevice() override;
    NullDevice(const NullDevice&) = delete;
    NullDevice& operator=(const NullDevice&) = delete;
    NullDevice(NullDevice&&) = delete;
    NullDevice& operator=(NullDevice&&) = delete;

    [[nodiscard]] core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                const SwapchainDesc& desc) override;
    [[nodiscard]] core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                std::uint32_t height) override;
    [[nodiscard]] core::Result<void> destroySwapchain(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> acquire(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> present(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc) override;
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
    // Ends a list that was submitted or discarded: its slot waits for the
    // pool's reset.
    void end(CommandListHandle commandList);

    NullInstance* instance_;
    DeviceDesc desc_;
    // Every pool is made at its maximum, so nothing after creation allocates.
    core::Pool<NullSwapchain, Swapchain> swapchains_;
    core::Pool<NullCommandPool, CommandPool> commandPools_;
    core::Pool<NullCommandList, CommandList> commandLists_;
    // A slot for each list the device holds, and the list recording in it.
    core::SlotChains slots_;
    std::vector<CommandListHandle> live_;
    std::vector<QueueType> queues_;
    std::vector<std::uint64_t> submitted_;
};

} // namespace bfr::rhi::null
