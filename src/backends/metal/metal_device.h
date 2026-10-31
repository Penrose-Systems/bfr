#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_util.h"

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <objc/message.h>
#include <objc/runtime.h>
#include <QuartzCore/QuartzCore.hpp>
#include <TargetConditionals.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace bfr::rhi::metal {

class MetalRhi;

/// A command pool: an allocator and a buffer for each list recording at once,
/// since a Metal 4 allocator serves one buffer at a time; how many are handed
/// out since the last reset, the lists still recording, and the event value its
/// last submit signals.
struct MetalCommandPool {
    struct Entry {
        Ref<MTL4::CommandAllocator> allocator;
        Ref<MTL4::CommandBuffer> buffer;
    };
    std::uint32_t queue = 0;
    std::vector<Entry> entries;
    std::uint32_t used = 0;
    std::vector<CommandListHandle> recording;
    std::uint64_t lastSubmit = 0;
};

/// One of the device's queues: its MTL4 command queue, the shared event each
/// submit signals the next value of, the value last submitted, the array a
/// submit is built in, and the allocators of pools destroyed while the GPU may
/// still run their lists, each with the value that frees them. Used by one
/// thread at a time under lock, which present shares; submitted is read from
/// any thread.
struct MetalQueue {
    QueueType type = QueueType::Graphics;
    Ref<MTL4::CommandQueue> queue;
    Ref<MTL::SharedEvent> event;
    std::atomic<std::uint64_t> submitted{0};
    std::vector<const MTL4::CommandBuffer*> buffers;
    std::vector<std::pair<std::vector<MetalCommandPool::Entry>, std::uint64_t>> retired;
    std::atomic<std::uint32_t> retiredCount{0};
    std::mutex lock;
};

/// Commands recorded on one buffer: the pool it came from, the encoder
/// rendering is open on, the swapchain rendering is open on, and every
/// swapchain rendered to, whose drawables the submit waits for and signals.
struct MetalCommandList {
    Ref<MTL4::CommandBuffer> buffer;
    CommandPoolHandle pool;
    Ref<MTL4::RenderCommandEncoder> encoder;
    std::optional<SwapchainHandle> rendering;
    std::vector<SwapchainHandle> rendered;
};

class MetalDevice final : public Device {
public:
    /// A device on device, which rhi listed, with a graphics queue and the
    /// queues asked for. Fails with RhiDeviceFailed when Metal gives no queue
    /// or event.
    [[nodiscard]] static core::Result<std::unique_ptr<MetalDevice>> create(MetalRhi& rhi, Ref<MTL::Device> device,
                                                                           const DeviceQueues& queues);

    /// For create.
    MetalDevice(MetalRhi& rhi, Ref<MTL::Device> device);
    ~MetalDevice() override;
    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;
    MetalDevice(MetalDevice&&) = delete;
    MetalDevice& operator=(MetalDevice&&) = delete;

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
    [[nodiscard]] core::Result<void> open(const DeviceQueues& queues);
    [[nodiscard]] core::Result<void> checkQueue(std::uint32_t queue) const;
    [[nodiscard]] core::Result<void> checkToken(Token token) const;
    void collectPools(MetalQueue& queue);
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    static void discard(MetalCommandList& list);
    [[nodiscard]] core::Result<MetalSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<MetalCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<MetalCommandList*> commandListFor(CommandListHandle commandList);

    MetalRhi* rhi_;
    Ref<MTL::Device> device_;
    std::vector<std::unique_ptr<MetalQueue>> queues_;
    // Room for a few windows' swapchains before the pool grows, and a limit no
    // tool should reach.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    static constexpr std::uint32_t kSwapchainLimit = 256;
    core::Pool<MetalSwapchain, Swapchain> swapchains_;
    static constexpr std::uint32_t kCommandPoolCapacity = 4;
    static constexpr std::uint32_t kCommandPoolLimit = 256;
    static constexpr std::uint32_t kCommandListCapacity = 8;
    static constexpr std::uint32_t kCommandListLimit = 4096;
    core::Pool<MetalCommandPool, CommandPool> commandPools_;
    core::Pool<MetalCommandList, CommandList> commandLists_;
};

} // namespace bfr::rhi::metal
