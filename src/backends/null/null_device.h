#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "core/slot_chains.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
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
    // The serial of the submit this list is gathered into, so a list named
    // twice in one submit is caught in a single pass.
    std::uint64_t queued = 0;
};

/// A buffer the GPU never sees: where it lives, and the bytes the CPU maps
/// when it is upload or readback memory.
struct NullBuffer {
    BufferMemory memory = BufferMemory::Device;
    std::vector<std::byte> bytes;
};

/// A texture, which holds no texels: no format has a use without a GPU.
struct NullTexture {
    TextureDesc desc;
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
    [[nodiscard]] core::Result<void> copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
                                                std::span<const BufferCopy> regions) override;
    [[nodiscard]] core::Result<void> barrier(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> textureBarrier(CommandListHandle commandList, TextureHandle texture,
                                                    TextureLayout before, TextureLayout after) override;
    [[nodiscard]] core::Result<void> copyBufferToTexture(CommandListHandle commandList, BufferHandle buffer,
                                                         TextureHandle texture,
                                                         std::span<const BufferTextureCopy> regions) override;
    [[nodiscard]] core::Result<void> copyTextureToBuffer(CommandListHandle commandList, TextureHandle texture,
                                                         BufferHandle buffer,
                                                         std::span<const BufferTextureCopy> regions) override;
    [[nodiscard]] core::Result<void> copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
                                                 std::span<const TextureCopy> regions) override;
    [[nodiscard]] core::Result<Token> submit(const SubmitDesc& desc) override;
    [[nodiscard]] core::Result<void> wait(Token token) override;
    [[nodiscard]] core::Result<bool> poll(Token token) override;
    [[nodiscard]] core::Result<void> destroyCommandList(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<BufferHandle> createBuffer(const BufferDesc& desc) override;
    [[nodiscard]] core::Result<std::span<std::byte>> mapBuffer(BufferHandle buffer) override;
    [[nodiscard]] core::Result<void> destroyBuffer(BufferHandle buffer) override;
    [[nodiscard]] core::Result<TextureHandle> createTexture(const TextureDesc& desc) override;
    [[nodiscard]] core::Result<void> destroyTexture(TextureHandle texture) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

    /// The queue refuses the next submit, as a driver's may: the lists named
    /// are spent, no token is made, and the device goes on. For the tests,
    /// which no real driver fails on demand.
    void refuseNextSubmit() { refuseNextSubmit_.store(true, std::memory_order_release); }

private:
    [[nodiscard]] core::Result<NullSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<NullCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<NullCommandList*> commandListFor(CommandListHandle commandList);
    [[nodiscard]] core::Result<NullBuffer*> bufferFor(BufferHandle buffer);
    [[nodiscard]] core::Result<NullTexture*> textureFor(TextureHandle texture);
    // Fails with RhiInvalidHandle when the list or either texture is gone.
    [[nodiscard]] core::Result<void> checkNames(CommandListHandle commandList, TextureHandle first,
                                                TextureHandle second);
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
    // Buffers are made off the frame path, so their pool may grow.
    static constexpr std::uint32_t kBufferCapacity = 256;
    static constexpr std::uint32_t kBufferLimit = 1U << 20U;
    core::Pool<NullBuffer, Buffer> buffers_;
    core::Pool<NullTexture, Texture> textures_;
    std::vector<QueueType> queues_;
    // The value last submitted on each queue, made once; submits on one queue
    // may come from several threads.
    std::vector<std::atomic<std::uint64_t>> submitted_;
    // Counts submits on every queue, so each has a serial of its own, even one
    // that fails, which the token value a queue hands out does not give.
    std::atomic<std::uint64_t> submitSerial_{0};
    std::atomic<bool> refuseNextSubmit_{false};
};

} // namespace bfr::rhi::null
