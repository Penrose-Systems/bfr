#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "core/fixed_vector.h"
#include "core/slot_chains.h"
#include "core/submissions.h"
#include "metal/metal_buffer.h"
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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace bfr::rhi::metal {

class MetalInstance;

/// A command pool: the queue it records for, its chain of list slots, each
/// holding an allocator and a buffer made with the pool, the next slot
/// beginCommands hands out, how many lists it handed out still record, and
/// the event value its last submit signals. Plain data: the device releases
/// the slots' objects once the GPU is done with them.
struct MetalCommandPool {
    std::uint32_t queue = 0;
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint32_t next = core::SlotChains::kEnd;
    std::uint32_t recording = 0;
    std::uint64_t lastSubmit = 0;
};

/// A list slot: an allocator and a buffer for it, since a Metal 4 allocator
/// serves one buffer at a time, and the list recording on it.
struct MetalListSlot {
    Ref<MTL4::CommandAllocator> allocator;
    Ref<MTL4::CommandBuffer> buffer;
    // Made with the slot and filled in by each beginRendering, so a pass
    // allocates no descriptor.
    Ref<MTL4::RenderPassDescriptor> pass;
    CommandListHandle live;
};

/// A destroyed pool the GPU may still run lists from, and the event value
/// that frees it.
struct MetalRetiredPool {
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint64_t value = 0;
};

/// One of the device's queues: its MTL4 command queue, the shared event each
/// submit signals the next value of, the value last submitted, the arrays a
/// submit is built in, and the pools destroyed while the GPU may still run
/// their lists. Used by one thread at a time under lock, which present
/// shares; submitted is read from any thread.
///
/// Why the arrays: submit runs every frame, so it fills these in place rather
/// than building vectors. They are made with the queue, when the device is
/// created, sized from the device's maximums, so no submit touches the heap.
/// The buffers they name are owned by their slots, which outlive the GPU's
/// use of them.
struct MetalQueue {
    QueueType type = QueueType::Graphics;
    Ref<MTL4::CommandQueue> queue;
    Ref<MTL::SharedEvent> event;
    std::atomic<std::uint64_t> submitted{0};
    core::FixedVector<const MTL4::CommandBuffer*> buffers;
    core::FixedVector<MetalSwapchain*> chains;
    core::FixedVector<MetalRetiredPool> retired;
    std::atomic<std::uint32_t> retiredCount{0};
    std::mutex lock;
};

/// Commands recorded on one buffer: the buffer, which its slot owns, the pool
/// and slot it came from, the encoder rendering is open on, the swapchain
/// rendering is open on, and the chain of render slots naming every swapchain
/// rendered to, whose drawables the submit waits for and signals.
struct MetalCommandList {
    MTL4::CommandBuffer* buffer = nullptr;
    CommandPoolHandle pool;
    std::uint32_t slot = 0;
    Ref<MTL4::RenderCommandEncoder> encoder;
    std::optional<SwapchainHandle> rendering;
    std::uint32_t rendered = core::SlotChains::kEnd;
    // The serial of the submit this list is gathered into, so a list named
    // twice in one submit is caught in a single pass.
    std::uint64_t queued = 0;
};

class MetalDevice final : public Device {
public:
    /// A device on device, which rhi listed, with a graphics queue and the
    /// queues asked for. Fails with RhiDeviceFailed when Metal gives no queue
    /// or event.
    [[nodiscard]] static core::Result<std::unique_ptr<MetalDevice>> create(MetalInstance& rhi, Ref<MTL::Device> device,
                                                                           const DeviceDesc& desc);

    /// For create.
    MetalDevice(MetalInstance& rhi, Ref<MTL::Device> device, const DeviceDesc& desc);
    ~MetalDevice() override;
    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;
    MetalDevice(MetalDevice&&) = delete;
    MetalDevice& operator=(MetalDevice&&) = delete;

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
    [[nodiscard]] core::Result<BufferHandle> createBuffer(const BufferDesc& desc) override;
    [[nodiscard]] core::Result<std::span<std::byte>> mapBuffer(BufferHandle buffer) override;
    [[nodiscard]] core::Result<void> destroyBuffer(BufferHandle buffer) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    [[nodiscard]] core::Result<void> open(const DeviceDesc& desc);
    [[nodiscard]] core::Result<void> checkQueue(std::uint32_t queue) const;
    [[nodiscard]] core::Result<void> checkToken(Token token) const;
    // Releases the retired pools the GPU is done with and frees their slots.
    // Runs on the frame path: it releases API objects and moves indices,
    // nothing more.
    void collectPools(MetalQueue& queue);
    // Releases the objects in a chain of list slots and gives the chain back.
    void releaseSlots(std::uint32_t first);
    // Ends a list that was submitted or discarded; its slot waits for the
    // pool's reset.
    void end(CommandListHandle commandList);
    // Destroys the buffers retired under submission IDs every queue has passed.
    void collectBuffers();
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    static void discard(MetalCommandList& list);
    [[nodiscard]] core::Result<MetalSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<MetalCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<MetalCommandList*> commandListFor(CommandListHandle commandList);
    [[nodiscard]] core::Result<MetalBuffer*> bufferFor(BufferHandle buffer);

    MetalInstance* instance_;
    Ref<MTL::Device> device_;
    DeviceDesc desc_;
    std::vector<std::unique_ptr<MetalQueue>> queues_;
    core::Pool<MetalSwapchain, Swapchain> swapchains_;
    core::Pool<MetalCommandPool, CommandPool> commandPools_;
    core::Pool<MetalCommandList, CommandList> commandLists_;
    // A slot for each list the device holds, and one for each swapchain a
    // recording list renders to, so beginCommands and beginRendering only
    // take indices. Both are made here, at the device's maximum of lists.
    core::SlotChains listSlots_;
    std::vector<MetalListSlot> lists_;
    core::SlotChains renderSlots_;
    std::vector<SwapchainHandle> renders_;
    // Live pools and destroyed ones not yet collected, held to
    // desc_.maxCommandPools so every queue's retired array has room.
    std::atomic<std::uint32_t> poolsHeld_{0};
    // Counts submits on every queue, so each has a serial of its own, even one
    // that fails, which the token value a queue hands out does not give.
    std::atomic<std::uint64_t> submitSerial_{0};

    core::Submissions submissions_;
    static constexpr std::uint32_t kBufferCapacity = 256;
    static constexpr std::uint32_t kBufferLimit = 1U << 20U;
    core::Pool<MetalBuffer, Buffer> buffers_;
};

} // namespace bfr::rhi::metal
