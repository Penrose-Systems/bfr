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
#include "metal/metal_pipeline.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_texture.h"
#include "metal/metal_util.h"
#include "rhi/bindless_slots.h"
#include "rhi/command_check.h"

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
    std::uint32_t pushConstantUpdates = 0;
    /// pushConstants calls since the pool's reset: the next entry of the ring
    /// its first slot holds.
    std::uint32_t pushed = 0;
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint32_t next = core::SlotChains::kEnd;
    std::uint32_t recording = 0;
    std::uint64_t lastSubmit = 0;
};

/// A list slot: an allocator and a buffer for it, since a Metal 4 allocator
/// serves one buffer at a time, the argument table its encoders bind the
/// tables and push constants through, and the list recording on it. The
/// pool's first slot also holds the pool's push constant ring: Metal 4 has no
/// setBytes, so each pushConstants copies into the ring and binds the copy.
/// All made with the pool and released with its slots, once the GPU is done.
struct MetalListSlot {
    Ref<MTL4::CommandAllocator> allocator;
    Ref<MTL4::CommandBuffer> buffer;
    // Made with the slot and filled in by each beginRendering, so a pass
    // allocates no descriptor.
    Ref<MTL4::RenderPassDescriptor> pass;
    Ref<MTL4::ArgumentTable> arguments;
    Ref<MTL::Buffer> pushRing;
    std::optional<Resident> pushResident;
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
/// and slot it came from, what it has bound, the encoder rendering is open on,
/// the swapchain the open rendering draws to, if one, and the chain of render
/// slots naming every swapchain rendered to, whose drawables the submit waits
/// for and signals. Metal binds pipelines on the encoder, so the pipelines
/// bound are kept by handle and set on the encoder at the draw or dispatch.
struct MetalCommandList {
    MTL4::CommandBuffer* buffer = nullptr;
    CommandPoolHandle pool;
    std::uint32_t slot = 0;
    RecordState state;
    PipelineHandle graphics;
    PipelineHandle compute;
    // The graphics pipeline the open encoder has been given, so a draw sets
    // the pipeline's state once.
    PipelineHandle encoderGraphics;
    MTL::IndexType indexType = MTL::IndexTypeUInt32;
    MTL::GPUAddress indexBuffer = 0;
    NS::UInteger indexBytes = 0;
    Ref<MTL4::RenderCommandEncoder> encoder;
    std::optional<SwapchainHandle> presenting;
    std::uint32_t rendered = core::SlotChains::kEnd;
    // The serial of the submit this list is gathered into, so a list named
    // twice in one submit is caught in a single pass.
    std::uint64_t queued = 0;
};

class MetalDevice final : public Device {
public:
    /// A device on device, which rhi listed, with a graphics queue and the
    /// queues asked for. Fails with RhiDeviceFailed when Metal gives no queue,
    /// event, compiler or bindless tables.
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
    [[nodiscard]] core::Result<void> setViewport(CommandListHandle commandList, const Viewport& viewport) override;
    [[nodiscard]] core::Result<void> setScissor(CommandListHandle commandList, const Rect& scissor) override;
    [[nodiscard]] core::Result<void> bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) override;
    [[nodiscard]] core::Result<void> pushConstants(CommandListHandle commandList,
                                                   std::span<const std::byte> data) override;
    [[nodiscard]] core::Result<void> bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
                                                     std::uint64_t offset, IndexType type) override;
    [[nodiscard]] core::Result<void> draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                          std::uint32_t instanceCount, std::uint32_t firstVertex,
                                          std::uint32_t firstInstance) override;
    [[nodiscard]] core::Result<void> drawIndexed(CommandListHandle commandList, std::uint32_t indexCount,
                                                 std::uint32_t instanceCount, std::uint32_t firstIndex,
                                                 std::int32_t vertexOffset, std::uint32_t firstInstance) override;
    [[nodiscard]] core::Result<void> dispatch(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                              std::uint32_t z) override;
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
    [[nodiscard]] core::Result<std::uint32_t> bufferSlot(BufferHandle buffer) override;
    [[nodiscard]] core::Result<void> destroyBuffer(BufferHandle buffer) override;
    [[nodiscard]] core::Result<TextureHandle> createTexture(const TextureDesc& desc) override;
    [[nodiscard]] core::Result<TextureSlots> textureSlots(TextureHandle texture) override;
    [[nodiscard]] core::Result<void> destroyTexture(TextureHandle texture) override;
    [[nodiscard]] core::Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc) override;
    [[nodiscard]] core::Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc) override;
    [[nodiscard]] core::Result<void> destroyPipeline(PipelineHandle pipeline) override;
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
    // Destroys the buffers, textures and pipelines retired under submission
    // IDs every queue has passed.
    void collectBuffers();
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    static void discard(MetalCommandList& list);
    [[nodiscard]] core::Result<MetalSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<MetalCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<MetalCommandList*> commandListFor(CommandListHandle commandList);
    [[nodiscard]] core::Result<MetalBuffer*> bufferFor(BufferHandle buffer);
    [[nodiscard]] core::Result<MetalTexture*> textureFor(TextureHandle texture);
    // The bound graphics pipeline, its state set on the open encoder once per
    // pipeline per rendering. Fails with RhiInvalidHandle when it is gone.
    [[nodiscard]] core::Result<const MetalPipeline*> applyGraphics(MetalCommandList& list);
    [[nodiscard]] core::Result<MetalPipeline*> pipelineFor(PipelineHandle pipeline);

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
    // Every buffer and texture, attached to every queue, so the GPU may use
    // any of them.
    // Declared before buffers_ and textures_, which take themselves out of it
    // as they go, under residencyLock_: Metal's residency sets are not
    // thread-safe.
    Ref<MTL::ResidencySet> residency_;
    std::mutex residencyLock_;
    static constexpr std::uint32_t kBufferCapacity = 256;
    static constexpr std::uint32_t kBufferLimit = 1U << 20U;
    core::Pool<MetalBuffer, Buffer> buffers_;
    core::Pool<MetalTexture, Texture> textures_;
    BindlessSlots bindless_;
    // The tables leave the residency set as the buffers do, under its lock.
    // Optional only because they are made after the device, in open.
    // Made in open, before any call reaches a device, so always there.
    std::optional<MetalBindlessTables> tables_;
    [[nodiscard]] const MetalBindlessTables& tables() const {
        return tables_.value(); // NOLINT(bugprone-unchecked-optional-access)
    }
    // Compiles every pipeline; Metal 4 makes pipelines through one.
    Ref<MTL4::Compiler> compiler_;
    core::Pool<MetalPipeline, Pipeline> pipelines_;
};

} // namespace bfr::rhi::metal
