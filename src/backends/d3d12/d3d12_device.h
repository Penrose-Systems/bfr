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
#include "d3d12/d3d12_buffer.h"
#include "d3d12/d3d12_pipeline.h"
#include "d3d12/d3d12_sampler.h"
#include "d3d12/d3d12_swapchain.h"
#include "d3d12/d3d12_texture.h"
#include "rhi/bindless_slots.h"
#include "rhi/command_check.h"

#include <d3d12.h>
#include <D3D12MemAlloc.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace bfr::rhi::d3d12 {

class D3D12Instance;

/// A command pool: the queue it records for, its chain of list slots, each
/// holding an allocator and a list made with the pool, the next slot
/// beginCommands hands out, how many lists it handed out still record, and
/// the fence value its last submit signals. Plain data: the device releases
/// the slots' objects once the GPU is done with them.
struct D3D12CommandPool {
    std::uint32_t queue = 0;
    std::uint32_t pushConstantUpdates = 0;
    std::uint32_t pushed = 0;
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint32_t next = core::SlotChains::kEnd;
    std::uint32_t recording = 0;
    std::uint64_t lastSubmit = 0;
};

/// A list slot: an allocator and a list for it, since D3D12 lets one list
/// record from an allocator at a time, and the list recording on it.
struct D3D12ListSlot {
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList7> list;
    CommandListHandle live;
};

/// A destroyed pool the GPU may still run lists from, and the fence value
/// that frees it.
struct D3D12RetiredPool {
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint64_t value = 0;
};

/// One of the device's queues: its ID3D12CommandQueue and the list type it
/// takes, the fence each submit signals the next value of with the event waits
/// use, the value last submitted, the arrays a submit is built in, and the
/// pools destroyed while the GPU may still run their lists. Used by one thread
/// at a time under lock, which present shares; submitted is read from any
/// thread.
///
/// Why the arrays: submit runs every frame, so it fills these in place rather
/// than building vectors. They are made with the queue, when the device is
/// created, sized from the device's maximums, so no submit touches the heap.
struct D3D12Queue {
    QueueType type = QueueType::Graphics;
    D3D12_COMMAND_LIST_TYPE listType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    // Waits share the event, so they take turns; lock stays free for submits.
    std::mutex waitLock;
    std::atomic<std::uint64_t> submitted{0};
    core::FixedVector<ID3D12CommandList*> lists;
    core::FixedVector<D3D12Swapchain*> chains;
    core::FixedVector<D3D12RetiredPool> retired;
    std::atomic<std::uint32_t> retiredCount{0};
    std::mutex lock;
};

/// Commands recorded on one list: the list, which its slot owns, the pool and
/// slot it came from, what it has bound, the swapchain the open rendering
/// draws to, if one, and what becomes of its image when rendering ends, the
/// attachments a rendering discards when it ends, and the chain of render
/// slots naming every swapchain rendered to, whose images the submit marks as
/// sent.
struct D3D12CommandList {
    ID3D12GraphicsCommandList7* list = nullptr;
    CommandPoolHandle pool;
    std::uint32_t slot = 0;
    RecordState state;
    std::optional<SwapchainHandle> presenting;
    StoreOp presentStoreOp = StoreOp::Store;
    std::array<ID3D12Resource*, kMaxColorTargets + 1> discards{};
    std::uint32_t discardCount = 0;
    std::uint32_t rendered = core::SlotChains::kEnd;
    // The serial of the submit this list is gathered into, so a list named
    // twice in one submit is caught in a single pass.
    std::uint64_t queued = 0;
};

class D3D12Device final : public Device {
public:
    /// A device on device, which rhi listed from adapter, with heapCount memory
    /// heaps, a graphics queue and the queues asked for. Fails with
    /// RhiDeviceFailed when D3D12 gives no queue, fence, root signature or
    /// descriptor heap.
    [[nodiscard]] static core::Result<std::unique_ptr<D3D12Device>> create(
        D3D12Instance& rhi, Microsoft::WRL::ComPtr<ID3D12Device> device, Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter,
        std::size_t heapCount, const DeviceDesc& desc);

    /// For create, with the number of queues create opens.
    D3D12Device(D3D12Instance& rhi, Microsoft::WRL::ComPtr<ID3D12Device> device,
                Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter, std::size_t heapCount, const DeviceDesc& desc);
    ~D3D12Device() override;
    D3D12Device(const D3D12Device&) = delete;
    D3D12Device& operator=(const D3D12Device&) = delete;
    D3D12Device(D3D12Device&&) = delete;
    D3D12Device& operator=(D3D12Device&&) = delete;

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
    [[nodiscard]] core::Result<SamplerHandle> createSampler(const SamplerDesc& desc) override;
    [[nodiscard]] core::Result<std::uint32_t> samplerSlot(SamplerHandle sampler) override;
    [[nodiscard]] core::Result<void> destroySampler(SamplerHandle sampler) override;
    [[nodiscard]] core::Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc) override;
    [[nodiscard]] core::Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc) override;
    [[nodiscard]] core::Result<void> destroyPipeline(PipelineHandle pipeline) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    [[nodiscard]] core::Result<void> open(const DeviceDesc& desc);
    [[nodiscard]] core::Result<void> waitFor(D3D12Queue& queue, std::uint64_t value);
    [[nodiscard]] core::Result<void> checkQueue(std::uint32_t queue) const;
    [[nodiscard]] core::Result<void> checkToken(Token token) const;
    // Releases the retired pools the GPU is done with and frees their slots.
    // Runs on the frame path: it releases API objects and moves indices,
    // nothing more.
    void collectPools(D3D12Queue& queue);
    // Releases the objects in a chain of list slots and gives the chain back.
    void releaseSlots(std::uint32_t first);
    // Ends a list that was submitted or discarded; its slot waits for the
    // pool's reset.
    void end(CommandListHandle commandList);
    // Destroys the buffers, textures, samplers and pipelines retired under submission
    // IDs every queue has passed.
    void collectBuffers();
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    void discard(ID3D12GraphicsCommandList7* list);
    [[nodiscard]] core::Result<D3D12Swapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<D3D12CommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<D3D12CommandList*> commandListFor(CommandListHandle commandList);
    [[nodiscard]] core::Result<D3D12Buffer*> bufferFor(BufferHandle buffer);
    [[nodiscard]] core::Result<D3D12Texture*> textureFor(TextureHandle texture);
    [[nodiscard]] core::Result<D3D12Pipeline*> pipelineFor(PipelineHandle pipeline);
    // The barrier barrier() records, ordering every resource on the list's queue.
    [[nodiscard]] core::Result<D3D12_GLOBAL_BARRIER> fullBarrierFor(const D3D12CommandList& list);

    D3D12Instance* instance_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    // Null when the runtime has no ID3D12Device11; samplers then go through
    // CreateSampler.
    Microsoft::WRL::ComPtr<ID3D12Device11> device11_;
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter_;
    std::size_t heapCount_;
    DeviceDesc desc_;
    std::vector<std::unique_ptr<D3D12Queue>> queues_;
    core::Pool<D3D12Swapchain, Swapchain> swapchains_;
    core::Pool<D3D12CommandPool, CommandPool> commandPools_;
    core::Pool<D3D12CommandList, CommandList> commandLists_;
    // A slot for each list the device holds, and one for each swapchain a
    // recording list renders to, so beginCommands and beginRendering only
    // take indices. Both are made here, at the device's maximum of lists.
    core::SlotChains listSlots_;
    std::vector<D3D12ListSlot> lists_;
    core::SlotChains renderSlots_;
    std::vector<SwapchainHandle> renders_;
    // Live pools and destroyed ones not yet collected, held to
    // desc_.maxCommandPools so every queue's retired array has room.
    std::atomic<std::uint32_t> poolsHeld_{0};
    // Counts submits on every queue, so each has a serial of its own, even one
    // that fails, which the token value a queue hands out does not give.
    std::atomic<std::uint64_t> submitSerial_{0};

    core::Submissions submissions_;
    Microsoft::WRL::ComPtr<D3D12MA::Allocator> allocator_;
    static constexpr std::uint32_t kBufferCapacity = 256;
    static constexpr std::uint32_t kBufferLimit = 1U << 20U;
    core::Pool<D3D12Buffer, Buffer> buffers_;
    core::Pool<D3D12Texture, Texture> textures_;
    // Every sampler holds a slot, so the pool is made at the table's size.
    core::Pool<D3D12Sampler, Sampler> samplers_;
    BindlessSlots bindless_;
    // Declared before pipelines_, which are made on it, so it outlives them.
    D3D12BindlessLayout layout_;
    core::Pool<D3D12Pipeline, Pipeline> pipelines_;
};

} // namespace bfr::rhi::d3d12
