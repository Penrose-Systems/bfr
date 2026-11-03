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
#include "vulkan/queue_families.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_buffer.h"
#include "vulkan/vulkan_pipeline.h"
#include "vulkan/vulkan_swapchain.h"
#include "vulkan/vulkan_texture.h"

#include <volk.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace bfr::rhi::vulkan {

class VulkanInstance;

/// A command pool: its VkCommandPool, the queue it records for, its chain of
/// list slots, each holding a buffer made with the pool, the next slot
/// beginCommands hands out, how many lists it handed out still record, and
/// the timeline value its last submit signals. Plain data: the device
/// destroys the VkCommandPool, once the GPU is done with it.
struct VulkanCommandPool {
    VkCommandPool pool = VK_NULL_HANDLE;
    std::uint32_t queue = 0;
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint32_t next = core::SlotChains::kEnd;
    std::uint32_t recording = 0;
    std::uint64_t lastSubmit = 0;
};

/// A list slot: the buffer its pool made in it, and the list recording on it.
struct VulkanListSlot {
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    CommandListHandle live;
};

/// A destroyed pool the GPU may still run lists from, and the timeline value
/// that frees it.
struct VulkanRetiredPool {
    VkCommandPool pool = VK_NULL_HANDLE;
    std::uint32_t first = core::SlotChains::kEnd;
    std::uint64_t value = 0;
};

/// One of the device's queues: its VkQueue and family, the timeline each
/// submit signals the next value of, the value last submitted, the arrays a
/// submit is built in, and the pools destroyed while the GPU may still run
/// their lists. Used by one thread at a time under lock, which present
/// shares; submitted is read from any thread.
///
/// Why the arrays: submit runs every frame, so it fills these in place rather
/// than building vectors. They are made with the queue, when the device is
/// created, sized from the device's maximums, so no submit touches the heap.
struct VulkanQueue {
    QueueType type = QueueType::Graphics;
    std::uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<std::uint64_t> submitted{0};
    core::FixedVector<VkCommandBufferSubmitInfo> buffers;
    core::FixedVector<VkSemaphoreSubmitInfo> waits;
    core::FixedVector<VkSemaphoreSubmitInfo> signals;
    core::FixedVector<VulkanSwapchain*> chains;
    // The largest value a submit waits for on each queue, by queue index.
    std::vector<std::uint64_t> waitValues;
    core::FixedVector<VulkanRetiredPool> retired;
    std::atomic<std::uint32_t> retiredCount{0};
    std::mutex lock;
};

/// Commands recorded on one buffer: the pool and slot it came from, the
/// swapchain rendering is open on, and the chain of render slots naming every
/// swapchain rendered to, whose images the submit waits for and signals.
struct VulkanCommandList {
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    CommandPoolHandle pool;
    std::uint32_t slot = 0;
    std::optional<SwapchainHandle> rendering;
    std::uint32_t rendered = core::SlotChains::kEnd;
    // The serial of the submit this list is gathered into, so a list named
    // twice in one submit is caught in a single pass.
    std::uint64_t queued = 0;
};

class VulkanDevice final : public Device {
public:
    /// A device over device, which rhi made on adapter with one graphics queue
    /// on families.graphics, queues.compute on families.compute and queues.copy
    /// on families.transfer, and with features turned on; rhi outlives it.
    /// hasBudget says whether adapter has VK_EXT_memory_budget; desc's
    /// maximums size everything the device holds. Fails with RhiDeviceFailed,
    /// destroying device, when the API refuses a queue's timeline or the
    /// bindless layout, and RhiLimitReached when it cannot hold the layout.
    [[nodiscard]] static core::Result<std::unique_ptr<VulkanDevice>> create(VulkanInstance& rhi,
                                                                            VkPhysicalDevice adapter, VkDevice device,
                                                                            const QueueFamilies& families,
                                                                            const DeviceDesc& desc, bool hasBudget,
                                                                            const RasterFeatures& features);

    /// For create, with the number of queues create opens.
    VulkanDevice(VulkanInstance& rhi, VkPhysicalDevice adapter, VkDevice device, const QueueFamilies& families,
                 const DeviceDesc& desc, bool hasBudget, const RasterFeatures& features);
    ~VulkanDevice() override;
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) = delete;
    VulkanDevice& operator=(VulkanDevice&&) = delete;

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
    [[nodiscard]] core::Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc) override;
    [[nodiscard]] core::Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc) override;
    [[nodiscard]] core::Result<void> destroyPipeline(PipelineHandle pipeline) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    [[nodiscard]] core::Result<void> openQueues(const QueueFamilies& families, const DeviceDesc& desc);
    [[nodiscard]] core::Result<std::uint64_t> completedValue(const VulkanQueue& queue) const;
    [[nodiscard]] core::Result<void> checkQueue(std::uint32_t queue) const;
    [[nodiscard]] core::Result<void> checkToken(Token token) const;
    // Destroys the retired pools the GPU is done with and frees their slots.
    // Runs on the frame path: it calls the API and moves indices, nothing more.
    void collectPools(VulkanQueue& queue);
    // Ends a list that was submitted or discarded; its slot waits for the
    // pool's reset.
    void end(CommandListHandle commandList);
    // Destroys the buffers, textures and pipelines retired under submission
    // IDs every queue has passed.
    void collectBuffers();
    // Warns of each kind of object the caller left when the device goes.
    void warnLeftovers() const;
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    [[nodiscard]] core::Result<VulkanSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<VulkanCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<VulkanCommandList*> commandListFor(CommandListHandle commandList);
    [[nodiscard]] core::Result<VulkanBuffer*> bufferFor(BufferHandle buffer);
    [[nodiscard]] core::Result<VulkanTexture*> textureFor(TextureHandle texture);
    [[nodiscard]] core::Result<VulkanPipeline*> pipelineFor(PipelineHandle pipeline);

    VulkanInstance* instance_;
    VkPhysicalDevice adapter_;
    VkDevice device_;
    // Each device calls through its own table: volk's global pointers hold one
    // device's functions, and two adapters can have different drivers.
    VolkDeviceTable table_{};
    std::uint32_t graphicsFamily_;
    std::uint32_t presentFamily_;
    bool hasBudget_;
    RasterFeatures rasterFeatures_;
    DeviceDesc desc_;
    std::vector<std::unique_ptr<VulkanQueue>> queues_;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VulkanQueue* presentOn_ = nullptr;
    core::Pool<VulkanSwapchain, Swapchain> swapchains_;
    core::Pool<VulkanCommandPool, CommandPool> commandPools_;
    core::Pool<VulkanCommandList, CommandList> commandLists_;
    // A slot for each list the device holds, and one for each swapchain a
    // recording list renders to, so beginCommands and beginRendering only
    // take indices. Both are made here, at the device's maximum of lists.
    core::SlotChains listSlots_;
    std::vector<VulkanListSlot> lists_;
    core::SlotChains renderSlots_;
    std::vector<SwapchainHandle> renders_;
    // Live pools and destroyed ones not yet collected, held to
    // desc_.maxCommandPools so every queue's retired array has room.
    std::atomic<std::uint32_t> poolsHeld_{0};
    // Counts submits on every queue, so each has a serial of its own, even one
    // that fails, which the token value a queue hands out does not give.
    std::atomic<std::uint64_t> submitSerial_{0};

    // Every queue family the device's queues are on, which each buffer is shared by.
    std::vector<std::uint32_t> families_;
    core::Submissions submissions_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    static constexpr std::uint32_t kBufferCapacity = 256;
    static constexpr std::uint32_t kBufferLimit = 1U << 20U;
    core::Pool<VulkanBuffer, Buffer> buffers_;
    core::Pool<VulkanTexture, Texture> textures_;
    // Declared before pipelines_, which are made on it, so it outlives them.
    VulkanBindlessLayout layout_;
    core::Pool<VulkanPipeline, Pipeline> pipelines_;
};

} // namespace bfr::rhi::vulkan
