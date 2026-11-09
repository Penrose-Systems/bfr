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
#include "rhi/bindless_slots.h"
#include "rhi/command_check.h"
#include "vulkan/queue_families.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_acceleration_structure.h"
#include "vulkan/vulkan_buffer.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_pipeline.h"
#include "vulkan/vulkan_pipeline_cache.h"
#include "vulkan/vulkan_sampler.h"
#include "vulkan/vulkan_sparse.h"
#include "vulkan/vulkan_swapchain.h"
#include "vulkan/vulkan_texture.h"
#include "vulkan/vulkan_timestamp.h"

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
/// What the instance read of the adapter and enabled on the device, which
/// the device's calls need.
struct VulkanDeviceSetup {
    /// The version the device uses: the lower of the instance's and the
    /// adapter's.
    std::uint32_t apiVersion = 0;
    /// The formats added by 1.3 the device has.
    Formats13 formats13{};
    bool hasBudget = false;
    bool hasSwapchain = false;
    std::uint32_t pushConstantsSize = 0;
    /// VkPhysicalDeviceLimits::timestampPeriod.
    float timestampPeriod = 0;
    /// The stages each kind of queue's timestamps take, and what
    /// vkGetCalibratedTimestampsKHR reads, none when the device has no
    /// calibrated timestamps extension enabled.
    Timestamps timestamps;
    /// Features::multiDrawIndirect and drawIndirectCount.
    bool multiDrawIndirect = false;
    bool drawIndirectCount = false;
    /// The features createGraphicsPipeline checks a desc against, and how
    /// depth-stencil attachments resolve.
    Features features{};
    ResolveCaps resolve{};
    /// The limits builds and instance records keep to.
    RayTracingLimits rayTracing{};
    /// The device enabled bufferDeviceAddress, so its buffers have addresses.
    bool deviceAddresses = false;
    /// The device has an acceleration structure table: the adapter's
    /// BindlessLimits::maxAccelerationStructures is not 0.
    bool accelerationStructureTable = false;
};

struct VulkanCommandPool {
    VkCommandPool pool = VK_NULL_HANDLE;
    std::uint32_t queue = 0;
    std::uint32_t pushConstantUpdates = 0;
    std::uint32_t pushed = 0;
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
    /// The family reports VK_QUEUE_SPARSE_BINDING_BIT.
    bool sparse = false;
    /// The family reports VK_QUEUE_COMPUTE_BIT.
    bool compute = false;
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
    // A tile mapping's binds and waits, made as a submit's arrays are, at the
    // device's maximums of regions and queues.
    core::FixedVector<VkSparseImageMemoryBind> imageBinds;
    core::FixedVector<VkSparseMemoryBind> tailBinds;
    core::FixedVector<VkSparseMemoryBind> bufferBinds;
    core::FixedVector<VkSemaphore> waitSemaphores;
    core::FixedVector<std::uint64_t> waitSemaphoreValues;
    std::mutex lock;
};

/// Commands recorded on one buffer: the pool and slot it came from, what it
/// has bound, the swapchain the open rendering draws to, if one, and the
/// chain of render slots naming every swapchain rendered to, whose images the
/// submit waits for and signals.
struct VulkanCommandList {
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    CommandPoolHandle pool;
    std::uint32_t slot = 0;
    std::optional<SwapchainHandle> presenting;
    std::uint32_t rendered = core::SlotChains::kEnd;
};

class VulkanDevice final : public Device {
public:
    /// A device over device, which rhi made on adapter with desc.graphics
    /// queues on families.graphics, desc.compute on families.compute and
    /// desc.copy on families.transfer, each family set where desc asks for its
    /// queues; rhi outlives it.
    /// hasBudget says whether adapter has VK_EXT_memory_budget; hasSwapchain
    /// whether device enabled VK_KHR_swapchain;
    /// pushConstantsSize is the adapter's maxPushConstantsSize; desc's
    /// maximums size everything the device holds. Fails with RhiDeviceFailed,
    /// destroying device, when the API refuses a queue's timeline or the
    /// bindless layout, and RhiLimitReached when it cannot hold the layout.
    [[nodiscard]] static core::Result<std::unique_ptr<VulkanDevice>> create(VulkanInstance& rhi,
                                                                            VkPhysicalDevice adapter, VkDevice device,
                                                                            const QueueFamilies& families,
                                                                            const DeviceDesc& desc,
                                                                            const VulkanDeviceSetup& setup);

    /// For create, with the number of queues create opens.
    VulkanDevice(VulkanInstance& rhi, VkPhysicalDevice adapter, VkDevice device, const DeviceDesc& desc,
                 const VulkanDeviceSetup& setup);
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
    [[nodiscard]] core::Result<SwapchainStatus> acquire(SwapchainHandle swapchain, std::uint64_t timeoutNs) override;
    [[nodiscard]] core::Result<SwapchainStatus> present(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc) override;
    [[nodiscard]] core::Result<void> resetCommandPool(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<void> destroyCommandPool(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<CommandListHandle> beginCommands(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                    const RenderingDesc& desc) override;
    [[nodiscard]] core::Result<void> endRendering(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> setViewport(CommandListHandle commandList, const Viewport& viewport) override;
    [[nodiscard]] core::Result<void> setScissor(CommandListHandle commandList, const Rect& scissor) override;
    [[nodiscard]] core::Result<void> setBlendConstants(CommandListHandle commandList,
                                                       const std::array<float, 4>& color) override;
    [[nodiscard]] core::Result<void> setStencilReference(CommandListHandle commandList,
                                                         std::uint32_t reference) override;
    [[nodiscard]] core::Result<void> bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) override;
    [[nodiscard]] core::Result<void> setTopology(CommandListHandle commandList, Topology topology) override;
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
    [[nodiscard]] core::Result<void> drawIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                  std::uint64_t offset, std::uint32_t drawCount,
                                                  std::uint32_t stride) override;
    [[nodiscard]] core::Result<void> drawIndexedIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                         std::uint64_t offset, std::uint32_t drawCount,
                                                         std::uint32_t stride) override;
    [[nodiscard]] core::Result<void> drawIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                       std::uint64_t offset, BufferHandle countBuffer,
                                                       std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                       std::uint32_t stride) override;
    [[nodiscard]] core::Result<void> drawIndexedIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                              std::uint64_t offset, BufferHandle countBuffer,
                                                              std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                              std::uint32_t stride) override;
    [[nodiscard]] core::Result<void> dispatchIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                      std::uint64_t offset) override;
    [[nodiscard]] core::Result<void> drawMesh(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                              std::uint32_t z) override;
    [[nodiscard]] core::Result<void> drawMeshIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                      std::uint64_t offset, std::uint32_t drawCount,
                                                      std::uint32_t stride) override;
    [[nodiscard]] core::Result<void> drawMeshIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                           std::uint64_t offset, BufferHandle countBuffer,
                                                           std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                           std::uint32_t stride) override;
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
    [[nodiscard]] core::Result<void> resolveTexture(CommandListHandle commandList, TextureHandle src,
                                                    std::uint32_t srcMipLevel, std::uint32_t srcLayer,
                                                    TextureHandle dst, std::uint32_t dstMipLevel,
                                                    std::uint32_t dstLayer) override;
    [[nodiscard]] core::Result<void> copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
                                                 std::span<const TextureCopy> regions) override;
    [[nodiscard]] core::Result<Token> submit(const SubmitDesc& desc) override;
    [[nodiscard]] core::Result<void> wait(Token token, std::uint64_t timeoutNs) override;
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
    [[nodiscard]] core::Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc& desc) override;
    [[nodiscard]] core::Result<std::size_t> pipelineCacheSize(PipelineCacheHandle cache) override;
    [[nodiscard]] core::Result<PipelineCacheWrite> pipelineCacheData(PipelineCacheHandle cache,
                                                                     std::span<std::byte> data) override;
    [[nodiscard]] core::Result<void> destroyPipelineCache(PipelineCacheHandle cache) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;
    [[nodiscard]] core::Result<TimestampPoolHandle> createTimestampPool(const TimestampPoolDesc& desc) override;
    [[nodiscard]] core::Result<void> destroyTimestampPool(TimestampPoolHandle pool) override;
    [[nodiscard]] core::Result<void> resetTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                     std::uint32_t first, std::uint32_t count) override;
    [[nodiscard]] core::Result<void> writeTimestamp(CommandListHandle commandList, TimestampPoolHandle pool,
                                                    std::uint32_t index, TimestampStage stage,
                                                    TimestampGranularity granularity) override;
    [[nodiscard]] core::Result<void> resolveTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                       std::uint32_t first, std::uint32_t count, BufferHandle buffer,
                                                       std::uint64_t offset) override;
    [[nodiscard]] core::Result<TimestampRate> timestampRate(std::uint32_t queue) const override;
    [[nodiscard]] core::Result<CalibratedTimestamps> calibrateTimestamps(std::uint32_t queue,
                                                                         HostClock clock) const override;
    [[nodiscard]] core::Result<SparseMemoryHandle> createSparseMemory(const SparseMemoryDesc& desc) override;
    [[nodiscard]] core::Result<void> destroySparseMemory(SparseMemoryHandle memory) override;
    [[nodiscard]] core::Result<TextureTiling> textureTiling(TextureHandle texture, TextureAspect aspect) override;
    [[nodiscard]] core::Result<BufferTiling> bufferTiling(BufferHandle buffer) override;
    [[nodiscard]] core::Result<SparseFormat> sparseFormat(const SparseFormatDesc& desc) const override;
    [[nodiscard]] core::Result<Token> updateTileMappings(const TileMappingDesc& desc) override;
    [[nodiscard]] core::Result<AccelerationStructureSizes> accelerationStructureSizes(
        const AccelerationStructureInputs& inputs) override;
    [[nodiscard]] core::Result<AccelerationStructureHandle> createAccelerationStructure(
        const AccelerationStructureDesc& desc) override;
    [[nodiscard]] core::Result<std::uint32_t> accelerationStructureSlot(AccelerationStructureHandle structure) override;
    [[nodiscard]] core::Result<std::uint64_t> accelerationStructureReference(
        AccelerationStructureHandle structure) override;
    [[nodiscard]] core::Result<void> destroyAccelerationStructure(AccelerationStructureHandle structure) override;
    [[nodiscard]] core::Result<void> writeInstances(std::span<const AccelerationStructureInstance> instances,
                                                    std::span<std::byte> out) override;
    [[nodiscard]] core::Result<void> buildAccelerationStructure(CommandListHandle commandList,
                                                                const AccelerationStructureBuild& build) override;
    [[nodiscard]] core::Result<void> buildAccelerationStructureIndirect(CommandListHandle commandList,
                                                                        const AccelerationStructureBuild& build,
                                                                        BufferHandle ranges, std::uint64_t offset,
                                                                        std::uint32_t stride) override;
    [[nodiscard]] core::Result<void> copyAccelerationStructure(CommandListHandle commandList,
                                                               AccelerationStructureHandle dst,
                                                               AccelerationStructureHandle src) override;
    [[nodiscard]] core::Result<AccelerationStructurePropertyPoolHandle> createAccelerationStructurePropertyPool(
        const AccelerationStructurePropertyPoolDesc& desc) override;
    [[nodiscard]] core::Result<void> destroyAccelerationStructurePropertyPool(
        AccelerationStructurePropertyPoolHandle pool) override;
    [[nodiscard]] core::Result<void> resetAccelerationStructureProperties(CommandListHandle commandList,
                                                                          AccelerationStructurePropertyPoolHandle pool,
                                                                          std::uint32_t first,
                                                                          std::uint32_t count) override;
    [[nodiscard]] core::Result<void> writeAccelerationStructureProperties(
        CommandListHandle commandList, AccelerationStructurePropertyPoolHandle pool, std::uint32_t first,
        std::span<const AccelerationStructureHandle> structures) override;
    [[nodiscard]] core::Result<void> resolveAccelerationStructureProperties(
        CommandListHandle commandList, AccelerationStructurePropertyPoolHandle pool, std::uint32_t first,
        std::uint32_t count, BufferHandle buffer, std::uint64_t offset) override;
    [[nodiscard]] core::Result<void> compactAccelerationStructure(CommandListHandle commandList,
                                                                  AccelerationStructureHandle dst,
                                                                  AccelerationStructureHandle src) override;
    [[nodiscard]] core::Result<void> serializeAccelerationStructure(CommandListHandle commandList,
                                                                    AccelerationStructureHandle src,
                                                                    BufferHandle buffer, std::uint64_t offset) override;
    [[nodiscard]] core::Result<void> deserializeAccelerationStructure(CommandListHandle commandList,
                                                                      BufferHandle buffer, std::uint64_t offset,
                                                                      AccelerationStructureHandle dst) override;
    [[nodiscard]] core::Result<bool> serializedCompatible(
        AccelerationStructureType type,
        std::span<const std::byte, kSerializedIdentifierSize> identifier) const override;
    [[nodiscard]] core::Result<std::uint64_t> rayTracingStackSize(PipelineHandle pipeline, std::uint32_t group,
                                                                  RayTracingGroupShader shader) override;
    [[nodiscard]] core::Result<ShaderHandle> createShader(const ShaderCode& code) override;
    [[nodiscard]] core::Result<void> destroyShader(ShaderHandle shader) override;
    [[nodiscard]] core::Result<PipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc& desc) override;
    [[nodiscard]] core::Result<void> shaderIdentifiers(PipelineHandle pipeline, std::uint32_t first,
                                                       std::uint32_t count, std::span<std::byte> out) override;
    [[nodiscard]] core::Result<void> captureReplayIdentifiers(PipelineHandle pipeline, std::uint32_t first,
                                                              std::uint32_t count, std::span<std::byte> out) override;
    [[nodiscard]] core::Result<void> traceRays(CommandListHandle commandList, const TraceRaysDesc& desc) override;
    [[nodiscard]] core::Result<void> traceRaysIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                       std::uint64_t offset) override;
    [[nodiscard]] core::Result<std::uint64_t> bufferAddress(BufferHandle buffer) override;

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
    // Destroys the buffers, textures, samplers and pipelines retired under submission
    // IDs every queue has passed.
    void collectBuffers();
    // Warns of each kind of object the caller left when the device goes.
    void warnLeftovers() const;
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    [[nodiscard]] core::Result<VulkanSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<VulkanCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<VulkanCommandList*> commandListFor(CommandListHandle commandList);
    // The kind of queue list records for. Fails with RhiInvalidHandle when its
    // pool is gone.
    [[nodiscard]] core::Result<QueueType> queueTypeOf(const VulkanCommandList& list);
    [[nodiscard]] core::Result<VulkanBuffer*> bufferFor(BufferHandle buffer);
    [[nodiscard]] core::Result<VulkanTexture*> textureFor(TextureHandle texture);
    [[nodiscard]] core::Result<VulkanPipeline*> pipelineFor(PipelineHandle pipeline);
    [[nodiscard]] core::Result<VulkanSparseMemory*> sparseMemoryFor(SparseMemoryHandle memory);
    [[nodiscard]] core::Result<VulkanAccelerationStructure*> accelerationStructureFor(
        AccelerationStructureHandle structure);
    // The list and pool of a property call, on a queue that runs compute.
    [[nodiscard]] core::Result<std::pair<VulkanCommandList*, VulkanPropertyPool*>> propertyCall(
        CommandListHandle commandList, AccelerationStructurePropertyPoolHandle pool);
    // Records build: with ranges' counts where ranges is given, a build
    // whose counts the GPU reads from ranges.first, ranges.second bytes
    // apart; else with the inputs' counts.
    [[nodiscard]] core::Result<void> recordBuild(CommandListHandle commandList, const AccelerationStructureBuild& build,
                                                 const std::pair<VkDeviceAddress, std::uint32_t>* ranges);
    // structure's VkMicromapEXT; VK_NULL_HANDLE for a null handle. Fails with
    // RhiInvalidCall for a structure that is not an opacity micromap, and
    // RhiInvalidHandle for one that is gone.
    [[nodiscard]] core::Result<VkMicromapEXT> micromapOf(AccelerationStructureHandle structure);
    // A copy of src into dst in mode, on a queue that runs compute.
    [[nodiscard]] core::Result<void> copyStructure(CommandListHandle commandList, AccelerationStructureHandle dst,
                                                   AccelerationStructureHandle src,
                                                   VkCopyAccelerationStructureModeKHR mode);
    // Fails with RhiUnsupported when list's queue runs no compute, which
    // builds and traces need, and RhiInvalidHandle when its pool is gone.
    [[nodiscard]] core::Result<void> checkRunsCompute(const VulkanCommandList& list);
    // buffer's device address, offset bytes in; 0 for a null buffer. Fails
    // with RhiInvalidHandle when the buffer is gone.
    [[nodiscard]] core::Result<VkDeviceAddress> addressOf(BufferHandle buffer, std::uint64_t offset);
    // region as Vulkan takes it, by its buffer's address; none for a null
    // buffer. Fails as addressOf does.
    [[nodiscard]] core::Result<VkStridedDeviceAddressRegionKHR> regionOf(const ShaderTableRegion& region);
    // VK_NULL_HANDLE for a null handle.
    [[nodiscard]] core::Result<VkPipelineCache> pipelineCacheFor(PipelineCacheHandle cache);
    // The binds of desc's texture and buffer regions into queue's arrays.
    [[nodiscard]] core::Result<void> addTextureBinds(VulkanQueue& queue, const TileMappingDesc& desc,
                                                     const VulkanSparseMemory* memory);
    [[nodiscard]] core::Result<void> addBufferBinds(VulkanQueue& queue, const TileMappingDesc& desc,
                                                    const VulkanSparseMemory* memory);

    VulkanInstance* instance_;
    VkPhysicalDevice adapter_;
    VkDevice device_;
    // Each device calls through its own table: volk's global pointers hold one
    // device's functions, and two adapters can have different drivers.
    VolkDeviceTable table_{};
    bool hasBudget_;
    bool hasSwapchain_;
    std::uint32_t pushConstantsSize_;
    float timestampPeriod_;
    Timestamps timestamps_;
    bool multiDrawIndirect_;
    bool drawIndirectCount_;
    Features features_;
    ResolveCaps resolve_;
    RayTracingLimits rayTracing_;
    BufferAddressing addressing_;
    Formats13 formats13_;
    DeviceDesc desc_;
    std::vector<std::unique_ptr<VulkanQueue>> queues_;
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

    // Every queue family the device's queues are on, which each buffer is shared by.
    std::vector<std::uint32_t> families_;
    core::Submissions submissions_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    static constexpr std::uint32_t kBufferCapacity = 256;
    static constexpr std::uint32_t kBufferLimit = 1U << 20U;
    core::Pool<VulkanBuffer, Buffer> buffers_;
    core::Pool<VulkanTexture, Texture> textures_;
    // Every sampler holds a slot, so the pool is made at the table's size.
    core::Pool<VulkanSampler, Sampler> samplers_;
    core::Pool<VulkanTimestampPool, TimestampPool> timestampPools_;
    core::Pool<VulkanSparseMemory, SparseMemory> sparseMemories_;
    core::Pool<VulkanPipelineCache, PipelineCache> pipelineCaches_;
    core::Pool<VulkanAccelerationStructure, AccelerationStructure> accelerationStructures_;
    core::Pool<VulkanPropertyPool, AccelerationStructurePropertyPool> propertyPools_;
    // What one build or size query fills, made at maxBuildGeometries; one
    // call at a time uses it, since pools record on many threads.
    std::mutex buildLock_;
    VulkanBuildArrays builds_;
    BindlessSlots bindless_;
    // Shaders' modules, which pipelines are made from and outlive.
    core::Pool<VulkanShader, Shader> shaders_;
    // Declared before pipelines_, which are made on it, so it outlives them.
    VulkanBindlessLayout layout_;
    core::Pool<VulkanPipeline, Pipeline> pipelines_;
};

} // namespace bfr::rhi::vulkan
