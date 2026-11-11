#pragma once

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/call_timing.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace bfr::rhi {

/// A count, total and longest time for each Call, added to from any thread
/// without a lock or an allocation.
class CallCounters {
public:
    /// Runs body, timing it on std::chrono::steady_clock as call, and gives
    /// what it returns.
    template <class Body>
    auto time(Call call, Body&& body) const {
        const auto start = std::chrono::steady_clock::now();
        auto result = body();
        add(call, std::chrono::steady_clock::now() - start);
        return result;
    }

    /// Each Call's counts, as many as timings holds.
    void write(std::span<CallTiming> timings) const;

private:
    struct Counter {
        std::atomic<std::uint64_t> count{0};
        std::atomic<std::uint64_t> totalNs{0};
        std::atomic<std::uint64_t> maxNs{0};
    };

    void add(Call call, std::chrono::steady_clock::duration taken) const;

    // Counting doesn't change what the device or instance does, so const
    // calls count too.
    mutable std::array<Counter, kCallCount> counters_{};
};

/// A Device that times each call it passes to the backend's, made when
/// InstanceDesc::callTimings is set.
class TimedDevice final : public Device {
public:
    explicit TimedDevice(std::unique_ptr<Device> device) : device_(std::move(device)) {}
    ~TimedDevice() override = default;
    TimedDevice(const TimedDevice&) = delete;
    TimedDevice& operator=(const TimedDevice&) = delete;
    TimedDevice(TimedDevice&&) = delete;
    TimedDevice& operator=(TimedDevice&&) = delete;

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
    [[nodiscard]] core::Result<void> copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
                                                 std::span<const TextureCopy> regions) override;
    [[nodiscard]] core::Result<void> resolveTexture(CommandListHandle commandList, TextureHandle src,
                                                    std::uint32_t srcMipLevel, std::uint32_t srcLayer,
                                                    TextureHandle dst, std::uint32_t dstMipLevel,
                                                    std::uint32_t dstLayer) override;
    [[nodiscard]] core::Result<Token> submit(const SubmitDesc& desc) override;
    [[nodiscard]] core::Result<void> wait(Token token, std::uint64_t timeoutNs) override;
    [[nodiscard]] core::Result<bool> poll(Token token) override;
    [[nodiscard]] core::Result<void> destroyCommandList(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<BufferHandle> createBuffer(const BufferDesc& desc) override;
    [[nodiscard]] core::Result<std::uint32_t> bufferSlot(BufferHandle buffer) override;
    [[nodiscard]] core::Result<std::span<std::byte>> mapBuffer(BufferHandle buffer) override;
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
    [[nodiscard]] core::Result<AccelerationStructureCapture> captureAccelerationStructure(
        AccelerationStructureHandle structure) override;
    [[nodiscard]] core::Result<void> destroyAccelerationStructure(AccelerationStructureHandle structure) override;
    [[nodiscard]] core::Result<void> writeInstances(std::span<const AccelerationStructureInstance> instances,
                                                    std::span<std::byte> out) override;
    [[nodiscard]] core::Result<void> writeMotionInstances(
        std::span<const AccelerationStructureMotionInstance> instances, std::span<const MotionKeyframe> keyframes,
        std::span<std::byte> out) override;
    [[nodiscard]] core::Result<void> buildAccelerationStructure(CommandListHandle commandList,
                                                                const AccelerationStructureBuild& build) override;
    [[nodiscard]] core::Result<void> buildAccelerationStructureIndirect(CommandListHandle commandList,
                                                                        const AccelerationStructureBuild& build,
                                                                        BufferHandle ranges, std::uint64_t offset,
                                                                        std::uint32_t stride) override;
    [[nodiscard]] core::Result<AccelerationStructureSizes> clusterSizes(const ClusterInputs& inputs) override;
    [[nodiscard]] core::Result<void> buildClusters(CommandListHandle commandList, const ClusterBuild& build) override;
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
    [[nodiscard]] core::Result<void> callTimings(std::span<CallTiming> timings) const override;

private:
    std::unique_ptr<Device> device_;
    CallCounters counters_;
};

/// An Instance that times each call it passes to the backend's, and whose
/// devices are TimedDevices, made when InstanceDesc::callTimings is set.
class TimedInstance final : public Instance {
public:
    TimedInstance(std::unique_ptr<Instance> instance, const RhiConfig& config)
        : Instance(config), instance_(std::move(instance)) {}
    ~TimedInstance() override = default;
    TimedInstance(const TimedInstance&) = delete;
    TimedInstance& operator=(const TimedInstance&) = delete;
    TimedInstance(TimedInstance&&) = delete;
    TimedInstance& operator=(TimedInstance&&) = delete;

    [[nodiscard]] std::span<const Adapter> adapters() const override;
    [[nodiscard]] core::Result<std::unique_ptr<Device>> createDevice(std::size_t adapter,
                                                                     const core::window::NativeWindow& presentTo,
                                                                     const DeviceDesc& desc) override;
    [[nodiscard]] core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const override;
    [[nodiscard]] core::Result<SwapchainCaps> swapchainCaps(std::size_t adapter,
                                                            const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<std::vector<SwapchainFormat>> swapchainFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<void> callTimings(std::span<CallTiming> timings) const override;

private:
    std::unique_ptr<Instance> instance_;
    CallCounters counters_;
};

} // namespace bfr::rhi
