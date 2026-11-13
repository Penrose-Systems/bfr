#include "rhi/timed_device.h"

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/call_timing.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/query.h"
#include "bfr/rhi/texture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace bfr::rhi {

void CallCounters::add(Call call, std::chrono::steady_clock::duration taken) const {
    const auto ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(taken).count());
    Counter& counter = counters_.at(static_cast<std::size_t>(call));
    counter.count.fetch_add(1, std::memory_order_relaxed);
    counter.totalNs.fetch_add(ns, std::memory_order_relaxed);
    std::uint64_t longest = counter.maxNs.load(std::memory_order_relaxed);
    while (ns > longest && !counter.maxNs.compare_exchange_weak(longest, ns, std::memory_order_relaxed)) {}
}

void CallCounters::write(std::span<CallTiming> timings) const {
    const std::size_t written = std::min(timings.size(), counters_.size());
    for (std::size_t call = 0; call < written; ++call) {
        const Counter& counter = counters_.at(call);
        timings[call] = {.count = counter.count.load(std::memory_order_relaxed),
                         .totalNs = counter.totalNs.load(std::memory_order_relaxed),
                         .maxNs = counter.maxNs.load(std::memory_order_relaxed)};
    }
}

core::Result<void> Device::callTimings(std::span<CallTiming> /*timings*/) const {
    return core::failure(core::ErrorCode::RhiInvalidCall, "call timings were not asked for in the InstanceDesc");
}

core::Result<void> Instance::callTimings(std::span<CallTiming> /*timings*/) const {
    return core::failure(core::ErrorCode::RhiInvalidCall, "call timings were not asked for in the InstanceDesc");
}

// Each call passes to the backend's as it is, timed.

core::Result<SwapchainHandle> TimedDevice::createSwapchain(const core::window::NativeWindow& window,
                                                           const SwapchainDesc& desc) {
    return counters_.time(Call::CreateSwapchain, [&] { return device_->createSwapchain(window, desc); });
}

core::Result<SwapchainImages> TimedDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                           std::uint32_t height) {
    return counters_.time(Call::ResizeSwapchain, [&] { return device_->resizeSwapchain(swapchain, width, height); });
}

core::Result<void> TimedDevice::destroySwapchain(SwapchainHandle swapchain) {
    return counters_.time(Call::DestroySwapchain, [&] { return device_->destroySwapchain(swapchain); });
}

core::Result<SwapchainStatus> TimedDevice::acquire(SwapchainHandle swapchain, std::uint64_t timeoutNs) {
    return counters_.time(Call::Acquire, [&] { return device_->acquire(swapchain, timeoutNs); });
}

core::Result<SwapchainStatus> TimedDevice::present(SwapchainHandle swapchain) {
    return counters_.time(Call::Present, [&] { return device_->present(swapchain); });
}

core::Result<CommandPoolHandle> TimedDevice::createCommandPool(const CommandPoolDesc& desc) {
    return counters_.time(Call::CreateCommandPool, [&] { return device_->createCommandPool(desc); });
}

core::Result<void> TimedDevice::resetCommandPool(CommandPoolHandle commandPool) {
    return counters_.time(Call::ResetCommandPool, [&] { return device_->resetCommandPool(commandPool); });
}

core::Result<void> TimedDevice::destroyCommandPool(CommandPoolHandle commandPool) {
    return counters_.time(Call::DestroyCommandPool, [&] { return device_->destroyCommandPool(commandPool); });
}

core::Result<CommandListHandle> TimedDevice::beginCommands(CommandPoolHandle commandPool) {
    return counters_.time(Call::BeginCommands, [&] { return device_->beginCommands(commandPool); });
}

core::Result<void> TimedDevice::beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                               const RenderingDesc& desc) {
    return counters_.time(Call::BeginRendering, [&] { return device_->beginRendering(commandList, swapchain, desc); });
}

core::Result<void> TimedDevice::endRendering(CommandListHandle commandList) {
    return counters_.time(Call::EndRendering, [&] { return device_->endRendering(commandList); });
}

core::Result<void> TimedDevice::setViewport(CommandListHandle commandList, const Viewport& viewport) {
    return counters_.time(Call::SetViewport, [&] { return device_->setViewport(commandList, viewport); });
}

core::Result<void> TimedDevice::setScissor(CommandListHandle commandList, const Rect& scissor) {
    return counters_.time(Call::SetScissor, [&] { return device_->setScissor(commandList, scissor); });
}

core::Result<void> TimedDevice::setBlendConstants(CommandListHandle commandList, const std::array<float, 4>& color) {
    return counters_.time(Call::SetBlendConstants, [&] { return device_->setBlendConstants(commandList, color); });
}

core::Result<void> TimedDevice::setStencilReference(CommandListHandle commandList, std::uint32_t reference) {
    return counters_.time(Call::SetStencilReference,
                          [&] { return device_->setStencilReference(commandList, reference); });
}

core::Result<void> TimedDevice::bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) {
    return counters_.time(Call::BindPipeline, [&] { return device_->bindPipeline(commandList, pipeline); });
}

core::Result<void> TimedDevice::pushConstants(CommandListHandle commandList, std::span<const std::byte> data) {
    return counters_.time(Call::PushConstants, [&] { return device_->pushConstants(commandList, data); });
}

core::Result<void> TimedDevice::bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
                                                std::uint64_t offset, IndexType type) {
    return counters_.time(Call::BindIndexBuffer,
                          [&] { return device_->bindIndexBuffer(commandList, buffer, offset, type); });
}

core::Result<void> TimedDevice::draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                     std::uint32_t instanceCount, std::uint32_t firstVertex,
                                     std::uint32_t firstInstance) {
    return counters_.time(Call::Draw, [&] {
        return device_->draw(commandList, vertexCount, instanceCount, firstVertex, firstInstance);
    });
}

core::Result<void> TimedDevice::drawIndexed(CommandListHandle commandList, std::uint32_t indexCount,
                                            std::uint32_t instanceCount, std::uint32_t firstIndex,
                                            std::int32_t vertexOffset, std::uint32_t firstInstance) {
    return counters_.time(Call::DrawIndexed, [&] {
        return device_->drawIndexed(commandList, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    });
}

core::Result<void> TimedDevice::dispatch(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                         std::uint32_t z) {
    return counters_.time(Call::Dispatch, [&] { return device_->dispatch(commandList, x, y, z); });
}

core::Result<void> TimedDevice::setTopology(CommandListHandle commandList, Topology topology) {
    return counters_.time(Call::SetTopology, [&] { return device_->setTopology(commandList, topology); });
}

core::Result<void> TimedDevice::drawIndirect(CommandListHandle commandList, BufferHandle buffer, std::uint64_t offset,
                                             std::uint32_t drawCount, std::uint32_t stride) {
    return counters_.time(Call::DrawIndirect,
                          [&] { return device_->drawIndirect(commandList, buffer, offset, drawCount, stride); });
}

core::Result<void> TimedDevice::drawIndexedIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                    std::uint64_t offset, std::uint32_t drawCount,
                                                    std::uint32_t stride) {
    return counters_.time(Call::DrawIndexedIndirect,
                          [&] { return device_->drawIndexedIndirect(commandList, buffer, offset, drawCount, stride); });
}

core::Result<void> TimedDevice::drawIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                  std::uint64_t offset, BufferHandle countBuffer,
                                                  std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                  std::uint32_t stride) {
    return counters_.time(Call::DrawIndirectCount, [&] {
        return device_->drawIndirectCount(commandList, buffer, offset, countBuffer, countOffset, maxDrawCount, stride);
    });
}

core::Result<void> TimedDevice::drawIndexedIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                         std::uint64_t offset, BufferHandle countBuffer,
                                                         std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                         std::uint32_t stride) {
    return counters_.time(Call::DrawIndexedIndirectCount, [&] {
        return device_->drawIndexedIndirectCount(commandList, buffer, offset, countBuffer, countOffset, maxDrawCount,
                                                 stride);
    });
}

core::Result<void> TimedDevice::dispatchIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                 std::uint64_t offset) {
    return counters_.time(Call::DispatchIndirect,
                          [&] { return device_->dispatchIndirect(commandList, buffer, offset); });
}

core::Result<void> TimedDevice::drawMesh(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                         std::uint32_t z) {
    return counters_.time(Call::DrawMesh, [&] { return device_->drawMesh(commandList, x, y, z); });
}

core::Result<void> TimedDevice::drawMeshIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                 std::uint64_t offset, std::uint32_t drawCount, std::uint32_t stride) {
    return counters_.time(Call::DrawMeshIndirect,
                          [&] { return device_->drawMeshIndirect(commandList, buffer, offset, drawCount, stride); });
}

core::Result<void> TimedDevice::drawMeshIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                      std::uint64_t offset, BufferHandle countBuffer,
                                                      std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                      std::uint32_t stride) {
    return counters_.time(Call::DrawMeshIndirectCount, [&] {
        return device_->drawMeshIndirectCount(commandList, buffer, offset, countBuffer, countOffset, maxDrawCount,
                                              stride);
    });
}

core::Result<void> TimedDevice::copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
                                           std::span<const BufferCopy> regions) {
    return counters_.time(Call::CopyBuffer, [&] { return device_->copyBuffer(commandList, src, dst, regions); });
}

core::Result<void> TimedDevice::barrier(CommandListHandle commandList) {
    return counters_.time(Call::Barrier, [&] { return device_->barrier(commandList); });
}

core::Result<void> TimedDevice::textureBarrier(CommandListHandle commandList, TextureHandle texture,
                                               TextureLayout before, TextureLayout after, const TextureRange& range) {
    return counters_.time(Call::TextureBarrier,
                          [&] { return device_->textureBarrier(commandList, texture, before, after, range); });
}

core::Result<void> TimedDevice::copyBufferToTexture(CommandListHandle commandList, BufferHandle buffer,
                                                    TextureHandle texture, std::span<const BufferTextureCopy> regions) {
    return counters_.time(Call::CopyBufferToTexture,
                          [&] { return device_->copyBufferToTexture(commandList, buffer, texture, regions); });
}

core::Result<void> TimedDevice::copyTextureToBuffer(CommandListHandle commandList, TextureHandle texture,
                                                    BufferHandle buffer, std::span<const BufferTextureCopy> regions) {
    return counters_.time(Call::CopyTextureToBuffer,
                          [&] { return device_->copyTextureToBuffer(commandList, texture, buffer, regions); });
}

core::Result<void> TimedDevice::copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
                                            std::span<const TextureCopy> regions) {
    return counters_.time(Call::CopyTexture, [&] { return device_->copyTexture(commandList, src, dst, regions); });
}

core::Result<void> TimedDevice::resolveTexture(CommandListHandle commandList, TextureHandle src,
                                               std::uint32_t srcMipLevel, std::uint32_t srcLayer, TextureHandle dst,
                                               std::uint32_t dstMipLevel, std::uint32_t dstLayer) {
    return counters_.time(Call::ResolveTexture, [&] {
        return device_->resolveTexture(commandList, src, srcMipLevel, srcLayer, dst, dstMipLevel, dstLayer);
    });
}

core::Result<Token> TimedDevice::submit(const SubmitDesc& desc) {
    return counters_.time(Call::Submit, [&] { return device_->submit(desc); });
}

core::Result<void> TimedDevice::wait(Token token, std::uint64_t timeoutNs) {
    return counters_.time(Call::Wait, [&] { return device_->wait(token, timeoutNs); });
}

core::Result<bool> TimedDevice::poll(Token token) {
    return counters_.time(Call::Poll, [&] { return device_->poll(token); });
}

core::Result<void> TimedDevice::runGarbageCollection() {
    return counters_.time(Call::RunGarbageCollection, [&] { return device_->runGarbageCollection(); });
}

core::Result<void> TimedDevice::destroyCommandList(CommandListHandle commandList) {
    return counters_.time(Call::DestroyCommandList, [&] { return device_->destroyCommandList(commandList); });
}

core::Result<BufferHandle> TimedDevice::createBuffer(const BufferDesc& desc) {
    return counters_.time(Call::CreateBuffer, [&] { return device_->createBuffer(desc); });
}

core::Result<BufferSlots> TimedDevice::bufferSlots(BufferHandle buffer) {
    return counters_.time(Call::BufferSlots, [&] { return device_->bufferSlots(buffer); });
}

core::Result<std::span<std::byte>> TimedDevice::mapBuffer(BufferHandle buffer) {
    return counters_.time(Call::MapBuffer, [&] { return device_->mapBuffer(buffer); });
}

core::Result<void> TimedDevice::destroyBuffer(BufferHandle buffer) {
    return counters_.time(Call::DestroyBuffer, [&] { return device_->destroyBuffer(buffer); });
}

core::Result<TextureHandle> TimedDevice::createTexture(const TextureDesc& desc) {
    return counters_.time(Call::CreateTexture, [&] { return device_->createTexture(desc); });
}

core::Result<TextureSlots> TimedDevice::textureSlots(TextureHandle texture) {
    return counters_.time(Call::TextureSlots, [&] { return device_->textureSlots(texture); });
}

core::Result<void> TimedDevice::destroyTexture(TextureHandle texture) {
    return counters_.time(Call::DestroyTexture, [&] { return device_->destroyTexture(texture); });
}

core::Result<SamplerHandle> TimedDevice::createSampler(const SamplerDesc& desc) {
    return counters_.time(Call::CreateSampler, [&] { return device_->createSampler(desc); });
}

core::Result<std::uint32_t> TimedDevice::samplerSlot(SamplerHandle sampler) {
    return counters_.time(Call::SamplerSlot, [&] { return device_->samplerSlot(sampler); });
}

core::Result<void> TimedDevice::destroySampler(SamplerHandle sampler) {
    return counters_.time(Call::DestroySampler, [&] { return device_->destroySampler(sampler); });
}

core::Result<PipelineHandle> TimedDevice::createComputePipeline(const ComputePipelineDesc& desc) {
    return counters_.time(Call::CreateComputePipeline, [&] { return device_->createComputePipeline(desc); });
}

core::Result<PipelineHandle> TimedDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    return counters_.time(Call::CreateGraphicsPipeline, [&] { return device_->createGraphicsPipeline(desc); });
}

core::Result<void> TimedDevice::destroyPipeline(PipelineHandle pipeline) {
    return counters_.time(Call::DestroyPipeline, [&] { return device_->destroyPipeline(pipeline); });
}

core::Result<PipelineCacheHandle> TimedDevice::createPipelineCache(const PipelineCacheDesc& desc) {
    return counters_.time(Call::CreatePipelineCache, [&] { return device_->createPipelineCache(desc); });
}

core::Result<std::size_t> TimedDevice::pipelineCacheSize(PipelineCacheHandle cache) {
    return counters_.time(Call::PipelineCacheSize, [&] { return device_->pipelineCacheSize(cache); });
}

core::Result<PipelineCacheWrite> TimedDevice::pipelineCacheData(PipelineCacheHandle cache, std::span<std::byte> data) {
    return counters_.time(Call::PipelineCacheData, [&] { return device_->pipelineCacheData(cache, data); });
}

core::Result<void> TimedDevice::destroyPipelineCache(PipelineCacheHandle cache) {
    return counters_.time(Call::DestroyPipelineCache, [&] { return device_->destroyPipelineCache(cache); });
}

core::Result<MemoryBudget> TimedDevice::budget() const {
    return counters_.time(Call::Budget, [&] { return device_->budget(); });
}

core::Result<TimestampPoolHandle> TimedDevice::createTimestampPool(const TimestampPoolDesc& desc) {
    return counters_.time(Call::CreateTimestampPool, [&] { return device_->createTimestampPool(desc); });
}

core::Result<void> TimedDevice::destroyTimestampPool(TimestampPoolHandle pool) {
    return counters_.time(Call::DestroyTimestampPool, [&] { return device_->destroyTimestampPool(pool); });
}

core::Result<void> TimedDevice::resetTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                std::uint32_t first, std::uint32_t count) {
    return counters_.time(Call::ResetTimestamps,
                          [&] { return device_->resetTimestamps(commandList, pool, first, count); });
}

core::Result<void> TimedDevice::writeTimestamp(CommandListHandle commandList, TimestampPoolHandle pool,
                                               std::uint32_t index, TimestampStage stage,
                                               TimestampGranularity granularity) {
    return counters_.time(Call::WriteTimestamp,
                          [&] { return device_->writeTimestamp(commandList, pool, index, stage, granularity); });
}

core::Result<void> TimedDevice::resolveTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                  std::uint32_t first, std::uint32_t count, BufferHandle buffer,
                                                  std::uint64_t offset) {
    return counters_.time(Call::ResolveTimestamps,
                          [&] { return device_->resolveTimestamps(commandList, pool, first, count, buffer, offset); });
}

core::Result<QueryPoolHandle> TimedDevice::createQueryPool(const QueryPoolDesc& desc) {
    return counters_.time(Call::CreateQueryPool, [&] { return device_->createQueryPool(desc); });
}

core::Result<void> TimedDevice::destroyQueryPool(QueryPoolHandle pool) {
    return counters_.time(Call::DestroyQueryPool, [&] { return device_->destroyQueryPool(pool); });
}

core::Result<QueryLayout> TimedDevice::queryLayout(QueryPoolHandle pool) {
    return counters_.time(Call::QueryLayout, [&] { return device_->queryLayout(pool); });
}

core::Result<void> TimedDevice::resetQueries(CommandListHandle commandList, QueryPoolHandle pool, std::uint32_t first,
                                             std::uint32_t count) {
    return counters_.time(Call::ResetQueries, [&] { return device_->resetQueries(commandList, pool, first, count); });
}

core::Result<void> TimedDevice::beginQuery(CommandListHandle commandList, QueryPoolHandle pool, std::uint32_t index,
                                           bool precise) {
    return counters_.time(Call::BeginQuery, [&] { return device_->beginQuery(commandList, pool, index, precise); });
}

core::Result<void> TimedDevice::endQuery(CommandListHandle commandList, QueryPoolHandle pool, std::uint32_t index) {
    return counters_.time(Call::EndQuery, [&] { return device_->endQuery(commandList, pool, index); });
}

core::Result<void> TimedDevice::resolveQueries(CommandListHandle commandList, QueryPoolHandle pool, std::uint32_t first,
                                               std::uint32_t count, BufferHandle buffer, std::uint64_t offset) {
    return counters_.time(Call::ResolveQueries,
                          [&] { return device_->resolveQueries(commandList, pool, first, count, buffer, offset); });
}

core::Result<void> TimedDevice::beginDebugLabel(CommandListHandle commandList, const DebugLabel& label) {
    return counters_.time(Call::BeginDebugLabel, [&] { return device_->beginDebugLabel(commandList, label); });
}

core::Result<void> TimedDevice::endDebugLabel(CommandListHandle commandList) {
    return counters_.time(Call::EndDebugLabel, [&] { return device_->endDebugLabel(commandList); });
}

core::Result<void> TimedDevice::insertDebugLabel(CommandListHandle commandList, const DebugLabel& label) {
    return counters_.time(Call::InsertDebugLabel, [&] { return device_->insertDebugLabel(commandList, label); });
}

core::Result<void> TimedDevice::beginQueueDebugLabel(std::uint32_t queue, const DebugLabel& label) {
    return counters_.time(Call::BeginQueueDebugLabel, [&] { return device_->beginQueueDebugLabel(queue, label); });
}

core::Result<void> TimedDevice::endQueueDebugLabel(std::uint32_t queue) {
    return counters_.time(Call::EndQueueDebugLabel, [&] { return device_->endQueueDebugLabel(queue); });
}

core::Result<void> TimedDevice::insertQueueDebugLabel(std::uint32_t queue, const DebugLabel& label) {
    return counters_.time(Call::InsertQueueDebugLabel, [&] { return device_->insertQueueDebugLabel(queue, label); });
}

core::Result<TimestampRate> TimedDevice::timestampRate(std::uint32_t queue) const {
    return counters_.time(Call::TimestampRate, [&] { return device_->timestampRate(queue); });
}

core::Result<CalibratedTimestamps> TimedDevice::calibrateTimestamps(std::uint32_t queue, HostClock clock) const {
    return counters_.time(Call::CalibrateTimestamps, [&] { return device_->calibrateTimestamps(queue, clock); });
}

core::Result<SparseMemoryHandle> TimedDevice::createSparseMemory(const SparseMemoryDesc& desc) {
    return counters_.time(Call::CreateSparseMemory, [&] { return device_->createSparseMemory(desc); });
}

core::Result<void> TimedDevice::destroySparseMemory(SparseMemoryHandle memory) {
    return counters_.time(Call::DestroySparseMemory, [&] { return device_->destroySparseMemory(memory); });
}

core::Result<TextureTiling> TimedDevice::textureTiling(TextureHandle texture, TextureAspect aspect) {
    return counters_.time(Call::TextureTiling, [&] { return device_->textureTiling(texture, aspect); });
}

core::Result<BufferTiling> TimedDevice::bufferTiling(BufferHandle buffer) {
    return counters_.time(Call::BufferTiling, [&] { return device_->bufferTiling(buffer); });
}

core::Result<SparseFormat> TimedDevice::sparseFormat(const SparseFormatDesc& desc) const {
    return counters_.time(Call::SparseFormat, [&] { return device_->sparseFormat(desc); });
}

core::Result<Token> TimedDevice::updateTileMappings(const TileMappingDesc& desc) {
    return counters_.time(Call::UpdateTileMappings, [&] { return device_->updateTileMappings(desc); });
}

core::Result<AccelerationStructureSizes> TimedDevice::accelerationStructureSizes(
    const AccelerationStructureInputs& inputs) {
    return counters_.time(Call::AccelerationStructureSizes,
                          [&] { return device_->accelerationStructureSizes(inputs); });
}

core::Result<AccelerationStructureHandle> TimedDevice::createAccelerationStructure(
    const AccelerationStructureDesc& desc) {
    return counters_.time(Call::CreateAccelerationStructure,
                          [&] { return device_->createAccelerationStructure(desc); });
}

core::Result<std::uint32_t> TimedDevice::accelerationStructureSlot(AccelerationStructureHandle structure) {
    return counters_.time(Call::AccelerationStructureSlot,
                          [&] { return device_->accelerationStructureSlot(structure); });
}

core::Result<std::uint64_t> TimedDevice::accelerationStructureReference(AccelerationStructureHandle structure) {
    return counters_.time(Call::AccelerationStructureReference,
                          [&] { return device_->accelerationStructureReference(structure); });
}

core::Result<AccelerationStructureCapture> TimedDevice::captureAccelerationStructure(
    AccelerationStructureHandle structure) {
    return counters_.time(Call::CaptureAccelerationStructure,
                          [&] { return device_->captureAccelerationStructure(structure); });
}

core::Result<void> TimedDevice::destroyAccelerationStructure(AccelerationStructureHandle structure) {
    return counters_.time(Call::DestroyAccelerationStructure,
                          [&] { return device_->destroyAccelerationStructure(structure); });
}

core::Result<void> TimedDevice::writeInstances(std::span<const AccelerationStructureInstance> instances,
                                               std::span<std::byte> out) {
    return counters_.time(Call::WriteInstances, [&] { return device_->writeInstances(instances, out); });
}

core::Result<void> TimedDevice::buildAccelerationStructure(CommandListHandle commandList,
                                                           const AccelerationStructureBuild& build) {
    return counters_.time(Call::BuildAccelerationStructure,
                          [&] { return device_->buildAccelerationStructure(commandList, build); });
}

core::Result<void> TimedDevice::writeMotionInstances(std::span<const AccelerationStructureMotionInstance> instances,
                                                     std::span<const MotionKeyframe> keyframes,
                                                     std::span<std::byte> out) {
    return counters_.time(Call::WriteMotionInstances,
                          [&] { return device_->writeMotionInstances(instances, keyframes, out); });
}

core::Result<void> TimedDevice::buildAccelerationStructureIndirect(CommandListHandle commandList,
                                                                   const AccelerationStructureBuild& build,
                                                                   BufferHandle ranges, std::uint64_t offset,
                                                                   std::uint32_t stride) {
    return counters_.time(Call::BuildAccelerationStructureIndirect, [&] {
        return device_->buildAccelerationStructureIndirect(commandList, build, ranges, offset, stride);
    });
}

core::Result<AccelerationStructureSizes> TimedDevice::clusterSizes(const ClusterInputs& inputs) {
    return counters_.time(Call::ClusterSizes, [&] { return device_->clusterSizes(inputs); });
}

core::Result<void> TimedDevice::buildClusters(CommandListHandle commandList, const ClusterBuild& build) {
    return counters_.time(Call::BuildClusters, [&] { return device_->buildClusters(commandList, build); });
}

core::Result<void> TimedDevice::copyAccelerationStructure(CommandListHandle commandList,
                                                          AccelerationStructureHandle dst,
                                                          AccelerationStructureHandle src) {
    return counters_.time(Call::CopyAccelerationStructure,
                          [&] { return device_->copyAccelerationStructure(commandList, dst, src); });
}

core::Result<AccelerationStructurePropertyPoolHandle> TimedDevice::createAccelerationStructurePropertyPool(
    const AccelerationStructurePropertyPoolDesc& desc) {
    return counters_.time(Call::CreateAccelerationStructurePropertyPool,
                          [&] { return device_->createAccelerationStructurePropertyPool(desc); });
}

core::Result<void> TimedDevice::destroyAccelerationStructurePropertyPool(AccelerationStructurePropertyPoolHandle pool) {
    return counters_.time(Call::DestroyAccelerationStructurePropertyPool,
                          [&] { return device_->destroyAccelerationStructurePropertyPool(pool); });
}

core::Result<void> TimedDevice::resetAccelerationStructureProperties(CommandListHandle commandList,
                                                                     AccelerationStructurePropertyPoolHandle pool,
                                                                     std::uint32_t first, std::uint32_t count) {
    return counters_.time(Call::ResetAccelerationStructureProperties, [&] {
        return device_->resetAccelerationStructureProperties(commandList, pool, first, count);
    });
}

core::Result<void> TimedDevice::writeAccelerationStructureProperties(
    CommandListHandle commandList, AccelerationStructurePropertyPoolHandle pool, std::uint32_t first,
    std::span<const AccelerationStructureHandle> structures) {
    return counters_.time(Call::WriteAccelerationStructureProperties, [&] {
        return device_->writeAccelerationStructureProperties(commandList, pool, first, structures);
    });
}

core::Result<void> TimedDevice::resolveAccelerationStructureProperties(CommandListHandle commandList,
                                                                       AccelerationStructurePropertyPoolHandle pool,
                                                                       std::uint32_t first, std::uint32_t count,
                                                                       BufferHandle buffer, std::uint64_t offset) {
    return counters_.time(Call::ResolveAccelerationStructureProperties, [&] {
        return device_->resolveAccelerationStructureProperties(commandList, pool, first, count, buffer, offset);
    });
}

core::Result<void> TimedDevice::compactAccelerationStructure(CommandListHandle commandList,
                                                             AccelerationStructureHandle dst,
                                                             AccelerationStructureHandle src) {
    return counters_.time(Call::CompactAccelerationStructure,
                          [&] { return device_->compactAccelerationStructure(commandList, dst, src); });
}

core::Result<void> TimedDevice::serializeAccelerationStructure(CommandListHandle commandList,
                                                               AccelerationStructureHandle src, BufferHandle buffer,
                                                               std::uint64_t offset) {
    return counters_.time(Call::SerializeAccelerationStructure,
                          [&] { return device_->serializeAccelerationStructure(commandList, src, buffer, offset); });
}

core::Result<void> TimedDevice::deserializeAccelerationStructure(CommandListHandle commandList, BufferHandle buffer,
                                                                 std::uint64_t offset,
                                                                 AccelerationStructureHandle dst) {
    return counters_.time(Call::DeserializeAccelerationStructure,
                          [&] { return device_->deserializeAccelerationStructure(commandList, buffer, offset, dst); });
}

core::Result<bool> TimedDevice::serializedCompatible(
    AccelerationStructureType type, std::span<const std::byte, kSerializedIdentifierSize> identifier) const {
    return counters_.time(Call::SerializedCompatible, [&] { return device_->serializedCompatible(type, identifier); });
}

core::Result<std::uint64_t> TimedDevice::rayTracingStackSize(PipelineHandle pipeline, std::uint32_t group,
                                                             RayTracingGroupShader shader) {
    return counters_.time(Call::RayTracingStackSize,
                          [&] { return device_->rayTracingStackSize(pipeline, group, shader); });
}

core::Result<ShaderHandle> TimedDevice::createShader(const ShaderCode& code) {
    return counters_.time(Call::CreateShader, [&] { return device_->createShader(code); });
}

core::Result<void> TimedDevice::destroyShader(ShaderHandle shader) {
    return counters_.time(Call::DestroyShader, [&] { return device_->destroyShader(shader); });
}

core::Result<PipelineHandle> TimedDevice::createRayTracingPipeline(const RayTracingPipelineDesc& desc) {
    return counters_.time(Call::CreateRayTracingPipeline, [&] { return device_->createRayTracingPipeline(desc); });
}

core::Result<void> TimedDevice::shaderIdentifiers(PipelineHandle pipeline, std::uint32_t first, std::uint32_t count,
                                                  std::span<std::byte> out) {
    return counters_.time(Call::ShaderIdentifiers,
                          [&] { return device_->shaderIdentifiers(pipeline, first, count, out); });
}

core::Result<void> TimedDevice::captureReplayIdentifiers(PipelineHandle pipeline, std::uint32_t first,
                                                         std::uint32_t count, std::span<std::byte> out) {
    return counters_.time(Call::CaptureReplayIdentifiers,
                          [&] { return device_->captureReplayIdentifiers(pipeline, first, count, out); });
}

core::Result<void> TimedDevice::traceRays(CommandListHandle commandList, const TraceRaysDesc& desc) {
    return counters_.time(Call::TraceRays, [&] { return device_->traceRays(commandList, desc); });
}

core::Result<void> TimedDevice::traceRaysIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                  std::uint64_t offset) {
    return counters_.time(Call::TraceRaysIndirect,
                          [&] { return device_->traceRaysIndirect(commandList, buffer, offset); });
}

core::Result<std::uint64_t> TimedDevice::bufferAddress(BufferHandle buffer) {
    return counters_.time(Call::BufferAddress, [&] { return device_->bufferAddress(buffer); });
}

core::Result<void> TimedDevice::callTimings(std::span<CallTiming> timings) const {
    counters_.write(timings);
    return {};
}

std::span<const Adapter> TimedInstance::adapters() const {
    return counters_.time(Call::InstanceAdapters, [&] { return instance_->adapters(); });
}

core::Result<std::unique_ptr<Device>> TimedInstance::createDevice(std::size_t adapter,
                                                                  const core::window::NativeWindow& presentTo,
                                                                  const DeviceDesc& desc) {
    auto made = counters_.time(Call::InstanceCreateDevice,
                               [&] { return instance_->createDevice(adapter, presentTo, desc); });
    if (!made) {
        return made;
    }
    return std::make_unique<TimedDevice>(std::move(*made));
}

core::Result<FormatCaps> TimedInstance::formatCaps(std::size_t adapter, Format format) const {
    return counters_.time(Call::InstanceFormatCaps, [&] { return instance_->formatCaps(adapter, format); });
}

core::Result<SwapchainCaps> TimedInstance::swapchainCaps(std::size_t adapter,
                                                         const core::window::NativeWindow& window) const {
    return counters_.time(Call::InstanceSwapchainCaps, [&] { return instance_->swapchainCaps(adapter, window); });
}

core::Result<std::vector<SwapchainFormat>> TimedInstance::swapchainFormats(
    std::size_t adapter, const core::window::NativeWindow& window) const {
    return counters_.time(Call::InstanceSwapchainFormats, [&] { return instance_->swapchainFormats(adapter, window); });
}

core::Result<void> TimedInstance::callTimings(std::span<CallTiming> timings) const {
    counters_.write(timings);
    return {};
}

} // namespace bfr::rhi
