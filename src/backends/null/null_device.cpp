#include "null/null_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"
#include "core/slot_chains.h"
#include "null/null_instance.h"
#include "rhi/buffer_check.h"
#include "rhi/pipeline_check.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace bfr::rhi::null {

NullDevice::NullDevice(NullInstance& rhi, const DeviceDesc& desc)
    : instance_(&rhi),
      desc_(desc),
      swapchains_("swapchains", desc.maxSwapchains, desc.maxSwapchains, rhi.logger()),
      commandPools_("command pools", desc.maxCommandPools, desc.maxCommandPools, rhi.logger()),
      commandLists_("command lists", desc.maxCommandLists, desc.maxCommandLists, rhi.logger()),
      slots_(desc.maxCommandLists),
      live_(desc.maxCommandLists),
      buffers_("buffers", kBufferCapacity, kBufferLimit, rhi.logger()),
      textures_("textures", kBufferCapacity, kBufferLimit, rhi.logger()),
      pipelines_("pipelines", kBufferCapacity, kBufferLimit, rhi.logger()) {
    queues_.push_back(QueueType::Graphics);
    queues_.insert(queues_.end(), desc.compute, QueueType::Compute);
    queues_.insert(queues_.end(), desc.copy, QueueType::Copy);
    submitted_ = std::vector<std::atomic<std::uint64_t>>(queues_.size());
}

NullDevice::~NullDevice() {
    // Swapchains are the device's children, so they go first.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    swapchains_.clear();
    if (buffers_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} buffers were left when the device went", buffers_.size());
    }
    if (textures_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} textures were left when the device went", textures_.size());
    }
    if (pipelines_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} pipelines were left when the device went", pipelines_.size());
    }
}

core::Result<SwapchainHandle> NullDevice::createSwapchain(const core::window::NativeWindow& window,
                                                          const SwapchainDesc& desc) {
    if (std::holds_alternative<std::monostate>(window)) {
        return core::failure(core::ErrorCode::RhiWindowFailed, "there is no window");
    }
    const SwapchainHandle swapchain = swapchains_.emplace(NullSwapchain{
        .desc = desc,
        .images = {.width = 0,
                   .height = 0,
                   .count = 0,
                   .format = desc.format,
                   .colorSpace = desc.colorSpace,
                   .presentMode = desc.presentMode,
                   .alphaMode = desc.alphaMode},
    });
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "a device holds at most {} swapchains",
                             desc_.maxSwapchains);
    }
    return swapchain;
}

core::Result<NullSwapchain*> NullDevice::swapchainFor(SwapchainHandle swapchain) {
    NullSwapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
    }
    return found;
}

core::Result<NullCommandList*> NullDevice::commandListFor(CommandListHandle commandList) {
    NullCommandList* found = commandLists_.find(commandList);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command list handle is stale");
    }
    return found;
}

core::Result<SwapchainImages> NullDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                          std::uint32_t height) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    NullSwapchain& chain = **found;
    // A zero size is a window with no area, which leaves the swapchain empty.
    const bool empty = width == 0 || height == 0;
    chain.images.width = empty ? 0 : width;
    chain.images.height = empty ? 0 : height;
    chain.images.count = empty ? 0 : chain.desc.imageCount;
    chain.acquired = false;
    return chain.images;
}

core::Result<void> NullDevice::destroySwapchain(SwapchainHandle swapchain) {
    if (swapchain.isNull()) {
        return {};
    }
    if (swapchains_.erase(swapchain)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

core::Result<void> NullDevice::acquire(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    NullSwapchain& chain = **found;
    if (chain.images.count == 0) {
        return core::failure(core::ErrorCode::RhiOutOfDate, "the swapchain has no images; resize it");
    }
    if (chain.acquired) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "an image is already acquired; present it first");
    }
    chain.acquired = true;
    return {};
}

core::Result<void> NullDevice::present(SwapchainHandle swapchain) {
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    NullSwapchain& chain = **found;
    if (!chain.acquired) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }
    // Presented nowhere.
    chain.acquired = false;
    return {};
}

core::Result<NullCommandPool*> NullDevice::commandPoolFor(CommandPoolHandle commandPool) {
    NullCommandPool* found = commandPools_.find(commandPool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the command pool handle is stale");
    }
    return found;
}

core::Result<void> NullDevice::checkQueue(std::uint32_t queue) const {
    if (queue >= queues_.size()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the device has no queue {}; it has {}", queue,
                             queues_.size());
    }
    return {};
}

core::Result<void> NullDevice::checkToken(Token token) const {
    if (auto checked = checkQueue(token.queue); !checked) {
        return checked;
    }
    const std::uint64_t last = submitted_[token.queue].load(std::memory_order_acquire);
    if (token.value > last) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "no submit has returned token {} on queue {}; the last is {}", token.value, token.queue,
                             last);
    }
    return {};
}

core::Result<CommandPoolHandle> NullDevice::createCommandPool(const CommandPoolDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const auto first = slots_.take(desc.listCount);
    if (!first) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the pool asks for {} lists; {} of the device's {} are free", desc.listCount,
                             slots_.available(), desc_.maxCommandLists);
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(
        NullCommandPool{.queue = desc.queue, .first = *first, .next = *first, .recording = 0});
    if (commandPool.isNull()) {
        slots_.give(*first);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command pools",
                             desc_.maxCommandPools);
    }
    return commandPool;
}

core::Result<void> NullDevice::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if ((*pool)->recording != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} lists from the pool are still recording; submit or destroy them", (*pool)->recording);
    }
    (*pool)->next = (*pool)->first;
    return {};
}

core::Result<void> NullDevice::destroyCommandPool(CommandPoolHandle commandPool) {
    if (commandPool.isNull()) {
        return {};
    }
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    // No GPU runs the lists, so the slots go back at once.
    for (std::uint32_t slot = (*pool)->first; slot != core::SlotChains::kEnd; slot = slots_.next(slot)) {
        commandLists_.erase(std::exchange(live_[slot], {}));
    }
    slots_.give((*pool)->first);
    commandPools_.erase(commandPool);
    return {};
}

core::Result<CommandListHandle> NullDevice::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const std::uint32_t slot = (*pool)->next;
    if (slot == core::SlotChains::kEnd) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "every list in the pool was handed out since its last reset");
    }
    // The pool's lists were made with it, so this only takes the next.
    const CommandListHandle commandList = commandLists_.emplace(
        NullCommandList{.pool = commandPool, .slot = slot, .rendering = false});
    if (commandList.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} command lists",
                             desc_.maxCommandLists);
    }
    (*pool)->next = slots_.next(slot);
    ++(*pool)->recording;
    live_[slot] = commandList;
    return commandList;
}

core::Result<void> NullDevice::beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                              const RenderingDesc& /*desc*/) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = swapchainFor(swapchain);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*list)->rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is already open");
    }
    if (!(*found)->acquired) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }
    (*list)->rendering = true;
    return {};
}

core::Result<void> NullDevice::endRendering(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (!(*list)->rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no rendering is open");
    }
    (*list)->rendering = false;
    return {};
}

// The null queues run nothing, so a copy is made as it is recorded: what the
// tests read back is what was written before the copy was recorded.
core::Result<void> NullDevice::copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
                                          std::span<const BufferCopy> regions) {
    if (auto list = commandListFor(commandList); !list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = bufferFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = bufferFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    for (const BufferCopy& region : regions) {
        std::ranges::copy(std::span{(*from)->bytes}.subspan(region.srcOffset, region.size),
                          std::span{(*to)->bytes}.subspan(region.dstOffset).begin());
    }
    return {};
}

core::Result<void> NullDevice::barrier(CommandListHandle commandList) {
    if (auto list = commandListFor(commandList); !list) {
        return std::unexpected(std::move(list.error()));
    }
    return {};
}

core::Result<Token> NullDevice::submit(const SubmitDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    for (const Token wait : desc.waits) {
        if (auto checked = checkToken(wait); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    // Each submit has a serial of its own, so a list named twice is caught
    // in this one pass, without searching what was gathered so far.
    const std::uint64_t serial = submitSerial_.fetch_add(1, std::memory_order_relaxed) + 1;
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
        }
        if ((*list)->queued == serial) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the list is named twice in the submit");
        }
        (*list)->queued = serial;
        if ((*list)->rendering) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
        }
        const NullCommandPool* pool = commandPools_.find((*list)->pool);
        if (pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the list was recorded for queue {}, not {}",
                                 pool->queue, desc.queue);
        }
    }
    // The lists are spent whether or not the submit goes through, as on every
    // backend; a refused one makes no token.
    const bool refused = refuseNextSubmit_.exchange(false, std::memory_order_acq_rel);
    for (const CommandListHandle commandList : desc.commandLists) {
        end(commandList);
    }
    if (refused) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "the queue refused the submit, as it was told to");
    }
    return Token{.queue = desc.queue, .value = submitted_[desc.queue].fetch_add(1, std::memory_order_acq_rel) + 1};
}

core::Result<void> NullDevice::wait(Token token) {
    return checkToken(token);
}

core::Result<bool> NullDevice::poll(Token token) {
    if (auto checked = checkToken(token); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return true;
}

core::Result<void> NullDevice::destroyCommandList(CommandListHandle commandList) {
    if (commandList.isNull()) {
        return {};
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    end(commandList);
    return {};
}

void NullDevice::end(CommandListHandle commandList) {
    const NullCommandList* list = commandLists_.find(commandList);
    if (list == nullptr) {
        return;
    }
    if (NullCommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
        --pool->recording;
    }
    live_[list->slot] = {};
    commandLists_.erase(commandList);
}

core::Result<NullBuffer*> NullDevice::bufferFor(BufferHandle buffer) {
    NullBuffer* found = buffers_.find(buffer);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    return found;
}

core::Result<BufferHandle> NullDevice::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Device memory holds bytes too, so copies through it carry them.
    NullBuffer made{.memory = desc.memory, .bytes = std::vector<std::byte>(desc.size)};
    const BufferHandle buffer = buffers_.emplace(std::move(made));
    if (buffer.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", kBufferLimit);
    }
    return buffer;
}

core::Result<std::span<std::byte>> NullDevice::mapBuffer(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*found)->memory == BufferMemory::Device) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the buffer is in device memory, which the CPU does not map");
    }
    return std::span{(*found)->bytes};
}

core::Result<void> NullDevice::destroyBuffer(BufferHandle buffer) {
    if (buffer.isNull()) {
        return {};
    }
    // The GPU runs nothing, so nothing can still be using it.
    if (!buffers_.erase(buffer)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    return {};
}

core::Result<NullTexture*> NullDevice::textureFor(TextureHandle texture) {
    NullTexture* found = textures_.find(texture);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the texture handle is stale");
    }
    return found;
}

// No format has a use without a GPU, so a texture holds no texels and its
// copies move none; the calls only name what they are given.
core::Result<TextureHandle> NullDevice::createTexture(const TextureDesc& desc) {
    const TextureHandle texture = textures_.emplace(NullTexture{.desc = desc});
    if (texture.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} textures", kBufferLimit);
    }
    return texture;
}

core::Result<void> NullDevice::destroyTexture(TextureHandle texture) {
    if (texture.isNull()) {
        return {};
    }
    // The GPU runs nothing, so nothing can still be using it.
    if (!textures_.erase(texture)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the texture handle is stale");
    }
    return {};
}

// No code runs without a GPU, so a pipeline keeps none; the calls check the
// desc as every backend does and name what they are given.
core::Result<PipelineHandle> NullDevice::createComputePipeline(const ComputePipelineDesc& desc) {
    if (auto checked = checkComputePipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(
        NullPipeline{.compute = true, .workgroupSize = desc.workgroupSize});
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines", kBufferLimit);
    }
    return pipeline;
}

core::Result<PipelineHandle> NullDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkGraphicsPipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(NullPipeline{.compute = false});
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines", kBufferLimit);
    }
    return pipeline;
}

core::Result<void> NullDevice::destroyPipeline(PipelineHandle pipeline) {
    if (pipeline.isNull()) {
        return {};
    }
    // The GPU runs nothing, so nothing can still be using it.
    if (!pipelines_.erase(pipeline)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    return {};
}

core::Result<void> NullDevice::textureBarrier(CommandListHandle commandList, TextureHandle texture, TextureLayout,
                                              TextureLayout) {
    return checkNames(commandList, texture, texture);
}

core::Result<void> NullDevice::copyBufferToTexture(CommandListHandle commandList, BufferHandle buffer,
                                                   TextureHandle texture, std::span<const BufferTextureCopy>) {
    if (auto found = bufferFor(buffer); !found) {
        return std::unexpected(std::move(found.error()));
    }
    return checkNames(commandList, texture, texture);
}

core::Result<void> NullDevice::copyTextureToBuffer(CommandListHandle commandList, TextureHandle texture,
                                                   BufferHandle buffer, std::span<const BufferTextureCopy>) {
    if (auto found = bufferFor(buffer); !found) {
        return std::unexpected(std::move(found.error()));
    }
    return checkNames(commandList, texture, texture);
}

core::Result<void> NullDevice::copyTexture(CommandListHandle commandList, TextureHandle src, TextureHandle dst,
                                           std::span<const TextureCopy>) {
    return checkNames(commandList, src, dst);
}

core::Result<void> NullDevice::checkNames(CommandListHandle commandList, TextureHandle first, TextureHandle second) {
    if (auto list = commandListFor(commandList); !list) {
        return std::unexpected(std::move(list.error()));
    }
    for (const TextureHandle texture : {first, second}) {
        if (auto found = textureFor(texture); !found) {
            return std::unexpected(std::move(found.error()));
        }
    }
    return {};
}

core::Result<MemoryBudget> NullDevice::budget() const {
    // No heaps, so nothing to budget.
    MemoryBudget budget;
    for (const MemoryHeap& heap : instance_->adapters()[0].caps.memory.heaps) {
        budget.heaps.push_back({.budget = heap.size, .usage = 0});
    }
    return budget;
}

} // namespace bfr::rhi::null
