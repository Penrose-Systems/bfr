#include "null/null_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/swapchain.h"
#include "null/null_instance.h"

#include <cstdint>
#include <expected>
#include <format>
#include <utility>
#include <variant>
#include <vector>

namespace bfr::rhi::null {

NullDevice::NullDevice(NullInstance& rhi, const DeviceQueues& queues)
    : instance_(&rhi),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandPools_("command pools", kCommandPoolCapacity, kCommandPoolLimit, rhi.logger()),
      commandLists_("command lists", kCommandListCapacity, kCommandListLimit, rhi.logger()) {
    queues_.push_back(QueueType::Graphics);
    queues_.insert(queues_.end(), queues.compute, QueueType::Compute);
    queues_.insert(queues_.end(), queues.copy, QueueType::Copy);
    submitted_.assign(queues_.size(), 0);
}

NullDevice::~NullDevice() {
    // Swapchains are the device's children, so they go first.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(instance_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    swapchains_.clear();
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
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("a device holds at most {} swapchains", kSwapchainLimit));
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
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("the device has no queue {}; it has {}", queue, queues_.size()));
    }
    return {};
}

core::Result<void> NullDevice::checkToken(Token token) const {
    if (auto checked = checkQueue(token.queue); !checked) {
        return checked;
    }
    if (token.value > submitted_[token.queue]) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("no submit has returned token {} on queue {}; the last is {}", token.value,
                                         token.queue, submitted_[token.queue]));
    }
    return {};
}

core::Result<CommandPoolHandle> NullDevice::createCommandPool(std::uint32_t queue) {
    if (auto checked = checkQueue(queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const CommandPoolHandle commandPool = commandPools_.emplace(NullCommandPool{.queue = queue, .recording = {}});
    if (commandPool.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command pools", kCommandPoolLimit));
    }
    return commandPool;
}

core::Result<void> NullDevice::resetCommandPool(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    if (!(*pool)->recording.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             std::format("{} lists from the pool are still recording; submit or destroy them",
                                         (*pool)->recording.size()));
    }
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
    for (const CommandListHandle list : (*pool)->recording) {
        commandLists_.erase(list);
    }
    commandPools_.erase(commandPool);
    return {};
}

core::Result<CommandListHandle> NullDevice::beginCommands(CommandPoolHandle commandPool) {
    auto pool = commandPoolFor(commandPool);
    if (!pool) {
        return std::unexpected(std::move(pool.error()));
    }
    const CommandListHandle commandList = commandLists_.emplace(
        NullCommandList{.pool = commandPool, .rendering = false});
    if (commandList.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("the device holds at most {} command lists", kCommandListLimit));
    }
    (*pool)->recording.push_back(commandList);
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

core::Result<Token> NullDevice::submit(const SubmitDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    for (const Token wait : desc.waits) {
        if (auto checked = checkToken(wait); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    for (const CommandListHandle commandList : desc.commandLists) {
        auto list = commandListFor(commandList);
        if (!list) {
            return std::unexpected(std::move(list.error()));
        }
        if ((*list)->rendering) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
        }
        const NullCommandPool* pool = commandPools_.find((*list)->pool);
        if (pool != nullptr && pool->queue != desc.queue) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 std::format("the list was recorded for queue {}, not {}", pool->queue, desc.queue));
        }
    }
    for (const CommandListHandle commandList : desc.commandLists) {
        if (NullCommandList* list = commandLists_.find(commandList); list != nullptr) {
            if (NullCommandPool* pool = commandPools_.find(list->pool); pool != nullptr) {
                std::erase(pool->recording, commandList);
            }
        }
        commandLists_.erase(commandList);
    }
    return Token{.queue = desc.queue, .value = ++submitted_[desc.queue]};
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
    if (NullCommandPool* pool = commandPools_.find((*list)->pool); pool != nullptr) {
        std::erase(pool->recording, commandList);
    }
    commandLists_.erase(commandList);
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
