#include "null/null_device.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/swapchain.h"
#include "null/null_rhi.h"
#include "rhi/swapchain_build.h"

#include <cstdint>
#include <expected>
#include <format>
#include <utility>
#include <variant>

namespace bfr::rhi::null {

NullDevice::NullDevice(NullRhi& rhi)
    : rhi_(&rhi),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()),
      commandLists_("command lists", kCommandListLimit, kCommandListLimit, rhi.logger()) {}

NullDevice::~NullDevice() {
    // Swapchains are the device's children, so they go first.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(rhi_->logger(), "{} swapchains were left when the device went", swapchains_.size());
    }
    swapchains_.clear();
}

core::Result<SwapchainHandle> NullDevice::createSwapchain(const core::window::NativeWindow& window,
                                                          const SwapchainRequest& request) {
    if (std::holds_alternative<std::monostate>(window)) {
        return core::failure(core::ErrorCode::RhiSurfaceFailed, "there is no window");
    }
    // The size is not asked: the window can change it before the first build.
    if (auto build = swapchainBuildFor(nullSurfaceCaps(), nullSurfaceFormats(), {}, request); !build) {
        return std::unexpected(std::move(build.error()));
    }
    const SwapchainHandle swapchain = swapchains_.emplace(NullSwapchain{
        .request = request,
        .images = {.width = 0,
                   .height = 0,
                   .count = 0,
                   .format = request.format,
                   .colorSpace = request.colorSpace,
                   .presentMode = request.presentMode,
                   .alphaMode = request.alphaMode},
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
    const auto build = swapchainBuildFor(nullSurfaceCaps(), nullSurfaceFormats(), {.width = width, .height = height},
                                         chain.request);
    if (!build) {
        return std::unexpected(build.error());
    }
    // A zero extent is a window with no area, which leaves the swapchain empty.
    const bool empty = build->extent.width == 0 || build->extent.height == 0;
    chain.images.width = build->extent.width;
    chain.images.height = build->extent.height;
    chain.images.count = empty ? 0 : build->imageCount;
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

core::Result<CommandListHandle> NullDevice::beginCommands() {
    const CommandListHandle commandList = commandLists_.emplace();
    if (commandList.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("{} command lists are already recording; submit or destroy one",
                                         kCommandListLimit));
    }
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

core::Result<void> NullDevice::submit(CommandListHandle commandList) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if ((*list)->rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "rendering is still open; end it first");
    }
    // Nothing runs; the list is done.
    commandLists_.erase(commandList);
    return {};
}

core::Result<void> NullDevice::destroyCommandList(CommandListHandle commandList) {
    if (commandList.isNull()) {
        return {};
    }
    if (commandLists_.erase(commandList)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the command list handle is stale");
}

core::Result<MemoryBudget> NullDevice::budget() const {
    // No heaps, so nothing to budget.
    MemoryBudget budget;
    for (const MemoryHeap& heap : rhi_->adapters()[0].caps.memory.heaps) {
        budget.heaps.push_back({.budget = heap.size, .usage = 0});
    }
    return budget;
}

} // namespace bfr::rhi::null
