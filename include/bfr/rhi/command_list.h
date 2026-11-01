#pragma once

#include "bfr/bfr.h"
#include "bfr/core/handle.h"

#include <array>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Where command lists are recorded from: the memory they record into, given
/// back by Device::resetCommandPool once the GPU has run them. Used by one
/// thread at a time, its lists included. Only named, by a CommandPoolHandle.
struct CommandPool;

using CommandPoolHandle = core::Handle<CommandPool>;

/// Commands recorded for the GPU, from Device::beginCommands until submit
/// sends them. Only named, by a CommandListHandle.
struct CommandList;

using CommandListHandle = core::Handle<CommandList>;

/// What a queue runs.
enum class QueueType : std::uint8_t {
    Graphics,
    Compute,
    Copy,
};

/// How to open a device: the queues it opens beside its graphics queue, and
/// how many command lists, command pools and swapchains it holds at once. The
/// device's queues are numbered: graphics is 0, then the compute queues, then
/// the copy queues. Room for every list, pool and swapchain is made with the
/// device, so nothing on the frame path allocates; past a maximum, calls fail
/// with RhiLimitReached, except a submit of more than maxCommandLists lists,
/// which fails with RhiInvalidCall. The maximums start at bfr.h's defaults,
/// which api/bfr.yaml owns.
struct DeviceDesc {
    std::uint32_t compute = 0;
    std::uint32_t copy = 0;
    std::uint32_t maxCommandLists = BFR_DEFAULT_MAX_COMMAND_LISTS;
    std::uint32_t maxCommandPools = BFR_DEFAULT_MAX_COMMAND_POOLS;
    std::uint32_t maxSwapchains = BFR_DEFAULT_MAX_SWAPCHAINS;
};

/// How to create a command pool: the queue its lists record for, and how many
/// lists it holds, all made with it. beginCommands hands them out in turn
/// until resetCommandPool gives them back.
struct CommandPoolDesc {
    std::uint32_t queue = 0;
    std::uint32_t listCount = 0;
};

/// A point on a queue's timeline: submit returns one, and the GPU passes it
/// once what was submitted has run. A default token has already completed.
struct Token {
    /// Index of the queue on the device; 0 is the graphics queue.
    std::uint32_t queue = 0;
    std::uint64_t value = 0;
    constexpr bool operator==(const Token&) const = default;
};

/// What Device::submit sends to one queue: lists recorded from pools on that
/// queue, run in this order after the GPU passes every wait.
struct SubmitDesc {
    std::uint32_t queue = 0;
    std::span<const CommandListHandle> commandLists;
    std::span<const Token> waits;
};

/// What an attachment holds when rendering begins.
enum class LoadOp : std::uint8_t {
    /// What the image held before.
    Load,
    /// The clear color.
    Clear,
    /// Anything; the renderer writes every pixel.
    DontCare,
};

/// What becomes of an attachment when rendering ends.
enum class StoreOp : std::uint8_t {
    /// What was rendered is kept.
    Store,
    /// It may be dropped.
    DontCare,
};

/// How to begin rendering to a swapchain's acquired image.
struct RenderingDesc {
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    /// Red, green, blue and alpha, 0 to 1, when loadOp is Clear.
    std::array<float, 4> clearColor{};
};

} // namespace bfr::rhi
