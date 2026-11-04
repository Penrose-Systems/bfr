#pragma once

#include "bfr/bfr.h"
#include "bfr/core/handle.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/texture.h"

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

/// How to open a device: the queues it opens beside its graphics queue, how
/// many command lists, command pools and swapchains it holds at once, and how
/// many slots each bindless table has. The device's queues are numbered:
/// graphics is 0, then the compute queues, then the copy queues. Room for
/// every list, pool, swapchain and slot is made with the device, so nothing on
/// the frame path allocates; past a maximum, calls fail with RhiLimitReached,
/// except a submit of more than maxCommandLists lists, which fails with
/// RhiInvalidCall. Every maximum is the caller's to set, since only the caller
/// knows what it uses; 0 is none. A table larger than the adapter holds fails
/// the device with RhiLimitReached; BindlessLimits says what it holds.
struct DeviceDesc {
    std::uint32_t compute = 0;
    std::uint32_t copy = 0;
    std::uint32_t maxCommandLists = 0;
    std::uint32_t maxCommandPools = 0;
    std::uint32_t maxSwapchains = 0;
    std::uint32_t maxBindlessTextures = 0;
    std::uint32_t maxBindlessStorageTextures = 0;
    std::uint32_t maxBindlessStorageBuffers = 0;
    std::uint32_t maxBindlessSamplers = 0;
};

/// How to create a command pool: the queue its lists record for, how many
/// lists it holds, all made with it, and the most pushConstants calls its
/// lists record between resets, which a backend that copies push constants
/// into memory makes room for with the pool; 0 is none. beginCommands hands the lists out
/// in turn until resetCommandPool gives them back.
struct CommandPoolDesc {
    std::uint32_t queue = 0;
    std::uint32_t listCount = 0;
    std::uint32_t pushConstantUpdates = 0;
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

/// One color target of a rendering: a texture with colorAttachment usage, in
/// layout ColorTarget, or, as attachment 0 when beginRendering is given a
/// swapchain, that swapchain's acquired image, with texture null.
struct ColorAttachment {
    TextureHandle texture;
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    /// Red, green, blue and alpha, 0 to 1, when loadOp is Clear.
    std::array<float, 4> clearColor{};
};

/// The depth-stencil target of a rendering: a texture with depthStencil
/// usage, in layout DepthStencilTarget, or none when texture is null. Its
/// depth and stencil are loaded and stored alike.
struct DepthStencilAttachment {
    TextureHandle texture;
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    /// 0 to 1, when loadOp is Clear.
    float clearDepth = 0;
    std::uint32_t clearStencil = 0;
};

/// How to begin rendering: the attachments drawn to, all of one size and
/// sample count, which the graphics pipelines drawn with were made for. The
/// first colorAttachmentCount color attachments are drawn to, in the order
/// the pipeline's targets are.
struct RenderingDesc {
    std::uint32_t colorAttachmentCount = 0;
    std::array<ColorAttachment, kMaxColorTargets> colorAttachments{};
    DepthStencilAttachment depthStencil;
};

/// Where on the attachments a rendering lands, in pixels from the top left,
/// and the depth range it maps to. Clip space is the same on every backend: x
/// right, y up, z 0 at the near plane and 1 at the far.
struct Viewport {
    float x = 0;
    float y = 0;
    float width = 0;
    float height = 0;
    float minDepth = 0;
    float maxDepth = 0;
};

/// A rectangle of pixels from the top left of the attachments.
struct Rect {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

/// How wide each index of an index buffer is. The values are the C API's.
enum class IndexType : std::uint8_t {
    Uint32 = 0,
    Uint16 = 1,
};

} // namespace bfr::rhi
