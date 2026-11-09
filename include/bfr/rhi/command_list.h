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

/// A queue's priority against every queue on the system, other processes'
/// included. Normal is the level a backend's queues have when none is set.
/// Queues::graphicsLevels and the others say which levels each kind takes; one
/// above normal may need privileges the process lacks, and then fails as the
/// backend says.
enum class QueueLevel : std::uint32_t {
    Normal = 0,
    Low = 1,
    High = 2,
    Realtime = 3,
};

/// One queue a device opens. rank, 0 to 1, is its priority against the
/// device's other queues, higher first, passed to the backend as given; where
/// Queues::ranks is 0 the backend doesn't rank a device's queues, so queues of
/// different ranks fail the device with RhiUnsupported. A level the backend
/// has no setting for fails it the same way, as do different levels in one
/// kind on Vulkan, which sets one level for all of a kind's queues.
struct QueueDesc {
    float rank = 0;
    QueueLevel level = QueueLevel::Normal;
};

/// How to open a device: the queues it opens, how many command lists, command
/// pools and swapchains it holds at once, and how many slots each bindless
/// table has. The queue spans are read during createDevice only. The device's
/// queues are numbered: the graphics queues from 0, then the compute queues,
/// then the copy queues, each in its span's order. Room for
/// every list, pool, swapchain and slot is made with the device, so nothing on
/// the frame path allocates; past a maximum, calls fail with RhiLimitReached,
/// except a submit of more than maxCommandLists lists, which fails with
/// RhiInvalidCall. Every maximum is the caller's to set, since only the caller
/// knows what it uses; 0 is none. Each table's size is passed to the backend
/// as given, and one larger than the adapter holds, as BindlessLimits says,
/// is the backend's to refuse, which Vulkan's validation does; D3D12 makes the three resource
/// tables in one heap, so there they fail with RhiLimitReached when together
/// they are more than it holds.
///
/// A graphics queue's lists run every call. A compute queue's run all but
/// rendering, draws, the calls that set their state or bind a graphics
/// pipeline or an index buffer, and resolveTexture; a copy queue's run only
/// copies, barriers and writeTimestamp. A copy between a buffer and a
/// texture's depth or stencil runs only in a graphics queue's lists, and one
/// in a copy queue's has a bufferOffset that is a multiple of 4.
/// TextureLayout says which layouts each kind's lists move a texture between.
struct DeviceDesc {
    std::span<const QueueDesc> graphics;
    std::span<const QueueDesc> compute;
    std::span<const QueueDesc> copy;
    std::uint32_t maxCommandLists = 0;
    std::uint32_t maxCommandPools = 0;
    std::uint32_t maxSwapchains = 0;
    std::uint32_t maxBindlessTextures = 0;
    std::uint32_t maxBindlessStorageTextures = 0;
    std::uint32_t maxBindlessStorageBuffers = 0;
    std::uint32_t maxBindlessSamplers = 0;
    std::uint32_t maxTimestampPools = 0;
    std::uint32_t maxSparseMemories = 0;
    /// The most texture regions and the most buffer regions one
    /// updateTileMappings call takes.
    std::uint32_t maxTileMappings = 0;
    /// The slots of the fifth bindless table, of top acceleration structures;
    /// with none, top structures are made with no slot. Where
    /// BindlessLimits::maxAccelerationStructures is 0 the adapter has no such
    /// table, and more than 0 fails with RhiLimitReached.
    std::uint32_t maxBindlessAccelerationStructures = 0;
    /// The most geometries one build takes, in buildAccelerationStructure
    /// and the size query; a top structure's instances are one geometry.
    std::uint32_t maxBuildGeometries = 0;
    /// The strides, in bytes, the device's indirect draws take, read during
    /// createDevice only: D3D12 fixes a stride in the command signature it
    /// makes for them, so it makes one for each here and fails an indirect
    /// draw of any other with RhiUnsupported. Vulkan and Metal take any
    /// stride as it is given.
    std::span<const std::uint32_t> indirectStrides{};
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
    /// Index of the queue on the device, numbered as DeviceDesc says.
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

/// The arguments of one indirect draw, as a buffer holds them for
/// Device::drawIndirect; the layout every backend reads.
struct DrawIndirectArgs {
    std::uint32_t vertexCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstVertex = 0;
    std::uint32_t firstInstance = 0;
};

/// The arguments of one indirect indexed draw, for Device::drawIndexedIndirect.
struct DrawIndexedIndirectArgs {
    std::uint32_t indexCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t firstInstance = 0;
};

/// The workgroups of one indirect dispatch, for Device::dispatchIndirect.
struct DispatchIndirectArgs {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
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

/// How the samples of a depth or stencil pixel become one when a rendering's
/// depth-stencil attachment is resolved; Caps::resolve says which the device
/// has. The values are the C API's.
enum class ResolveMode : std::uint8_t {
    /// Not resolved.
    None = 0,
    SampleZero = 1,
    Average = 2,
    Min = 3,
    Max = 4,
};

/// One color target of a rendering: a texture with colorAttachment usage, in
/// layout ColorTarget, or, as attachment 0 when beginRendering is given a
/// swapchain, that swapchain's acquired image, with texture null. Given a
/// swapchain, attachment 0 may instead name a multisampled texture of the
/// swapchain's format, which is then resolved into the acquired image.
struct ColorAttachment {
    TextureHandle texture;
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    /// Red, green, blue and alpha, 0 to 1, when loadOp is Clear.
    std::array<float, 4> clearColor{};
    /// The target holds integers: clearColorInteger applies instead. D3D12
    /// clears with floats, so there a value a 32-bit float does not hold
    /// exactly fails with RhiInvalidCall.
    bool clearInteger = false;
    /// Red, green, blue and alpha bits, when loadOp is Clear; a signed format
    /// reads them as int32.
    std::array<std::uint32_t, 4> clearColorInteger{};
    /// A single-sample texture of texture's format and size, with
    /// colorAttachment usage, in layout ColorTarget, that texture's samples
    /// are resolved into as rendering ends; null for none, and null for the
    /// attachment the swapchain's image is resolved into. Non-integer formats
    /// average and integer ones take sample zero, as the backend resolves;
    /// FormatCaps::resolve says which formats resolve.
    TextureHandle resolveTexture{};
};

/// The depth-stencil target of a rendering: a texture with depthStencil
/// usage, in layout DepthStencilTarget, or DepthStencilRead when readOnly, or
/// none when texture is null. Depth and stencil each load and store as their
/// ops say; an op of an aspect the format lacks is ignored.
struct DepthStencilAttachment {
    TextureHandle texture;
    LoadOp depthLoadOp = LoadOp::Load;
    StoreOp depthStoreOp = StoreOp::Store;
    LoadOp stencilLoadOp = LoadOp::Load;
    StoreOp stencilStoreOp = StoreOp::Store;
    /// 0 to 1, when depthLoadOp is Clear.
    float clearDepth = 0;
    /// When stencilLoadOp is Clear. The low 8 bits count.
    std::uint32_t clearStencil = 0;
    /// The rendering tests depth and stencil but never writes them, and
    /// shaders may sample the texture meanwhile; every op is then Load and
    /// Store. The pipelines drawn with write neither depth nor stencil.
    bool readOnly = false;
    /// A single-sample texture of texture's format and size, with
    /// depthStencil usage, in layout DepthStencilTarget, that the samples are
    /// resolved into as rendering ends, by depthResolve and stencilResolve;
    /// null for none.
    TextureHandle resolveTexture{};
    ResolveMode depthResolve = ResolveMode::None;
    ResolveMode stencilResolve = ResolveMode::None;
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
