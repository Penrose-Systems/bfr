#pragma once

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/call_timing.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/sampler.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"
#include "bfr/rhi/timestamp.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// A logical GPU on one of the Instance's adapters.
class Device {
public:
    /// Waits for the GPU to finish its work, then releases the device and the
    /// swapchains left on it.
    virtual ~Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    /// An empty swapchain on window, which must stay open until the swapchain
    /// is destroyed; resizeSwapchain builds its images. Fails with
    /// RhiWindowFailed when there is no window, when its window system's
    /// surface extension is unavailable, or when the API refuses the window,
    /// RhiPresentUnsupported when the device or desc.queue can't present to
    /// it, RhiInvalidCall when the device has no queue desc.queue or an enum
    /// in desc is not one BFR has, and RhiLimitReached when the device holds
    /// maxSwapchains already.
    [[nodiscard]] virtual core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                        const SwapchainDesc& desc) = 0;

    /// Builds the swapchain's images at width by height pixels, replacing any
    /// it has once the GPU is done with them. The window system may set the size
    /// itself. A window with no area leaves the swapchain empty, which is not a
    /// failure. Builds as the desc says. Fails with RhiInvalidHandle when
    /// swapchain is destroyed, RhiUnsupported when the backend has no setting
    /// for a value of the desc, and RhiSwapchainFailed when the API refuses;
    /// a failure leaves it empty.
    [[nodiscard]] virtual core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                        std::uint32_t height) = 0;

    /// Destroys the swapchain once the GPU is done with its images, before
    /// returning: its window is then free to close. A null handle does
    /// nothing. Fails with RhiInvalidHandle when the swapchain is already
    /// destroyed.
    [[nodiscard]] virtual core::Result<void> destroySwapchain(SwapchainHandle swapchain) = 0;

    /// Takes the swapchain's next image, for beginRendering, and gives what
    /// the backend said of the swapchain as it did. timeoutNs goes to the
    /// backend's acquire; it is unused where acquire doesn't wait, and where
    /// the backend can only wait with no limit or for a time of its own, the
    /// largest value waits with no limit and any other for the backend's time.
    /// Fails with RhiTimeout when no image came in time, RhiOutOfDate when the
    /// swapchain has no images or no longer matches its window, which
    /// resizeSwapchain mends, RhiInvalidCall when an image is already
    /// acquired, and RhiInvalidHandle when the swapchain is destroyed.
    [[nodiscard]] virtual core::Result<SwapchainStatus> acquire(SwapchainHandle swapchain, std::uint64_t timeoutNs) = 0;

    /// Shows the acquired image once what submit sent for it is done, and
    /// gives what the backend said of the swapchain as it did. Fails
    /// with RhiOutOfDate when the swapchain no longer matches its window, the
    /// image still presented when it could be, and RhiInvalidCall when no image
    /// is acquired.
    [[nodiscard]] virtual core::Result<SwapchainStatus> present(SwapchainHandle swapchain) = 0;

    /// A command pool as desc says, with its lists made. Fails with
    /// RhiInvalidCall for a queue the device does not have, RhiLimitReached at
    /// the device's maximum of pools or when fewer than desc.listCount of its
    /// lists are free, and RhiDeviceFailed when the API refuses.
    [[nodiscard]] virtual core::Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc) = 0;

    /// Gives every list the pool made back to it, so beginCommands may hand
    /// them out and use their memory again. A list from the pool still
    /// recording ends with it on Vulkan; on D3D12 and Metal it fails with
    /// RhiInvalidCall. Fails with RhiInvalidCall while the GPU has not passed
    /// the token of the pool's last submit, and RhiInvalidHandle when the pool
    /// is destroyed.
    [[nodiscard]] virtual core::Result<void> resetCommandPool(CommandPoolHandle commandPool) = 0;

    /// Destroys the pool and every list it made once the GPU has finished
    /// with them; the handles are invalid at once. A null handle does
    /// nothing. Fails with RhiInvalidHandle when the pool is already destroyed.
    [[nodiscard]] virtual core::Result<void> destroyCommandPool(CommandPoolHandle commandPool) = 0;

    /// Starts recording the pool's next list. Fails with RhiInvalidHandle when
    /// the pool is destroyed, RhiLimitReached when every list it holds was
    /// handed out since it was last reset, and RhiDeviceFailed when the API
    /// refuses.
    [[nodiscard]] virtual core::Result<CommandListHandle> beginCommands(CommandPoolHandle commandPool) = 0;

    /// Begins rendering to desc's attachments, with the image acquire took
    /// from swapchain as color attachment 0 when swapchain is not null, and
    /// sets the viewport and scissor to the whole of them; ends with
    /// endRendering. Only draw and drawIndexed, and the state they take, are
    /// recorded in between. Fails with RhiInvalidCall when no attachment is
    /// given, when a color attachment but the swapchain's names no texture,
    /// when a swapchain is given with no color attachment or with attachment 0
    /// naming a texture, when no image is acquired, when it was already
    /// submitted, and RhiInvalidHandle when a
    /// texture is gone.
    [[nodiscard]] virtual core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                            const RenderingDesc& desc) = 0;

    /// Ends the rendering beginRendering began; a swapchain image is then
    /// ready to present, and a texture is in the layout it was given in.
    /// Fails with RhiInvalidCall on Metal when none is open, as it has no
    /// encoder to end.
    [[nodiscard]] virtual core::Result<void> endRendering(CommandListHandle commandList) = 0;

    /// Sets the viewport the draws after it use. Fails with RhiInvalidCall
    /// outside rendering on Metal, which sets it on the rendering's encoder,
    /// and RhiInvalidHandle when the list is gone.
    [[nodiscard]] virtual core::Result<void> setViewport(CommandListHandle commandList, const Viewport& viewport) = 0;

    /// Sets the scissor the draws after it keep to. Fails as setViewport does.
    [[nodiscard]] virtual core::Result<void> setScissor(CommandListHandle commandList, const Rect& scissor) = 0;

    /// Sets the color, red, green, blue and alpha, the constant blend factors
    /// of the draws after it read. Vulkan has no color until it is set, and
    /// Metal sets it on the rendering's encoder, so a draw that reads it needs
    /// it set in its rendering. Fails as setViewport does.
    [[nodiscard]] virtual core::Result<void> setBlendConstants(CommandListHandle commandList,
                                                               const std::array<float, 4>& color) = 0;

    /// Sets the value the stencil test of the draws after it compares against,
    /// and StencilOp::Replace writes, on both faces; the low 8 bits count.
    /// Vulkan has no value until it is set, and Metal sets it on the
    /// rendering's encoder, so a draw that tests stencil needs it set in its
    /// rendering. Fails as setViewport does.
    [[nodiscard]] virtual core::Result<void> setStencilReference(CommandListHandle commandList,
                                                                 std::uint32_t reference) = 0;

    /// Makes pipeline the one draws or dispatches after it run, a graphics
    /// pipeline for draws and a compute one for dispatches, until another of
    /// its kind is bound. Draws assemble the list topology of the pipeline's
    /// class. Fails with RhiInvalidHandle when the list or pipeline is gone.
    [[nodiscard]] virtual core::Result<void> bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) = 0;

    /// Sets the first data.size() bytes of the push constants every shader
    /// reads, for the draws and dispatches after it. Fails with RhiInvalidCall
    /// when the size is not a multiple of 4 or is more than the caps'
    /// limits.maxPushConstantsSize,
    /// RhiLimitReached past the pool's pushConstantUpdates, and
    /// RhiInvalidHandle when the list is gone.
    [[nodiscard]] virtual core::Result<void> pushConstants(CommandListHandle commandList,
                                                           std::span<const std::byte> data) = 0;

    /// Makes buffer, which has index usage, the one drawIndexed reads indices
    /// of type from, starting offset bytes in. Fails with RhiInvalidHandle when
    /// the list or buffer is gone.
    [[nodiscard]] virtual core::Result<void> bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
                                                             std::uint64_t offset, IndexType type) = 0;

    /// Records a draw of vertexCount vertices, instanceCount times, with the
    /// bound graphics pipeline, inside rendering. The vertex shader sees
    /// vertex indices from firstVertex; whether it sees instance indices from
    /// firstInstance or from 0 is the backend's, as api/bfr.yaml says under
    /// bfr_draw. Fails with RhiInvalidCall outside rendering on Metal, which
    /// draws on the rendering's encoder, and RhiInvalidHandle when the list
    /// or the bound pipeline is gone.
    [[nodiscard]] virtual core::Result<void> draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                                  std::uint32_t instanceCount, std::uint32_t firstVertex,
                                                  std::uint32_t firstInstance) = 0;

    /// Records a draw of indexCount indices read from the bound index buffer
    /// from firstIndex, each plus vertexOffset giving the vertex index the
    /// vertex shader sees, instanceCount times, inside rendering. Fails as
    /// draw does.
    [[nodiscard]] virtual core::Result<void> drawIndexed(CommandListHandle commandList, std::uint32_t indexCount,
                                                         std::uint32_t instanceCount, std::uint32_t firstIndex,
                                                         std::int32_t vertexOffset, std::uint32_t firstInstance) = 0;

    /// Records a dispatch of x by y by z workgroups of the bound compute
    /// pipeline, outside rendering; commands after a barrier see what it
    /// wrote. Fails with RhiInvalidHandle when the list or the bound pipeline
    /// is gone.
    [[nodiscard]] virtual core::Result<void> dispatch(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                                      std::uint32_t z) = 0;

    /// Records copies of regions from src to dst, outside rendering. src has
    /// copySrc usage and is device or upload memory; dst has copyDst usage and
    /// is device or readback memory. Commands after a barrier in the list, or
    /// in a later submit that waits on its token, see what was copied. Fails
    /// with RhiInvalidHandle when the list or a buffer is gone.
    [[nodiscard]] virtual core::Result<void> copyBuffer(CommandListHandle commandList, BufferHandle src,
                                                        BufferHandle dst, std::span<const BufferCopy> regions) = 0;

    /// Records a barrier, outside rendering: commands recorded after it in the
    /// list start once those before it have finished, and see what they
    /// wrote. Fails with RhiInvalidHandle when the list is gone.
    [[nodiscard]] virtual core::Result<void> barrier(CommandListHandle commandList) = 0;

    /// Records a barrier, as barrier does, that also moves every layer and
    /// mip level of texture from layout before to layout after. Outside
    /// rendering. Fails with RhiInvalidHandle when the list or texture is gone.
    [[nodiscard]] virtual core::Result<void> textureBarrier(CommandListHandle commandList, TextureHandle texture,
                                                            TextureLayout before, TextureLayout after) = 0;

    /// Records copies of regions from buffer to texture, which is in layout
    /// CopyDst, outside rendering; api/bfr.yaml states the alignment the
    /// regions keep. Seen as copyBuffer's are. Fails with RhiInvalidHandle
    /// when the list, buffer or texture is gone.
    [[nodiscard]] virtual core::Result<void> copyBufferToTexture(CommandListHandle commandList, BufferHandle buffer,
                                                                 TextureHandle texture,
                                                                 std::span<const BufferTextureCopy> regions) = 0;

    /// Records copies of regions from texture, which is in layout CopySrc, to
    /// buffer, outside rendering. Fails as copyBufferToTexture does.
    [[nodiscard]] virtual core::Result<void> copyTextureToBuffer(CommandListHandle commandList, TextureHandle texture,
                                                                 BufferHandle buffer,
                                                                 std::span<const BufferTextureCopy> regions) = 0;

    /// Records copies of regions from src, in layout CopySrc, to dst, in
    /// layout CopyDst, of one format and sample count, outside rendering.
    /// Fails with RhiInvalidHandle when the list or a texture is gone.
    [[nodiscard]] virtual core::Result<void> copyTexture(CommandListHandle commandList, TextureHandle src,
                                                         TextureHandle dst, std::span<const TextureCopy> regions) = 0;

    /// Sends desc's command lists to its queue, which runs them after what it
    /// was sent before and after the GPU passes the waits, ends the lists, and
    /// returns the token their completion signals. Calls for one queue must
    /// not overlap. Fails with RhiInvalidCall for a queue the device does not
    /// have or a list from a pool on another queue, and RhiInvalidHandle for a
    /// list that is gone. A wait on a token
    /// no submit has returned yet waits until one does.
    [[nodiscard]] virtual core::Result<Token> submit(const SubmitDesc& desc) = 0;

    /// Blocks until the GPU has passed token, which may be one no submit has
    /// returned yet, or until timeoutNs has passed. timeoutNs goes to the
    /// backend's wait; one that counts in milliseconds is given them rounded
    /// up, and on D3D12 one past its longest count waits with no limit. Fails
    /// with RhiTimeout when the time runs out, RhiInvalidCall for a token on a
    /// queue the device does not have, and RhiDeviceFailed when the wait fails.
    [[nodiscard]] virtual core::Result<void> wait(Token token, std::uint64_t timeoutNs) = 0;

    /// Whether the GPU has passed token, without blocking. Fails as wait does.
    [[nodiscard]] virtual core::Result<bool> poll(Token token) = 0;

    /// Discards a command list that was not submitted; the pool hands it out
    /// again after its next reset. A null handle does nothing. Fails with RhiInvalidHandle when the list is already
    /// gone.
    [[nodiscard]] virtual core::Result<void> destroyCommandList(CommandListHandle commandList) = 0;

    /// A buffer as desc says, with a slot in the storage buffer table when
    /// its usage is storage. Cold: it frees buffers the GPU is done with, so
    /// it is not for the frame path. Fails with RhiInvalidCall for a desc of
    /// no bytes or no usage, RhiOutOfMemory when the API says the memory ran
    /// out, RhiLimitReached at the device's maximum of buffers or of storage
    /// buffer slots, and RhiDeviceFailed when the API refuses the desc or
    /// gives no buffer.
    [[nodiscard]] virtual core::Result<BufferHandle> createBuffer(const BufferDesc& desc) = 0;

    /// The buffer's index in the storage buffer table, held from creation to
    /// destruction. Fails with RhiInvalidCall when the buffer has no storage
    /// usage and RhiInvalidHandle when it is destroyed.
    [[nodiscard]] virtual core::Result<std::uint32_t> bufferSlot(BufferHandle buffer) = 0;

    /// Where the CPU reaches an upload or readback buffer, borrowed until the
    /// buffer is destroyed. CPU writes reach the GPU, and GPU writes reach the
    /// CPU once their token completes, without a flush. Fails with
    /// RhiInvalidCall for device memory and RhiInvalidHandle when the buffer
    /// is destroyed.
    [[nodiscard]] virtual core::Result<std::span<std::byte>> mapBuffer(BufferHandle buffer) = 0;

    /// Destroys the buffer once the GPU has passed every submit made before
    /// the call; the handle is invalid at once. Cold: it records the point
    /// the buffer waits for and frees what the GPU is done with, so it is not
    /// for the frame path. A null handle does nothing. Fails with
    /// RhiInvalidHandle when the buffer is already destroyed.
    [[nodiscard]] virtual core::Result<void> destroyBuffer(BufferHandle buffer) = 0;

    /// A texture as desc says, in device memory, with a slot in each bindless
    /// table its usage names. Cold: it frees textures the GPU is done with.
    /// Fails with RhiOutOfMemory when the API says memory ran out,
    /// RhiLimitReached at the device's maximum of textures or of a table's
    /// slots, and RhiDeviceFailed when the API refuses the desc or gives no
    /// texture.
    [[nodiscard]] virtual core::Result<TextureHandle> createTexture(const TextureDesc& desc) = 0;

    /// Where the texture is in the bindless tables. Fails with
    /// RhiInvalidHandle when it is destroyed.
    [[nodiscard]] virtual core::Result<TextureSlots> textureSlots(TextureHandle texture) = 0;

    /// Destroys the texture once the GPU has passed every submit made before
    /// the call; the handle is invalid at once. Cold, as destroyBuffer is. A
    /// null handle does nothing. Fails with RhiInvalidHandle when the texture
    /// is already destroyed.
    [[nodiscard]] virtual core::Result<void> destroyTexture(TextureHandle texture) = 0;

    /// A sampler as desc says, with a slot in the sampler table. Cold: it
    /// frees samplers the GPU is done with. Fails with RhiUnsupported when
    /// the backend has no setting for a member, RhiLimitReached when the
    /// sampler table is full, and RhiDeviceFailed when the API refuses the
    /// desc.
    [[nodiscard]] virtual core::Result<SamplerHandle> createSampler(const SamplerDesc& desc) = 0;

    /// The sampler's index in the sampler table, held from creation to
    /// destruction. Fails with RhiInvalidHandle when it is destroyed.
    [[nodiscard]] virtual core::Result<std::uint32_t> samplerSlot(SamplerHandle sampler) = 0;

    /// Destroys the sampler once the GPU has passed every submit made before
    /// the call; the handle is invalid at once. Cold, as destroyBuffer is. A
    /// null handle does nothing. Fails with RhiInvalidHandle when the sampler
    /// is already destroyed.
    [[nodiscard]] virtual core::Result<void> destroySampler(SamplerHandle sampler) = 0;

    /// A compute pipeline as desc says, made before returning. Cold: it
    /// compiles, and frees pipelines the GPU is done with. Fails with
    /// RhiLimitReached at the device's maximum, and RhiDeviceFailed when the
    /// API refuses the code or the desc, with what it said.
    [[nodiscard]] virtual core::Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc) = 0;

    /// A graphics pipeline as desc says, made before returning. Cold, as
    /// createComputePipeline is. A target of format Undefined is an unused
    /// one. Fails with RhiInvalidCall for a desc with an enum value or format
    /// BFR doesn't have or more than kMaxColorTargets targets, RhiLimitReached at the device's maximum, and
    /// RhiDeviceFailed when the API refuses the code or the desc, with what it
    /// said.
    [[nodiscard]] virtual core::Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc) = 0;

    /// Destroys the pipeline once the GPU has passed every submit made before
    /// the call; the handle is invalid at once. Cold, as destroyBuffer is. A
    /// null handle does nothing. Fails with RhiInvalidHandle when the pipeline
    /// is already destroyed.
    [[nodiscard]] virtual core::Result<void> destroyPipeline(PipelineHandle pipeline) = 0;

    /// How much of each heap the process may use and uses now. Fails with
    /// RhiUnsupported when the driver can't say.
    [[nodiscard]] virtual core::Result<MemoryBudget> budget() const = 0;

    /// A pool of desc.count GPU timestamps, written by lists on desc.queue.
    /// Cold. Fails with RhiInvalidCall when the device has no queue
    /// desc.queue, RhiLimitReached at maxTimestampPools, and RhiDeviceFailed
    /// when the API refuses.
    [[nodiscard]] virtual core::Result<TimestampPoolHandle> createTimestampPool(const TimestampPoolDesc& desc) = 0;

    /// Destroys the pool once the GPU has passed every submit made before the
    /// call; the handle is invalid at once. Cold. A null handle does nothing.
    /// Fails with RhiInvalidHandle when the pool is already destroyed.
    [[nodiscard]] virtual core::Result<void> destroyTimestampPool(TimestampPoolHandle pool) = 0;

    /// Records a reset of count timestamps from first, which a backend that
    /// needs one before a timestamp is written again is given; the others
    /// record nothing. Outside rendering. Fails with RhiInvalidHandle for a
    /// list or pool that is gone.
    [[nodiscard]] virtual core::Result<void> resetTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                             std::uint32_t first, std::uint32_t count) = 0;

    /// Records the GPU's timestamp at stage into the pool's timestamp index,
    /// with granularity where the backend takes one. Fails with
    /// RhiInvalidHandle for a list or pool that is gone, RhiInvalidCall for a
    /// stage or granularity BFR doesn't have, and RhiUnsupported for a stage
    /// the backend has no setting for where the list is, as Caps::timestamps
    /// says.
    [[nodiscard]] virtual core::Result<void> writeTimestamp(CommandListHandle commandList, TimestampPoolHandle pool,
                                                            std::uint32_t index, TimestampStage stage,
                                                            TimestampGranularity granularity) = 0;

    /// Records a copy of count timestamps from first into buffer at offset,
    /// as raw ticks of Caps::timestamps.entryBytes each, once the GPU has
    /// written them. The buffer needs copyDst usage. Outside rendering. Fails
    /// with RhiInvalidHandle for a list, pool or buffer that is gone, and
    /// RhiInvalidCall inside rendering on a backend with nothing to record it
    /// in there.
    [[nodiscard]] virtual core::Result<void> resolveTimestamps(CommandListHandle commandList, TimestampPoolHandle pool,
                                                               std::uint32_t first, std::uint32_t count,
                                                               BufferHandle buffer, std::uint64_t offset) = 0;

    /// How long a timestamp tick lasts on queue, as the backend gives it.
    /// Fails with RhiInvalidCall when the device has no such queue, and
    /// RhiDeviceFailed when the API can't say.
    [[nodiscard]] virtual core::Result<TimestampRate> timestampRate(std::uint32_t queue) const = 0;

    /// A GPU timestamp on queue and clock's reading, taken together. Fails
    /// with RhiInvalidCall when the device has no such queue or clock is not
    /// one BFR has, RhiUnsupported for a clock Caps::timestamps.hostClocks
    /// doesn't list, and RhiDeviceFailed when the API refuses.
    [[nodiscard]] virtual core::Result<CalibratedTimestamps> calibrateTimestamps(std::uint32_t queue,
                                                                                 HostClock clock) const = 0;

    /// The CPU time spent in each of the device's calls since it was made,
    /// written to timings by Call, as many as it holds. Fails with
    /// RhiInvalidCall when InstanceDesc::callTimings was not set.
    [[nodiscard]] virtual core::Result<void> callTimings(std::span<CallTiming> timings) const;

protected:
    Device() = default;
};

} // namespace bfr::rhi
