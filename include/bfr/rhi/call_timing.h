#pragma once

#include <cstddef>
#include <cstdint>

namespace bfr::rhi {

/// Each call of Device and Instance that call timings count, named as the
/// call is, an Instance's prefixed by Instance.
enum class Call : std::uint32_t {
    // Device calls.
    CreateSwapchain = 0,
    ResizeSwapchain = 1,
    DestroySwapchain = 2,
    Acquire = 3,
    Present = 4,
    CreateCommandPool = 5,
    ResetCommandPool = 6,
    DestroyCommandPool = 7,
    BeginCommands = 8,
    BeginRendering = 9,
    EndRendering = 10,
    SetViewport = 11,
    SetScissor = 12,
    SetBlendConstants = 13,
    SetStencilReference = 14,
    BindPipeline = 15,
    PushConstants = 16,
    BindIndexBuffer = 17,
    Draw = 18,
    DrawIndexed = 19,
    Dispatch = 20,
    CopyBuffer = 21,
    Barrier = 22,
    TextureBarrier = 23,
    CopyBufferToTexture = 24,
    CopyTextureToBuffer = 25,
    CopyTexture = 26,
    Submit = 27,
    Wait = 28,
    Poll = 29,
    DestroyCommandList = 30,
    CreateBuffer = 31,
    BufferSlot = 32,
    MapBuffer = 33,
    DestroyBuffer = 34,
    CreateTexture = 35,
    TextureSlots = 36,
    DestroyTexture = 37,
    CreateSampler = 38,
    SamplerSlot = 39,
    DestroySampler = 40,
    CreateComputePipeline = 41,
    CreateGraphicsPipeline = 42,
    DestroyPipeline = 43,
    Budget = 44,
    CreateTimestampPool = 45,
    DestroyTimestampPool = 46,
    ResetTimestamps = 47,
    WriteTimestamp = 48,
    ResolveTimestamps = 49,
    TimestampRate = 50,
    CalibrateTimestamps = 51,
    // Instance calls.
    InstanceAdapters = 52,
    InstanceCreateDevice = 53,
    InstanceFormatCaps = 54,
    InstanceSwapchainCaps = 55,
    InstanceSwapchainFormats = 56,
};

/// How many values Call has.
inline constexpr std::size_t kCallCount = 57;

/// The CPU time one kind of call has taken, read from std::chrono::steady_clock
/// around each call: how many there were, their nanoseconds in all, and the
/// longest one's.
struct CallTiming {
    std::uint64_t count = 0;
    std::uint64_t totalNs = 0;
    std::uint64_t maxNs = 0;
};

} // namespace bfr::rhi
