#pragma once

#include "bfr/core/handle.h"

#include <array>
#include <cstdint>

namespace bfr::rhi {

/// Commands recorded for the GPU, from Device::beginCommands until submit
/// sends them. Only named, by a CommandListHandle.
struct CommandList;

using CommandListHandle = core::Handle<CommandList>;

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
