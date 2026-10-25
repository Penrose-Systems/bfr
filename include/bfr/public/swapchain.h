#pragma once

#include "handle.h"

#include <cstdint>

namespace bfr::rhi {

/// The images a device presents to one surface in turn. Only named, by a
/// SwapchainHandle.
struct Swapchain;

using SwapchainHandle = core::Handle<Swapchain>;

/// A swapchain's images: their size in pixels and how many there are. All
/// zero while the swapchain is empty.
struct SwapchainImages {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t count = 0;
};

} // namespace bfr::rhi
