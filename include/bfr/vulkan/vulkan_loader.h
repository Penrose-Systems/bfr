#pragma once

#include "error.h"

namespace bfr::rhi::vulkan {

/// Opens the system's Vulkan loader and points volk's global functions at it.
/// The library stays loaded for the life of the process, because every
/// instance shares those functions. Only the first call opens it; later calls,
/// from any thread, return once it has. Fails with RhiLoaderFailed, giving the
/// OS's reason for each library tried.
[[nodiscard]] core::Result<void> openLoader();

} // namespace bfr::rhi::vulkan
