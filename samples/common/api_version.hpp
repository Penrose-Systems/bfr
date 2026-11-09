#pragma once

// The API version the samples ask each backend for.

#include "bfr.hpp"

#include <cstdint>

namespace sample {

struct ApiVersion {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
};

/// Vulkan 1.4, the newest the samples are written for. D3D12 and Metal read
/// no version yet.
inline ApiVersion apiVersionFor(bfr::Backend backend) {
    constexpr ApiVersion kVulkan{.major = 1, .minor = 4};
    return backend == bfr::Backend::Vulkan ? kVulkan : ApiVersion{};
}

} // namespace sample
