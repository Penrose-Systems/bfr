#pragma once

#include "error.h"

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <expected>
#include <format>
#include <string>
#include <utility>

namespace bfr::rhi::vulkan {

/// A Vulkan call that failed, and what it returned.
struct VulkanFailure {
    std::string call;
    VkResult result = VK_SUCCESS;

    /// "call: VK_ERROR_...".
    [[nodiscard]] std::string message() const { return std::format("{}: {}", call, string_VkResult(result)); }
};

/// The code for a call that failed with result: running out of memory, a lost
/// device and a lost surface keep codes of their own, since the caller deals
/// with each differently; anything else is code.
[[nodiscard]] inline core::ErrorCode codeFor(core::ErrorCode code, VkResult result) {
    switch (result) {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return core::ErrorCode::RhiOutOfMemory;
        case VK_ERROR_DEVICE_LOST:
            return core::ErrorCode::RhiDeviceLost;
        case VK_ERROR_SURFACE_LOST_KHR:
            return core::ErrorCode::RhiSurfaceLost;
        default:
            return code;
    }
}

/// The error for failed, coded as codeFor says.
[[nodiscard]] inline std::unexpected<core::Error> failure(core::ErrorCode code, const VulkanFailure& failed) {
    return core::failure(codeFor(code, failed.result), failed.message());
}

} // namespace bfr::rhi::vulkan
