#include "error.h"
#include "instance_table.h"
#include "vulkan_surface.h"
#include "window.h"

#include <catch2/catch_test_macros.hpp>
#include <vulkan/vulkan_core.h>

using namespace bfr::rhi::vulkan;

TEST_CASE("Vulkan surface fails when the instance lacks the window system's extension", "[rhi][unit]") {
    const InstanceTable vk;
    const auto surface = createSurface(VK_NULL_HANDLE, vk, {}, bfr::core::window::Headless{});

    REQUIRE_FALSE(surface.has_value());
    CHECK(surface.error().code == bfr::core::ErrorCode::RhiSurfaceFailed);
    CHECK(surface.error().message == "VK_EXT_headless_surface is not available");
}
