#include "swapchain_settings.h"

#include <catch2/catch_test_macros.hpp>
#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <limits>

using namespace bfr::rhi::vulkan;

namespace {

constexpr std::uint32_t kUndefined = std::numeric_limits<std::uint32_t>::max();
constexpr VkExtent2D kWindow{.width = 800, .height = 600};
constexpr std::uint32_t kImages = 3;
constexpr std::uint32_t kMaxImages = 8;
constexpr std::uint32_t kMaxExtent = 4096;

// A surface that leaves the extent to the swapchain, takes 1 to 8 images and
// composites opaquely.
VkSurfaceCapabilitiesKHR capabilities() {
    VkSurfaceCapabilitiesKHR surface{};
    surface.minImageCount = 1;
    surface.maxImageCount = kMaxImages;
    surface.currentExtent = {.width = kUndefined, .height = kUndefined};
    surface.minImageExtent = {.width = 1, .height = 1};
    surface.maxImageExtent = {.width = kMaxExtent, .height = kMaxExtent};
    surface.supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    return surface;
}

constexpr VkSurfaceFormatKHR format(VkFormat format) {
    return {.format = format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
}

const auto kFormats = std::to_array({format(VK_FORMAT_B8G8R8A8_UNORM)});

} // namespace

TEST_CASE("Swapchain settings take BGRA8 in the sRGB colour space when the surface lists it", "[rhi][unit]") {
    const auto formats = std::to_array({format(VK_FORMAT_R8G8B8A8_UNORM), format(VK_FORMAT_B8G8R8A8_UNORM)});

    const SwapchainSettings settings = chooseSwapchainSettings(capabilities(), formats, kWindow, kImages);

    CHECK(settings.format.format == VK_FORMAT_B8G8R8A8_UNORM);
    CHECK(settings.format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
}

TEST_CASE("Swapchain settings take RGBA8 when the surface lacks BGRA8", "[rhi][unit]") {
    const auto formats = std::to_array({format(VK_FORMAT_A2B10G10R10_UNORM_PACK32), format(VK_FORMAT_R8G8B8A8_UNORM)});

    const SwapchainSettings settings = chooseSwapchainSettings(capabilities(), formats, kWindow, kImages);

    CHECK(settings.format.format == VK_FORMAT_R8G8B8A8_UNORM);
}

TEST_CASE("Swapchain settings take the surface's first format when it lists neither 8-bit format", "[rhi][unit]") {
    const auto formats = std::to_array(
        {format(VK_FORMAT_A2B10G10R10_UNORM_PACK32), format(VK_FORMAT_R16G16B16A16_SFLOAT)});

    const SwapchainSettings settings = chooseSwapchainSettings(capabilities(), formats, kWindow, kImages);

    CHECK(settings.format.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32);
}

TEST_CASE("Swapchain settings take the surface's extent when it reports one", "[rhi][unit]") {
    constexpr VkExtent2D kReported{.width = 640, .height = 480};
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.currentExtent = kReported;

    const SwapchainSettings settings = chooseSwapchainSettings(surface, kFormats, kWindow, kImages);

    CHECK(settings.extent.width == kReported.width);
    CHECK(settings.extent.height == kReported.height);
}

TEST_CASE("Swapchain settings clamp the window's size when the surface leaves the extent to them", "[rhi][unit]") {
    // Smaller than the window both ways.
    constexpr VkExtent2D kLargest{.width = 700, .height = 500};
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.maxImageExtent = kLargest;

    const SwapchainSettings settings = chooseSwapchainSettings(surface, kFormats, kWindow, kImages);

    CHECK(settings.extent.width == kLargest.width);
    CHECK(settings.extent.height == kLargest.height);
}

TEST_CASE("Swapchain settings have no extent when the window has no area", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR minimised = capabilities();
    minimised.currentExtent = {.width = 0, .height = 0};

    const SwapchainSettings reported = chooseSwapchainSettings(minimised, kFormats, kWindow, kImages);
    const SwapchainSettings asked = chooseSwapchainSettings(capabilities(), kFormats, {.width = 0, .height = 600},
                                                            kImages);

    CHECK(reported.extent.width == 0);
    CHECK(reported.extent.height == 0);
    CHECK(asked.extent.width == 0);
    CHECK(asked.extent.height == 0);
}

TEST_CASE("Swapchain settings ask for the image count within the surface's range", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR surface = capabilities();

    CHECK(chooseSwapchainSettings(surface, kFormats, kWindow, kImages).imageCount == kImages);
    CHECK(chooseSwapchainSettings(surface, kFormats, kWindow, kMaxImages + 2).imageCount == kMaxImages);
    surface.minImageCount = 4;
    CHECK(chooseSwapchainSettings(surface, kFormats, kWindow, kImages).imageCount == 4);
    // No maximum.
    surface.maxImageCount = 0;
    CHECK(chooseSwapchainSettings(surface, kFormats, kWindow, kMaxImages + 2).imageCount == kMaxImages + 2);
}

TEST_CASE("Swapchain settings ask for at least two images", "[rhi][unit]") {
    CHECK(chooseSwapchainSettings(capabilities(), kFormats, kWindow, 1).imageCount == 2);
}

TEST_CASE("Swapchain settings composite opaquely when the surface can", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.supportedCompositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR | VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    CHECK(chooseSwapchainSettings(surface, kFormats, kWindow, kImages).compositeAlpha ==
          VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
}

TEST_CASE("Swapchain settings composite as the surface supports when it can't composite opaquely", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.supportedCompositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR | VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;

    CHECK(chooseSwapchainSettings(surface, kFormats, kWindow, kImages).compositeAlpha ==
          VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);
}

TEST_CASE("Swapchain settings composite opaquely when the surface reports no way to composite", "[rhi][unit]") {
    // Vulkan promises at least one way; a driver that breaks that still gets one.
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.supportedCompositeAlpha = 0;

    CHECK(chooseSwapchainSettings(surface, kFormats, kWindow, kImages).compositeAlpha ==
          VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
}
