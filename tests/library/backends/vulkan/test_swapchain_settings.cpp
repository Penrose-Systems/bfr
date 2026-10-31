// The Vulkan names of a swapchain request, each value as given.

#include "bfr/rhi/format.h"
#include "bfr/rhi/swapchain.h"
#include "vulkan/swapchain_settings.h"

#include <catch2/catch_test_macros.hpp>
#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <limits>

using namespace bfr::rhi;
using namespace bfr::rhi::vulkan;

namespace {

constexpr std::uint32_t kUndefined = std::numeric_limits<std::uint32_t>::max();
constexpr VkExtent2D kWindow{.width = 800, .height = 600};
constexpr std::uint32_t kMaxImages = 8;
constexpr std::uint32_t kMaxExtent = 4096;

// A surface that leaves the extent to the swapchain, takes 1 to 8 images and
// composites opaquely or premultiplied.
VkSurfaceCapabilitiesKHR capabilities() {
    VkSurfaceCapabilitiesKHR surface{};
    surface.minImageCount = 1;
    surface.maxImageCount = kMaxImages;
    surface.currentExtent = {.width = kUndefined, .height = kUndefined};
    surface.minImageExtent = {.width = 1, .height = 1};
    surface.maxImageExtent = {.width = kMaxExtent, .height = kMaxExtent};
    surface.supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR | VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    return surface;
}

} // namespace

TEST_CASE("Vulkan swapchain settings name the request in Vulkan's terms", "[vulkan][unit]") {
    const SwapchainSettings settings = swapchainSettingsFor(capabilities(), kWindow,
                                                            {.imageCount = 2,
                                                             .presentMode = PresentMode::Immediate,
                                                             .format = Format::Rgba8Unorm,
                                                             .alphaMode = AlphaMode::Premultiplied});

    CHECK(settings.format.format == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(settings.format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    CHECK(settings.imageCount == 2);
    CHECK(settings.extent.width == kWindow.width);
    CHECK(settings.extent.height == kWindow.height);
    CHECK(settings.presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR);
    CHECK(settings.compositeAlpha == VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR);
}

TEST_CASE("Vulkan swapchain settings pass on what the surface does not allow", "[vulkan][unit]") {
    const SwapchainSettings settings = swapchainSettingsFor(
        capabilities(), kWindow, {.imageCount = kMaxImages + 1, .alphaMode = AlphaMode::Inherit});

    CHECK(settings.imageCount == kMaxImages + 1);
    CHECK(settings.compositeAlpha == VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
}

TEST_CASE("Vulkan swapchain settings take the surface's extent when it reports one", "[vulkan][unit]") {
    constexpr VkExtent2D kReported{.width = 640, .height = 480};
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.currentExtent = kReported;

    const SwapchainSettings settings = swapchainSettingsFor(surface, kWindow, {});

    CHECK(settings.extent.width == kReported.width);
    CHECK(settings.extent.height == kReported.height);
}

TEST_CASE("Vulkan present modes and composite alphas are named one to one", "[vulkan][unit]") {
    CHECK(presentModeFor(PresentMode::Fifo) == VK_PRESENT_MODE_FIFO_KHR);
    CHECK(presentModeFor(PresentMode::FifoRelaxed) == VK_PRESENT_MODE_FIFO_RELAXED_KHR);
    CHECK(presentModeFor(PresentMode::Mailbox) == VK_PRESENT_MODE_MAILBOX_KHR);
    CHECK(presentModeFor(PresentMode::Immediate) == VK_PRESENT_MODE_IMMEDIATE_KHR);
    CHECK(compositeAlphaFor(AlphaMode::Opaque) == VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
    CHECK(compositeAlphaFor(AlphaMode::Premultiplied) == VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR);
    CHECK(compositeAlphaFor(AlphaMode::Postmultiplied) == VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);
    CHECK(compositeAlphaFor(AlphaMode::Inherit) == VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
}
