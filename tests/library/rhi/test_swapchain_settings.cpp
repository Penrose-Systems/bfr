#include "error.h"
#include "format.h"
#include "swapchain.h"
#include "swapchain_settings.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <limits>

using namespace bfr::rhi;
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

const auto kBgraOnly = std::to_array({format(VK_FORMAT_B8G8R8A8_UNORM)});
const auto kFifoOnly = std::to_array({VK_PRESENT_MODE_FIFO_KHR});

// The settings for request, the defaults unless it says otherwise.
bfr::core::Result<SwapchainSettings> settingsFor(const VkSurfaceCapabilitiesKHR& surface, VkExtent2D size = kWindow,
                                                 const SwapchainRequest& request = {}) {
    return swapchainSettingsFor(surface, kBgraOnly, kFifoOnly, size, request);
}

} // namespace

TEST_CASE("Swapchain settings are what the request says when the surface allows it", "[rhi][unit]") {
    const auto settings = settingsFor(capabilities());

    REQUIRE(settings.has_value());
    CHECK(settings->format.format == VK_FORMAT_B8G8R8A8_UNORM);
    CHECK(settings->format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    CHECK(settings->imageCount == 1);
    CHECK(settings->presentMode == VK_PRESENT_MODE_FIFO_KHR);
    CHECK(settings->compositeAlpha == VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
}

TEST_CASE("Swapchain settings take exactly the format and color space asked for", "[rhi][unit]") {
    const auto formats = std::to_array({format(VK_FORMAT_B8G8R8A8_UNORM), format(VK_FORMAT_R8G8B8A8_UNORM)});

    const auto settings = swapchainSettingsFor(capabilities(), formats, kFifoOnly, kWindow,
                                               {.format = Format::Rgba8Unorm});

    REQUIRE(settings.has_value());
    CHECK(settings->format.format == VK_FORMAT_R8G8B8A8_UNORM);
}

TEST_CASE("Swapchain settings refuse a format the surface doesn't list, and take no other", "[rhi][unit]") {
    const auto formats = std::to_array({format(VK_FORMAT_R8G8B8A8_UNORM), format(VK_FORMAT_A2B10G10R10_UNORM_PACK32)});

    const auto settings = swapchainSettingsFor(capabilities(), formats, kFifoOnly, kWindow, {});

    REQUIRE_FALSE(settings.has_value());
    CHECK(settings.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(settings.error().message, Catch::Matchers::ContainsSubstring("does not list the format"));
}

TEST_CASE("Swapchain settings refuse a color space the surface lists the format without", "[rhi][unit]") {
    const auto settings = swapchainSettingsFor(capabilities(), kBgraOnly, kFifoOnly, kWindow,
                                               {.colorSpace = ColorSpace::Hdr10Pq});

    REQUIRE_FALSE(settings.has_value());
    CHECK(settings.error().code == bfr::core::ErrorCode::RhiUnsupported);
}

TEST_CASE("Swapchain settings take the surface's extent when it reports one", "[rhi][unit]") {
    constexpr VkExtent2D kReported{.width = 640, .height = 480};
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.currentExtent = kReported;

    const auto settings = settingsFor(surface);

    REQUIRE(settings.has_value());
    CHECK(settings->extent.width == kReported.width);
    CHECK(settings->extent.height == kReported.height);
}

TEST_CASE("Swapchain settings take the window's size when the surface leaves the extent to them", "[rhi][unit]") {
    const auto settings = settingsFor(capabilities());

    REQUIRE(settings.has_value());
    CHECK(settings->extent.width == kWindow.width);
    CHECK(settings->extent.height == kWindow.height);
}

TEST_CASE("Swapchain settings refuse a size the surface can't take, and don't clamp it", "[rhi][unit]") {
    // Smaller than the window both ways.
    constexpr VkExtent2D kLargest{.width = 700, .height = 500};
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.maxImageExtent = kLargest;

    const auto settings = settingsFor(surface);

    REQUIRE_FALSE(settings.has_value());
    CHECK(settings.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(settings.error().message, Catch::Matchers::ContainsSubstring("1x1 to 700x500 pixels; 800x600"));
}

TEST_CASE("Swapchain settings have no extent when the window has no area", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR minimised = capabilities();
    minimised.currentExtent = {.width = 0, .height = 0};

    const auto reported = settingsFor(minimised);
    const auto asked = settingsFor(capabilities(), {.width = 0, .height = 600});

    REQUIRE(reported.has_value());
    REQUIRE(asked.has_value());
    CHECK(reported->extent.width == 0);
    CHECK(reported->extent.height == 0);
    CHECK(asked->extent.width == 0);
    CHECK(asked->extent.height == 0);
}

TEST_CASE("Swapchain settings take the fewest images the surface allows when none are asked for", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.minImageCount = 4;

    CHECK(settingsFor(surface)->imageCount == 4);
}

TEST_CASE("Swapchain settings take the image count asked for when the surface allows it", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR surface = capabilities();

    CHECK(settingsFor(surface, kWindow, {.imageCount = kImages})->imageCount == kImages);
    CHECK(settingsFor(surface, kWindow, {.imageCount = kMaxImages})->imageCount == kMaxImages);
    // One, though two would let a frame draw while another shows: that is the caller's call.
    CHECK(settingsFor(surface, kWindow, {.imageCount = 1})->imageCount == 1);
    // No maximum.
    surface.maxImageCount = 0;
    CHECK(settingsFor(surface, kWindow, {.imageCount = kMaxImages + 2})->imageCount == kMaxImages + 2);
}

TEST_CASE("Swapchain settings refuse an image count outside the surface's range, and say the range", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.minImageCount = 4;

    const auto below = settingsFor(surface, kWindow, {.imageCount = kImages});
    const auto above = settingsFor(surface, kWindow, {.imageCount = kMaxImages + 2});
    surface.maxImageCount = 0;
    const auto belowWithoutMaximum = settingsFor(surface, kWindow, {.imageCount = kImages});

    REQUIRE_FALSE(below.has_value());
    CHECK(below.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(below.error().message, Catch::Matchers::ContainsSubstring("4 to 8 images; 3 were asked"));
    REQUIRE_FALSE(above.has_value());
    REQUIRE_FALSE(belowWithoutMaximum.has_value());
    CHECK_THAT(belowWithoutMaximum.error().message, Catch::Matchers::ContainsSubstring("4 or more images"));
}

TEST_CASE("Swapchain settings refuse a present mode the surface doesn't list", "[rhi][unit]") {
    const auto modes = std::to_array({VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR});

    const auto immediate = swapchainSettingsFor(capabilities(), kBgraOnly, modes, kWindow,
                                                {.presentMode = PresentMode::Immediate});
    const auto mailbox = swapchainSettingsFor(capabilities(), kBgraOnly, modes, kWindow,
                                              {.presentMode = PresentMode::Mailbox});

    REQUIRE(immediate.has_value());
    CHECK(immediate->presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR);
    REQUIRE_FALSE(mailbox.has_value());
    CHECK(mailbox.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(mailbox.error().message, Catch::Matchers::ContainsSubstring("mailbox"));
}

TEST_CASE("Swapchain settings refuse an alpha mode the surface doesn't support, and take no other", "[rhi][unit]") {
    VkSurfaceCapabilitiesKHR surface = capabilities();
    surface.supportedCompositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR | VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;

    const auto premultiplied = settingsFor(surface, kWindow, {.alphaMode = AlphaMode::Premultiplied});
    const auto opaque = settingsFor(surface, kWindow, {.alphaMode = AlphaMode::Opaque});

    REQUIRE(premultiplied.has_value());
    CHECK(premultiplied->compositeAlpha == VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR);
    REQUIRE_FALSE(opaque.has_value());
    CHECK(opaque.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(opaque.error().message, Catch::Matchers::ContainsSubstring("opaque"));
}
