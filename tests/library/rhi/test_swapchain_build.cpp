// How every backend builds, or refuses, a swapchain request from what its
// surface reports. Pure: no backend, no GPU.

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/swapchain.h"
#include "rhi/swapchain_build.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cstdint>

using namespace bfr::rhi;

namespace {

constexpr Extent kWindow{.width = 800, .height = 600};
constexpr std::uint32_t kImages = 3;
constexpr std::uint32_t kMaxImages = 8;
constexpr std::uint32_t kMaxExtent = 4096;

// A surface that leaves the size to the swapchain, takes 1 to 8 images,
// presents fifo and composites opaquely.
SurfaceCaps surface() {
    return {
        .minImageCount = 1,
        .maxImageCount = kMaxImages,
        .width = kSizeFromSwapchain,
        .height = kSizeFromSwapchain,
        .minWidth = 1,
        .minHeight = 1,
        .maxWidth = kMaxExtent,
        .maxHeight = kMaxExtent,
        .rotation = SurfaceRotation::None,
        .fifo = true,
        .fifoRelaxed = false,
        .mailbox = false,
        .immediate = false,
        .alphaOpaque = true,
        .alphaPremultiplied = false,
        .alphaPostmultiplied = false,
        .alphaInherit = false,
    };
}

constexpr SurfaceFormat srgb(Format format) {
    return {.format = format, .colorSpace = ColorSpace::Srgb};
}

constexpr std::array kBgraOnly{srgb(Format::Bgra8Unorm)};

// The build for request on caps, the defaults unless it says otherwise.
bfr::core::Result<SwapchainBuild> buildFor(const SurfaceCaps& caps, Extent size = kWindow,
                                           const SwapchainRequest& request = {}) {
    return swapchainBuildFor(caps, kBgraOnly, size, request);
}

} // namespace

TEST_CASE("Swapchain build is what the request says when the surface allows it", "[rhi][unit]") {
    const auto build = buildFor(surface());

    REQUIRE(build.has_value());
    CHECK(build->imageCount == 1);
    CHECK(build->extent.width == kWindow.width);
    CHECK(build->extent.height == kWindow.height);
}

TEST_CASE("Swapchain build takes exactly the format and color space asked for", "[rhi][unit]") {
    constexpr std::array kFormats{srgb(Format::Bgra8Unorm), srgb(Format::Rgba8Unorm)};

    const auto build = swapchainBuildFor(surface(), kFormats, kWindow, {.format = Format::Rgba8Unorm});

    CHECK(build.has_value());
}

TEST_CASE("Swapchain build refuses a format the surface doesn't list, and takes no other", "[rhi][unit]") {
    constexpr std::array kFormats{srgb(Format::Rgba8Unorm), srgb(Format::Rgb10a2Unorm)};

    const auto build = swapchainBuildFor(surface(), kFormats, kWindow, {});

    REQUIRE_FALSE(build.has_value());
    CHECK(build.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(build.error().message, Catch::Matchers::ContainsSubstring("does not list the format"));
}

TEST_CASE("Swapchain build refuses a color space the surface lists the format without", "[rhi][unit]") {
    const auto build = buildFor(surface(), kWindow, {.colorSpace = ColorSpace::Hdr10Pq});

    REQUIRE_FALSE(build.has_value());
    CHECK(build.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(build.error().message, Catch::Matchers::ContainsSubstring("does not list the format"));
}

TEST_CASE("Swapchain build takes the surface's size when it reports one", "[rhi][unit]") {
    constexpr Extent kReported{.width = 640, .height = 480};
    SurfaceCaps caps = surface();
    caps.width = kReported.width;
    caps.height = kReported.height;

    const auto build = buildFor(caps);

    REQUIRE(build.has_value());
    CHECK(build->extent.width == kReported.width);
    CHECK(build->extent.height == kReported.height);
}

TEST_CASE("Swapchain build refuses a size the surface can't take, and doesn't clamp it", "[rhi][unit]") {
    // Smaller than the window both ways.
    constexpr Extent kLargest{.width = 700, .height = 500};
    SurfaceCaps caps = surface();
    caps.maxWidth = kLargest.width;
    caps.maxHeight = kLargest.height;

    const auto build = buildFor(caps);

    REQUIRE_FALSE(build.has_value());
    CHECK(build.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(build.error().message, Catch::Matchers::ContainsSubstring("1x1 to 700x500 pixels; 800x600"));
}

TEST_CASE("Swapchain build has no extent when the window has no area", "[rhi][unit]") {
    SurfaceCaps minimised = surface();
    minimised.width = 0;
    minimised.height = 0;

    const auto reported = buildFor(minimised);
    const auto asked = buildFor(surface(), {.width = 0, .height = kWindow.height});

    REQUIRE(reported.has_value());
    REQUIRE(asked.has_value());
    CHECK(reported->extent.width == 0);
    CHECK(reported->extent.height == 0);
    CHECK(asked->extent.width == 0);
    CHECK(asked->extent.height == 0);
}

TEST_CASE("Swapchain build takes the fewest images the surface allows when none are asked for", "[rhi][unit]") {
    SurfaceCaps caps = surface();
    caps.minImageCount = 4;

    CHECK(buildFor(caps)->imageCount == 4);
}

TEST_CASE("Swapchain build takes the image count asked for when the surface allows it", "[rhi][unit]") {
    SurfaceCaps caps = surface();

    CHECK(buildFor(caps, kWindow, {.imageCount = kImages})->imageCount == kImages);
    CHECK(buildFor(caps, kWindow, {.imageCount = kMaxImages})->imageCount == kMaxImages);
    // One, though two would let a frame draw while another shows: that is the caller's call.
    CHECK(buildFor(caps, kWindow, {.imageCount = 1})->imageCount == 1);
    // No maximum.
    caps.maxImageCount = 0;
    CHECK(buildFor(caps, kWindow, {.imageCount = kMaxImages + 2})->imageCount == kMaxImages + 2);
}

TEST_CASE("Swapchain build refuses an image count outside the surface's range, and says the range", "[rhi][unit]") {
    SurfaceCaps caps = surface();
    caps.minImageCount = 4;

    const auto below = buildFor(caps, kWindow, {.imageCount = kImages});
    const auto above = buildFor(caps, kWindow, {.imageCount = kMaxImages + 2});
    caps.maxImageCount = 0;
    const auto belowWithoutMaximum = buildFor(caps, kWindow, {.imageCount = kImages});

    REQUIRE_FALSE(below.has_value());
    CHECK(below.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(below.error().message, Catch::Matchers::ContainsSubstring("4 to 8 images; 3 were asked"));
    REQUIRE_FALSE(above.has_value());
    REQUIRE_FALSE(belowWithoutMaximum.has_value());
    CHECK_THAT(belowWithoutMaximum.error().message, Catch::Matchers::ContainsSubstring("4 or more images"));
}

TEST_CASE("Swapchain build refuses a present mode the surface doesn't allow, and names it", "[rhi][unit]") {
    SurfaceCaps caps = surface();
    caps.immediate = true;

    const auto immediate = buildFor(caps, kWindow, {.presentMode = PresentMode::Immediate});
    const auto mailbox = buildFor(caps, kWindow, {.presentMode = PresentMode::Mailbox});

    CHECK(immediate.has_value());
    REQUIRE_FALSE(mailbox.has_value());
    CHECK(mailbox.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(mailbox.error().message, Catch::Matchers::ContainsSubstring("mailbox"));
}

TEST_CASE("Swapchain build refuses an alpha mode the surface doesn't allow, and takes no other", "[rhi][unit]") {
    SurfaceCaps caps = surface();
    caps.alphaOpaque = false;
    caps.alphaPremultiplied = true;
    caps.alphaInherit = true;

    const auto premultiplied = buildFor(caps, kWindow, {.alphaMode = AlphaMode::Premultiplied});
    const auto opaque = buildFor(caps, kWindow, {.alphaMode = AlphaMode::Opaque});

    CHECK(premultiplied.has_value());
    REQUIRE_FALSE(opaque.has_value());
    CHECK(opaque.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(opaque.error().message, Catch::Matchers::ContainsSubstring("opaque"));
}
