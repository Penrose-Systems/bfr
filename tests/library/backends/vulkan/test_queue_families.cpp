#include "vulkan/queue_families.h"

#include <catch2/catch_test_macros.hpp>
#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <array>

using namespace bfr::rhi::vulkan;

namespace {

VkQueueFamilyProperties family(VkQueueFlags flags) {
    VkQueueFamilyProperties properties{};
    properties.queueFlags = flags;
    properties.queueCount = 1;
    return properties;
}

} // namespace

TEST_CASE("Queue families give compute and transfer their own families when the adapter has them", "[vulkan][unit]") {
    // The RTX 3070's families: graphics, transfer, compute, video decode,
    // video encode, transfer.
    const auto families = std::to_array({
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT),
        family(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT),
        family(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT),
        family(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_VIDEO_DECODE_BIT_KHR),
        family(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_VIDEO_ENCODE_BIT_KHR),
        family(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT),
    });

    const auto chosen = chooseQueueFamilies(families, {});

    REQUIRE(chosen.has_value());
    CHECK(chosen->graphics == 0);
    CHECK(chosen->compute == 2);
    CHECK(chosen->transfer == 1);
}

TEST_CASE("Queue families give transfer a compute family when no family only transfers", "[vulkan][unit]") {
    const auto families = std::to_array({
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
        family(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
    });

    const auto chosen = chooseQueueFamilies(families, {});

    REQUIRE(chosen.has_value());
    CHECK(chosen->transfer == 1);
}

TEST_CASE("Queue families give transfer a compute family that does not report the transfer bit", "[vulkan][unit]") {
    // Vulkan makes reporting the transfer bit optional on a compute family,
    // which transfers regardless.
    const auto families = std::to_array({
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
        family(VK_QUEUE_COMPUTE_BIT),
    });

    const auto chosen = chooseQueueFamilies(families, {});

    REQUIRE(chosen.has_value());
    CHECK(chosen->transfer == 1);
}

TEST_CASE("Queue families give graphics a family that also runs compute", "[vulkan][unit]") {
    const auto families = std::to_array({
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT),
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
    });

    const auto chosen = chooseQueueFamilies(families, {});

    REQUIRE(chosen.has_value());
    CHECK(chosen->graphics == 1);
    CHECK(chosen->compute == 1);
}

TEST_CASE("Queue families give graphics a family that presents to the surface", "[vulkan][unit]") {
    const auto families = std::to_array({
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
    });
    const auto presents = std::to_array<VkBool32>({VK_FALSE, VK_TRUE});

    const auto chosen = chooseQueueFamilies(families, presents);

    REQUIRE(chosen.has_value());
    CHECK(chosen->graphics == 1);
    CHECK(chosen->present == 1);
}

TEST_CASE("Queue families give presenting its own family when no graphics family presents", "[vulkan][unit]") {
    const auto families = std::to_array({
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
        family(VK_QUEUE_TRANSFER_BIT),
    });
    const auto presents = std::to_array<VkBool32>({VK_FALSE, VK_TRUE});

    const auto chosen = chooseQueueFamilies(families, presents);

    REQUIRE(chosen.has_value());
    CHECK(chosen->graphics == 0);
    CHECK(chosen->present == 1);
}

TEST_CASE("Queue families have no present family when no family presents to the surface", "[vulkan][unit]") {
    const auto families = std::to_array({family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)});
    const auto presents = std::to_array<VkBool32>({VK_FALSE});

    const auto chosen = chooseQueueFamilies(families, presents);

    REQUIRE(chosen.has_value());
    CHECK_FALSE(chosen->present.has_value());
}

TEST_CASE("Queue families present on the graphics family without a surface", "[vulkan][unit]") {
    const auto families = std::to_array({
        family(VK_QUEUE_TRANSFER_BIT),
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT),
    });

    const auto chosen = chooseQueueFamilies(families, {});

    REQUIRE(chosen.has_value());
    CHECK(chosen->present == 1);
}
