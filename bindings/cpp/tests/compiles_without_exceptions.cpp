// Instantiates every call in bfr.hpp, for a build without exceptions or RTTI.
// Compiled only, never run.
#include "bfr.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace {

// Whether every call succeeded.
[[maybe_unused]] bool callEverything() {
    const auto backends = bfr::backends();
    if (!backends || backends->empty()) {
        return false;
    }
    // Vulkan 1.2, BFR's minimum there; never run, so any version compiles.
    constexpr std::uint32_t kMajor = 1;
    constexpr std::uint32_t kMinor = 2;
    auto instance = bfr::Instance::create({.backend = backends->front(),
                                           .apiMajor = kMajor,
                                           .apiMinor = kMinor,
                                           .validation = false,
                                           .messageCallback = [](bfr::MessageSeverity, std::string_view, void*) {}});
    if (!instance) {
        return false;
    }
    const auto adapters = instance->adapters();
    const bfr::Window headlessWindow{.system = bfr::WindowSystem::Headless};
    const auto families = instance->queueFamilies(0);
    const auto family = instance->findQueueFamily(0, {.graphics = true, .presents = &headlessWindow});
    const auto presents = instance->queueFamilyPresents(0, 0, {.window = headlessWindow});
    if (!families || !family || !presents) {
        return false;
    }
    const std::array<bfr::QueueDesc, 1> queue{bfr::QueueDesc{.family = *family}};
    auto device = instance->createDevice({.adapter = 0, .queues = queue});
    if (!adapters || !device) {
        return false;
    }
    const auto surface = instance->swapchainCaps(0, {.window = headlessWindow});
    if (!surface) {
        return false;
    }
    auto headless = device->createSwapchain({.window = headlessWindow, .imageCount = surface->minImageCount});
    if (!headless) {
        return false;
    }
    return headless->resize(1, 1).has_value();
}

} // namespace
