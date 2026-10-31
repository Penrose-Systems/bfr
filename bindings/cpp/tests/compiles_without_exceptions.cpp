// Instantiates every call in bfr.hpp, for a build without exceptions or RTTI.
// Compiled only, never run.
#include "bfr.hpp"

#include <string_view>

namespace {

// Whether every call succeeded.
[[maybe_unused]] bool callEverything() {
    const auto backends = bfr::backends();
    if (!backends || backends->empty()) {
        return false;
    }
    auto instance = bfr::Instance::create({.backend = backends->front(),
                                           .validation = false,
                                           .messageCallback = [](bfr::MessageSeverity, std::string_view) {
                                           }});
    if (!instance) {
        return false;
    }
    const auto adapters = instance->adapters();
    const bfr::Window headlessWindow{.system = bfr::WindowSystem::Headless};
    auto device = instance->createDevice({.adapter = 0, .window = headlessWindow});
    if (!adapters || !device) {
        return false;
    }
    const auto surface = instance->surfaceCaps(0, {.window = headlessWindow});
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
