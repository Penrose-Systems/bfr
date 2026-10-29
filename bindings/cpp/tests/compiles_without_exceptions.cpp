// Instantiates every call in bfr.hpp, for a build without exceptions or RTTI.
// Compiled only, never run.
#include "bfr.hpp"

#include <string_view>

namespace {

// Whether every call succeeded.
[[maybe_unused]] bool callEverything() {
    auto instance = bfr::Instance::create({.backend = bfr::Backend::Vulkan,
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
    auto headless = device->createSwapchain({.window = headlessWindow, .imageCount = 0});
    if (!headless) {
        return false;
    }
    return headless->resize(1, 1).has_value();
}

} // namespace
