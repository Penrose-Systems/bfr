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
    auto device = instance->createDevice({.adapter = 0}, bfr::WindowHeadless{});
    if (!adapters || !device) {
        return false;
    }
    const auto xlib = device->createSwapchain({.imageCount = 0}, bfr::WindowXlib{});
    const auto wayland = device->createSwapchain({.imageCount = 0}, bfr::WindowWayland{});
    const auto win32 = device->createSwapchain({.imageCount = 0}, bfr::WindowWin32{});
    const auto metal = device->createSwapchain({.imageCount = 0}, bfr::WindowMetal{});
    auto headless = device->createSwapchain({.imageCount = 0}, bfr::WindowHeadless{});
    if (!xlib || !wayland || !win32 || !metal || !headless) {
        return false;
    }
    return headless->resize(1, 1).has_value();
}

} // namespace
