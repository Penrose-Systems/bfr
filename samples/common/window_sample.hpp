// What the windowing samples share once a window exists: pick an adapter that
// presents to it, report what it can present, make a swapchain, and keep the
// swapchain the size of the window until the window closes. Nothing is drawn:
// BFR has no present call yet, so the window stays blank.
#pragma once

#include "bfr.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sample {

struct Size {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

namespace detail {

constexpr std::size_t kLabelWidth = 13;

// label, then one item to a line, the rest under the first, then a blank line.
inline void row(std::string_view label, const std::vector<std::string>& items) {
    bool first = true;
    for (const std::string& item : items) {
        std::cout << std::format("  {:<{}}{}\n", first ? label : std::string_view{}, kLabelWidth, item);
        first = false;
    }
    std::cout << "\n";
}

inline int fail(std::string_view what, const bfr::Error& error) {
    std::cerr << what << ": " << error.message << "\n";
    return 1;
}

inline void printSurface(const bfr::SurfaceCaps& caps, const std::vector<bfr::SurfaceFormat>& formats) {
    const std::string most = caps.maxImageCount == 0 ? "any" : std::to_string(caps.maxImageCount);
    row("Images", {std::format("{} to {}", caps.minImageCount, most)});
    row("Sizes", {std::format("{}x{} to {}x{}", caps.minWidth, caps.minHeight, caps.maxWidth, caps.maxHeight)});
    std::vector<std::string> modes;
    for (const auto& [has, mode] : {std::pair{caps.fifo, "fifo"}, std::pair{caps.fifoRelaxed, "fifo relaxed"},
                                    std::pair{caps.mailbox, "mailbox"}, std::pair{caps.immediate, "immediate"}}) {
        if (has) {
            modes.emplace_back(mode);
        }
    }
    row("Present", modes);
    std::vector<std::string> names;
    names.reserve(formats.size());
    for (const bfr::SurfaceFormat& format : formats) {
        names.push_back(std::format("{} {}", bfr::name(format.format), bfr::name(format.colorSpace)));
    }
    row("Swap formats", names);
}

} // namespace detail

// Runs until poll returns false. poll waits for window events for a short
// while, so the loop doesn't spin. size is the window's size in pixels.
inline int show(const bfr::Window& window, const std::function<bool()>& poll, const std::function<Size()>& size) {
    auto instance = bfr::Instance::create(
        {.backend = bfr::Backend::Vulkan,
         .validation = true,
         .messageCallback = [](bfr::MessageSeverity /*severity*/, std::string_view message) {
             std::cerr << message << "\n";
         }});
    if (!instance) {
        return detail::fail("can't create an instance", instance.error());
    }
    const auto adapters = instance->adapters();
    if (!adapters) {
        return detail::fail("can't list adapters", adapters.error());
    }

    bfr::Device device;
    std::uint32_t adapter = 0;
    for (; adapter < adapters->size(); ++adapter) {
        auto opened = instance->createDevice({.adapter = adapter, .window = window});
        if (opened) {
            device = std::move(*opened);
            break;
        }
        if (opened.error().code != bfr::Result::ErrorPresentUnsupported) {
            return detail::fail("can't open " + adapters->at(adapter).name, opened.error());
        }
    }
    if (!device) {
        std::cerr << "no adapter presents to this window\n";
        return 1;
    }
    detail::row("Adapter", {adapters->at(adapter).name});

    const auto caps = instance->surfaceCaps(adapter, {.window = window});
    if (!caps) {
        return detail::fail("can't ask what the window presents", caps.error());
    }
    const auto formats = instance->surfaceFormats(adapter, {.window = window});
    if (!formats) {
        return detail::fail("can't list the window's formats", formats.error());
    }
    detail::printSurface(*caps, *formats);

    auto swapchain = device.createSwapchain({.window = window});
    if (!swapchain) {
        return detail::fail("can't create a swapchain", swapchain.error());
    }
    Size last;
    while (poll()) {
        const Size now = size();
        // A minimised window has no size to make a swapchain for.
        if ((now.width == 0 || now.height == 0) || (now.width == last.width && now.height == last.height)) {
            continue;
        }
        const auto images = swapchain->resize(now.width, now.height);
        if (!images) {
            return detail::fail("can't resize the swapchain", images.error());
        }
        last = now;
        detail::row("Swapchain", {std::format("{}x{}, {} images", images->width, images->height, images->count),
                                  std::format("{} {}", bfr::name(images->format), bfr::name(images->colorSpace)),
                                  std::format("{} presenting", bfr::name(images->presentMode)),
                                  std::format("{} alpha", bfr::name(images->alphaMode))});
    }
    return 0;
}

} // namespace sample
