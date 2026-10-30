// What the windowing samples share once a window exists: pick an adapter that
// presents to it, report what it can present, make a swapchain, and clear and
// present it every frame, in a color that cycles, until the window closes.
#pragma once

#include "bfr.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sample {

struct Size {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

/// The backend the sample runs on: the one `--backend <name>` in args names,
/// else the first the build lists. Fails, printing why, when there is none
/// or the name is not a backend the build has.
inline std::optional<bfr::Backend> backendFrom(std::span<char*> args) {
    const auto backends = bfr::backends();
    if (!backends || backends->empty()) {
        std::cerr << "this build has no backend\n";
        return std::nullopt;
    }
    std::string_view wanted;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (std::string_view{args[i]} == "--backend" && i + 1 < args.size()) {
            wanted = args[++i];
        }
    }
    if (wanted.empty()) {
        return backends->front();
    }
    for (const bfr::Backend backend : *backends) {
        if (bfr::name(backend) == wanted) {
            return backend;
        }
    }
    std::cerr << "no backend named " << wanted << " in this build; it has:";
    for (const bfr::Backend backend : *backends) {
        std::cerr << ' ' << bfr::name(backend);
    }
    std::cerr << "\n";
    return std::nullopt;
}

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

// The clear color at time t, in seconds: a hue that goes round once every
// few seconds.
inline std::array<float, 4> colorAt(double t) {
    constexpr double kPeriodSeconds = 6.0;
    constexpr double kTau = 6.283185307179586;
    const double phase = t / kPeriodSeconds * kTau;
    const auto channel = [phase](double offset) {
        return static_cast<float>(0.5 + 0.5 * std::sin(phase + offset));
    };
    return {channel(0.0), channel(kTau / 3), channel(2 * kTau / 3), 1.0F};
}

} // namespace detail

// Runs on backend until poll returns false, clearing and presenting a frame
// each time it returns true; poll handles the window's events. size is the
// window's size in pixels.
inline int show(bfr::Backend backend, const bfr::Window& window, const std::function<bool()>& poll,
                const std::function<Size()>& size) {
    detail::row("Backend", {std::string{bfr::name(backend)}});
    auto instance = bfr::Instance::create(
        {.backend = backend,
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
    const auto started = std::chrono::steady_clock::now();
    Size last;
    bool resize = true;
    // With nothing to present, the loop waits instead of spinning.
    constexpr std::chrono::milliseconds kIdle{10};
    while (poll()) {
        const Size now = size();
        // A minimised window has no size to make a swapchain for.
        if (now.width == 0 || now.height == 0) {
            std::this_thread::sleep_for(kIdle);
            continue;
        }
        if (resize || now.width != last.width || now.height != last.height) {
            const auto images = swapchain->resize(now.width, now.height);
            if (!images) {
                return detail::fail("can't resize the swapchain", images.error());
            }
            last = now;
            resize = false;
            detail::row("Swapchain", {std::format("{}x{}, {} images", images->width, images->height, images->count),
                                      std::format("{} {}", bfr::name(images->format), bfr::name(images->colorSpace)),
                                      std::format("{} presenting", bfr::name(images->presentMode)),
                                      std::format("{} alpha", bfr::name(images->alphaMode))});
            if (images->count == 0) {
                std::this_thread::sleep_for(kIdle);
                continue;
            }
        }
        // The window changed under the swapchain: rebuild it next time round.
        const auto acquired = swapchain->acquire();
        if (!acquired) {
            if (acquired.error().code == bfr::Result::ErrorOutOfDate) {
                resize = true;
                continue;
            }
            return detail::fail("can't acquire an image", acquired.error());
        }
        auto commands = device.beginCommands();
        if (!commands) {
            return detail::fail("can't begin commands", commands.error());
        }
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
        if (const auto begun = commands->beginRendering(*swapchain, {.loadOp = bfr::LoadOp::Clear,
                                                                     .clearColor = detail::colorAt(elapsed.count())});
            !begun) {
            return detail::fail("can't begin rendering", begun.error());
        }
        if (const auto ended = commands->endRendering(); !ended) {
            return detail::fail("can't end rendering", ended.error());
        }
        if (const auto submitted = commands->submit(); !submitted) {
            return detail::fail("can't submit", submitted.error());
        }
        const auto presented = swapchain->present();
        if (!presented) {
            if (presented.error().code == bfr::Result::ErrorOutOfDate) {
                resize = true;
                continue;
            }
            return detail::fail("can't present", presented.error());
        }
    }
    return 0;
}

} // namespace sample
