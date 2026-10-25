#include "display.h"
#include "display_test_support.h"
#include "error.h"
#include "event.h"
#include "window.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_video.h>

#include <cstdint>
#include <string_view>
#include <variant>
#include <vector>

using namespace bfr::platform;
using namespace bfr::platform::test;
using bfr::core::ErrorCode;
namespace window = bfr::core::window;

namespace {

// The windowed size the fullscreen tests leave and return to.
constexpr std::uint32_t kWindowedWidth = 640;
constexpr std::uint32_t kWindowedHeight = 360;
// A size other than the default, carried by the resize events.
constexpr std::uint32_t kResizedWidth = 320;
constexpr std::uint32_t kResizedHeight = 200;

// Whether native is the struct the video driver fills, with its handles set.
bool holdsHandlesOf(std::string_view driver, const window::NativeWindow& native) {
    if (driver == "x11") {
        const auto* xlib = std::get_if<window::Xlib>(&native);
        return xlib != nullptr && xlib->display != nullptr && xlib->window != 0;
    }
    if (driver == "wayland") {
        const auto* wayland = std::get_if<window::Wayland>(&native);
        return wayland != nullptr && wayland->display != nullptr && wayland->surface != nullptr;
    }
    if (driver == "windows") {
        const auto* win32 = std::get_if<window::Win32>(&native);
        return win32 != nullptr && win32->instance != nullptr && win32->window != nullptr;
    }
    if (driver == "cocoa") {
        const auto* metal = std::get_if<window::Metal>(&native);
        return metal != nullptr && metal->layer != nullptr;
    }
    return driver == "offscreen" && std::holds_alternative<window::Headless>(native);
}

} // namespace

// --- Lifecycle ---

TEST_CASE("Display::create returns a display", "[platform][unit]") {
    REQUIRE(Display::create() != nullptr);
}

TEST_CASE("Display init succeeds with a valid config", "[platform][unit]") {
    auto display = Display::create();
    REQUIRE(display->init(hiddenConfig()).has_value());
    display->shutdown();
}

TEST_CASE("Display creates its window without an OpenGL context", "[platform][unit]") {
    auto display = openDisplay();
    const SDL_WindowFlags flags = SDL_GetWindowFlags(sdlWindow());
    CHECK((flags & SDL_WINDOW_OPENGL) == 0);
}

TEST_CASE("Display init twice reports DisplayAlreadyInitialised", "[platform][unit]") {
    auto display = openDisplay();
    const auto second = display->init(hiddenConfig());
    REQUIRE_FALSE(second.has_value());
    CHECK(second.error().code == ErrorCode::DisplayAlreadyInitialised);
    CHECK_FALSE(second.error().message.empty());
}

TEST_CASE("Display init rejects a zero extent", "[platform][unit]") {
    auto display = Display::create();
    auto config = hiddenConfig();
    config.width = 0;
    const auto result = display->init(config);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::DisplayWindowFailed);
    CHECK(std::holds_alternative<std::monostate>(display->nativeWindow()));
}

TEST_CASE("Display reports no native window before init or after shutdown", "[platform][unit]") {
    // Another open display keeps SDL's video driver up, so only this
    // display's own state decides.
    const auto other = openDisplay();
    auto display = Display::create();
    CHECK(std::holds_alternative<std::monostate>(display->nativeWindow()));
    REQUIRE(display->init(hiddenConfig()).has_value());
    display->shutdown();
    CHECK(std::holds_alternative<std::monostate>(display->nativeWindow()));
}

TEST_CASE("Display reports the native window of its video driver", "[platform][unit]") {
    auto display = openDisplay();
    const std::string_view driver = SDL_GetCurrentVideoDriver();

    const window::NativeWindow native = display->nativeWindow();

    INFO("video driver " << driver);
    CHECK(holdsHandlesOf(driver, native));
}

TEST_CASE("Display tick before init and after shutdown is harmless", "[platform][unit]") {
    auto display = Display::create();
    display->tick();
    REQUIRE(display->init(hiddenConfig()).has_value());
    display->shutdown();
    display->tick();
}

TEST_CASE("Display shutdown is idempotent", "[platform][unit]") {
    auto display = openDisplay();
    display->shutdown();
    display->shutdown();
}

TEST_CASE("Display recreates its window after shutdown", "[platform][unit]") {
    constexpr int kRecreations = 3;
    for (int recreation = 0; recreation < kRecreations; ++recreation) {
        auto display = openDisplay();
        CHECK_FALSE(std::holds_alternative<std::monostate>(display->nativeWindow()));
        CHECK(display->isOpen());
        display->shutdown();
        CHECK(std::holds_alternative<std::monostate>(display->nativeWindow()));
        CHECK_FALSE(display->isOpen());
    }
}

TEST_CASE("Display setFullscreen before init reports DisplayNotInitialised", "[platform][unit]") {
    auto display = Display::create();
    const auto result = display->setFullscreen(true);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::DisplayNotInitialised);
}

// --- Dimensions ---

TEST_CASE("Display width and height match the config after init", "[platform][unit]") {
    constexpr std::uint32_t kConfigWidth = 1024;
    constexpr std::uint32_t kConfigHeight = 768;
    auto config = hiddenConfig();
    config.width = kConfigWidth;
    config.height = kConfigHeight;
    auto display = openDisplay(config);
    CHECK(display->width() == kConfigWidth);
    CHECK(display->height() == kConfigHeight);
}

TEST_CASE("Display vsyncEnabled matches the config", "[platform][unit]") {
    auto config = hiddenConfig();
    config.vsync = false;
    auto display = openDisplay(config);
    CHECK_FALSE(display->vsyncEnabled());
}

TEST_CASE("Display fullscreen toggle fills the monitor and restores the window", "[platform][unit]") {
    auto config = hiddenConfig();
    config.width = kWindowedWidth;
    config.height = kWindowedHeight;
    config.visible = true;
    auto display = openDisplay(config);
    const BlankWaylandFrame frame{*display};
    const VideoMode mode = display->currentMode();
    Recorder recorder;
    recorder.attach(*display);

    REQUIRE(display->setFullscreen(true).has_value());
    REQUIRE(tickUntilSize(*display, mode.width, mode.height, &frame));
    CHECK(display->isFullscreen());
    REQUIRE(recorder.count<WindowResized>() >= 1);
    CHECK(recorder.last<WindowResized>()->width == mode.width);
    CHECK(recorder.last<WindowResized>()->height == mode.height);

    REQUIRE(display->setFullscreen(false).has_value());
    REQUIRE(tickUntilSize(*display, kWindowedWidth, kWindowedHeight, &frame));
    CHECK_FALSE(display->isFullscreen());
    REQUIRE(recorder.count<WindowResized>() >= 2);
    CHECK(recorder.last<WindowResized>()->width == kWindowedWidth);
    CHECK(recorder.last<WindowResized>()->height == kWindowedHeight);
}

TEST_CASE("Display honours fullscreen at init and returns to the authored size", "[platform][unit]") {
    auto config = hiddenConfig();
    config.width = kWindowedWidth;
    config.height = kWindowedHeight;
    config.fullscreen = true;
    config.visible = true;
    auto display = openDisplay(config);
    const BlankWaylandFrame frame{*display};
    const VideoMode mode = display->currentMode();
    // A window manager may finish fullscreen after init returns; the size
    // arrives as events.
    REQUIRE(tickUntilSize(*display, mode.width, mode.height, &frame));
    CHECK(display->isFullscreen());

    REQUIRE(display->setFullscreen(false).has_value());
    REQUIRE(tickUntilSize(*display, kWindowedWidth, kWindowedHeight, &frame));
    CHECK_FALSE(display->isFullscreen());
}

// --- Video modes ---

TEST_CASE("Display videoModes lists at least one mode", "[platform][unit]") {
    auto display = openDisplay();
    CHECK_FALSE(display->videoModes().empty());
}

TEST_CASE("Display currentMode has a non-zero extent", "[platform][unit]") {
    auto display = openDisplay();
    const VideoMode mode = display->currentMode();
    CHECK(mode.width > 0);
    CHECK(mode.height > 0);
}

// --- Window events ---

TEST_CASE("Display close request closes the window and reports WindowClosed", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushWindowEvent(windowId(), SDL_EVENT_WINDOW_CLOSE_REQUESTED);
    display->tick();
    CHECK_FALSE(display->isOpen());
    CHECK(recorder.count<WindowClosed>() == 1);
}

TEST_CASE("Display pixel resize updates the size and reports WindowResized", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushWindowEvent(windowId(), SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED, static_cast<Sint32>(kResizedWidth),
                    static_cast<Sint32>(kResizedHeight));
    display->tick();
    CHECK(display->width() == kResizedWidth);
    CHECK(display->height() == kResizedHeight);
    REQUIRE(recorder.last<WindowResized>() != nullptr);
    CHECK(recorder.last<WindowResized>()->width == kResizedWidth);
    CHECK(recorder.last<WindowResized>()->height == kResizedHeight);
}

TEST_CASE("Display ignores a resize in window coordinates", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushWindowEvent(windowId(), SDL_EVENT_WINDOW_RESIZED, static_cast<Sint32>(kResizedWidth),
                    static_cast<Sint32>(kResizedHeight));
    display->tick();
    CHECK(display->width() == kWidth);
    CHECK(display->height() == kHeight);
    CHECK(recorder.count<WindowResized>() == 0);
}

TEST_CASE("Display focus and minimise events are reported", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    const SDL_WindowID window = windowId();
    pushWindowEvent(window, SDL_EVENT_WINDOW_FOCUS_LOST);
    pushWindowEvent(window, SDL_EVENT_WINDOW_FOCUS_GAINED);
    pushWindowEvent(window, SDL_EVENT_WINDOW_MINIMIZED);
    pushWindowEvent(window, SDL_EVENT_WINDOW_RESTORED);
    display->tick();
    CHECK(recorder.count<WindowFocusLost>() == 1);
    CHECK(recorder.count<WindowFocusGained>() == 1);
    CHECK(recorder.count<WindowMinimized>() == 1);
    REQUIRE(recorder.last<WindowRestored>() != nullptr);
    CHECK(recorder.last<WindowRestored>()->width == kWidth);
    CHECK(recorder.last<WindowRestored>()->height == kHeight);
}

TEST_CASE("Display ignores window events for another window", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushWindowEvent(windowId() + 1, SDL_EVENT_WINDOW_CLOSE_REQUESTED);
    pushKey(windowId() + 1, SDL_SCANCODE_A, true);
    display->tick();
    CHECK(display->isOpen());
    CHECK(recorder.events.empty());
}
