#pragma once

// Helpers for the Display tests: they drive the real SDL backend by pushing
// SDL events and by attaching SDL virtual gamepads.

#include "display.h"
#include "error.h"
#include "event.h"

#include <catch2/catch_test_macros.hpp>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <string>
#include <variant>
#include <vector>

#ifdef __linux__
    #include "window.h"

    #include <sys/mman.h>
    #include <unistd.h>
    #include <wayland-client.h>

    #include <cstring>
    #include <string_view>
#endif

namespace bfr::platform::test {

inline constexpr std::uint32_t kWidth = 800;
inline constexpr std::uint32_t kHeight = 600;

inline DisplayConfig hiddenConfig() {
    return {.title = "Test",
            .width = kWidth,
            .height = kHeight,
            .fullscreen = false,
            .vsync = true,
            .resizable = false,
            .visible = false,
            .input = {}};
}

/// Ticks until the display reports the size, or the time runs out. The
/// offscreen driver resizes at once; a real window manager can finish after
/// SDL_SyncWindow returns, passing through intermediate sizes.
inline std::unique_ptr<Display> openDisplay(const DisplayConfig& config = hiddenConfig()) {
    auto display = Display::create();
    REQUIRE(display != nullptr);
    const core::Result<void> opened = display->init(config);
    INFO((opened.has_value() ? std::string{}
                             : std::string{core::name(opened.error().code)} + ": " + opened.error().message));
    REQUIRE(opened.has_value());
    return display;
}

// The open display's SDL window. The tests open one display at a time, so it
// is SDL's only window.
inline SDL_Window* sdlWindow() {
    int count = 0;
    SDL_Window** windows = SDL_GetWindows(&count);
    SDL_Window* window = count == 1 ? *windows : nullptr;
    SDL_free(static_cast<void*>(windows));
    REQUIRE(window != nullptr);
    return window;
}

inline SDL_WindowID windowId() {
    return SDL_GetWindowID(sdlWindow());
}

/// A blank 1x1 frame for the display's window when it is a Wayland window, and
/// nothing otherwise. A Wayland compositor applies size and fullscreen changes
/// only to a window that has a buffer and commits after each change, which a
/// renderer does every frame in real use; the tests draw nothing. Destroy it
/// before the display shuts down.
class BlankWaylandFrame {
public:
    explicit BlankWaylandFrame(const Display& display) {
#ifdef __linux__
        const core::window::NativeWindow native = display.nativeWindow();
        const auto* window = std::get_if<core::window::Wayland>(&native);
        if (window == nullptr) {
            return;
        }
        display_ = static_cast<wl_display*>(window->display);
        // Its own queue, so its roundtrip never dispatches SDL's events. The
        // registry is made on a wrapper already on that queue, so no event
        // reaches the default queue first.
        queue_ = wl_display_create_queue(display_);
        void* wrapper = wl_proxy_create_wrapper(display_);
        wl_proxy_set_queue(static_cast<wl_proxy*>(wrapper), queue_);
        registry_ = wl_display_get_registry(static_cast<wl_display*>(wrapper));
        wl_proxy_wrapper_destroy(wrapper);
        static constexpr wl_registry_listener kListener{
            .global =
                [](void* data, wl_registry* registry, std::uint32_t name, const char* interface, std::uint32_t) {
                    if (std::string_view{interface} == wl_shm_interface.name) {
                        *static_cast<wl_shm**>(data) = static_cast<wl_shm*>(
                            wl_registry_bind(registry, name, &wl_shm_interface, 1));
                    }
                },
            .global_remove = [](void*, wl_registry*, std::uint32_t) {},
        };
        wl_registry_add_listener(registry_, &kListener, static_cast<void*>(&shm_));
        wl_display_roundtrip_queue(display_, queue_);
        REQUIRE(shm_ != nullptr);

        constexpr std::int32_t kStride = 4; // one XRGB8888 pixel
        const int file = memfd_create("bfr-blank-frame", MFD_CLOEXEC);
        REQUIRE(file >= 0);
        REQUIRE(ftruncate(file, kStride) == 0);
        pool_ = wl_shm_create_pool(shm_, file, kStride);
        close(file);
        buffer_ = wl_shm_pool_create_buffer(pool_, 0, 1, 1, kStride, WL_SHM_FORMAT_XRGB8888);
        surface_ = static_cast<wl_surface*>(window->surface);
        present();
#else
        static_cast<void>(display);
#endif
    }

    ~BlankWaylandFrame() {
#ifdef __linux__
        if (buffer_ != nullptr) {
            wl_buffer_destroy(buffer_);
        }
        if (pool_ != nullptr) {
            wl_shm_pool_destroy(pool_);
        }
        if (shm_ != nullptr) {
            wl_shm_destroy(shm_);
        }
        if (registry_ != nullptr) {
            wl_registry_destroy(registry_);
        }
        if (queue_ != nullptr) {
            wl_event_queue_destroy(queue_);
        }
#endif
    }

    /// Attaches the frame and commits, as a renderer presents each frame.
    void present() const {
#ifdef __linux__
        if (surface_ == nullptr) {
            return;
        }
        wl_surface_attach(surface_, buffer_, 0, 0);
        wl_surface_damage(surface_, 0, 0, 1, 1);
        wl_surface_commit(surface_);
        wl_display_flush(display_);
#endif
    }

    BlankWaylandFrame(const BlankWaylandFrame&) = delete;
    BlankWaylandFrame& operator=(const BlankWaylandFrame&) = delete;
    BlankWaylandFrame(BlankWaylandFrame&&) = delete;
    BlankWaylandFrame& operator=(BlankWaylandFrame&&) = delete;

private:
#ifdef __linux__
    wl_display* display_ = nullptr;
    wl_event_queue* queue_ = nullptr;
    wl_registry* registry_ = nullptr;
    wl_shm* shm_ = nullptr;
    wl_shm_pool* pool_ = nullptr;
    wl_buffer* buffer_ = nullptr;
    wl_surface* surface_ = nullptr;
#endif
};

inline bool tickUntilSize(Display& display, std::uint32_t width, std::uint32_t height,
                          const BlankWaylandFrame* frame = nullptr) {
    constexpr Uint64 kTimeoutMs = 2000;
    constexpr Uint32 kPollMs = 10;
    const Uint64 deadline = SDL_GetTicks() + kTimeoutMs;
    for (;;) {
        display.tick();
        if (frame != nullptr) {
            frame->present();
        }
        if (display.width() == width && display.height() == height) {
            return true;
        }
        if (SDL_GetTicks() >= deadline) {
            return false;
        }
        SDL_Delay(kPollMs);
    }
}

/// Collects every event the display delivers.
struct Recorder {
    std::vector<Event> events;

    void attach(Display& display) {
        display.setEventCallback([this](const Event& event) { events.push_back(event); });
    }

    template <class T>
    [[nodiscard]] const T* last() const {
        for (const Event& event : events | std::views::reverse) {
            if (const T* match = std::get_if<T>(&event); match != nullptr) {
                return match;
            }
        }
        return nullptr;
    }

    template <class T>
    [[nodiscard]] std::size_t count() const {
        std::size_t total = 0;
        for (const Event& event : events) {
            if (std::holds_alternative<T>(event)) {
                ++total;
            }
        }
        return total;
    }
};

inline void pushWindowEvent(SDL_WindowID window, SDL_EventType type, Sint32 data1 = 0, Sint32 data2 = 0) {
    SDL_Event event;
    SDL_zero(event);
    event.type = type;
    event.window.windowID = window;
    event.window.data1 = data1;
    event.window.data2 = data2;
    REQUIRE(SDL_PushEvent(&event));
}

inline void pushKey(SDL_WindowID window, SDL_Scancode scancode, bool down, SDL_Keymod mod = SDL_KMOD_NONE,
                    bool repeat = false) {
    SDL_Event event;
    SDL_zero(event);
    event.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
    event.key.windowID = window;
    event.key.scancode = scancode;
    event.key.mod = mod;
    event.key.down = down;
    event.key.repeat = repeat;
    REQUIRE(SDL_PushEvent(&event));
}

inline void pushText(SDL_WindowID window, const char* text) {
    SDL_Event event;
    SDL_zero(event);
    event.type = SDL_EVENT_TEXT_INPUT;
    event.text.windowID = window;
    event.text.text = text;
    REQUIRE(SDL_PushEvent(&event));
}

inline void pushMouseMotion(SDL_WindowID window, float x, float y, float dx, float dy) {
    SDL_Event event;
    SDL_zero(event);
    event.type = SDL_EVENT_MOUSE_MOTION;
    event.motion.windowID = window;
    event.motion.x = x;
    event.motion.y = y;
    event.motion.xrel = dx;
    event.motion.yrel = dy;
    REQUIRE(SDL_PushEvent(&event));
}

inline void pushMouseButton(SDL_WindowID window, Uint8 button, bool down, float x, float y) {
    SDL_Event event;
    SDL_zero(event);
    event.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
    event.button.windowID = window;
    event.button.button = button;
    event.button.down = down;
    event.button.x = x;
    event.button.y = y;
    REQUIRE(SDL_PushEvent(&event));
}

inline void pushWheel(SDL_WindowID window, float x, float y) {
    SDL_Event event;
    SDL_zero(event);
    event.type = SDL_EVENT_MOUSE_WHEEL;
    event.wheel.windowID = window;
    event.wheel.x = x;
    event.wheel.y = y;
    REQUIRE(SDL_PushEvent(&event));
}

inline void pushGamepadButton(SDL_JoystickID which, Uint8 button, bool down) {
    SDL_Event event;
    SDL_zero(event);
    event.type = down ? SDL_EVENT_GAMEPAD_BUTTON_DOWN : SDL_EVENT_GAMEPAD_BUTTON_UP;
    event.gbutton.which = which;
    event.gbutton.button = button;
    event.gbutton.down = down;
    REQUIRE(SDL_PushEvent(&event));
}

/// A virtual gamepad SDL drives like a real one, so connect, buttons, axes
/// and disconnect all flow through the display's real translation path.
class VirtualGamepad {
public:
    VirtualGamepad() {
        SDL_VirtualJoystickDesc desc;
        SDL_INIT_INTERFACE(&desc);
        desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
        desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
        desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
        desc.name = "Virtual Pad";
        id_ = SDL_AttachVirtualJoystick(&desc);
        REQUIRE(id_ != 0);
        joystick_ = SDL_OpenJoystick(id_);
        REQUIRE(joystick_ != nullptr);
    }
    ~VirtualGamepad() { detach(); }
    VirtualGamepad(const VirtualGamepad&) = delete;
    VirtualGamepad& operator=(const VirtualGamepad&) = delete;
    VirtualGamepad(VirtualGamepad&&) = delete;
    VirtualGamepad& operator=(VirtualGamepad&&) = delete;

    void detach() {
        if (joystick_ != nullptr) {
            SDL_CloseJoystick(joystick_);
            joystick_ = nullptr;
        }
        if (id_ != 0) {
            SDL_DetachVirtualJoystick(id_);
            id_ = 0;
        }
    }

    void press(SDL_GamepadButton button, bool down) { REQUIRE(SDL_SetJoystickVirtualButton(joystick_, button, down)); }
    void move(SDL_GamepadAxis axis, Sint16 value) { REQUIRE(SDL_SetJoystickVirtualAxis(joystick_, axis, value)); }

private:
    SDL_JoystickID id_ = 0;
    SDL_Joystick* joystick_ = nullptr;
};

/// Ticks until the virtual pad connects and returns its event. Real pads on
/// the machine may hold other slots, so tests filter by this id.
inline GamepadConnected connectVirtualPad(Display& display, Recorder& recorder) {
    display.tick();
    for (const Event& event : recorder.events) {
        if (const auto* connected = std::get_if<GamepadConnected>(&event);
            connected != nullptr && connected->name == "Virtual Pad") {
            const GamepadConnected result = *connected;
            recorder.events.clear();
            return result;
        }
    }
    FAIL("the virtual pad never connected");
    return {};
}

template <class T>
std::vector<T> padEvents(const Recorder& recorder, std::uint8_t gamepadId) {
    std::vector<T> matches;
    for (const Event& event : recorder.events) {
        if (const T* match = std::get_if<T>(&event); match != nullptr && match->gamepadId == gamepadId) {
            matches.push_back(*match);
        }
    }
    return matches;
}

} // namespace bfr::platform::test
