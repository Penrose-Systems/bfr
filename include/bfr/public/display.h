#pragma once

#include "error.h"
#include "event.h"
#include "window.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bfr::platform {

inline constexpr std::uint32_t kDefaultDisplayWidth = 1280;
inline constexpr std::uint32_t kDefaultDisplayHeight = 720;
inline constexpr float kDefaultGamepadNoiseFloor = 0.01F;

/// A display mode in pixels.
struct VideoMode {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t refreshRate = 0;
};

/// Thresholds applied to raw input before it becomes an event. Not a gameplay
/// deadzone: it only suppresses the jitter a stick reports while untouched.
struct InputConfig {
    /// An axis is reported once it moves this far from its last reported value.
    float gamepadNoiseFloor = kDefaultGamepadNoiseFloor;
};

struct DisplayConfig {
    std::string title = "BFR";
    /// The window size the OS shows, in its window coordinates. Everything the
    /// display reports is in pixels; on a high-density screen the two differ.
    std::uint32_t width = kDefaultDisplayWidth;
    std::uint32_t height = kDefaultDisplayHeight;
    /// Borderless, on the window's monitor, at the desktop mode.
    bool fullscreen = false;
    bool vsync = true;
    bool resizable = false;
    bool visible = true;
    InputConfig input;
};

/// One OS window and its input. Every call must come from the main thread,
/// because the OS window systems require it. Sizes and positions are pixels.
class Display {
public:
    using EventCallback = std::function<void(const Event&)>;

    /// The backend for this build. Never null.
    [[nodiscard]] static std::unique_ptr<Display> create();

    virtual ~Display() = default;
    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;
    Display(Display&&) = delete;
    Display& operator=(Display&&) = delete;

    /// Opens the window. Fails with DisplayAlreadyInitialised while open,
    /// DisplayWindowFailed for a zero extent or a window the OS refuses, and
    /// DisplayBackendFailed when the window system cannot start.
    [[nodiscard]] virtual core::Result<void> init(const DisplayConfig& config) = 0;
    /// Closes the window. Safe to repeat; init may follow.
    virtual void shutdown() = 0;
    /// Delivers every pending event to the callback, in order. Does nothing
    /// before init.
    virtual void tick() = 0;

    /// False once the user or the OS asks to close, and before init.
    [[nodiscard]] virtual bool isOpen() const = 0;
    [[nodiscard]] virtual std::uint32_t width() const = 0;
    [[nodiscard]] virtual std::uint32_t height() const = 0;
    [[nodiscard]] virtual bool vsyncEnabled() const = 0;
    /// Enters or leaves borderless fullscreen without changing the monitor's
    /// mode; leaving restores the window's previous size. Fails with
    /// DisplayNotInitialised before init and DisplayFullscreenFailed when
    /// refused.
    [[nodiscard]] virtual core::Result<void> setFullscreen(bool fullscreen) = 0;
    [[nodiscard]] virtual bool isFullscreen() const = 0;

    /// The modes of the window's monitor, or the primary monitor before init.
    [[nodiscard]] virtual std::vector<VideoMode> videoModes() const = 0;
    [[nodiscard]] virtual VideoMode currentMode() const = 0;

    /// The window's handles for surface creation; std::monostate when not open.
    /// Valid until shutdown.
    [[nodiscard]] virtual core::window::NativeWindow nativeWindow() const = 0;

    /// Replaces the callback; an empty one discards events. It runs inside
    /// tick().
    virtual void setEventCallback(EventCallback callback) = 0;

protected:
    Display() = default;
};

} // namespace bfr::platform
