#pragma once

#include "display.h"
#include "error.h"
#include "event.h"
#include "input_codes.h"
#include "window.h"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_video.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bfr::platform {

/// The desktop display, through SDL3.
class SdlDisplay final : public Display {
public:
    SdlDisplay() = default;
    ~SdlDisplay() override;
    SdlDisplay(const SdlDisplay&) = delete;
    SdlDisplay& operator=(const SdlDisplay&) = delete;
    SdlDisplay(SdlDisplay&&) = delete;
    SdlDisplay& operator=(SdlDisplay&&) = delete;

    [[nodiscard]] core::Result<void> init(const DisplayConfig& config) override;
    void shutdown() override;
    void tick() override;

    [[nodiscard]] bool isOpen() const override;
    [[nodiscard]] std::uint32_t width() const override;
    [[nodiscard]] std::uint32_t height() const override;
    [[nodiscard]] bool vsyncEnabled() const override;
    [[nodiscard]] core::Result<void> setFullscreen(bool fullscreen) override;
    [[nodiscard]] bool isFullscreen() const override;

    [[nodiscard]] std::vector<VideoMode> videoModes() const override;
    [[nodiscard]] VideoMode currentMode() const override;

    [[nodiscard]] core::window::NativeWindow nativeWindow() const override;

    void setEventCallback(EventCallback callback) override;

private:
    static constexpr std::size_t kMaxGamepads = 16;
    static constexpr auto kAxisCount = static_cast<std::size_t>(GamepadAxis::Count);

    /// An open gamepad in its slot; the slot index is its gamepadId.
    struct Gamepad {
        SDL_Gamepad* handle = nullptr;
        SDL_JoystickID device = 0;
        std::array<float, kAxisCount> reported{};
    };

    void dispatch(const Event& event) const;
    void close();
    void resize(std::uint32_t pixelWidth, std::uint32_t pixelHeight);
    void readPixelSize();
    void logOpened() const;
    [[nodiscard]] float pixelDensity() const;
    [[nodiscard]] SDL_DisplayID monitor() const;

    void handle(const SDL_Event& event);
    void handleWindow(const SDL_WindowEvent& event);
    void handleKey(const SDL_KeyboardEvent& event) const;
    void handleText(const SDL_TextInputEvent& event) const;
    void handleMotion(const SDL_MouseMotionEvent& event) const;
    void handleButton(const SDL_MouseButtonEvent& event) const;
    void handleWheel(const SDL_MouseWheelEvent& event) const;
    void handleGamepadAdded(SDL_JoystickID device);
    void handleGamepadRemoved(SDL_JoystickID device);
    void handleGamepadButton(const SDL_GamepadButtonEvent& event) const;
    void handleGamepadAxis(const SDL_GamepadAxisEvent& event);
    [[nodiscard]] std::size_t slotOf(SDL_JoystickID device) const;
    void closeGamepads();

    SDL_Window* window_ = nullptr;
    // Only with SDL's cocoa driver: the view that holds the window's Metal layer.
    SDL_MetalView metalView_ = nullptr;
    SDL_WindowID windowId_ = 0;
    DisplayConfig config_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    bool open_ = false;
    EventCallback callback_;
    std::array<Gamepad, kMaxGamepads> gamepads_{};
};

} // namespace bfr::platform
