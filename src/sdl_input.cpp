#include "sdl_input.h"

#include "input_codes.h"

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <utility>

namespace bfr::platform {

namespace {

constexpr std::array kScancodes = std::to_array<std::pair<SDL_Scancode, Key>>({
    {SDL_SCANCODE_A, Key::A},
    {SDL_SCANCODE_B, Key::B},
    {SDL_SCANCODE_C, Key::C},
    {SDL_SCANCODE_D, Key::D},
    {SDL_SCANCODE_E, Key::E},
    {SDL_SCANCODE_F, Key::F},
    {SDL_SCANCODE_G, Key::G},
    {SDL_SCANCODE_H, Key::H},
    {SDL_SCANCODE_I, Key::I},
    {SDL_SCANCODE_J, Key::J},
    {SDL_SCANCODE_K, Key::K},
    {SDL_SCANCODE_L, Key::L},
    {SDL_SCANCODE_M, Key::M},
    {SDL_SCANCODE_N, Key::N},
    {SDL_SCANCODE_O, Key::O},
    {SDL_SCANCODE_P, Key::P},
    {SDL_SCANCODE_Q, Key::Q},
    {SDL_SCANCODE_R, Key::R},
    {SDL_SCANCODE_S, Key::S},
    {SDL_SCANCODE_T, Key::T},
    {SDL_SCANCODE_U, Key::U},
    {SDL_SCANCODE_V, Key::V},
    {SDL_SCANCODE_W, Key::W},
    {SDL_SCANCODE_X, Key::X},
    {SDL_SCANCODE_Y, Key::Y},
    {SDL_SCANCODE_Z, Key::Z},
    {SDL_SCANCODE_0, Key::Digit0},
    {SDL_SCANCODE_1, Key::Digit1},
    {SDL_SCANCODE_2, Key::Digit2},
    {SDL_SCANCODE_3, Key::Digit3},
    {SDL_SCANCODE_4, Key::Digit4},
    {SDL_SCANCODE_5, Key::Digit5},
    {SDL_SCANCODE_6, Key::Digit6},
    {SDL_SCANCODE_7, Key::Digit7},
    {SDL_SCANCODE_8, Key::Digit8},
    {SDL_SCANCODE_9, Key::Digit9},
    {SDL_SCANCODE_SPACE, Key::Space},
    {SDL_SCANCODE_APOSTROPHE, Key::Apostrophe},
    {SDL_SCANCODE_COMMA, Key::Comma},
    {SDL_SCANCODE_MINUS, Key::Minus},
    {SDL_SCANCODE_PERIOD, Key::Period},
    {SDL_SCANCODE_SLASH, Key::Slash},
    {SDL_SCANCODE_SEMICOLON, Key::Semicolon},
    {SDL_SCANCODE_EQUALS, Key::Equal},
    {SDL_SCANCODE_LEFTBRACKET, Key::LeftBracket},
    {SDL_SCANCODE_BACKSLASH, Key::Backslash},
    {SDL_SCANCODE_RIGHTBRACKET, Key::RightBracket},
    {SDL_SCANCODE_GRAVE, Key::GraveAccent},
    {SDL_SCANCODE_F1, Key::F1},
    {SDL_SCANCODE_F2, Key::F2},
    {SDL_SCANCODE_F3, Key::F3},
    {SDL_SCANCODE_F4, Key::F4},
    {SDL_SCANCODE_F5, Key::F5},
    {SDL_SCANCODE_F6, Key::F6},
    {SDL_SCANCODE_F7, Key::F7},
    {SDL_SCANCODE_F8, Key::F8},
    {SDL_SCANCODE_F9, Key::F9},
    {SDL_SCANCODE_F10, Key::F10},
    {SDL_SCANCODE_F11, Key::F11},
    {SDL_SCANCODE_F12, Key::F12},
    {SDL_SCANCODE_F13, Key::F13},
    {SDL_SCANCODE_F14, Key::F14},
    {SDL_SCANCODE_F15, Key::F15},
    {SDL_SCANCODE_F16, Key::F16},
    {SDL_SCANCODE_F17, Key::F17},
    {SDL_SCANCODE_F18, Key::F18},
    {SDL_SCANCODE_F19, Key::F19},
    {SDL_SCANCODE_F20, Key::F20},
    {SDL_SCANCODE_F21, Key::F21},
    {SDL_SCANCODE_F22, Key::F22},
    {SDL_SCANCODE_F23, Key::F23},
    {SDL_SCANCODE_F24, Key::F24},
    {SDL_SCANCODE_ESCAPE, Key::Escape},
    {SDL_SCANCODE_RETURN, Key::Enter},
    {SDL_SCANCODE_TAB, Key::Tab},
    {SDL_SCANCODE_BACKSPACE, Key::Backspace},
    {SDL_SCANCODE_INSERT, Key::Insert},
    {SDL_SCANCODE_DELETE, Key::Delete},
    {SDL_SCANCODE_HOME, Key::Home},
    {SDL_SCANCODE_END, Key::End},
    {SDL_SCANCODE_PAGEUP, Key::PageUp},
    {SDL_SCANCODE_PAGEDOWN, Key::PageDown},
    {SDL_SCANCODE_PRINTSCREEN, Key::PrintScreen},
    {SDL_SCANCODE_SCROLLLOCK, Key::ScrollLock},
    {SDL_SCANCODE_PAUSE, Key::Pause},
    {SDL_SCANCODE_CAPSLOCK, Key::CapsLock},
    {SDL_SCANCODE_NUMLOCKCLEAR, Key::NumLock},
    {SDL_SCANCODE_UP, Key::ArrowUp},
    {SDL_SCANCODE_DOWN, Key::ArrowDown},
    {SDL_SCANCODE_LEFT, Key::ArrowLeft},
    {SDL_SCANCODE_RIGHT, Key::ArrowRight},
    {SDL_SCANCODE_LSHIFT, Key::LeftShift},
    {SDL_SCANCODE_RSHIFT, Key::RightShift},
    {SDL_SCANCODE_LCTRL, Key::LeftControl},
    {SDL_SCANCODE_RCTRL, Key::RightControl},
    {SDL_SCANCODE_LALT, Key::LeftAlt},
    {SDL_SCANCODE_RALT, Key::RightAlt},
    {SDL_SCANCODE_LGUI, Key::LeftSuper},
    {SDL_SCANCODE_RGUI, Key::RightSuper},
    {SDL_SCANCODE_APPLICATION, Key::Menu},
    {SDL_SCANCODE_MENU, Key::Menu},
    {SDL_SCANCODE_KP_0, Key::Numpad0},
    {SDL_SCANCODE_KP_1, Key::Numpad1},
    {SDL_SCANCODE_KP_2, Key::Numpad2},
    {SDL_SCANCODE_KP_3, Key::Numpad3},
    {SDL_SCANCODE_KP_4, Key::Numpad4},
    {SDL_SCANCODE_KP_5, Key::Numpad5},
    {SDL_SCANCODE_KP_6, Key::Numpad6},
    {SDL_SCANCODE_KP_7, Key::Numpad7},
    {SDL_SCANCODE_KP_8, Key::Numpad8},
    {SDL_SCANCODE_KP_9, Key::Numpad9},
    {SDL_SCANCODE_KP_PERIOD, Key::NumpadDecimal},
    {SDL_SCANCODE_KP_DIVIDE, Key::NumpadDivide},
    {SDL_SCANCODE_KP_MULTIPLY, Key::NumpadMultiply},
    {SDL_SCANCODE_KP_MINUS, Key::NumpadSubtract},
    {SDL_SCANCODE_KP_PLUS, Key::NumpadAdd},
    {SDL_SCANCODE_KP_ENTER, Key::NumpadEnter},
    {SDL_SCANCODE_KP_EQUALS, Key::NumpadEqual},
});

/// Indexed by scancode; every unnamed scancode is Key::Unknown.
constexpr std::array<Key, SDL_SCANCODE_COUNT> kKeys = [] {
    std::array<Key, SDL_SCANCODE_COUNT> keys{};
    for (const auto& [scancode, key] : kScancodes) {
        keys.at(static_cast<std::size_t>(scancode)) = key;
    }
    return keys;
}();

struct ModifierBit {
    SDL_Keymod sdl = SDL_KMOD_NONE;
    Modifier modifier = Modifier::None;
};

constexpr std::array kModifiers = std::to_array<ModifierBit>({
    {.sdl = SDL_KMOD_SHIFT, .modifier = Modifier::Shift},
    {.sdl = SDL_KMOD_CTRL, .modifier = Modifier::Control},
    {.sdl = SDL_KMOD_ALT, .modifier = Modifier::Alt},
    {.sdl = SDL_KMOD_GUI, .modifier = Modifier::Super},
    {.sdl = SDL_KMOD_CAPS, .modifier = Modifier::CapsLock},
    {.sdl = SDL_KMOD_NUM, .modifier = Modifier::NumLock},
});

constexpr auto kAxisMax = static_cast<float>(SDL_JOYSTICK_AXIS_MAX);

} // namespace

Key toKey(SDL_Scancode scancode) {
    if (scancode < 0 || scancode >= SDL_SCANCODE_COUNT) {
        return Key::Unknown;
    }
    return kKeys.at(static_cast<std::size_t>(scancode));
}

ModifierFlags toModifiers(SDL_Keymod mod) {
    ModifierFlags flags = 0;
    for (const ModifierBit& bit : kModifiers) {
        if ((mod & bit.sdl) != 0) {
            flags |= static_cast<ModifierFlags>(bit.modifier);
        }
    }
    return flags;
}

std::optional<MouseButton> toMouseButton(Uint8 button) {
    switch (button) {
        case SDL_BUTTON_LEFT:
            return MouseButton::Left;
        case SDL_BUTTON_RIGHT:
            return MouseButton::Right;
        case SDL_BUTTON_MIDDLE:
            return MouseButton::Middle;
        case SDL_BUTTON_X1:
            return MouseButton::Button4;
        case SDL_BUTTON_X2:
            return MouseButton::Button5;
        default:
            return std::nullopt;
    }
}

std::optional<GamepadButton> toGamepadButton(SDL_GamepadButton button) {
    switch (button) {
        case SDL_GAMEPAD_BUTTON_SOUTH:
            return GamepadButton::South;
        case SDL_GAMEPAD_BUTTON_EAST:
            return GamepadButton::East;
        case SDL_GAMEPAD_BUTTON_WEST:
            return GamepadButton::West;
        case SDL_GAMEPAD_BUTTON_NORTH:
            return GamepadButton::North;
        case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER:
            return GamepadButton::LeftBumper;
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER:
            return GamepadButton::RightBumper;
        case SDL_GAMEPAD_BUTTON_LEFT_STICK:
            return GamepadButton::LeftStick;
        case SDL_GAMEPAD_BUTTON_RIGHT_STICK:
            return GamepadButton::RightStick;
        case SDL_GAMEPAD_BUTTON_DPAD_UP:
            return GamepadButton::DPadUp;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
            return GamepadButton::DPadDown;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
            return GamepadButton::DPadLeft;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
            return GamepadButton::DPadRight;
        case SDL_GAMEPAD_BUTTON_START:
            return GamepadButton::Start;
        case SDL_GAMEPAD_BUTTON_BACK:
            return GamepadButton::Select;
        case SDL_GAMEPAD_BUTTON_GUIDE:
            return GamepadButton::Home;
        case SDL_GAMEPAD_BUTTON_TOUCHPAD:
            return GamepadButton::Touchpad;
        default:
            return std::nullopt;
    }
}

std::optional<GamepadAxis> toGamepadAxis(SDL_GamepadAxis axis) {
    switch (axis) {
        case SDL_GAMEPAD_AXIS_LEFTX:
            return GamepadAxis::LeftStickX;
        case SDL_GAMEPAD_AXIS_LEFTY:
            return GamepadAxis::LeftStickY;
        case SDL_GAMEPAD_AXIS_RIGHTX:
            return GamepadAxis::RightStickX;
        case SDL_GAMEPAD_AXIS_RIGHTY:
            return GamepadAxis::RightStickY;
        case SDL_GAMEPAD_AXIS_LEFT_TRIGGER:
            return GamepadAxis::LeftTrigger;
        case SDL_GAMEPAD_AXIS_RIGHT_TRIGGER:
            return GamepadAxis::RightTrigger;
        default:
            return std::nullopt;
    }
}

float normaliseAxis(GamepadAxis axis, Sint16 value) {
    const bool trigger = axis == GamepadAxis::LeftTrigger || axis == GamepadAxis::RightTrigger;
    return std::clamp(static_cast<float>(value) / kAxisMax, trigger ? 0.F : -1.F, 1.F);
}

} // namespace bfr::platform
