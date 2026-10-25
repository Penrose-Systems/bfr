#pragma once

#include "input_codes.h"

#include <cstdint>
#include <string>
#include <variant>

namespace bfr::platform {

// Sizes and positions are in pixels.

// --- Window ---

struct WindowResized {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};
struct WindowClosed {};
struct WindowFocusGained {};
struct WindowFocusLost {};
struct WindowMinimized {};
struct WindowRestored {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// --- Keyboard ---

struct KeyPressed {
    Key key = Key::Unknown;
    ModifierFlags modifiers = 0;
};
struct KeyReleased {
    Key key = Key::Unknown;
    ModifierFlags modifiers = 0;
};
struct KeyRepeat {
    Key key = Key::Unknown;
    ModifierFlags modifiers = 0;
};
struct CharacterTyped {
    std::uint32_t codepoint = 0;
};

// --- Mouse ---

struct MouseMoved {
    float x = 0.F;
    float y = 0.F;
    float deltaX = 0.F;
    float deltaY = 0.F;
};
struct MouseButtonPressed {
    MouseButton button = MouseButton::Left;
    ModifierFlags modifiers = 0;
    float x = 0.F;
    float y = 0.F;
};
struct MouseButtonReleased {
    MouseButton button = MouseButton::Left;
    ModifierFlags modifiers = 0;
    float x = 0.F;
    float y = 0.F;
};
struct MouseScrolled {
    float deltaX = 0.F;
    float deltaY = 0.F;
};

// --- Touch ---

struct TouchBegin {
    std::uint32_t fingerId = 0;
    float x = 0.F;
    float y = 0.F;
    float pressure = 0.F;
};
struct TouchMoved {
    std::uint32_t fingerId = 0;
    float x = 0.F;
    float y = 0.F;
    float pressure = 0.F;
    float deltaX = 0.F;
    float deltaY = 0.F;
};
struct TouchEnd {
    std::uint32_t fingerId = 0;
    float x = 0.F;
    float y = 0.F;
};
struct TouchCancelled {
    std::uint32_t fingerId = 0;
};

// --- Gamepad ---

/// `gamepadId` is a slot, reused once its gamepad disconnects.
struct GamepadConnected {
    std::uint8_t gamepadId = 0;
    std::string name;
    bool hasGamepadMapping = false;
};
struct GamepadDisconnected {
    std::uint8_t gamepadId = 0;
};
struct GamepadButtonPressed {
    std::uint8_t gamepadId = 0;
    GamepadButton button = GamepadButton::South;
};
struct GamepadButtonReleased {
    std::uint8_t gamepadId = 0;
    GamepadButton button = GamepadButton::South;
};
struct GamepadAxisMoved {
    std::uint8_t gamepadId = 0;
    GamepadAxis axis = GamepadAxis::LeftStickX;
    float value = 0.F;
};

// ---

using Event =
    std::variant<WindowResized, WindowClosed, WindowFocusGained, WindowFocusLost, WindowMinimized, WindowRestored,
                 KeyPressed, KeyReleased, KeyRepeat, CharacterTyped, MouseMoved, MouseButtonPressed,
                 MouseButtonReleased, MouseScrolled, TouchBegin, TouchMoved, TouchEnd, TouchCancelled, GamepadConnected,
                 GamepadDisconnected, GamepadButtonPressed, GamepadButtonReleased, GamepadAxisMoved>;

} // namespace bfr::platform
