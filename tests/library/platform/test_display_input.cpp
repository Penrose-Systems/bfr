#include "display.h"
#include "display_test_support.h"
#include "event.h"
#include "input_codes.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_video.h>

#include <cstdint>
#include <variant>
#include <vector>

using namespace bfr::platform;
using namespace bfr::platform::test;

// --- Keyboard ---

TEST_CASE("Display key down with a modifier reports KeyPressed", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushKey(windowId(), SDL_SCANCODE_A, true, SDL_KMOD_LSHIFT);
    display->tick();
    REQUIRE(recorder.last<KeyPressed>() != nullptr);
    CHECK(recorder.last<KeyPressed>()->key == Key::A);
    CHECK(recorder.last<KeyPressed>()->modifiers == static_cast<ModifierFlags>(Modifier::Shift));
}

TEST_CASE("Display key up reports KeyReleased", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushKey(windowId(), SDL_SCANCODE_RETURN, false);
    display->tick();
    REQUIRE(recorder.last<KeyReleased>() != nullptr);
    CHECK(recorder.last<KeyReleased>()->key == Key::Enter);
    CHECK(recorder.last<KeyReleased>()->modifiers == static_cast<ModifierFlags>(Modifier::None));
}

TEST_CASE("Display repeated key down reports KeyRepeat", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushKey(windowId(), SDL_SCANCODE_SPACE, true, SDL_KMOD_RCTRL, true);
    display->tick();
    CHECK(recorder.count<KeyPressed>() == 0);
    REQUIRE(recorder.last<KeyRepeat>() != nullptr);
    CHECK(recorder.last<KeyRepeat>()->key == Key::Space);
    CHECK(recorder.last<KeyRepeat>()->modifiers == static_cast<ModifierFlags>(Modifier::Control));
}

TEST_CASE("Display translates every key family", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    const SDL_WindowID window = windowId();
    pushKey(window, SDL_SCANCODE_Z, true);
    pushKey(window, SDL_SCANCODE_0, true);
    pushKey(window, SDL_SCANCODE_F12, true);
    pushKey(window, SDL_SCANCODE_LEFT, true);
    pushKey(window, SDL_SCANCODE_KP_ENTER, true);
    pushKey(window, SDL_SCANCODE_LGUI, true);
    pushKey(window, SDL_SCANCODE_GRAVE, true);
    pushKey(window, SDL_SCANCODE_LANG1, true);
    display->tick();
    std::vector<Key> keys;
    keys.reserve(recorder.events.size());
    for (const Event& event : recorder.events) {
        keys.push_back(std::get<KeyPressed>(event).key);
    }
    CHECK(keys == std::vector<Key>{Key::Z, Key::Digit0, Key::F12, Key::ArrowLeft, Key::NumpadEnter, Key::LeftSuper,
                                   Key::GraveAccent, Key::Unknown});
}

TEST_CASE("Display combines every modifier", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    const SDL_Keymod mods = SDL_KMOD_RSHIFT | SDL_KMOD_LCTRL | SDL_KMOD_LALT | SDL_KMOD_RGUI | SDL_KMOD_CAPS |
                            SDL_KMOD_NUM;
    pushKey(windowId(), SDL_SCANCODE_B, true, mods);
    display->tick();
    REQUIRE(recorder.last<KeyPressed>() != nullptr);
    const ModifierFlags expected = static_cast<ModifierFlags>(Modifier::Shift) |
                                   static_cast<ModifierFlags>(Modifier::Control) |
                                   static_cast<ModifierFlags>(Modifier::Alt) |
                                   static_cast<ModifierFlags>(Modifier::Super) |
                                   static_cast<ModifierFlags>(Modifier::CapsLock) |
                                   static_cast<ModifierFlags>(Modifier::NumLock);
    CHECK(recorder.last<KeyPressed>()->modifiers == expected);
}

TEST_CASE("Display text input reports one CharacterTyped per code point", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushText(windowId(), "A\xC3\xA9\xE2\x82\xAC");
    display->tick();
    REQUIRE(recorder.count<CharacterTyped>() == 3);
    CHECK(std::get<CharacterTyped>(recorder.events[0]).codepoint == 0x41U);
    CHECK(std::get<CharacterTyped>(recorder.events[1]).codepoint == 0xE9U);
    CHECK(std::get<CharacterTyped>(recorder.events[2]).codepoint == 0x20ACU);
}

// --- Mouse ---

TEST_CASE("Display mouse motion reports position and delta", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    constexpr float kX = 100.F;
    constexpr float kY = 200.F;
    constexpr float kDeltaX = 5.F;
    constexpr float kDeltaY = -3.F;
    pushMouseMotion(windowId(), kX, kY, kDeltaX, kDeltaY);
    display->tick();
    REQUIRE(recorder.last<MouseMoved>() != nullptr);
    CHECK(recorder.last<MouseMoved>()->x == kX);
    CHECK(recorder.last<MouseMoved>()->y == kY);
    CHECK(recorder.last<MouseMoved>()->deltaX == kDeltaX);
    CHECK(recorder.last<MouseMoved>()->deltaY == kDeltaY);
}

TEST_CASE("Display mouse buttons report button and position", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    constexpr float kPressX = 320.F;
    constexpr float kPressY = 240.F;
    constexpr float kReleaseX = 100.F;
    constexpr float kReleaseY = 200.F;
    // Only the button is checked for the rest.
    constexpr float kOtherX = 1.F;
    constexpr float kOtherY = 2.F;
    const SDL_WindowID window = windowId();
    pushMouseButton(window, SDL_BUTTON_LEFT, true, kPressX, kPressY);
    pushMouseButton(window, SDL_BUTTON_RIGHT, false, kReleaseX, kReleaseY);
    pushMouseButton(window, SDL_BUTTON_MIDDLE, true, kOtherX, kOtherY);
    pushMouseButton(window, SDL_BUTTON_X1, true, kOtherX, kOtherY);
    pushMouseButton(window, SDL_BUTTON_X2, true, kOtherX, kOtherY);
    display->tick();
    REQUIRE(recorder.count<MouseButtonPressed>() == 4);
    REQUIRE(recorder.count<MouseButtonReleased>() == 1);
    const auto& pressed = std::get<MouseButtonPressed>(recorder.events[0]);
    CHECK(pressed.button == MouseButton::Left);
    CHECK(pressed.x == kPressX);
    CHECK(pressed.y == kPressY);
    const auto& released = std::get<MouseButtonReleased>(recorder.events[1]);
    CHECK(released.button == MouseButton::Right);
    CHECK(released.x == kReleaseX);
    CHECK(released.y == kReleaseY);
    CHECK(std::get<MouseButtonPressed>(recorder.events[2]).button == MouseButton::Middle);
    CHECK(std::get<MouseButtonPressed>(recorder.events[3]).button == MouseButton::Button4);
    CHECK(std::get<MouseButtonPressed>(recorder.events[4]).button == MouseButton::Button5);
}

TEST_CASE("Display mouse wheel reports both axes", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    constexpr float kWheelX = -1.F;
    constexpr float kWheelY = 3.F;
    pushWheel(windowId(), kWheelX, kWheelY);
    display->tick();
    REQUIRE(recorder.last<MouseScrolled>() != nullptr);
    CHECK(recorder.last<MouseScrolled>()->deltaX == kWheelX);
    CHECK(recorder.last<MouseScrolled>()->deltaY == kWheelY);
}

// --- Callback wiring ---

TEST_CASE("Display with no callback discards events", "[platform][unit]") {
    auto display = openDisplay();
    pushKey(windowId(), SDL_SCANCODE_SPACE, true);
    display->tick();
}

TEST_CASE("Display second callback replaces the first", "[platform][unit]") {
    auto display = openDisplay();
    int first = 0;
    int second = 0;
    display->setEventCallback([&](const Event&) { ++first; });
    display->setEventCallback([&](const Event&) { ++second; });
    pushKey(windowId(), SDL_SCANCODE_A, true);
    display->tick();
    CHECK(first == 0);
    CHECK(second == 1);
}

TEST_CASE("Display delivers events in order", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    const SDL_WindowID window = windowId();
    pushKey(window, SDL_SCANCODE_A, true);
    pushKey(window, SDL_SCANCODE_A, false);
    pushWheel(window, 0.F, 1.F);
    display->tick();
    REQUIRE(recorder.events.size() == 3);
    CHECK(std::holds_alternative<KeyPressed>(recorder.events[0]));
    CHECK(std::holds_alternative<KeyReleased>(recorder.events[1]));
    CHECK(std::holds_alternative<MouseScrolled>(recorder.events[2]));
}

// --- Gamepad ---

TEST_CASE("Display reports a gamepad connecting and disconnecting", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    GamepadConnected connected;
    {
        const VirtualGamepad pad;
        connected = connectVirtualPad(*display, recorder);
    }
    CHECK(connected.hasGamepadMapping);
    display->tick();
    CHECK(padEvents<GamepadDisconnected>(recorder, connected.gamepadId).size() == 1);
}

TEST_CASE("Display reuses a disconnected gamepad's slot", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    std::uint8_t first = 0;
    {
        const VirtualGamepad pad;
        first = connectVirtualPad(*display, recorder).gamepadId;
    }
    display->tick();
    const VirtualGamepad again;
    CHECK(connectVirtualPad(*display, recorder).gamepadId == first);
}

TEST_CASE("Display translates gamepad buttons", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    VirtualGamepad pad;
    const std::uint8_t id = connectVirtualPad(*display, recorder).gamepadId;
    pad.press(SDL_GAMEPAD_BUTTON_SOUTH, true);
    pad.press(SDL_GAMEPAD_BUTTON_BACK, true);
    pad.press(SDL_GAMEPAD_BUTTON_GUIDE, true);
    pad.press(SDL_GAMEPAD_BUTTON_DPAD_LEFT, true);
    display->tick();
    pad.press(SDL_GAMEPAD_BUTTON_SOUTH, false);
    display->tick();
    std::vector<GamepadButton> pressed;
    for (const GamepadButtonPressed& event : padEvents<GamepadButtonPressed>(recorder, id)) {
        pressed.push_back(event.button);
    }
    CHECK(pressed == std::vector<GamepadButton>{GamepadButton::South, GamepadButton::Select, GamepadButton::Home,
                                                GamepadButton::DPadLeft});
    const auto released = padEvents<GamepadButtonReleased>(recorder, id);
    REQUIRE(released.size() == 1);
    CHECK(released[0].button == GamepadButton::South);
}

TEST_CASE("Display normalises gamepad sticks to [-1, 1] and triggers to [0, 1]", "[platform][unit]") {
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    VirtualGamepad pad;
    const std::uint8_t id = connectVirtualPad(*display, recorder).gamepadId;
    pad.move(SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
    pad.move(SDL_GAMEPAD_AXIS_RIGHTY, SDL_JOYSTICK_AXIS_MIN);
    pad.move(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, SDL_JOYSTICK_AXIS_MAX);
    display->tick();
    const auto moves = padEvents<GamepadAxisMoved>(recorder, id);
    REQUIRE(moves.size() == 3);
    CHECK(moves[0].axis == GamepadAxis::LeftStickX);
    CHECK(moves[0].value == Catch::Approx(1.F));
    CHECK(moves[1].axis == GamepadAxis::RightStickY);
    CHECK(moves[1].value == Catch::Approx(-1.F));
    CHECK(moves[2].axis == GamepadAxis::RightTrigger);
    CHECK(moves[2].value == Catch::Approx(1.F));
}

TEST_CASE("Display reports an axis only once it moves the noise floor from the last report", "[platform][unit]") {
    constexpr Sint16 kJitter = 1000;
    constexpr Sint16 kSecondJitter = 2000;
    auto config = hiddenConfig();
    constexpr float kNoiseFloor = 0.1F;
    config.input.gamepadNoiseFloor = kNoiseFloor;
    auto display = openDisplay(config);
    Recorder recorder;
    recorder.attach(*display);
    VirtualGamepad pad;
    const std::uint8_t id = connectVirtualPad(*display, recorder).gamepadId;
    pad.move(SDL_GAMEPAD_AXIS_LEFTX, kJitter);
    display->tick();
    pad.move(SDL_GAMEPAD_AXIS_LEFTX, kSecondJitter);
    display->tick();
    CHECK(padEvents<GamepadAxisMoved>(recorder, id).empty());
    pad.move(SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
    display->tick();
    REQUIRE(padEvents<GamepadAxisMoved>(recorder, id).size() == 1);
    CHECK(padEvents<GamepadAxisMoved>(recorder, id).back().value == Catch::Approx(1.F));
    pad.move(SDL_GAMEPAD_AXIS_LEFTX, kJitter);
    display->tick();
    const auto moves = padEvents<GamepadAxisMoved>(recorder, id);
    REQUIRE(moves.size() == 2);
    CHECK(moves.back().value == Catch::Approx(static_cast<float>(kJitter) / SDL_JOYSTICK_AXIS_MAX));
}

TEST_CASE("Display ignores gamepad events for an unknown device", "[platform][unit]") {
    constexpr SDL_JoystickID kUnknownDevice = 999;
    auto display = openDisplay();
    Recorder recorder;
    recorder.attach(*display);
    pushGamepadButton(kUnknownDevice, SDL_GAMEPAD_BUTTON_SOUTH, true);
    display->tick();
    CHECK(recorder.count<GamepadButtonPressed>() == 0);
}
