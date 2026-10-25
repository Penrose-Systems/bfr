#pragma once

#include "input_codes.h"

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>

#include <optional>

namespace bfr::platform {

// SDL input codes to BFR's. Codes BFR has no name for map to Key::Unknown
// or to nullopt, and the event is dropped.

[[nodiscard]] Key toKey(SDL_Scancode scancode);
[[nodiscard]] ModifierFlags toModifiers(SDL_Keymod mod);
[[nodiscard]] std::optional<MouseButton> toMouseButton(Uint8 button);
[[nodiscard]] std::optional<GamepadButton> toGamepadButton(SDL_GamepadButton button);
[[nodiscard]] std::optional<GamepadAxis> toGamepadAxis(SDL_GamepadAxis axis);

/// Sticks to [-1, 1], triggers to [0, 1].
[[nodiscard]] float normaliseAxis(GamepadAxis axis, Sint16 value);

} // namespace bfr::platform
