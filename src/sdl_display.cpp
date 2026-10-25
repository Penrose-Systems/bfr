#include "sdl_display.h"

#include "display.h"
#include "error.h"
#include "event.h"
#include "input_codes.h"
#include "log.h"
#include "sdl_input.h"
#include "window.h"

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_version.h>
#include <SDL3/SDL_video.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::platform {

namespace {

constexpr SDL_InitFlags kSubsystems = SDL_INIT_VIDEO | SDL_INIT_GAMEPAD;

// SDL's name for its video driver, such as "x11"; empty before init.
std::string_view videoDriver() {
    const char* name = SDL_GetCurrentVideoDriver();
    return name != nullptr ? name : "";
}

std::uint32_t toPixels(float size, float density) {
    return static_cast<std::uint32_t>(std::lround(size * density));
}

SDL_Window* createWindow(const DisplayConfig& config) {
    const SDL_PropertiesID properties = SDL_CreateProperties();
    if (properties == 0) {
        return nullptr;
    }
    SDL_SetStringProperty(properties, SDL_PROP_WINDOW_CREATE_TITLE_STRING, config.title.c_str());
    SDL_SetNumberProperty(properties, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, config.width);
    SDL_SetNumberProperty(properties, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, config.height);
    SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN, true);
    SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, config.resizable);
    SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_HIDDEN_BOOLEAN, !config.visible);
    SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_FULLSCREEN_BOOLEAN, config.fullscreen);
    // The RHI creates the graphics context. Without this, SDL makes every
    // macOS window an OpenGL window.
    SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_EXTERNAL_GRAPHICS_CONTEXT_BOOLEAN, true);
    SDL_Window* window = SDL_CreateWindowWithProperties(properties);
    SDL_DestroyProperties(properties);
    return window;
}

VideoMode toVideoMode(const SDL_DisplayMode& mode) {
    return {.width = toPixels(static_cast<float>(mode.w), mode.pixel_density),
            .height = toPixels(static_cast<float>(mode.h), mode.pixel_density),
            .refreshRate = static_cast<std::uint32_t>(std::lround(mode.refresh_rate))};
}

} // namespace

std::unique_ptr<Display> Display::create() {
    return std::make_unique<SdlDisplay>();
}

SdlDisplay::~SdlDisplay() {
    shutdown();
}

core::Result<void> SdlDisplay::init(const DisplayConfig& config) {
    if (window_ != nullptr) {
        return core::failure(core::ErrorCode::DisplayAlreadyInitialised, "the display is already open");
    }
    if (config.width == 0 || config.height == 0) {
        return core::failure(core::ErrorCode::DisplayWindowFailed, "the window extent must be non-zero");
    }
    if (!SDL_InitSubSystem(kSubsystems)) {
        return core::failure(core::ErrorCode::DisplayBackendFailed, SDL_GetError());
    }

    window_ = createWindow(config);
    if (window_ == nullptr) {
        std::string error = SDL_GetError();
        SDL_QuitSubSystem(kSubsystems);
        return core::failure(core::ErrorCode::DisplayWindowFailed, std::move(error));
    }

    // A Vulkan surface on macOS draws to a CAMetalLayer, which SDL gives
    // through a Metal view.
    if (videoDriver() == "cocoa") {
        metalView_ = SDL_Metal_CreateView(window_);
        if (metalView_ == nullptr) {
            std::string error = SDL_GetError();
            shutdown();
            return core::failure(core::ErrorCode::DisplayWindowFailed, std::move(error));
        }
    }

    windowId_ = SDL_GetWindowID(window_);
    config_ = config;
    open_ = true;
    // SDL delivers typed text only after this.
    SDL_StartTextInput(window_);
    // A window manager applies the requested state later; init returns with it.
    SDL_SyncWindow(window_);
    readPixelSize();
    logOpened();
    return {};
}

void SdlDisplay::shutdown() {
    if (window_ == nullptr) {
        return;
    }
    closeGamepads();
    SDL_StopTextInput(window_);
    if (metalView_ != nullptr) {
        SDL_Metal_DestroyView(metalView_);
        metalView_ = nullptr;
    }
    SDL_DestroyWindow(window_);
    window_ = nullptr;
    windowId_ = 0;
    width_ = 0;
    height_ = 0;
    open_ = false;
    SDL_QuitSubSystem(kSubsystems);
    // Platform is SDL's only owner, so once its subsystems are down SDL's
    // global state goes too; otherwise the leak checker reports it.
    if (SDL_WasInit(0) == 0) {
        SDL_Quit();
    }
    BFR_LOG_INFO("display closed");
}

void SdlDisplay::tick() {
    if (window_ == nullptr) {
        return;
    }
    SDL_Event event{};
    while (SDL_PollEvent(&event)) {
        handle(event);
    }
}

bool SdlDisplay::isOpen() const {
    return open_;
}

std::uint32_t SdlDisplay::width() const {
    return width_;
}

std::uint32_t SdlDisplay::height() const {
    return height_;
}

bool SdlDisplay::vsyncEnabled() const {
    return config_.vsync;
}

core::Result<void> SdlDisplay::setFullscreen(bool fullscreen) {
    if (window_ == nullptr) {
        return core::failure(core::ErrorCode::DisplayNotInitialised, "the display is not open");
    }
    if (!SDL_SetWindowFullscreen(window_, fullscreen)) {
        return core::failure(core::ErrorCode::DisplayFullscreenFailed, SDL_GetError());
    }
    // The new size arrives as an event, even when a window manager finishes
    // after this wait times out.
    SDL_SyncWindow(window_);
    BFR_LOG_INFO("fullscreen {}", fullscreen ? "on" : "off");
    return {};
}

bool SdlDisplay::isFullscreen() const {
    return window_ != nullptr && (SDL_GetWindowFlags(window_) & SDL_WINDOW_FULLSCREEN) != 0;
}

std::vector<VideoMode> SdlDisplay::videoModes() const {
    const SDL_DisplayID display = monitor();
    int count = 0;
    const std::unique_ptr<SDL_DisplayMode*, decltype(&SDL_free)> modes(SDL_GetFullscreenDisplayModes(display, &count),
                                                                       &SDL_free);
    std::vector<VideoMode> result;
    if (modes != nullptr) {
        const std::span<SDL_DisplayMode*> list(modes.get(), static_cast<std::size_t>(count));
        result.reserve(list.size());
        for (const SDL_DisplayMode* mode : list) {
            result.push_back(toVideoMode(*mode));
        }
    }
    // A display that lists no modes still runs its desktop mode, which is the
    // one borderless fullscreen uses.
    if (result.empty()) {
        if (const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(display); desktop != nullptr) {
            result.push_back(toVideoMode(*desktop));
        }
    }
    return result;
}

VideoMode SdlDisplay::currentMode() const {
    const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(monitor());
    return mode != nullptr ? toVideoMode(*mode) : VideoMode{};
}

core::window::NativeWindow SdlDisplay::nativeWindow() const {
    if (window_ == nullptr) {
        return {};
    }
    // Each SDL video driver reports its handles as window properties.
    const std::string_view driver = videoDriver();
    const SDL_PropertiesID properties = SDL_GetWindowProperties(window_);
    const auto pointer = [properties](const char* name) {
        return SDL_GetPointerProperty(properties, name, nullptr);
    };
    if (driver == "x11") {
        return core::window::Xlib{
            .display = pointer(SDL_PROP_WINDOW_X11_DISPLAY_POINTER),
            .window = static_cast<std::uint64_t>(
                SDL_GetNumberProperty(properties, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0)),
        };
    }
    if (driver == "wayland") {
        return core::window::Wayland{.display = pointer(SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER),
                                     .surface = pointer(SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER)};
    }
    if (driver == "windows") {
        return core::window::Win32{.instance = pointer(SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER),
                                   .window = pointer(SDL_PROP_WINDOW_WIN32_HWND_POINTER)};
    }
    if (driver == "cocoa") {
        return core::window::Metal{.layer = SDL_Metal_GetLayer(metalView_)};
    }
    if (driver == "offscreen") {
        return core::window::Headless{};
    }
    return {};
}

void SdlDisplay::setEventCallback(EventCallback callback) {
    callback_ = std::move(callback);
}

void SdlDisplay::dispatch(const Event& event) const {
    if (callback_) {
        callback_(event);
    }
}

void SdlDisplay::close() {
    if (!open_) {
        return;
    }
    open_ = false;
    dispatch(WindowClosed{});
}

void SdlDisplay::resize(std::uint32_t pixelWidth, std::uint32_t pixelHeight) {
    // SDL also reports the size a window opens at; only a change is a resize.
    if (pixelWidth == width_ && pixelHeight == height_) {
        return;
    }
    width_ = pixelWidth;
    height_ = pixelHeight;
    dispatch(WindowResized{.width = width_, .height = height_});
}

void SdlDisplay::readPixelSize() {
    int pixelWidth = 0;
    int pixelHeight = 0;
    SDL_GetWindowSizeInPixels(window_, &pixelWidth, &pixelHeight);
    width_ = static_cast<std::uint32_t>(pixelWidth);
    height_ = static_cast<std::uint32_t>(pixelHeight);
}

float SdlDisplay::pixelDensity() const {
    return SDL_GetWindowPixelDensity(window_);
}

void SdlDisplay::logOpened() const {
    // Everything here only feeds the log, so a build with logging off skips it.
    if constexpr (!core::kLoggingEnabled) {
        return;
    }

    const int linked = SDL_GetVersion();
    const char* driver = SDL_GetCurrentVideoDriver();
    BFR_LOG_INFO("SDL {}.{}.{} (compiled {}.{}.{}), video driver {}", SDL_VERSIONNUM_MAJOR(linked),
                 SDL_VERSIONNUM_MINOR(linked), SDL_VERSIONNUM_MICRO(linked), SDL_MAJOR_VERSION, SDL_MINOR_VERSION,
                 SDL_MICRO_VERSION, driver != nullptr ? driver : "none");

    const SDL_DisplayID display = monitor();
    const char* name = SDL_GetDisplayName(display);
    const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(display);
    if (desktop != nullptr) {
        BFR_LOG_INFO("monitor \"{}\" {}x{} @ {:.2f} Hz, scale {:.2f}", name != nullptr ? name : "", desktop->w,
                     desktop->h, desktop->refresh_rate, SDL_GetDisplayContentScale(display));
    }

    int width = 0;
    int height = 0;
    SDL_GetWindowSize(window_, &width, &height);
    BFR_LOG_INFO("window {}x{} {} ({}x{} px, density {:.2f}), vsync {}", width, height,
                 isFullscreen() ? "fullscreen" : "windowed", width_, height_, pixelDensity(),
                 config_.vsync ? "on" : "off");
}

SDL_DisplayID SdlDisplay::monitor() const {
    return window_ != nullptr ? SDL_GetDisplayForWindow(window_) : SDL_GetPrimaryDisplay();
}

void SdlDisplay::handle(const SDL_Event& event) {
    if (event.type >= SDL_EVENT_WINDOW_FIRST && event.type <= SDL_EVENT_WINDOW_LAST) {
        handleWindow(event.window);
        return;
    }
    switch (event.type) {
        case SDL_EVENT_QUIT:
            close();
            break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
            handleKey(event.key);
            break;
        case SDL_EVENT_TEXT_INPUT:
            handleText(event.text);
            break;
        case SDL_EVENT_MOUSE_MOTION:
            handleMotion(event.motion);
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            handleButton(event.button);
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            handleWheel(event.wheel);
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
            handleGamepadAdded(event.gdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            handleGamepadRemoved(event.gdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        case SDL_EVENT_GAMEPAD_BUTTON_UP:
            handleGamepadButton(event.gbutton);
            break;
        case SDL_EVENT_GAMEPAD_AXIS_MOTION:
            handleGamepadAxis(event.gaxis);
            break;
        default:
            break;
    }
}

void SdlDisplay::handleWindow(const SDL_WindowEvent& event) {
    if (event.windowID != windowId_) {
        return;
    }
    switch (event.type) {
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            close();
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            resize(static_cast<std::uint32_t>(event.data1), static_cast<std::uint32_t>(event.data2));
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED:
            dispatch(WindowFocusGained{});
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            dispatch(WindowFocusLost{});
            break;
        case SDL_EVENT_WINDOW_MINIMIZED:
            dispatch(WindowMinimized{});
            break;
        case SDL_EVENT_WINDOW_RESTORED:
            dispatch(WindowRestored{.width = width_, .height = height_});
            break;
        default:
            break;
    }
}

void SdlDisplay::handleKey(const SDL_KeyboardEvent& event) const {
    if (event.windowID != windowId_) {
        return;
    }
    const Key key = toKey(event.scancode);
    const ModifierFlags modifiers = toModifiers(event.mod);
    if (!event.down) {
        dispatch(KeyReleased{.key = key, .modifiers = modifiers});
    }
    else if (event.repeat) {
        dispatch(KeyRepeat{.key = key, .modifiers = modifiers});
    }
    else {
        dispatch(KeyPressed{.key = key, .modifiers = modifiers});
    }
}

void SdlDisplay::handleText(const SDL_TextInputEvent& event) const {
    if (event.windowID != windowId_ || event.text == nullptr) {
        return;
    }
    const char* text = event.text;
    std::size_t length = SDL_strlen(text);
    while (length > 0) {
        dispatch(CharacterTyped{.codepoint = SDL_StepUTF8(&text, &length)});
    }
}

void SdlDisplay::handleMotion(const SDL_MouseMotionEvent& event) const {
    if (event.windowID != windowId_) {
        return;
    }
    const float density = pixelDensity();
    dispatch(MouseMoved{.x = event.x * density,
                        .y = event.y * density,
                        .deltaX = event.xrel * density,
                        .deltaY = event.yrel * density});
}

void SdlDisplay::handleButton(const SDL_MouseButtonEvent& event) const {
    if (event.windowID != windowId_) {
        return;
    }
    const std::optional<MouseButton> button = toMouseButton(event.button);
    if (!button.has_value()) {
        return;
    }
    const float density = pixelDensity();
    const ModifierFlags modifiers = toModifiers(SDL_GetModState());
    if (event.down) {
        dispatch(MouseButtonPressed{
            .button = *button, .modifiers = modifiers, .x = event.x * density, .y = event.y * density});
    }
    else {
        dispatch(MouseButtonReleased{
            .button = *button, .modifiers = modifiers, .x = event.x * density, .y = event.y * density});
    }
}

void SdlDisplay::handleWheel(const SDL_MouseWheelEvent& event) const {
    if (event.windowID != windowId_) {
        return;
    }
    const float direction = event.direction == SDL_MOUSEWHEEL_FLIPPED ? -1.F : 1.F;
    dispatch(MouseScrolled{.deltaX = event.x * direction, .deltaY = event.y * direction});
}

void SdlDisplay::handleGamepadAdded(SDL_JoystickID device) {
    if (slotOf(device) != kMaxGamepads) {
        return;
    }
    const std::size_t slot = slotOf(0);
    if (slot == kMaxGamepads) {
        return;
    }
    SDL_Gamepad* opened = SDL_OpenGamepad(device);
    if (opened == nullptr) {
        return;
    }
    gamepads_.at(slot) = Gamepad{.handle = opened, .device = device, .reported = {}};
    const char* name = SDL_GetGamepadName(opened);
    BFR_LOG_INFO("gamepad {} connected: {}", slot, name != nullptr ? name : "");
    dispatch(GamepadConnected{.gamepadId = static_cast<std::uint8_t>(slot),
                              .name = name != nullptr ? name : "",
                              .hasGamepadMapping = true});
}

void SdlDisplay::handleGamepadRemoved(SDL_JoystickID device) {
    const std::size_t slot = slotOf(device);
    if (slot == kMaxGamepads) {
        return;
    }
    SDL_CloseGamepad(gamepads_.at(slot).handle);
    gamepads_.at(slot) = Gamepad{};
    BFR_LOG_INFO("gamepad {} disconnected", slot);
    dispatch(GamepadDisconnected{.gamepadId = static_cast<std::uint8_t>(slot)});
}

void SdlDisplay::handleGamepadButton(const SDL_GamepadButtonEvent& event) const {
    const std::size_t slot = slotOf(event.which);
    const std::optional<GamepadButton> button = toGamepadButton(static_cast<SDL_GamepadButton>(event.button));
    if (slot == kMaxGamepads || !button.has_value()) {
        return;
    }
    const auto gamepadId = static_cast<std::uint8_t>(slot);
    if (event.down) {
        dispatch(GamepadButtonPressed{.gamepadId = gamepadId, .button = *button});
    }
    else {
        dispatch(GamepadButtonReleased{.gamepadId = gamepadId, .button = *button});
    }
}

void SdlDisplay::handleGamepadAxis(const SDL_GamepadAxisEvent& event) {
    const std::size_t slot = slotOf(event.which);
    const std::optional<GamepadAxis> axis = toGamepadAxis(static_cast<SDL_GamepadAxis>(event.axis));
    if (slot == kMaxGamepads || !axis.has_value()) {
        return;
    }
    const float value = normaliseAxis(*axis, event.value);
    float& reported = gamepads_.at(slot).reported.at(static_cast<std::size_t>(*axis));
    if (std::abs(value - reported) < config_.input.gamepadNoiseFloor) {
        return;
    }
    reported = value;
    dispatch(GamepadAxisMoved{.gamepadId = static_cast<std::uint8_t>(slot), .axis = *axis, .value = value});
}

std::size_t SdlDisplay::slotOf(SDL_JoystickID device) const {
    for (std::size_t slot = 0; slot < kMaxGamepads; ++slot) {
        if (gamepads_.at(slot).device == device) {
            return slot;
        }
    }
    return kMaxGamepads;
}

void SdlDisplay::closeGamepads() {
    for (Gamepad& gamepad : gamepads_) {
        if (gamepad.handle != nullptr) {
            SDL_CloseGamepad(gamepad.handle);
        }
        gamepad = Gamepad{};
    }
}

} // namespace bfr::platform
