// Opens an SDL window, makes a BFR swapchain for it and clears and presents it
// every frame, resizing the swapchain as the window resizes. Close the window
// to exit.
#include "bfr.hpp"
#include "window_sample.hpp"

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;

// What BFR needs from the window. SDL's video driver decides which handles it
// gives; metalView is made only on Apple, and the caller destroys it.
bfr::Window nativeWindow(SDL_Window* window, SDL_MetalView& metalView) {
    const SDL_PropertiesID properties = SDL_GetWindowProperties(window);
    const char* name = SDL_GetCurrentVideoDriver();
    const std::string_view driver = name != nullptr ? name : "";
    if (driver == "windows") {
        return {.window = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr)};
    }
    if (driver == "x11") {
        // An X11 window is a number, which BFR takes as a pointer.
        const auto xid = static_cast<std::uintptr_t>(
            SDL_GetNumberProperty(properties, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
        return {.window = reinterpret_cast<void*>(xid),
                .display = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr)};
    }
    if (driver == "wayland") {
        return {.window = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr),
                .display = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr),
                .system = bfr::WindowSystem::Wayland};
    }
    if (driver == "cocoa") {
        metalView = SDL_Metal_CreateView(window);
        return {.window = SDL_Metal_GetLayer(metalView)};
    }
    return {};
}

} // namespace

int main(int argc, char* argv[]) {
    const auto backend = sample::backendFrom({argv, static_cast<std::size_t>(argc)});
    if (!backend) {
        return 2;
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "can't start SDL: " << SDL_GetError() << "\n";
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("bfr_sdl", kWidth, kHeight,
                                          SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (window == nullptr) {
        std::cerr << "can't open a window: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    int result = 0;
    SDL_MetalView metalView = nullptr;
    {
        bool open = true;
        result = sample::show(
            *backend, nativeWindow(window, metalView),
            [&open] {
                // Presenting paces the loop, so events are polled, not waited for.
                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                        open = false;
                    }
                }
                return open;
            },
            [window] {
                int width = 0;
                int height = 0;
                SDL_GetWindowSizeInPixels(window, &width, &height);
                return sample::Size{.width = static_cast<std::uint32_t>(width),
                                    .height = static_cast<std::uint32_t>(height)};
            });
    }
    // The swapchain and device are gone by here, so the window can go.
    if (metalView != nullptr) {
        SDL_Metal_DestroyView(metalView);
    }
    SDL_DestroyWindow(window);
    SDL_Quit();
    return result;
}
