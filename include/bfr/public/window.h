#pragma once

#include <cstdint>
#include <variant>

/// A window's handles, one struct per window system, for creating a graphics
/// surface. Platform fills them; the RHI reads them. They stay valid while the
/// window that gave them is open.
namespace bfr::core::window {

struct Xlib {
    void* display = nullptr; // Display*
    std::uint64_t window = 0; // Window, an XID
};

struct Wayland {
    void* display = nullptr; // wl_display*
    void* surface = nullptr; // wl_surface*
};

struct Win32 {
    void* instance = nullptr; // HINSTANCE
    void* window = nullptr; // HWND
};

struct Metal {
    void* layer = nullptr; // CAMetalLayer*
};

/// No window: images are presented nowhere, for tests and offscreen work.
struct Headless {};

/// std::monostate when there is no window, or its window system has no struct
/// here.
using NativeWindow = std::variant<std::monostate, Xlib, Wayland, Win32, Metal, Headless>;

} // namespace bfr::core::window
