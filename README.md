# BFR

BFR (Babel Fish RHI) is a C++ rendering hardware interface: one GPU API that
runs on Vulkan, D3D12 and Metal. It is a standalone library for any C++
renderer, aimed at real-time 3D, GPU-driven rendering, ray tracing and tools.

## Status

Early. The API below is a design, not yet code, and will change.

What exists today is a Vulkan base, developed and tested on Linux:

- listing adapters and creating a device
- swapchains for native windows (Xlib, Wayland, Win32, Metal layer)
- an SDL3 window and input layer
- JSON configuration, logging and error results

Planned: the core API described below, a render graph, D3D12 and Metal
backends, a C API, and bindings for Rust, Python and C#.

## How it works

BFR is one library with two layers over three backends:

- **The core** turns your calls into native GPU work and does only what you
  ask. You state barriers and choose the order. One thing it does own: no
  GPU memory is freed while the GPU might still use it.
- **The render graph** is the usual way to build a frame. You describe passes
  and what each reads and writes; it orders them, places barriers and
  manages temporary textures. It uses only the public core API, so you can
  skip it and use your own graph instead.
- **Backends** (Vulkan 1.3, D3D12, Metal 4) implement the core only.

A program creates an **instance** for one backend, picks an **adapter**,
creates a **device** with the queues it needs, and a **swapchain** per
window. Every object is a small integer **handle**, so a stale or wrong
handle is caught instead of crashing.

The main ideas:

- **Bindless.** Every pipeline shares one layout: a table of texture and
  buffer views, a sampler table and up to 128 bytes of push constants.
  Shaders reach resources by index; vertices are pulled, not bound.
- **Explicit barriers**, stated before and after, shaped after Vulkan
  synchronization 2 and D3D12 enhanced barriers.
- **Timelines.** Each queue has a counter the GPU advances. A submission
  returns a completion token `{queue, value}` that other queues or the CPU
  can wait on.
- **Deferred destruction.** Destroying an object invalidates its handle at
  once; the memory is freed after the GPU finishes the work already
  submitted.
- **Threading.** Create and destroy objects from any thread; record each
  command list on one thread; lists run in the order you submit them.

[`docs/how-it-works.md`](docs/how-it-works.md) walks through all of this,
with a frame end to end and how each concept maps to each backend.

## Requirements

Target hardware: Vulkan 1.3 with descriptor indexing, D3D12 Shader Model 6.6
with enhanced barriers (Windows 10 1909+), or Metal 4 (macOS and iOS 26+,
Apple GPU family 7+).

To build: CMake 3.28, Ninja, GCC 13 or Clang 19, and a
[vcpkg](https://github.com/microsoft/vcpkg) checkout in `VCPKG_ROOT`. On
macOS, Vulkan also needs MoltenVK, from the LunarG SDK or Homebrew.

## Using BFR in your project

Add BFR to your CMake build and link the `bfr` target:

```cmake
add_subdirectory(bfr)
target_link_libraries(your_app PRIVATE bfr)
```

BFR's dependencies come from vcpkg; add them to your own vcpkg manifest
(see [`vcpkg.json`](vcpkg.json)): SDL3, nlohmann-json, spdlog, volk and the
Vulkan utility libraries. Your project keeps control of its build type and
sanitizers.

`-DBFR_LOGGING=OFF` compiles BFR's log calls out entirely.

## Building BFR on its own

```
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Debug builds run the address, undefined-behaviour and leak sanitizers;
`-DBFR_SANITIZE=none` turns them off. Tests that need a GPU are labelled
`gpu`; `ctest --preset debug -LE gpu` skips them.

## License

MIT. See [`LICENSE`](LICENSE).
