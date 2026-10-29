# BFR

Babel Fish RHI: one GPU API over Vulkan, D3D12 and Metal, in C, with wrappers
for C++, C#, Rust and Python.

## Status (2026-10-26)

Early. Today: a C API, [`bfr.h`](include/bfr/public/bfr.h), and its C++
wrapper, [`bfr.hpp`](bindings/cpp/include/bfr.hpp), over a Vulkan backend
tested on Linux. They cover instances, adapters and what each can do,
devices and swapchains.

Planned: the rest of the core API below, a render graph, D3D12 and Metal
backends, and the C#, Rust and Python wrappers.

## Design

- **Core**: explicit barriers and submission order; GPU memory is freed only
  after the GPU is done with it.
- **Render graph**: orders passes and places barriers, built on the public
  core. Optional.
- **Bindless**: one layout for every pipeline. Shaders index resource tables
  and pull vertices.
- **Timelines**: each submission returns a `{queue, value}` token to wait on.
- **Handles**: resources are small integers, so stale handles are caught.

Details: [`docs/how-it-works.md`](docs/how-it-works.md).

## Requirements

| | |
|---|---|
| GPU | Vulkan 1.3 · D3D12 SM 6.6 with enhanced barriers · Metal 4 (Apple GPU family 7+) |
| Build | CMake 3.28, Ninja, GCC 13 or Clang 19, vcpkg in `VCPKG_ROOT` |
| Linux | X11 and Wayland development headers |
| macOS | MoltenVK, for Vulkan |
| Checks | clang-format 19, clang-tidy 19, Python 3 with PyYAML |

## Use in your project

```cmake
add_subdirectory(bfr)
target_link_libraries(your_app PRIVATE bfr)   # static
# or bfr_shared: libbfr.so, exporting only the bfr_* functions
```

Include `bfr.h`. Dependencies come from vcpkg; see [`vcpkg.json`](vcpkg.json).
The API itself is described in [`api/bfr.yaml`](api/bfr.yaml), which
`bfr.h` and `bfr.hpp` are generated from.

From C++23, link `bfr_cpp` as well and include `bfr.hpp`. Objects destroy
what they own, and calls return `std::expected`; nothing throws, so it
builds without exceptions.

```cpp
// Each call returns std::expected; error().code and error().message say why it failed.
auto instance = bfr::Instance::create({.backend = bfr::Backend::Vulkan});
if (!instance) {
    return report(instance.error());
}
// The handles your windowing code gave you: HWND, X11 Window, CAMetalLayer* and so on.
const bfr::Window window{.window = nativeWindow, .display = nativeDisplay};
auto device = instance->createDevice({.adapter = 0, .window = window});
if (!device) {
    return report(device.error());
}
auto swapchain = device->createSwapchain({.window = window, .imageCount = 3});
if (!swapchain) {
    return report(swapchain.error());
}
auto images = swapchain->resize(1280, 720);
```

A swapchain borrows the device that made it, so destroy it first.

Ask what an adapter can do before opening it: `caps` returns its limits,
features, memory and queues in one struct; `formats`, `surfaceCaps`,
`surfaceFormats` answer the rest, and a device's `budget` gives
its memory use now.

```cpp
auto caps = instance->caps(0);
if (caps && caps->features.meshShader) {
    // ...
}
```

## Samples

[`samples/`](samples) shows the API in use, through the C++ wrapper.

- [`caps`](samples/caps/main.cpp): what each adapter is and can do. Run
  `bfr_caps` for a summary, `--formats` for every format's uses, `--window`
  for what a swapchain may be, and `--all` for every field.

## Build and test

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug            # -LE gpu skips GPU tests
python3 tools/check_format.py
python3 tools/gen_c_api.py      # after editing api/bfr.yaml
python3 tools/gen_cpp_api.py
```

Presets: `debug`, `asan`, `tsan`, `releasedebinfo`, `release`, `ci`.

| Option | Default | |
|---|---|---|
| `BFR_LOGGING` | `ON` | Compile log calls in |
| `BFR_SANITIZE` | `none` | `address`, `thread` or `none` |
| `BFR_CLANG_TIDY` | `OFF` | Run clang-tidy on every source |
| `BFR_BUILD_TESTS` | `ON` | Build the tests |
| `BFR_BUILD_SAMPLES` | `ON` | Build the samples |
| `BFR_USE_CCACHE` | `ON` | Use ccache if installed |

As a subproject, the parent project sets the build type and sanitizers, and
tests and samples are off.

## License

MIT
