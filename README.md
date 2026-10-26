# BFR

Babel Fish RHI: one GPU API over Vulkan, D3D12 and Metal, in C, with wrappers
for C++, C#, Rust and Python.

## Status (2026-10-25)

Early. Today: a C API, [`bfr.h`](include/bfr/public/bfr.h), over a Vulkan
backend tested on Linux. It covers instances, adapters, devices and
swapchains.

Planned: the rest of the core API below, a render graph, D3D12 and Metal
backends, and the C++, C#, Rust and Python wrappers.

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
`bfr.h` is generated from.

## Build and test

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug            # -LE gpu skips GPU tests
python3 tools/check_format.py
python3 tools/gen_c_api.py      # after editing api/bfr.yaml
```

Presets: `debug`, `tsan`, `releasedebinfo`, `release`, `ci`.

| Option | Default | |
|---|---|---|
| `BFR_LOGGING` | `ON` | Compile log calls in |
| `BFR_SANITIZE` | `address` in Debug | `address`, `thread` or `none` |
| `BFR_CLANG_TIDY` | `OFF` | Run clang-tidy on every source |
| `BFR_BUILD_TESTS` | `ON` | Build the tests |
| `BFR_USE_CCACHE` | `ON` | Use ccache if installed |

As a subproject, the parent project sets the build type and sanitizers, and
tests are off.

## License

MIT
