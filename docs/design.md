# BFR design

Current choices and their reasons, started 2026-10-24. Nothing is final; a
choice is reopened when a later question exposes a problem, and earlier
proposals are input like everything else. What other
engines and RHIs do is in [`research.md`](research.md).

## Requirements

- Standalone library first; any C++ renderer is a first-class consumer.
- Minimum requirements: each `backend` value in `api/bfr.yaml`. Adapters
  below them are not listed. Reason: every call is built on bindless
  binding, dynamic rendering and explicit barriers; the rest is in caps.
- The caller names the instance's API version, passed through as is. Below
  the minimum, instance creation fails with RhiUnsupported; above it,
  nothing is capped: a device runs at the lower of that version and its
  adapter's and turns on what that version has, using core or extension form as the
  device has it. D3D12 and Metal don't read the version yet (paused while
  Vulkan takes shape, owner, 2026-11-08).
- Bindless binding only.
- Platforms: Windows, Linux, macOS, Android (Vulkan), iOS (Metal).
- Backends: Vulkan, D3D12, Metal. Not WebGPU: no explicit barriers,
  bindless or push constants, and one queue.
- Workloads: real-time 3D games, GPU-driven rendering (indirect, compute culling,
  mesh shaders), ray tracing, tools and editors (many swapchains, offscreen
  rendering, readback).

## Current choices

- C++23 for the implementation; a C++26 feature once MSVC, Apple Clang and the Android NDK all
  ship it. Reason: MSVC and Apple Clang are required for D3D12 and Metal;
  C++26 reflection is GCC-only as of 2026-10.
- Namespaces: `bfr` (API), `bfr::core` (the base: results, handles, pools, logging,
  window handles), `bfr::rhi` (backends).
- Public API is C, with C++, C#, Rust and Python wrappers in the project
  ([below](#c-api-and-wrappers)). Reason: C++ has no stable ABI; every FFI
  speaks C.
- Two layers in one library. The core has explicit barriers (caller states access before
  and after) and owns GPU lifetime (resources retire against completion
  tokens, not CPU release). The render graph always ships, is the primary
  path, uses only the public core API; users may bypass it with their own graph
  on the core. Reason: callers
  with their own graph need barriers nothing guesses under; bindless and
  GPU-driven work hide accesses from inference, so access is declared either
  way; backends implement
  only the core.
- One authority per resource at a time: what the graph didn't create is imported with
  its current state and exported with its final one.
- No automatic barrier tracker in the first cut. Planned: an opt-in layer
  over explicit barriers, switchable per command list (NVRHI's
  `setEnableAutomaticBarriers` is the model). Reason: the graph already
  knows every access, so the graph and the explicit path ship first; a
  tracker is a third authority bound by the same import/export
  rule.

## Core RHI (draft)

Objects, ownership and contracts; no signatures yet.

### Object map

```
Instance ──> Adapter (value: info, limits, features)
   │
   └──> Device ──> Queue (graphics | compute | copy)
          ├──> Swapchain
          ├──> Buffer, Texture, Sampler, AccelerationStructure
          ├──> TextureView, BufferView   (each owns a bindless slot)
          ├──> Heap                      (placed resources, graph aliasing)
          ├──> Pipeline (graphics | mesh | compute)
          ├──> CommandPool ──> CommandList
          └──> Timeline                  (queue progress, CPU waits)
```

Ownership points down. A child never holds its parent; what it needs is
copied in at creation.

### Backend selection

- Picked at run time when the `Instance` is made; each compiled in by build
  option; instances of different backends may coexist. Reason: Windows has
  two backends and tools may want both; one indirect call per command is
  small next to driver cost.
- No null backend: agnostic tests run on every built backend and on
  Mesa's lavapipe (CPU Vulkan), so no GPU is needed. Reason: a backend that draws
  nothing tests no driver, and every call would need an empty twin.
- No numeric minimum is promised; apps read caps and decide.
- Device rebuild (not built until resources exist): onto another backend or
  adapter, or the same one after loss, between frames. BFR waits for the
  GPU, reads contents back if the old device works, destroys everything,
  recreates each object from its stored desc and re-uploads. Handles don't
  change, so app code doesn't either; after loss the contents are gone and
  BFR reports which resources need refilling.
  Unsubmitted lists are dropped; pipelines need bytecode for every backend
  the app may switch to. Reason: a settings-menu switch and loss recovery
  are one job, and BFR is what knows every object.
- Commands translate to native calls as recorded, not buffered for replay.
  Reason: no double recording. Revisit if Metal encoder switching needs
  lookahead.

### Device and queues

- A device turns on what its adapter reports in its version's core structs
  and the extensions BFR enables, except robust access (changes
  out-of-bounds behaviour, costs speed). Caps report each feature beyond the
  minimum.
- Caps are per adapter, readable before any device. Reason: features are
  requested at device creation; choosing a GPU shouldn't mean opening each.
  One call returns everything fixed (info, limits, bindless, compute, mesh,
  ray tracing, features, memory, queues); formats, a window's swapchain caps
  and formats, and the live memory budget are separate calls, since they
  vary per format, per window or over time. A limit for a missing feature
  is 0.
- Queues: graphics, compute, copy, opened as the desc asks, refused when the
  adapter has fewer (caps give the count), numbered graphics from 0, then
  compute, then copy. A pool records for one queue. A submit names the
  queue, lists in order and tokens to wait on, and returns the token it
  signals. Each queue has its own timeline and is externally synchronized,
  so submits to different queues never contend.

### Synchronization

- One progress primitive: a timeline (Vulkan timeline semaphore, D3D12
  fence, Metal shared event). Token = `{queue, value}`; submits wait on and
  signal tokens; the CPU waits or polls. Binary semaphores live only inside
  the swapchain.
- Barriers are explicit and full: a list's later commands start after its
  earlier ones finish and see their writes. Native on each backend (Vulkan memory
  barrier, D3D12 global barrier, Metal 4 consumer barrier). A finer form
  (resource, stage, access) only if measurement shows the full one costs
  too much.
- A texture barrier is a full barrier that also moves a texture between
  layouts the caller names (from and to); BFR keeps no record. On Metal,
  which has no layouts, it is just the full barrier.
- D3D12 uses enhanced barriers (minimum requirement). Reason: a legacy path
  doubles the sync code.
- Metal targets Metal 4, macOS/iOS 26+ (minimum requirement). Reason: its
  explicit barriers, residency sets and command allocators match this core.

### Lifetime and residency

- `destroy(handle)` invalidates the handle at once; the object is freed when
  every queue has passed what was submitted before. The device keeps one
  64-bit submission ID (not a queue timeline value) that every submit
  advances; tables retire against the latest and free an ID's objects once every
  queue has passed the value it submitted under that ID. Before any submit, destroy frees
  at once.
- Except swapchains: destroy waits for the GPU's use of it and returns
  with it gone.
  Reason: the app closes its window when destroy returns; deferring would
  keep the window needed until a moment the app can't see. Rare, so the
  wait is cheap.
- Destroying an object an unsubmitted list references is a usage error,
  which a usage check reports.
- Residency is part of the core contract, not a backend detail: bindless shaders reach resources no draw names, and
  Metal 4 needs them in a residency set. The device keeps one of all live
  bindless resources, updated by deferred destruction and graph aliasing.

### Resources and memory

- Buffer, Texture, Sampler, AccelerationStructure, allocated by default
  through VMA, D3D12MA or Metal heaps.
- `Heap` and placed resources for explicit memory; public because the
  graph's transient aliasing is built on them.
- Upload and readback helpers are in the core and issue their own barriers.
  Readback returns a token; blocking waits are explicit.

### Binding model

- One global layout: a bindless table of texture and buffer views, a
  sampler table, a small push-constant block. No per-pipeline layouts or
  descriptor sets. Each backend's shader-side layout is in `api/bfr.yaml`
  under `pipeline`; table sizes are `DeviceDesc`'s `max_bindless_*`, made
  with the device and passed through. A uniform buffer table where the backend
  updates uniform buffers after bind; caps say where.
- Shaders reach buffers and textures by descriptor index. Reason: HLSL on
  D3D12 has no pointer dereference; indices work on all three backends. Every buffer has a GPU address where
  the device gives them, for the calls that read by address (ray tracing build inputs, indirect arguments).
- A resource takes a slot in each table its usage names at creation,
  reported by `get_buffer_slots`, `get_texture_slots` and `get_sampler_slot`;
  it is the same
  number on every backend so a shader index means one thing. On D3D12 BFR sets
  each descriptor range's offset from the table's start to its region's, so indices in spaces 1 and 2
  are table-local. A slot is freed when its resource is collected (after
  every submit before the destroy has run), never at destroy, so pending
  work never sees it reused.
- A sampler holds one sampler-table slot. Every desc member reaches the
  backend unchanged; only one the backend has no setting for fails, with
  `error_unsupported`. Identical samplers are not shared: each create reaches the backend
  and holds its own slot. Vulkan turns on each optional sampler feature its
  caps report, so the caps say what it has.
- Metal 4 has no push constants or encoder `setBytes`; it binds only through
  `MTL4ArgumentTable`, snapshotted per draw or dispatch. The backend copies
  push constants into a resident ring owned by the command pool, sized by
  its `push_constant_updates`, reset with the pool, bound by `setAddress`
  before the draw. No default: only the caller knows how many it records.
  Vulkan and D3D12 need no ring and pass push constants straight through;
  it is on the pool because a caller declares capacity, not a backend.
- Push constants are sized by `bfr_caps_limits.max_push_constants_size`
  (device limit on Vulkan, root signature room on D3D12, 128 on Metal);
  samplers by `bfr_caps_bindless.max_samplers`.

### Pipelines

- Graphics, mesh and compute pipelines take per-backend bytecode, fixed
  state and attachment formats; no render pass objects (dynamic rendering). The pipeline fixes
  the topology class; exact topology, viewport, scissor, blend constants and
  stencil reference are set while recording, as every backend allows.
- With one global layout, BFR needs almost no reflection: push-constant
  and workgroup size.
- Creation is asynchronous: pending, ready or failed; recording with a
  pipeline not ready is a usage error; fallback is the renderer's choice.
  Not yet: today `createComputePipeline` and `createGraphicsPipeline` return
  the pipeline or its failure.
- Pipeline cache: blobs loaded and stored through a host hook.
- Vertex pulling only: shaders fetch vertices by index. Reason: simplest core, fits GPU-driven work. Revisit
  if measurements on tile-based mobile GPUs show fixed-function fetch winning for
  position-only shading.
- One clip space: x right, y up, z 0 near to 1 far, native on D3D12 and
  Metal. Vulkan gets negative-height viewports offset by the height (core
  since 1.1), which also matches winding. Reason: one vertex shader and
  front face mean the same everywhere. Hazel does the same; Godot flips in
  shader translation, treating Vulkan as canonical, which BFR can't since the caller compiles shaders.
- `first_instance` as a shader sees it is the backend's: Vulkan's
  `gl_InstanceIndex` and Metal's `instance_id` include it; D3D12's
  `SV_InstanceID` starts at 0. BFR passes the value through and documents
  the difference rather than patching bytecode it didn't compile.

### Command recording

- A `CommandPool` belongs to one thread and frame slot and is reset after
  its last submit's token completes (earlier is a usage error); only that
  thread records its lists. D3D12 and Metal 4 allow one recording list per
  allocator, so there a pool holds one allocator per recording list.
- Lists from many threads are submitted in the order given to `submit`,
  whatever the recording order.
- Rendering scopes carry attachments and load/store actions explicitly.
  Reason: tile-based GPUs depend on them.
- Commands: barriers, copies, clears, draws (direct, indexed, indirect,
  indirect count), mesh and compute dispatch (direct, indirect),
  acceleration structure builds, push constants, pipeline bind, viewport and
  scissor, timestamps and queries, debug markers. Today: barriers, copies,
  rendering to swapchain images and textures with load/store actions,
  viewport and scissor, pipeline bind,
  push constants, index buffer bind, direct and indexed draws, direct
  dispatches.
- Every backend keeps the same record of what a list has bound
  (`src/rhi/command_check.h`). The few checks BFR makes of a call live there
  once, for the backends that need them: Metal's for the push constants it
  copies into its pool and the indirect draw counts its call has none of.

### Swapchain

- Made from a device, a native window handle and a presenting queue; any
  number per device (tools).
- Acquire returns the image's texture and an internal wait; present waits on
  a render token.
- Vulkan: one present semaphore per image, so none is reused while the
  presentation engine holds it; swapchain maintenance extensions where
  present.
- Resize and window loss are results, not fatal errors.

### Ray tracing

- Core: acceleration structures and inline ray queries from any stage, on
  all three backends.
- Ray tracing pipelines with binding tables sit behind a capability, out of
  the first cut.

### Handles

- Instance and device are opaque pointers; an app holds one of each, named
  in every call.
- Everything else is a 64-bit generational handle (24-bit slot, 40-bit
  generation) scoped to one device; every call names the device, so it needs no
  device ID. A struct per
  type, so C rejects a buffer for a texture, and a wrapper class per type,
  so no type tag. Reason: C-compatible, stale-safe, no reference counts.
  Generations grow by two per reuse and the free list reuses the last slot
  freed, so 32 bits would wrap after 2^31 cycles on one slot (days of tool
  churn); 40 bits never do.

### Threading

- Create and destroy are safe from any thread; destroying a handle already
  gone does nothing. Reason: C# and Python destroy from GC finalizers.
- Handle lookup is lock-free, since recording resolves handles on every
  command from many threads. Create and destroy lock only to take or return
  a slot; objects are built and destroyed outside the lock. Each type has a fixed maximum per device; past it creation fails
  with `limit_reached`. Reason: a table sized once never moves under a
  lookup.
- A command list is used by one thread at a time; a swapchain is externally
  synchronized; queues as in [Device and queues](#device-and-queues).
- Destroying a device destroys its handles; nothing may be called on them
  after. A GC wrapper may finalize device and handles in any order, so it
  keeps the order: the device destroys the handles it still tracks, under a
  lock, and a handle finalizer finding its device gone does nothing. C#:
  each handle adds a reference to the device's `SafeHandle`, so the device
  goes last. Python: each handle holds its device. Rust and C++ borrow it.

### Failure

- Creation returns `bfr_result` and writes the object to an out parameter;
  `bfr_get_last_error` gives the thread's last failure message. Recording
  calls return nothing; usage checks report misuse through the message
  callback. Wrappers map failures to their language's errors.
- Device loss is sticky: every later call reports it and a host callback
  fires once, until a [rebuild](#backend-selection).
- Validation is set at instance creation; messages go to a host callback.

## C API and wrappers

The C API mainly serves the wrappers, all of which this project writes; direct C use works but is
secondary.

### Source of truth

A YAML description generates `bfr.h` and most of every wrapper. It records
what a C header can't: which pointer is an array and which field its count,
what may be null, what owns what, which thread a callback runs on. Reason:
one description keeps four languages in step; hand-written wrappers drift.

### C rules

- Names: `bfr_` prefix, snake_case (`bfr_device_create_buffer`,
  `bfr_buffer_desc`, `BFR_FORMAT_RGBA8_UNORM`).
- Desc structs start with `const bfr_chain* next`. After 1.0 structs never
  change; new fields come as chained structs. Before 1.0 anything may break.
- Enums have `_FORCE32 = 0x7FFFFFFF`, so they are 32 bits on every compiler
  and generators keep their type. Flags are `uint64_t`. `bfr_bool` is
  `uint8_t` like C++'s `bool`, so structs of numbers and bools pass between C and
  the C++ wrapper uncopied;
  C# binds it as `byte` or `[MarshalAs(UnmanagedType.U1)] bool`.
- Strings are `{data, length}`; arrays a pointer and a count.
- The caller owns what it passes; BFR copies what it keeps. Returned lists
  take two calls: count, then fill.
- No function-like macros, unions, bitfields, anonymous structs or flexible
  array members. Reason: bindgen, ClangSharp and cffi handle them badly.
- Callbacks (function pointer plus `void* user_data`) are kept to the
  message callback; the rest is polled (pipeline readiness by query, device
  loss by result). Reason: calling into C# or Python from a driver thread is
  costly and error-prone.
- Destroys may come in any order: a device keeps its instance alive.
  Reason: GCs finalize in no fixed order. So the instance's message callback
  and `user_data` must outlive its last device, and wrappers tie them to
  that device, not the instance object.
- What would be one call per item takes an array. Reason: a Python call
  costs microseconds.
- `bfr.h` compiles as C11 and C++. The shared library exports only `bfr_*`,
  by linker export list as well as hidden visibility (libstdc++ gives its
  templates default visibility). No exception crosses the boundary: each entry point turns one into a
  result.
- One set of objects, two libraries: static `bfr`, which also gives the
  tests the C++ classes behind the C API, and shared `bfr_shared` (`libbfr`), C API only,
  which the wrappers load.
- A setting belonging to one object goes on its desc, not the instance
  (swapchain image count is on `bfr_swapchain_desc`).

### Wrappers

- Generated from the description: raw bindings, enums, flags, scoped
  ownership (RAII in C++, `IDisposable` in C#, `Drop` in Rust, context
  managers in Python), errors as
  exceptions or results. Hand-written only where a language deserves its own
  shape, such as command recording.
- C++ (`bfr.hpp`, C++23) returns `std::expected` and never throws, so it
  works with exceptions off. Reason: many engines disable exceptions, the C
  API already returns results, and the library needs C++23 anyway.
  Extension structs (Windows and others) follow the desc as extra
  arguments; one the desc doesn't take won't compile. A handle borrows its
  device, which must outlive it. Structs name handles by `Ref`
  (`BufferRef`), a borrowed value laid out as the C handle, so structs and
  arrays pass to C uncopied.
  Object methods are `const` because many threads call them; handle methods
  aren't, because calls on one handle can't overlap.
- Each in `bindings/<language>`, tested through the real library.
- Order: C++ (the tests use it), C#, then Rust and Python.
- No package publishing (NuGet, crates.io, PyPI) until the API settles.
  Reason: each needs prebuilt libraries and CI on every platform.

## Open questions (rough order)

1. Mobile descriptor and update-after-bind limits (Mali, Adreno) for sizing
   the bindless tables.
2. Render graph scope: which of dependencies, barriers, transient pooling,
   aliasing, async compute and pass culling ship first; graph lifetime (rebuilt per frame or cached);
   how pass callbacks record.
3. Shader contract: source language and compiler (Slang is the candidate),
   per-backend bytecode packaging.
4. Ray tracing pipelines on Metal: mapping its intersection and visible
   function tables to a binding table.
5. Host hooks: logging, allocators, pipeline cache, imports, diagnostics.
6. Repo, build, tests, first backend, CI.
