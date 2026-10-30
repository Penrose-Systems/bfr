# BFR design

Ground-up design, started 2026-10-24. Nothing here is final. Every entry is
the current choice and its reason; when a later question exposes a problem,
the earlier choice is reopened. Earlier proposals are input like everything
else. [`research.md`](research.md) has what other engines and RHIs do.

## Requirements

- Standalone library first. Any C++ renderer is a first-class consumer;
  Vesper is one of them.
- Hardware floor: modern desktop and recent mobile. Vulkan 1.3 with
  descriptor indexing, D3D12 Shader Model 6.6, Metal 3 (Apple GPU family 7+).
  Bindless is available everywhere; no non-bindless binding path.
- Platforms: Windows, Linux, macOS, Android (Vulkan), iOS (Metal).
- OS floor: macOS/iOS 26+ (Metal 4). Windows 10 1909+ with D3D12 enhanced
  barriers in the driver.
- Backends: Vulkan, D3D12, Metal. No WebGPU: it has no explicit barriers, no
  bindless, no push constants and one queue.
- Workloads: real-time 3D games, GPU-driven rendering (indirect, compute
  culling, mesh shaders), ray tracing, tools and editors (many swapchains,
  offscreen rendering, readback).

## Current choices

- Language: C++23 for the implementation. A C++26 feature is adopted only
  once MSVC, Apple Clang and the Android NDK all ship it. Reason: MSVC and
  Apple Clang are required for D3D12 and Metal; C++26 reflection is GCC-only
  as of 2026-10.
- Namespaces: `bfr` for the API, `bfr::core` for the base (results,
  handles, pools, logging, window handles), `bfr::rhi` for the backends.
- Public API: C. Wrappers for C++, C#, Rust and Python are part of the
  project. See [C API and wrappers](#c-api-and-wrappers). Reason: C++ has no
  stable ABI; every FFI speaks the C calling convention.

- Layering: two layers in one library. The core exposes explicit barriers
  (caller states access before and after) and owns GPU lifetime (resources
  retire against completion tokens, never on CPU release). The render graph
  always ships and is the primary path; it uses only the public core API.
  Users may bypass it with their own graph on the core. Reason: standalone
  consumers with their own graph need barriers nothing guesses under; bindless
  and GPU-driven work hide accesses from inference, so access is declared
  either way; backends implement only the core.
- One authority per resource at a time. A resource the graph did not create
  is imported with its current state and exported with its final state.
- No automatic barrier tracker. Reason: the graph already holds every access
  a tracker would infer; a second convenience path doubles the sync code to
  test. Upload and readback helpers issue their own barriers in the core.

## Core RHI (draft)

First pass at the base layer. Objects, ownership and contracts only; no
signatures yet.

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

Runtime, chosen at `Instance` creation; each backend compiled in by build
option. Several instances of different backends may coexist in one process.
Reason: Windows has two backends and tools may want both; one indirect call
per command is small next to driver cost.

A null backend is in every build, inside the library: one adapter that
reports nothing (every limit 0, no heaps, no queues, no optional feature), a
device that accepts every call and draws nothing. The C API neither lists nor
accepts it. Reason: every agnostic test then runs on two backends on every
machine, and anything shaped like one API fails against the other; to an app
a backend that draws nothing is no use. BFR promises no numeric minimum: an
app reads the caps and decides.

A device can be rebuilt while the app runs: on another backend, on another
adapter, or on the same one after device loss. BFR waits for the GPU, reads
back resource contents if the old device still works, destroys everything,
recreates every object from its stored desc on the new device and uploads
the contents again. Handles stay the same, so the app's code doesn't change.
After device loss the contents are gone, so BFR reports which resources need
refilling. A rebuild happens between frames; unsubmitted command lists are
dropped. Pipelines need bytecode for every backend the app may switch to.
Reason: switching backend or GPU in a settings menu, and recovering from
device loss, are the same job, and BFR is the one that knows every object.
Not built until resources exist.

Commands are translated to native calls as they are recorded, not buffered
and replayed. Reason: avoids recording everything twice. Revisit if Metal
encoder switching needs lookahead.

### Device and queues

- The device is created from an adapter with the queues and optional
  features it wants. An adapter missing a floor feature (Vulkan: descriptor
  indexing with update-after-bind, timelines, synchronization2, dynamic
  rendering, buffer device address) is not listed; every device turns the
  floor's features on. Optional ones are negotiated and queryable.
- Caps are per adapter and readable before any device exists. Reason:
  optional features are asked for when the device is created, and choosing
  a GPU shouldn't mean opening each one. One call returns everything fixed
  (info, limits, bindless, compute, mesh, ray tracing, features, memory,
  queues) as one struct. Formats, a window's surface caps and formats, the
  live memory budget are calls of their own: per
  format, per window, or changing while the app runs. A limit for a
  feature the adapter lacks is 0.
- Queue types: graphics, compute, copy. Start with one graphics queue in the
  first backend; async compute and copy queues come later on the same model.

### Synchronization

- One primitive for all queue progress: a timeline (Vulkan timeline
  semaphore, D3D12 fence, Metal shared event). A completion token is
  `{queue, value}`. Submissions wait on and signal tokens; the CPU waits on
  or polls them. Binary semaphores exist only inside the swapchain.
- Barriers are explicit and batched: per resource (with subresource range)
  and global memory barriers, each with stage and access before and after,
  plus a texture layout. Shaped after Vulkan synchronization 2 and D3D12
  enhanced barriers, which map almost one to one.
- D3D12 requires enhanced barriers; adapters without them are
  rejected. Reason: a legacy-barrier path doubles the sync code.
  Enhanced barriers are an optional driver feature checked at runtime; the
  Agility SDK reaches Windows 10 1909+.
- Metal backend targets Metal 4 (macOS/iOS 26+). Reason: Metal 4
  has explicit barriers, residency sets and command allocators, which match
  this core directly, and runs on Apple GPU family 7+ (A14, M1), the same
  hardware as our floor. On Metal 3 explicit barriers would be emulated with
  untracked heaps, fences and memory barriers.

### Lifetime and residency

- `destroy(handle)` is deferred: the object is freed once every queue has
  passed the values submitted before the call. The handle is invalid at once.
  Each handle table retires an object against the last submitted value and
  frees it when that value completes. Until queues submit, every value has
  completed, so objects are freed during the destroy call. A handle table's
  token is one 64-bit number, not a queue's timeline value: the device keeps
  a submission serial that every submit advances, retires against the
  latest, and collects a serial once every queue has passed the value it
  submitted under it.
- Swapchains are the exception: destroying one waits for the GPU's use of
  it and destroys it before returning. Reason: its window belongs to the
  application, which closes it once the destroy call returns; a deferred
  destruction would need the window kept open until a moment the
  application cannot observe. Destroying a swapchain is rare, so the wait
  costs nothing that matters.
- Destroying an object an unsubmitted command list references is a usage
  error, caught by validation.
- Residency is part of the core contract, not a backend detail. Bindless
  means shaders reach resources the API never sees per draw; Metal 4 needs
  them in a residency set. The device keeps a residency set of all live
  bindless resources; deferred destruction and graph aliasing update it.

### Resources and memory

- Buffer, Texture, Sampler, AccelerationStructure. Default allocation goes
  through an internal allocator (VMA, D3D12MA, Metal heaps).
- `Heap` plus placed resources for explicit memory control. The graph's
  transient aliasing is built on this, so it stays public.
- Upload and readback helpers live in the core and issue their own barriers.
  Readback returns a completion token; blocking waits are explicit.

### Binding model

- One global layout for every pipeline: a bindless table of texture and
  buffer views, a sampler table, and a small push-constant block. No
  per-pipeline layouts or descriptor sets.
- Shaders reach buffers and textures by descriptor index. Reason: HLSL on
  D3D12 has no pointer dereference; indices work on all three backends. GPU
  addresses appear only where the native APIs require them (ray tracing
  build inputs, indirect arguments).
- Metal 4 has no push constants and no `setBytes` on its encoders; resources
  bind only through `MTL4ArgumentTable`, which Metal snapshots at each draw
  or dispatch. The Metal backend copies push-constant data into a linear
  upload ring owned by the command pool and binds that address with
  `setAddress` before the draw. The ring is recycled with the pool and kept
  resident.
- Creating a view allocates its bindless slot; destroying it retires the
  slot through deferred destruction.
- Limits set the sizes: push constants at most 128 bytes (Vulkan guarantee,
  within the D3D12 64-DWORD root signature); samplers at most 2048 (D3D12
  shader-visible sampler heap). Mobile descriptor limits still to check.

### Pipelines

- Graphics, mesh and compute pipelines take per-backend shader bytecode plus
  fixed state and attachment formats. No render pass objects (dynamic
  rendering).
- With one global layout, BFR needs almost no reflection: push-constant size
  and workgroup size.
- Creation is asynchronous. A pipeline is pending, ready or failed; recording
  with a pipeline that is not ready is a usage error. Fallback choice belongs
  to the renderer.
- Pipeline cache: load and store as blobs through a host hook.
- Vertex input: vertex pulling only; shaders fetch vertices by index.
  Reason: simplest core, fits GPU-driven work. Revisit if measurements on
  tile-based mobile GPUs show fixed-function fetch winning for position-only
  shading.

### Command recording

- A `CommandPool` belongs to one thread and one frame slot; it is reset once
  its work completes. `CommandList`s from it are recorded by that thread only.
- Lists recorded on many threads are submitted in the order given to
  `submit`; recording order across threads means nothing.
- Rendering scopes carry attachments and load and store actions explicitly.
  Reason: tile-based GPUs depend on them.
- Commands: barriers, copies, clears, draw (direct, indexed, indirect,
  indirect count), mesh dispatch, compute dispatch (direct, indirect),
  acceleration structure builds, push constants, pipeline bind, viewport and
  scissor, timestamps and queries, debug markers.

### Swapchain

- Created from a device, a native window handle and a presenting queue. Any
  number per device (tools).
- Acquire returns the image's texture and an internal wait; present waits on
  a render token.
- Vulkan: one present semaphore per image, so binary semaphores are never
  reused while the presentation engine holds them; use swapchain
  maintenance extensions where present.
- Resize and surface loss are reported as results, not fatal errors.

### Ray tracing

- Core: acceleration structures and inline ray queries from any shader
  stage. All three backends support them.
- Ray-tracing pipelines with shader binding tables sit behind a capability
  and stay out of the first cut. Open: how Metal's intersection and visible
  function tables map to a binding table.

### Handles

- Instance, device and command list are opaque pointers. Reason: recording
  is the hot path and a pointer needs no lookup.
- Everything else is a generational 64-bit handle: a 24-bit slot index under
  a 40-bit generation. Each type is its own struct, so the C compiler rejects
  a buffer where a texture is expected, and every wrapper has a class per
  type, so a handle carries no type tag. Scoped to one device; every call
  names the device, so a handle needs no device ID. Reason: C-compatible,
  safe against stale use, no reference counts. The generation grows by two
  per reuse of a slot and the free list reuses the last slot freed, so 32
  bits would come round after 2^31 destroy-and-create cycles on one slot, a
  few days of churn for a tool; 40 bits never do.

### Threading

- Creation and destruction on a device are safe from any thread.
  Destroying a handle that is already gone does nothing. Reason: C# and
  Python wrappers destroy from garbage-collector finalizers.
- Looking up a handle takes no lock: recording resolves handles on every
  command, from many threads. Creation and destruction lock only to take or
  give back a slot; objects are built and destroyed outside the lock. Each
  object type has a fixed maximum per device; past it creation fails with
  `limit_reached`. Reason: the slot table is sized once, so lookups never see
  it move.
- A command list is used by one thread at a time.
- Submission to one queue is externally synchronized; different queues may
  submit concurrently.
- A swapchain is externally synchronized.
- A handle's calls name its device, and destroying the device destroys its
  handles, so nothing may be called on a handle once its device is
  destroyed. The device and handle objects of a garbage-collected wrapper
  may be finalized in either order, so the wrapper keeps the order itself:
  its device destroys the handles it still tracks, under a lock, and a
  handle finalizer that finds its device gone does nothing. In C# the
  device is a `SafeHandle` each handle wrapper adds a reference to, so the
  device is released last whatever order the finalizers run; in Python
  each handle holds its device; Rust and C++ borrow it.

### Failure

- Creation returns a `bfr_result` and writes the object through an out
  parameter; `bfr_get_last_error` gives the calling thread's message for its
  last failure. Recording calls return nothing; misuse is caught by
  validation. Wrappers turn failures into their language's errors.
- Device loss is sticky: every later call reports it, and a host callback
  fires once, until the device is rebuilt (see Backend selection).
- Validation is configured at instance creation; messages go to a host
  callback.

## C API and wrappers

The C API is mostly a target for wrappers, all of which this project writes.
Direct use from C works but is not the main case.

### Source of truth

A YAML description of the API generates `bfr.h` and the bulk of every
wrapper. It records what a C header cannot: which pointer is an array and
which field is its count, what may be null, what owns what, which thread a
callback runs on. Reason: one description keeps four languages from drifting;
hand-written wrappers would each drift from the header.

### C rules

- Names: `bfr_` prefix, snake_case: `bfr_device_create_buffer`,
  `bfr_buffer_desc`, `BFR_FORMAT_RGBA8_UNORM`.
- Description structs start with `const bfr_chain* next`. After 1.0 a struct
  never changes; new fields arrive as chained structs. Before 1.0 anything
  may break.
- Enums are C enums with a `_FORCE32 = 0x7FFFFFFF` member, so they are 32
  bits on every compiler and generators keep their type. Flags are
  `uint64_t`. `bfr_bool` is `uint32_t`. Reason: P/Invoke marshals `bool` as
  4 bytes by default.
- Strings are `{data, length}`; arrays are a pointer and a count.
- The caller owns what it passes; BFR copies what it keeps. Lists BFR
  returns use two calls: one for the count, one to fill the caller's array.
- No function-like macros, unions, bitfields, anonymous structs or flexible
  array members. Reason: bindgen, ClangSharp and cffi handle them badly.
- Callbacks are a function pointer and `void* user_data`, kept to the
  message callback. Everything else is polled: pipeline readiness by query,
  device loss by result. Reason: calling into C# or Python from a driver
  thread is costly and easy to get wrong.
- Destroy calls may come in any order: a device keeps its instance alive
  until the device is destroyed. Reason: garbage collectors finalize in no
  fixed order. The instance's message callback and `user_data` must then
  stay valid until its last device is destroyed, so a wrapper ties them to
  the last device, not to the instance object.
- Calls that would be made once per item take arrays. Reason: a call from
  Python costs microseconds.
- `bfr.h` compiles as C11 and C++. The shared library exports only `bfr_*`
  symbols, through a linker export list as well as hidden visibility
  (libstdc++ gives its templates default visibility). No exception crosses
  the boundary: each entry point turns one into a result.
- Two libraries from one set of objects: static `bfr`, which also exposes
  the C++ classes behind the C API to the tests, and shared `bfr_shared`
  (`libbfr`), which exposes only the C API. The wrappers load the shared one.
- Settings that belong to one object go on that object's desc, not the
  instance: the swapchain's image count is on `bfr_swapchain_desc`.

### Wrappers

- Generated from the description: raw bindings, enums, flags, scoped
  ownership (RAII in C++, `IDisposable` in C#, `Drop` in Rust, context
  managers in Python), errors as exceptions or results. Hand-written only
  where a language deserves its own shape, such as command recording.
- C++ (`bfr.hpp`): C++23. Calls return `std::expected` and nothing
  throws, so it works with exceptions off. Windows and other extension
  structs go after the desc as extra arguments; one the desc doesn't take
  won't compile. A handle borrows its device, so the device must outlive
  it. Object methods are `const` because many threads can call them;
  handle methods aren't, because calls on one handle can't overlap.
  Reason: lots of engines turn exceptions off, the C API already returns
  results, and the library needs C++23 anyway.
- In `bindings/<language>`, each tested through the real library.
- Order: C++ first, which the tests and Vesper use; then C#; then Rust and
  Python.
- No package publishing (NuGet, crates.io, PyPI) until the API settles.
  Reason: each needs prebuilt libraries and CI on every platform.

## Open questions (rough order)

1. Mobile descriptor and update-after-bind limits (Mali, Adreno) to size the
   bindless tables.
2. Render graph scope: which of dependencies, barriers, transient pooling,
   aliasing, async compute and pass culling ship first; graph lifetime
   (rebuilt per frame or cached); how pass callbacks record.
3. Shader contract: source language and compiler (Slang is the candidate),
   per-backend bytecode packaging.
4. Ray-tracing pipelines on Metal.
5. Host hooks: logging, allocators, pipeline cache, imports, diagnostics.
6. Repo, build, tests, first backend, CI.
