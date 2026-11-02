# Research

What other engines and RHIs do, gathered while designing BFR. Findings only;
the choices are in [`design.md`](design.md).

## Host hooks found

Surveyed: Unreal, Unity (public plugin interfaces only), Godot, O3DE,
Frostbite/Halcyon (talks), The Forge, Diligent, NVRHI, SDL3 GPU, Dawn, wgpu,
Filament, Granite. Findings only; each row is a decision.

| Hook | Who has it |
|---|---|
| Message callback (errors, warnings, validation) | Dawn, wgpu, Diligent, NVRHI, The Forge, SDL3, Unity `IUnityLog` |
| Validation switches at creation | All but Filament; GPU-based validation in The Forge, wgpu, Diligent |
| Extra instance and device extensions | SDL3, The Forge |
| Imported instance and device | NVRHI, Granite, wgpu, Filament |
| Capability queries | 7 of 8 standalone, Unreal `GRHISupports*`, Unity `SystemInfo` |
| Pipeline cache as blob load/store callbacks | Dawn, Filament, The Forge, Diligent, Granite, wgpu |
| CPU allocator | SDL3 (process-wide), Diligent, NVRHI (Vulkan callbacks), The Forge (compile-time), Unreal (Vulkan callbacks), Godot (Vulkan callbacks, tracking only). Not Unity, O3DE, Dawn, wgpu, Filament, Granite |
| GPU memory knobs (VMA, D3D12MA) | Diligent, Godot, Unreal |
| Device-lost callback | Dawn, wgpu; O3DE flags it internally; Unity reset events |
| Debug names and GPU markers | Godot, The Forge, NVRHI, Dawn, wgpu, Unreal, Unity |
| Worker pool or job hook | Dawn, Granite, Diligent (async shader compile) |
| No-GPU backend | Halcyon Mock, O3DE Null |

CPU allocator options on the table: one allocator struct (allocate with size
and alignment, deallocate, user data) used by pools and pages and forwarded to
Vulkan, cold code on standard containers; the same struct for everything via
pmr; no hook.

From Halcyon: generational, type-checked handles owned by the app; adapter
enumeration apart from device creation; backends mixable in one process; the
render graph above the RHI, never inside it.

## Public APIs of full RHIs

Grouped by what the host calls.

### Godot `RenderingDevice`

One class, everything by RID.

- Lifecycle: `initialize(context, main_window)`, `finalize`, `submit`, `sync`,
  `swap_buffers`, `create_local_device`.
- Resources: `buffer_*` (vertex, index, uniform, storage, texture buffer;
  copy, update, clear, get_data_async), `texture_*` (create, shared,
  from_extension, update, copy, clear, resolve), `sampler_create`,
  `vertex_format_create`, `vertex_array_create`, `index_array_create`,
  `free_rid`.
- Shaders and binding: `shader_create_from_spirv`, `shader_create_from_bytecode`,
  `uniform_set_create`.
- Pipelines: `framebuffer_format_create`, `framebuffer_create`,
  `render_pipeline_create`, `compute_pipeline_create`,
  `raytracing_pipeline_create`, `update_pipeline_cache`.
- Recording: `draw_list_begin`, `_bind_render_pipeline`, `_bind_uniform_set`,
  `_bind_vertex_array`, `_set_push_constant`, `_draw`, `_draw_indirect`,
  `_end`; the same shape for `compute_list_*` and `raytracing_list_*`;
  `barrier`, `full_barrier`.
- Screen: `screen_create`, `screen_prepare_for_drawing`,
  `screen_get_width/height/format`, `screen_free`,
  `draw_list_begin_for_screen`.
- Ray tracing: `blas_create`, `tlas_create`, `blas_build`, `tlas_build`,
  `hit_sbt_*`.
- Debug and info: `set_resource_name`, `draw_command_begin/end_label`,
  `capture_timestamp`, `get_device_name`, `get_device_api_version`,
  `has_feature`, `get_driver_and_device_memory_report`.
- No queues, fences or semaphores exposed. One implicit frame; barriers are
  explicit but coarse.

### NVRHI

One interface per object, ref-counted handles.

- `IDevice`: `createHeap`, `createTexture`, `createStagingTexture`,
  `createBuffer`, `mapBuffer`, `bindTextureMemory`, `bindBufferMemory`
  (placed resources), `createShader`, `createShaderLibrary`, `createSampler`,
  `createInputLayout`, `createFramebuffer`, `createGraphicsPipeline`,
  `createComputePipeline`, `createMeshletPipeline`,
  `createRayTracingPipeline`, `createBindingLayout`, `createBindlessLayout`,
  `createBindingSet`, `createDescriptorTable`, `createAccelStruct`,
  `createEventQuery`, `createTimerQuery`, `createCommandList`,
  `executeCommandLists(lists, queue)`, `queueWaitForCommandList`,
  `waitForIdle`, `runGarbageCollection`, `queryFeatureSupport`,
  `queryFormatSupport`, `getNativeQueue`, `getMessageCallback`.
- `ICommandList`: `open`, `close`, `writeBuffer`, `writeTexture`, `copy*`,
  `clear*`, `setGraphicsState`, `setComputeState`, `setMeshletState`,
  `setRayTracingState` (one state struct binds pipeline, framebuffer,
  bindings, buffers, viewport), `setPushConstants`, `draw*`, `dispatch*`,
  `dispatchMesh*`, `dispatchRays`, `buildBottomLevelAccelStruct`,
  `beginTimerQuery`, `beginMarker`, `endMarker`; barriers:
  `setEnableAutomaticBarriers`, `setTextureState`, `setBufferState`,
  `setPermanentTextureState`, `commitBarriers`.
- Resource interfaces (`ITexture`, `IBuffer`, `ISampler`, `IShader`,
  `IFramebuffer`, `IBindingSet`, pipelines, queries) expose `getDesc()` and
  `getNativeObject()` only.
- Three queues as an enum (graphics, compute, copy); submission returns a
  fence value. No swapchain: the host's DeviceManager owns it and wraps the
  backbuffer with `createHandleForNativeTexture`.

### The Forge

C functions over opaque structs.

- Lifecycle: `initRendererContext`, `initRenderer`, `initQueue(desc)`,
  `initFence`, `initSemaphore`, `addSwapChain(desc with window)`,
  `acquireNextImage`, `queueSubmit(desc: cmds, wait and signal semaphores,
  fence)`, `queuePresent`, `waitForFences`, `waitQueueIdle`.
- Resources: `addResourceHeap`, `addRenderTarget`, `addSampler`,
  `addShaderBinary`, `addPipeline`, `addPipelineCache`,
  `getPipelineCacheData`, `addDescriptorSet`, `addQueryPool`; buffers and
  textures through a separate `IResourceLoader` (`addResource`,
  `updateResource`) that does staging and uploads.
- Recording: `initCmdPool`, `initCmd`, `beginCmd`, `endCmd`,
  `cmdBindRenderTargets`, `cmdSetViewport`, `cmdSetScissor`,
  `cmdBindPipeline`, `cmdBindDescriptorSet`, `cmdBindVertexBuffer`,
  `cmdBindIndexBuffer`, `cmdDraw*`, `cmdDispatch`, `cmdExecuteIndirect`,
  `cmdResourceBarrier(buffer, texture, render-target barriers)`,
  `cmdBeginQuery`, `cmdEndQuery`, `cmdBeginDebugMarker`, `cmdEndDebugMarker`,
  `cmdAddDebugMarker`, `cmdWriteMarker`.
- Everything explicit: queues, fences, semaphores, barriers with before and
  after states. Closest of the three to raw Vulkan and D3D12.

### Unreal

From memory, not checked against source (needs an Epic account).

- `FDynamicRHI`, one per backend: `RHICreateTexture`, `RHICreateBuffer`,
  `RHICreateShader`, `RHICreateGraphicsPipelineState`, `RHICreateViewport`,
  `RHICreateUniformBuffer`.
- `FRHICommandList`: `SetGraphicsPipelineState`, `SetShaderParameters`,
  `DrawPrimitive`, `DispatchComputeShader`, `Transition(FRHITransitionInfo)`,
  `BeginRenderPass`.
- `FRHIResource` ref-counted with `TRefCountPtr`; queues are an enum
  (graphics, async compute, copy); the swapchain is `FRHIViewport`. The Render
  Dependency Graph sits above the RHI and emits the transitions.

## Buffer copies and barriers

Facts behind a buffer copy command and a parameterless full barrier. Findings
only. Each item is SETTLED (primary source named) or OPEN (no primary source
found; what was searched is named). Spec text was read at: DirectX-Specs
`d3d/D3D12EnhancedBarriers.md` (master), Vulkan-Docs `chapters/synchronization.adoc`
and `appendices/memorymodel.adoc` (main), Vulkan registry and
Vulkan-ValidationLayers at header version 357, Apple documentation as served
on 2026-11-01, metal-cpp as vendored in the build. Godot is cited at
commit e7b12e7 unless another commit is named.

### D3D12 enhanced barriers

**a. One global barrier as a full barrier.**

- SETTLED: `SyncBefore = D3D12_BARRIER_SYNC_ALL` "indicates ALL PRECEDING work
  must complete before executing the barrier"; `SyncAfter = SYNC_ALL`
  "indicates ALL SUBSEQUENT work must wait for the barrier to complete".
  (Enhanced Barriers, "D3D12_BARRIER_SYNC_ALL")
- SETTLED: a global barrier "applies to ALL resource memory" and does not
  transition texture layouts. (Enhanced Barriers, "D3D12_BARRIER_TYPE")
  It "cannot be used to transition" a texture where the access change would
  also need a layout change. (Enhanced Barriers, "Global Barriers")
- SETTLED: `ACCESS_COMMON` as `AccessAfter`: "subresource data must be
  available for any layout-compatible access after a barrier. Since buffers
  have no layout, `D3D12_BARRIER_ACCESS_COMMON` simply means any
  buffer-compatible access." (Enhanced Barriers, "Access Transitions")
- SETTLED: `ACCESS_COMMON` as `AccessBefore` "implies the set of all
  write-access types", is "strongly discouraged", and "A debug layer warning
  is issued when `AccessBefore` is set to `D3D12_BARRIER_ACCESS_COMMON`".
  (Enhanced Barriers, "Access Transitions"; also "App developers should avoid
  using `D3D12_BARRIER_ACCESS_COMMON` as a barrier `AccessBefore` value",
  "D3D12_BARRIER_ACCESS_COMMON")
- SETTLED: `ACCESS_COMMON` is compatible with "Any valid sync bits". (Enhanced
  Barriers, "Access Bits Barrier Sync Compatibility")
- SETTLED: `SYNC_ALL` is listed for direct, compute, copy and video command
  list types. (Enhanced Barriers, "Command Queue Sync Compatibility")
- SETTLED: a legacy NULL UAV barrier "is equivalent to enhanced Barrier global
  barriers". (Enhanced Barriers, "UAV Barriers")
- SETTLED: access transitions alone do not synchronise: "Access transitions do
  not perform synchronization." (Enhanced Barriers, "Access Transitions")
- SETTLED (from the items above): `SYNC_ALL`/`SYNC_ALL` with `ACCESS_COMMON`/`ACCESS_COMMON` is valid and
  covers all prior writes and all later accesses on that queue, with a debug
  layer warning for the `AccessBefore` value. The spec offers no other
  single-value "all writes" `AccessBefore`; naming write bits explicitly is
  what it recommends.
- SETTLED: "Resource Accesses MUST be compatible with the type of Queue
  accessing the resource." Write accesses valid in `AccessBefore`: direct
  queue, all but the video ones; compute queue, `UNORDERED_ACCESS`,
  `COPY_DEST`, `RAYTRACING_ACCELERATION_STRUCTURE_WRITE`; copy queue,
  `COPY_DEST` only. (Enhanced Barriers, "Command Queue Access Compatibility")
- SETTLED: `D3D12_BARRIER_ACCESS_COMMON` as `AccessAfter` draws no warning; it
  "may be used to return a resource back to common accessibility".
  (Enhanced Barriers, "Access Transitions")
- OPEN: whether `RAYTRACING_ACCELERATION_STRUCTURE_WRITE` in `AccessBefore` is
  rejected on a device without raytracing.

**b. Mixing `Barrier()` and `ResourceBarrier()` in one command list.**

- SETTLED: allowed. "Interop between enhanced Barrier API's and legacy
  `D3D12_RESOURCE_STATES` is supported. However, mixing legacy and enhanced
  barriers on the same subresource can introduce extra performance overhead
  ... it is strongly recommended that applications avoid mixing barrier types
  on the same subresource as much as possible." (Enhanced Barriers, "Interop
  with legacy ResourceBarrier")
- SETTLED: per-resource rule: resources "assigned any legacy state (either via
  state promotion or `ResourceBarrier`) other than
  `D3D12_RESOURCE_STATE_COMMON` must be transitioned to
  `D3D12_RESOURCE_STATE_COMMON` before being referenced using an enhanced
  Barrier"; textures with an enhanced layout must be put in `LAYOUT_COMMON`
  "before being referenced by a legacy `ResourceBarrier`"; "at the start of
  any given `ExecuteCommandLists` scope, all buffer resources and
  simultaneous-access texture subresources are implicitly in
  `D3D12_RESOURCE_STATE_COMMON`". (same section)
- SETTLED: "The D3D12 runtime internally translates all `ResourceBarrier`
  calls to equivalent Enhanced Barriers at the driver interface." (Enhanced
  Barriers, "Compatibility with legacy D3D12_RESOURCE_STATES")
- OPEN: whether a global barrier counts as "referencing" a swapchain texture
  that is in a legacy non-COMMON state (for example `RENDER_TARGET`) at the
  time. The spec states the rule per resource and says global barriers "do not
  transition texture layout"; it does not address this case. Searched
  "Interop", "Global Barriers", "D3D12_BARRIER_TYPE", "Legacy layouts".
- OPEN: whether a buffer implicitly promoted by `CopyBufferRegion` (legacy
  promotion) inside a list that otherwise uses only enhanced barriers counts as
  "assigned any legacy state ... via state promotion" for the rule above.
  Not addressed; same sections searched.

**c. Buffers created with a legacy initial state; promotion and decay.**

- SETTLED: enhanced barriers apply to buffers created with legacy states.
  "Despite the fact that legacy resource creation API's have an Initial
  State, buffers do not have a layout, and thus are treated as though they
  have an initial state of `D3D12_RESOURCE_STATE_COMMON`. This includes
  Upload Heap and Readback Heap buffers". (Enhanced Barriers, "Initial
  Resource State")
- SETTLED: buffers have no layout: "buffers have only a single subresource and
  do not have a transitionable layout". (Enhanced Barriers, "Buffer
  Barriers") With the new creation APIs, "Buffers may only use
  `D3D12_BARRIER_LAYOUT_UNDEFINED` as an initial layout." (Enhanced Barriers,
  "New Resource Creation API's") So `CreateResource3` is allowed for buffers
  but not required by the spec.
- SETTLED: enhanced barriers are optional: "Developers must check for optional
  driver support before using command list Barrier API's or resource `Create`
  methods using `InitialLayout`", through
  `D3D12_FEATURE_DATA_D3D12_OPTIONS12::EnhancedBarriersSupported`. (Enhanced
  Barriers, "Hardware Requirements"; DirectX-Headers `d3d12.h`)
- SETTLED: D3D12MA `Allocator::CreateResource3` takes `D3D12_BARRIER_LAYOUT
  InitialLayout`, "internally uses `ID3D12Device10::CreateCommittedResource3`
  or `ID3D12Device10::CreatePlacedResource2`", and returns `E_NOINTERFACE`
  if `ID3D12Device10` is unavailable. (D3D12MA `D3D12MemAlloc.h`, as vendored
  in Godot `thirdparty/d3d12ma`)
- SETTLED (enhanced): buffers need no barrier for first use in an
  `ExecuteCommandLists` scope: "buffers MAY be initially accessed in an
  `ExecuteCommandLists` scope without a Barrier", for listed accesses
  including `COPY_DEST` and `COPY_SOURCE`; "barriers MUST be used to flush
  sequential writes to the same resource". (Enhanced Barriers, "Barrier-Free
  Access")
- SETTLED (enhanced): "Since there must be no pending commands or cache flush
  operations between `ExecuteCommandLists` boundaries". (Enhanced Barriers,
  "Barrier-Free Access")
- SETTLED (legacy): "All buffer resources ... are implicitly promoted from
  D3D12_RESOURCE_STATE_COMMON to the relevant state on first GPU access";
  buffers may be promoted to `COPY_DEST` and `COPY_SOURCE`; "promotion from
  one promoted read state into multiple read state is valid, but this is not
  the case for write states". (Microsoft Learn, "Using resource barriers ...",
  "Common state promotion")
- SETTLED (legacy): "Buffer resources on any queue type" decay to COMMON when
  an `ExecuteCommandLists` completes on the GPU; "Decay does not occur between
  command lists executed together in the same ExecuteCommandLists call";
  buffers decay "regardless of whether they were explicitly transitioned".
  (same page, "State decay to common")
- SETTLED, both models: an upload-heap buffer cannot be a copy destination and
  a readback-heap buffer cannot be a copy source.
  - Legacy: "resources created on UPLOAD heaps must start in and cannot change
    from the GENERIC_READ state ... committed resources created in READBACK
    heaps must start in and cannot change from the COPY_DEST state."
    (Microsoft Learn, "Using resource barriers ...", "Initial states for
    resources"; also `D3D12_HEAP_TYPE`)
  - Enhanced: upload heap resources "allow only" vertex, constant, index,
    shader resource, indirect argument, `COPY_SOURCE` and `RESOLVE_SOURCE`
    access; readback heap resources support only `COPY_DEST` and
    `RESOLVE_DEST`. (Enhanced Barriers, "Upload Heap Resources", "Readback
    Heap Resources")
- SETTLED: readback heap buffers "may require barriers to manage
  write-after-write hazards". (Enhanced Barriers, "Readback Heap Resources")
- SETTLED: `CopyBufferRegion` documents no alignment rule for `DstOffset`,
  `SrcOffset` or `NumBytes`. (Microsoft Learn,
  `ID3D12GraphicsCommandList::CopyBufferRegion`) A buffer may be both source
  and destination of one copy under enhanced barriers if the regions do not
  intersect. (Enhanced Barriers, "Self Resource Copy")

**d. Engines.**

- SETTLED, Godot (`drivers/d3d12/rendering_device_driver_d3d12.cpp`):
  - Buffers are made with D3D12MA `CreateResource3(..., D3D12_BARRIER_LAYOUT_UNDEFINED, ...)`
    when enhanced barriers are supported, else `CreateResource(..., initial_state, ...)`
    (944-961; support flag from `options12.EnhancedBarriersSupported`, 6452).
    Legacy initial states: default heap `COMMON` (890), upload `GENERIC_READ`,
    readback `COPY_DEST` (895-913).
  - Pipeline barriers exist only with enhanced barriers ("Enhanced barriers
    are a requirement for this function.", 2426-2428). Each memory barrier
    becomes one `D3D12_GLOBAL_BARRIER`; buffer barriers use `Offset = 0`,
    `Size = UINT64_MAX` (2446-2464). `PIPELINE_STAGE_ALL_COMMANDS_BIT` maps to
    `D3D12_BARRIER_SYNC_ALL` (2282); access starts as `ACCESS_COMMON` and has
    no case for `MEMORY_READ`/`MEMORY_WRITE` (2189-2191); incompatible
    sync/access falls back to `SYNC_ALL`/`ACCESS_COMMON` (2366-2368).
  - Its full barrier is `ALL_COMMANDS`/`MEMORY_READ|MEMORY_WRITE` on both sides
    (`servers/rendering/rendering_device_graph.cpp:1455-1460`), so on D3D12
    it becomes one global barrier `SYNC_ALL`/`SYNC_ALL`,
    `ACCESS_COMMON`/`ACCESS_COMMON`.
  - `command_copy_buffer` adds legacy `COPY_SOURCE`/`COPY_DEST` transitions
    only without enhanced barriers, then calls `CopyBufferRegion` once per
    region (4058-4072).
  - Mixes both models in one list: screen framebuffers get legacy transitions
    "even if enhanced barriers are supported" (4521-4522) and go back to
    `PRESENT` by legacy transition at the end of the pass (4709-4714).
  - No comments on promotion or decay (grepped `promot|decay|implicit` in
    `drivers/d3d12`).
- SETTLED, NVRHI: enhanced barriers without a global barrier. Buffers are made
  with `CreateCommittedResource3(..., D3D12_BARRIER_LAYOUT_UNDEFINED, ...)`
  when supported (`src/d3d12/d3d12-buffer.cpp:147-157`), placed ones with
  `CreatePlacedResource2` (353-361); buffer barriers cover the whole buffer
  ("NVRHI doesn't support partial buffer barriers",
  `src/d3d12/d3d12-state-tracking.cpp:182`); no `D3D12_GLOBAL_BARRIER`
  anywhere (grepped). Legacy UAV barriers in `src/d3d12/d3d12-upload.cpp:211-215`
  are not gated on enhanced support.
- SETTLED, Diligent, The Forge, O3DE: no enhanced barriers (grepped
  `GraphicsCommandList7`, `D3D12_GLOBAL_BARRIER`, `D3D12_BARRIER_SYNC_ALL`,
  `CreateResource3`, `CreateCommittedResource3`, `CreatePlacedResource2`,
  `EnhancedBarriersSupported`). All three create upload buffers in
  `GENERIC_READ` and readback buffers in `COPY_DEST` (Diligent
  `Graphics/GraphicsEngineD3D12/src/BufferD3D12Impl.cpp:126-133`; The Forge
  `Common_3/Graphics/Direct3D12/Direct3D12.c:3456-3464`; O3DE
  `Gems/Atom/RHI/DX12/Code/Source/RHI/Conversions.cpp:725-731`).

### Host visibility after a CPU wait

**a. Vulkan.**

- SETTLED: a fence wait is an execution dependency only for the host: "Signaling
  a fence and waiting on the host does not guarantee that the results of
  memory accesses will be visible to the host, as the access scope of a
  memory dependency defined by a fence only includes device access. A memory
  barrier or other memory dependency must be used to guarantee this."
  (Vulkan spec, "Waiting on Fences", `synchronization-fences-waiting`)
- SETTLED: a semaphore signal operation has first access scope "all memory
  access performed by the device" and an empty second access scope; the same
  holds for fence signals. (Vulkan spec, "Semaphore Signaling", "Fence
  Signaling")
- OPEN: the spec has no note for `vkWaitSemaphores` like the fence note above.
  Searched `synchronization.adoc` for host waits on semaphores, "Host
  Operations on Semaphores" and `vkWaitSemaphores`.
- SETTLED: what makes device writes reach the host: a memory dependency whose
  destination access mask includes `HOST_READ` or `HOST_WRITE` "includes a
  memory domain operation from device domain to host domain". (Vulkan spec,
  Memory Model, "Availability, Visibility, and Domain Operations")
  `VkMemoryBarrier2` "defines a memory dependency affecting all device
  memory". (Vulkan spec, "Global Memory Barriers")
- SETTLED: "If dstAccessMask includes VK_ACCESS_2_HOST_READ_BIT, dstStageMask
  must include VK_PIPELINE_STAGE_2_HOST_BIT". (VUID-VkMemoryBarrier2-dstAccessMask-03916)
- SETTLED: `HOST_BIT` is "a pseudo-stage indicating execution on the host of
  reads/writes of device memory. This stage is not invoked by any commands
  recorded in a command buffer." `ALL_COMMANDS_BIT` "specifies all operations
  performed by all commands supported on the queue it is used with". (Vulkan
  spec, "Pipeline Stages") So a barrier with `ALL_COMMANDS` as destination does
  not include `HOST`.
- SETTLED: `HOST_COHERENT` covers only the last step: without it
  "vkInvalidateMappedMemoryRanges must be called to guarantee that writes
  which are available to the host domain are made visible to host
  operations"; with it, "writes made available to the host domain are
  automatically made visible to the host". (Vulkan spec, "Host Access Types",
  `synchronization-host-access-types`) The domain operation above is still
  needed either way.
- SETTLED: the other direction is automatic: a queue submission's first access
  scope "includes all host writes to mappable device memory that are
  available to the host memory domain". (Vulkan spec, "Host Write Ordering
  Guarantees", `synchronization-submission-host-writes`)
- SETTLED, engines: Granite records a `COPY`/`TRANSFER_WRITE` to
  `HOST`/`HOST_READ` barrier before the fenced submit of a readback
  (`renderer/utils/image_utils.cpp:259-261`). NVRHI notes the barrier is
  missing: "TODO: there should be a barrier... But there can't be a command
  list here" (`src/vulkan/vulkan-buffer.cpp:609-610`). Godot records no HOST
  barrier; readback buffers require `HOST_COHERENT|HOST_VISIBLE`
  (`drivers/vulkan/rendering_device_driver_vulkan.cpp:2040-2047`) and
  `buffer_get_data` stalls for all frames before mapping
  (`servers/rendering/rendering_device.cpp:1383-1386`). Diligent invalidates
  mapped ranges for non-coherent readback memory
  (`Graphics/GraphicsEngineVulkan/src/DeviceContextVkImpl.cpp:2900`).

**b. D3D12.**

- SETTLED: `Map` "invalidates the CPU cache, when necessary, so that CPU reads
  to this address reflect any modifications made by the GPU". (Microsoft Learn,
  `ID3D12Resource::Map`)
- SETTLED: `Map` "doesn't synchronize with the GPU"; "You use a fence to detect
  when the GPU completes processing"; "After the data is ready (including any
  necessary resource barrier), call ID3D12Resource::Map to make the readback
  data visible to the CPU." (Microsoft Learn, "Readback data using heaps")
- SETTLED: readback heap "CPU cache behavior is write-back". (Microsoft Learn,
  `D3D12_HEAP_TYPE`)
- SETTLED: persistent mapping is allowed for all CPU-accessible heaps;
  the documented condition is about CPU writes before GPU use. (Microsoft
  Learn, `ID3D12Resource::Map`, Remarks)
- OPEN: whether a pointer kept from an earlier `Map` (persistent map) of a
  READBACK buffer sees GPU writes made after that `Map` without a new
  `Map`/`Unmap`. The docs tie CPU cache invalidation to `Map` and say nothing
  for the persistent read case. Searched the `Map`, `Unmap`, `D3D12_HEAP_TYPE`
  and "Readback data using heaps" pages.

**c. Metal.**

- SETTLED: for `MTLStorageModeShared`, "you're responsible for synchronizing
  access ... Ensure that all changes you schedule on either the CPU or GPU for
  a resource that uses shared memory complete before accessing that resource
  on the other processor." (Apple, `MTLStorageMode.shared`) No flush or
  invalidate call is documented for shared buffers.
- SETTLED: `MTLSharedEvent` "synchronizes memory operations to one or more
  resources across multiple CPUs, GPUs, and processes"; shared events
  synchronize "resource memory operations from other parts of the app,
  including: Code that runs on the CPU". (Apple, `MTLSharedEvent`; "Resource
  synchronization")
- SETTLED: `MTL4CommandQueue.signalEvent(_:value:)` "Schedules an operation to
  signal a GPU event with a specific value after all GPU work prior to this
  point is complete." (Apple, `MTL4CommandQueue/signalEvent(_:value:)`)
- SETTLED: `MTLStorageModeManaged` needs an explicit GPU-to-CPU step: after a
  GPU write, "encode a synchronize(resource:) command ... Before executing
  this command, the modified buffer's data on the CPU is in an undefined
  state." Managed is macOS only. (Apple, "Synchronizing a managed resource in
  macOS"; `MTLStorageMode.managed`)
- SETTLED: `MTL4ComputeCommandEncoder` has no `synchronize` method; the
  synchronize commands are on `MTLBlitCommandEncoder`. (metal-cpp
  `MTL4ComputeCommandEncoder.hpp`, `MTLBlitCommandEncoder.hpp`; Apple,
  `MTL4ComputeCommandEncoder` command table)
- SETTLED: Metal 4 "is supported by devices equipped with the Apple M1 and
  later, as well as the A14 Bionic and later." (WWDC25 "Discover Metal 4")
- OPEN: whether Managed buffers are usable with Metal 4 queues, and how they
  would be synchronised. Searched the Metal 4 core API article, the
  `MTL4ComputeCommandEncoder` and `MTLStorageMode` pages.

### Metal 4

**a. Hazard tracking.**

- SETTLED: none. "In Metal 4, the framework considers all resources
  untracked." (Apple, "Understanding the Metal 4 core API", "Barriers") "The
  value of an MTLResource instance's hazardTrackingMode property has no effect
  on the work you submit to an MTL4CommandQueue." (Apple, "Resource
  synchronization")
- SETTLED: commands inside one MTL4 compute pass may run concurrently: "An
  MTL4ComputeCommandEncoder instance creates a compute pass that runs commands
  concurrently on the GPU". (Apple, "Resource synchronization")
- SETTLED: residency sets do not track hazards. (Apple, `MTLResidencySet`)

**b. Barrier semantics.**

- SETTLED: `barrierAfterEncoderStages:beforeEncoderStages:` is intra-pass: it
  orders "subsequent work you encode in the current command encoder,
  corresponding to beforeEncoderStages" after "all prior commands in this
  command encoder, corresponding to afterEncoderStages". The stages must be
  ones "for which this encoder can encode commands"; for a compute encoder
  "any combination of dispatch, blit and accelerationStructure". (Apple,
  `MTL4CommandEncoder/barrier(afterEncoderStages:beforeEncoderStages:visibilityOptions:)`)
  "An intrapass barrier has no effect on any other pass." (Apple,
  "Synchronizing stages within a pass")
- SETTLED: "Metal doesn't support intrapass barriers that wait for the tile or
  fragment stages on devices that have a tile-based deferred rendering (TBDR)
  architecture, such as Apple silicon GPUs." The documented route is to end
  the render pass and use a queue barrier or fence. (Apple, "Synchronizing
  stages within a pass")
- SETTLED: `barrierAfterQueueStages:beforeStages:` (consumer barrier) is called
  on the consuming encoder. `afterQueueStages` "applies to work ... you encode
  in prior command encoders, and not for the current encoder"; `beforeStages`
  "applies to work you encode in the current command encoder". It may name
  stages unrelated to the current encoder. "Don't use this method for
  synchronizing resource access within the same pass." (Apple,
  `MTL4CommandEncoder/barrier(afterQueueStages:beforeStages:visibilityOptions:)`)
  It blocks `beforeStages` "in the current pass, and all subsequent passes in
  the same queue". (Apple, "Synchronizing passes with consumer barriers")
- SETTLED: `barrierAfterStages:beforeQueueStages:` (producer barrier) is called
  on the producing encoder. `afterStages` covers "commands you previously
  encode in the current encoder (and prior encoders)"; `beforeQueueStages`
  "applies to subsequent encoders and not to work in the current command
  encoder". Any stages may be passed. (Apple,
  `MTL4CommandEncoder/barrier(afterStages:beforeQueueStages:visibilityOptions:)`)
  "Producer barriers are only available to Metal 4 encoder types."
  (Apple, "Synchronizing passes with producer barriers")
- SETTLED: both queue barriers cover passes "that you submit to the same
  command queue, including passes from other command buffers". (Apple,
  "Synchronizing passes with producer barriers", "... consumer barriers")
  Events are the cross-queue mechanism. (Apple, "Resource synchronization")
- SETTLED: "Metal can reorder the exact point where it applies the barrier, so
  encode the barrier as close to the command that consumes the resource as
  possible." (Apple, consumer barrier method page)
- SETTLED: `MTL4VisibilityOptions`: `VisibilityOptionNone = 0`,
  `VisibilityOptionDevice = 1` ("Flushes caches to the GPU (device) memory
  coherence point"), `VisibilityOptionResourceAlias = 1 << 1` ("Flushes caches
  to ensure that aliased virtual addresses are memory consistent"). The Swift
  declarations default to `[ .device ]`; metal-cpp has no default argument.
  (Apple, `MTL4VisibilityOptions`; metal-cpp `MTL4CommandEncoder.hpp`)
- SETTLED: `MTLStages` values: Vertex, Fragment, Tile, Object, Mesh,
  ResourceState, Dispatch, Blit, AccelerationStructure, MachineLearning, and
  `StageAll` ("Convenience mask representing all stages of GPU work"). Buffer
  copies run in the blit stage. (Apple, `MTLStages`,
  `MTL4ComputeCommandEncoder` command table; metal-cpp `MTLCommandEncoder.hpp`)
- SETTLED (from the items above): a full barrier between a copy and later work:
  - later work in the same compute encoder: an intra-pass barrier on that
    encoder (blit/dispatch/accelerationStructure stages only);
  - later work in a later encoder of the same command buffer, or in a later
    command buffer committed to the same queue: a producer barrier on the
    copying encoder before it ends, or a consumer barrier on the later
    encoder. A consumer barrier on the encoder that recorded the copy does not
    cover that copy.
- OPEN: whether `StageAll` is accepted in `afterEncoderStages`/
  `beforeEncoderStages` of an intra-pass barrier. The method page says the
  stages must be ones the encoder can encode; nothing says whether `StageAll`
  is filtered or invalid. Searched the method page, `MTLStages/all` and
  "Synchronizing stages within a pass".
- SETTLED: barriers can only be recorded on an encoder. They are methods of
  `MTL4CommandEncoder`; `MTL4CommandBuffer` has none. (metal-cpp
  `MTL4CommandBuffer.hpp`, `MTL4CommandEncoder.hpp`) `endEncoding()` "Declares
  that all command generation from this encoder is complete", so a producer
  barrier has to be recorded before it. (Apple, `MTL4CommandEncoder`)

**c. Buffer copies.**

- SETTLED: Metal 4 has no blit encoder; `MTL4CommandBuffer` makes compute,
  render and machine-learning encoders only. (metal-cpp
  `MTL4CommandBuffer.hpp`) `MTL4ComputeCommandEncoder` "combines the
  functionality of" `MTLBlitCommandEncoder`, `MTLComputeCommandEncoder` and
  `MTLAccelerationStructureCommandEncoder`. (Apple, "Understanding the Metal 4
  core API")
- SETTLED: buffer copies are
  `MTL4ComputeCommandEncoder::copyFromBuffer(source, sourceOffset, destination, destinationOffset, size)`,
  stage `blit`. (Apple, `MTL4ComputeCommandEncoder/copy(sourceBuffer:sourceOffset:destinationBuffer:destinationOffset:size:)`;
  metal-cpp `MTL4ComputeCommandEncoder.hpp`)

**d. Residency.**

- SETTLED: MTL4 encoders have no `useResource`/`useHeap` methods (metal-cpp:
  present in `MTLComputeCommandEncoder.hpp` and `MTLRenderCommandEncoder.hpp`,
  absent from every `MTL4*.hpp`). `MTL4RenderCommandEncoder` "differs ... by
  removing methods that manage resource bindings and residency sets" and lets
  you "Add a residency set to either an encoder's command buffer, or the
  command queue". (Apple, "Understanding the Metal 4 core API")
- SETTLED: "In Metal 4, your app uses residency sets to specify the resources
  that Metal should make resident ... make sure that the residency set contains
  all the required resources for the command buffer you commit." (WWDC25
  "Discover Metal 4") "Anytime you need a resource on the GPU, add it to a
  residency set." (WWDC25 "Explore Metal 4 games")
- OPEN: an Apple reference page stating that an MTL4 command using a buffer not
  in any attached residency set is invalid, and what happens. Searched
  `MTLResidencySet`, "Simplifying GPU resource management with residency
  sets", `MTL4CommandQueue`, `MTL4CommandBuffer/useResidencySet`, the Metal 4
  core API article.
- SETTLED: queue sets: `MTL4CommandQueue::addResidencySet` "Applies a residency
  set to a queue, which Metal applies to the queue's command buffers as you
  commit them"; up to 32 per queue. Command buffer sets:
  `MTL4CommandBuffer::useResidencySet`, up to 32 per command buffer. (Apple,
  method pages)
- SETTLED: changes take effect only on `commit()`: "A residency set's addition
  and removal methods don't take effect until you call this method." Methods
  "aren't thread-safe". Metal "doesn't synchronize the state of the residency
  set between the CPU and the GPU", so allocations can be added while the GPU
  runs. An allocation from a heap makes the whole heap resident. (Apple,
  `MTLResidencySet`)
- SETTLED: removing a set from a queue leaves it attached to in-flight command
  buffers. (Apple, "Simplifying GPU resource management with residency sets")
- SETTLED: no local engine implements Metal 4 (grepped `MTL4`,
  `MTLResidencySet`, `useResidencySet`, `addResidencySet`,
  `barrierAfter*Stages` in Godot, The Forge, O3DE, Diligent, NVRHI, Granite,
  Hazel, Doriax). Godot has only a forward declaration of an MTL4 command
  buffer (`drivers/metal/rendering_device_driver_metal.h:57-64`).
- SETTLED, Godot at commit 3000096 (older than e7b12e7), Metal 3 with
  residency sets and queue barriers, opt-in through `GODOT_MTL_FORCE_BARRIERS=1`
  on OS 26 ("Barriers are still experimental in Metal 3",
  `drivers/metal/rendering_device_driver_metal3.cpp:109-127`):
  - one "Main Residency Set" added to the queue with `addResidencySet`, and
    resources made `Untracked` (same lines);
  - every buffer is added at creation and removed at free
    (`rendering_device_driver_metal.cpp:149`, `_track_resource` at 2832-2842);
    textures and argument buffers too;
  - pending additions and removals are applied and `commit()` is called at
    submit, only when something changed (`rendering_device_driver_metal3.cpp:202-216`);
  - a per-command-buffer set is attached with `useResidencySet`
    (`metal3_objects.cpp:134-136`);
  - inter-pass barriers are recorded on the next encoder that must wait,
    with `barrierAfterQueueStages` ("barrierAfterQueueStages is called on the
    encoder that must wait", `metal3_objects.cpp:163`, 198-213);
    `ALL_COMMANDS` maps to `MTL::StageAll` (`metal_objects_shared.h:531-532`).
- SETTLED, Godot at e7b12e7: no residency set is ever assigned (no
  `main_residency_set =` in `drivers/metal`); residency comes from
  `useHeaps`/`useResources` and ordering from fences
  (`drivers/metal/metal3_objects.cpp:72-149`).

### Metal buffer copy alignment

- SETTLED for `MTLBlitCommandEncoder`: "In macOS, `sourceOffset` needs to be a
  multiple of `4`, but can be any value in iOS and tvOS"; the same for
  `destinationOffset` and `size`. Overlapping copies within one buffer "may
  result in unexpected behavior". (Apple,
  `MTLBlitCommandEncoder/copy(from:sourceOffset:to:destinationOffset:size:)`)
- OPEN for `MTL4ComputeCommandEncoder`: its buffer copy page states no
  alignment rule. Searched that page and the `MTL4ComputeCommandEncoder`
  overview.
- SETTLED, for comparison: `vkCmdCopyBuffer` has no alignment VUID (offsets below buffer
  size, size within bounds, size > 0, regions not overlapping); D3D12
  `CopyBufferRegion` documents none. (Vulkan-ValidationLayers 1.4.357
  `vk_validation_error_messages.cpp`; Microsoft Learn)
- SETTLED, engines: no local engine handles a multiple-of-4 rule for Metal buffer copies
  (Godot `MDCommandBuffer::copy_buffer`, `drivers/metal/metal3_objects.cpp:721-731`;
  The Forge `Common_3/Graphics/Metal/MetalRenderer.mm:4580-4584`; grepped
  for alignment checks). Godot checks multiple-of-4 only for `buffer_clear`
  (`servers/rendering/rendering_device.cpp:1327-1328`).

### Vulkan barriers inside dynamic rendering

- SETTLED: `vkCmdPipelineBarrier2` is `renderpass="both"`; `vkCmdCopyBuffer`
  and `vkCmdCopyBuffer2` are `renderpass="outside"`
  (VUID-vkCmdCopyBuffer-renderpass: "This command must only be called outside
  of a render pass instance"). (vk.xml 357)
- SETTLED: inside a `vkCmdBeginRendering` instance, from
  Vulkan-ValidationLayers 1.4.357 `vk_validation_error_messages.cpp`:
  - VUID-vkCmdPipelineBarrier2-None-09553: unless a
    `shaderTileImage*ReadAccess` feature or `dynamicRenderingLocalRead` is
    enabled, it "must not be called within a render pass instance started with
    vkCmdBeginRendering".
  - VUID-vkCmdPipelineBarrier2-None-09554: without `dynamicRenderingLocalRead`,
    "there must be no buffer or image memory barriers".
  - VUID-vkCmdPipelineBarrier2-None-09586: without `dynamicRenderingLocalRead`,
    memory barriers may only use the color and depth/stencil attachment
    read/write access bits.
  - VUID-vkCmdPipelineBarrier2-srcStageMask-09556: "must only specify
    framebuffer-space stages in srcStageMask and dstStageMask".
  - VUID-vkCmdPipelineBarrier2-image-09555: an attachment image in an image
    barrier must be in `RENDERING_LOCAL_READ` or `GENERAL`.
  - For any render pass instance: VUID-...-None-07892 (graphics stages only),
    -dependencyFlags-07891 (`BY_REGION` when source has framebuffer-space
    stages), -None-07890, -oldLayout-01181 (no layout change),
    -srcQueueFamilyIndex-01182 (no queue transfer), -None-07893 (`VIEW_LOCAL`
    with multiview), -suspended ("must not be called between suspended render
    pass instances").
- SETTLED (from the items above): `ALL_COMMANDS`, `HOST`, `COPY` and buffer barriers are not allowed inside
  a dynamic rendering instance.

## Textures and texture copies

Facts behind creating and destroying textures and copying buffer to texture,
texture to buffer and texture to texture. Findings only. Each item is SETTLED
(primary source named) or OPEN (no primary source found; what was searched is
named). Spec text was read at: Vulkan-Docs `chapters/resources.adoc`,
`copies.adoc`, `formats.adoc`, `limits.adoc`, `devsandqueues.adoc` and
`proposals/VK_KHR_unified_image_layouts.adoc` (main, 2026-11-02); vk.xml and
Vulkan-ValidationLayers at header version 357; DirectX-Specs
`d3d/D3D12EnhancedBarriers.md` and `d3d/VulkanOn12.md` (master);
DirectX-Headers as vendored in Godot `thirdparty/directx_headers`, and
`src/d3dx12_property_format_table.cpp` (main); Microsoft Learn and Apple
documentation as served on 2026-11-02; metal-cpp as vendored in the build.
Godot is cited at commit e7b12e7, NVRHI at 8e8c36e, Diligent at bb821b7, The
Forge at a00d6d3, O3DE at 94790ad, Granite at 7c59ad8.

### Texture layouts

**a. Vulkan.**

- SETTLED: every image subresource is in a layout, and every command that
  touches an image names the layout it is in: "for each mechanism of
  accessing an image in the API the application must specify a parameter or
  structure member that indicates which image layout the image
  subresource(s) are considered to be in". (Vulkan spec, "Image Layouts")
- SETTLED: copies accept only the transfer layout or `GENERAL`:
  "dstImageLayout must be VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR,
  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, or VK_IMAGE_LAYOUT_GENERAL"
  (VUID-vkCmdCopyBufferToImage-dstImageLayout-01396); the same with
  `TRANSFER_SRC_OPTIMAL` for `vkCmdCopyImageToBuffer` (-srcImageLayout-01397)
  and both sides of `vkCmdCopyImage` (-srcImageLayout-01917,
  -dstImageLayout-01395). A copy within one subresource range needs `GENERAL`
  or `SHARED_PRESENT` on both sides (VUID-vkCmdCopyImage-srcImage-09460).
  (Vulkan-ValidationLayers 1.4.357)
- SETTLED: `GENERAL` is valid for any access on core Vulkan, with a possible
  cost: "VK_IMAGE_LAYOUT_GENERAL supports all types of device access, unless
  specified otherwise." "After initialization, applications need not use any
  layout other than the general layout, though this may produce suboptimal
  performance on some implementations." Exceptions named: `PRESENT_SRC_KHR`,
  `SHARED_PRESENT_KHR` and the video layouts. (Vulkan spec, "Image Layouts")
- SETTLED: `VK_KHR_unified_image_layouts` is a ratified KHR device extension
  (number 528, `ratified="vulkan"`, depends on
  `VK_KHR_get_physical_device_properties2,VK_VERSION_1_1`) with features
  `unifiedImageLayouts` and `unifiedImageLayoutsVideo`. (vk.xml 357) With
  the feature enabled, "the VK_IMAGE_LAYOUT_GENERAL image layout may be used
  in place of the other layouts where allowed with no loss of performance".
  (Vulkan spec, "Image Layouts") The proposal: layouts "are, barring a few
  exceptions, effectively all identical"; present layouts "are different from
  VK_IMAGE_LAYOUT_GENERAL"; `UNDEFINED` is kept, and `GENERAL` "is not
  allowed when creating an image". (Vulkan-Docs
  `proposals/VK_KHR_unified_image_layouts.adoc`) It is a feature a device may
  or may not report; Vulkan 1.3 core does not include it.
- SETTLED: the spec does not say how much slower `GENERAL` is without the
  extension; it says only "may produce suboptimal performance on some
  implementations". (Vulkan spec, "Image Layouts")
- SETTLED: initial layout: "initialLayout must be VK_IMAGE_LAYOUT_UNDEFINED
  or VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT or VK_IMAGE_LAYOUT_PREINITIALIZED"
  (VUID-VkImageCreateInfo-initialLayout-00993). `UNDEFINED`: "the contents of
  the data are considered to be undefined, and the transition away from this
  layout is not guaranteed to preserve that data." `PREINITIALIZED` is for
  linear images whose contents the host writes. "For either of these initial
  layouts, any image subresources must be transitioned to another layout
  before they are accessed by the device." "Image memory cannot be
  transitioned into" `UNDEFINED` or `PREINITIALIZED`. (Vulkan spec, "Image
  Layouts")
- SETTLED: a layout transition is an image memory barrier with
  `oldLayout`/`newLayout`; the parameterless full barrier BFR has today (a
  global memory barrier) changes no layout. (Vulkan spec,
  `synchronization.adoc`, "Image Memory Barriers"; see "Buffer copies and
  barriers")

**b. D3D12 enhanced barriers.**

- SETTLED: access allowed per layout ("Layout Access Compatibility" table):
  - `UNDEFINED`: "Any access bits (only when BOTH `LayoutBefore` AND
    `LayoutAfter` are `UNDEFINED`)", else `NO_ACCESS`.
  - `COMMON`: `SHADER_RESOURCE`, `COPY_DEST`, `COPY_SOURCE`.
  - `DIRECT_QUEUE_COMMON`, `COMPUTE_QUEUE_COMMON`: `COPY_SOURCE`,
    `COPY_DEST`, `SHADER_RESOURCE`, `UNORDERED_ACCESS`.
  - `GENERIC_READ`, `COMPUTE_QUEUE_GENERIC_READ`: `SHADER_RESOURCE`,
    `COPY_SOURCE`. `DIRECT_QUEUE_GENERIC_READ` adds `DEPTH_STENCIL_READ`,
    `SHADING_RATE_SOURCE`, `RESOLVE_SOURCE`.
  - `RENDER_TARGET`: `RENDER_TARGET` only. `UNORDERED_ACCESS` (and queue
    variants): `UNORDERED_ACCESS` only. `DEPTH_STENCIL_WRITE`:
    `DEPTH_STENCIL_READ`, `DEPTH_STENCIL_WRITE`. `DEPTH_STENCIL_READ`,
    `SHADER_RESOURCE`, `COPY_SOURCE`, `COPY_DEST`, `RESOLVE_*`,
    `SHADING_RATE_SOURCE` (and queue variants): the one matching access.
  - `PRESENT`: "Alias for `D3D12_BARRIER_LAYOUT_COMMON`."
  (Enhanced Barriers, "Layout Access Compatibility",
  "D3D12_BARRIER_LAYOUT_PRESENT")
- SETTLED: no layout allows render target together with copy or shader
  resource for an ordinary texture; `RENDER_TARGET` and the depth layouts
  allow only attachment access. (same table)
- SETTLED: the one exception is a simultaneous-access texture: "textures
  created using the `D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS` flag can
  be used without barrier" for `SHADER_RESOURCE`, `COPY_DEST`, `COPY_SOURCE`,
  `RENDER_TARGET`, `UNORDERED_ACCESS`, `RESOLVE_DEST`, `RESOLVE_SOURCE`
  (Enhanced Barriers, "COMMON Layout"); they "have an immutable layout, and
  therefore never require a layout transition", but "accessing
  simultaneous-access textures is generally much slower than standard texture
  accesses" ("Single-Queue Simultaneous Access"). The flag "Can't be used
  with MSAA textures" nor with `ALLOW_DEPTH_STENCIL`. (Microsoft Learn,
  `D3D12_RESOURCE_FLAGS`)
- SETTLED: a texture can stay in `COMMON` for copies and shader reads:
  "Any texture subresource in `D3D12_BARRIER_LAYOUT_COMMON` can be used
  without a layout transition for any combination of" `SHADER_RESOURCE`,
  `COPY_DEST`, `COPY_SOURCE`. "Subresources with this layout are readable in
  any queue type without requiring a layout change. They are also writable as
  a copy dest in any queue type." The queue-specific common layouts add
  `UNORDERED_ACCESS`, offer "reduced decompress operations", and "Using a
  queue-specific common layout is strongly recommended whenever possible."
  (Enhanced Barriers, "COMMON Layout", "D3D12_BARRIER_LAYOUT_COMMON",
  "Barrier-Free Access")
- SETTLED: layouts are tied to queue types. "Both `LayoutBefore` and
  `LayoutAfter` must be compatible with the type of queue performing the
  Barrier." ("Layout Transitions") Direct queues may use every layout except
  `COMPUTE_QUEUE_*`; compute queues may not use `RENDER_TARGET`, depth or
  `DIRECT_QUEUE_*` layouts (except
  `DIRECT_QUEUE_GENERIC_READ_COMPUTE_QUEUE_ACCESSIBLE`); copy queues may use
  `COMMON` only: "Copy queues do not support layout transition Barriers".
  ("Command Queue Layout Compatibility", "Copy Queues")
  `DIRECT_QUEUE_COMMON` "cannot be used for Present"; `COMMON` "is still the
  required layout for Presentation". ("D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON")
- SETTLED: on some hardware a layout carries compression: "textures are often
  compressed when used as a render target or depth stencil and are often
  uncompressed for shader read or copy commands"; a transition to a
  queue-specific layout "can be significantly faster" because decompression
  may be skipped. ("Layout Transitions", "Queue type specific layouts")
- SETTLED: initial layout. `CreateCommittedResource3` and
  `CreatePlacedResource2` take `InitialLayout`: "Initial layout of texture
  resource, `D3D12_BARRIER_LAYOUT_UNDEFINED` for buffers". ("ID3D12Device10
  CreateCommittedResource3", "... CreatePlacedResource2") Legacy creation
  states imply a layout: "For texture resources, this initial state implies
  an initial layout according to the table" (`COMMON` → `COMMON`,
  `RENDER_TARGET` → `RENDER_TARGET`, `COPY_DEST` → internal
  `LEGACY_COPY_DEST`, ...); the legacy layouts "are not exposed in the public
  API and may not be used in enhanced `Barrier` API calls". ("Initial
  Resource State", "Equivalent D3D12_BARRIER_LAYOUT for each
  D3D12_RESOURCE_STATES bit", "Legacy layouts")
- SETTLED: leaving `UNDEFINED`: a barrier with only one side `UNDEFINED`
  "must set the corresponding `AccessBefore` or `AccessAfter` value to
  `D3D12_BARRIER_ACCESS_NO_ACCESS`"; `TEXTURE_BARRIER_FLAG_DISCARD` "Can only
  be used when `LayoutBefore` is `D3D12_BARRIER_LAYOUT_UNDEFINED`"; a
  transition from `UNDEFINED` "to any potentially-compressed layout ... MUST
  initialize compression metadata when `D3D12_TEXTURE_BARRIER_FLAG_DISCARD`
  is present". Barrier-free first access needs "The subresource layout is
  compatible with the access type" and "Any necessary compression metadata
  has been initialized". ("D3D12_BARRIER_LAYOUT_UNDEFINED",
  "D3D12_TEXTURE_BARRIER_FLAG_DISCARD", "Placed Resource Metadata
  Initialization", "Barrier-Free Access")
- OPEN: whether a texture created with `InitialLayout = UNDEFINED` may be
  copied into without a barrier first. The rules above require a compatible
  layout; `UNDEFINED` is compatible with no access unless both sides of a
  barrier are `UNDEFINED`. No sentence addresses first use after creation.
  Searched "Barrier-Free Access", "New Resource Creation API's",
  "D3D12_BARRIER_LAYOUT_UNDEFINED", "Initial Resource State".
- SETTLED: the global barrier BFR records today does not change layouts (see
  "Buffer copies and barriers"); a layout change needs a
  `D3D12_TEXTURE_BARRIER`. ("Layout Transitions")

**c. Metal 4.**

- SETTLED: no texture layouts. metal-cpp has no layout type or parameter for
  textures; the copy methods of `MTL4::ComputeCommandEncoder` take no layout,
  and the only `MTL4*.hpp` header mentioning "layout" is
  `MTL4AccelerationStructure.hpp`. (metal-cpp, grepped `layout` in `MTL4*.hpp`,
  `MTLTexture.hpp`, `MTLCommandEncoder.hpp`) Metal 4 synchronisation is by
  access conflict only: "Synchronize commands submitted to an
  MTL4CommandQueue instance when they have an access conflict with a
  resource." (Apple, "Resource synchronization")
- SETTLED: the related concern is lossless compression, which Metal manages.
  "By default, Metal attempts to optimize a texture's data for both GPU and
  CPU memory operations based on the texture's storage mode and usage
  options." Lossless compression is not applied when the format is block
  compressed, when usage includes `unknown`, `shaderWrite` or
  `pixelFormatView`, or when the texture uses an `MTLBuffer`. Private storage
  or `renderTarget` usage makes Metal optimise for GPU access by default;
  otherwise the app may encode `optimizeContentsForGPUAccess`.
  `allowGPUOptimizedContents = false` opts out. (Apple, "Optimizing texture
  data"; `MTLTextureUsage`: "Only set usage options that correspond to a
  texture's intended use.")
- SETTLED: `MTL4ComputeCommandEncoder` has `optimizeContentsForGPUAccess`
  (whole texture, or slice and level). (metal-cpp
  `MTL4ComputeCommandEncoder.hpp:96-97`; Apple,
  `MTL4ComputeCommandEncoder/optimizeContents(forGPUAccess:)`)
- SETTLED: copies require `isFramebufferOnly` false on both textures. (Apple,
  MTL4 copy method pages)

**d. Engines: who tracks layouts.**

- SETTLED, Godot: the driver interface takes explicit layouts and the render
  graph tracks them; callers never see layouts.
  - `RenderingDeviceDriver::TextureLayout`: `UNDEFINED`, `GENERAL`,
    `STORAGE_OPTIMAL`, `COLOR_ATTACHMENT_OPTIMAL`,
    `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, `DEPTH_STENCIL_READ_ONLY_OPTIMAL`,
    `SHADER_READ_ONLY_OPTIMAL`, `COPY_SRC_OPTIMAL`, `COPY_DST_OPTIMAL`,
    `RESOLVE_SRC_OPTIMAL`, `RESOLVE_DST_OPTIMAL`, two VRS layouts
    (`servers/rendering/rendering_device_driver.h:220-234`). `TextureBarrier`
    holds `prev_layout`, `next_layout` and a subresource range (381-388);
    `command_copy_texture` and the buffer/texture copies take the layouts as
    arguments (581-587).
  - The graph keeps one `ResourceUsage` per resource tracker
    (`servers/rendering/rendering_device_graph.h:199`; slices get their own
    trackers, `rendering_device.cpp:7569-7573`), maps usage to layout in
    `_usage_to_image_layout` (`rendering_device_graph.cpp:138-173`) and adds
    a texture barrier when the usage changes (602-611, 786-787). Copies are
    replayed with `COPY_SRC_OPTIMAL`/`COPY_DST_OPTIMAL` (1239-1254).
  - Public barriers are gone: `barrier()` and `full_barrier()` print
    "Deprecated. Barriers are automatically inserted by RenderingDevice."
    (`rendering_device.cpp:7230-7236`)
  - Vulkan maps `GENERAL` and `STORAGE_OPTIMAL` to `VK_IMAGE_LAYOUT_GENERAL`,
    copy and resolve layouts to `TRANSFER_SRC/DST_OPTIMAL`
    (`drivers/vulkan/rendering_device_driver_vulkan.cpp:336-350`). No use of
    `VK_KHR_unified_image_layouts` (grepped `unified_image_layout`).
  - D3D12 maps `GENERAL` → `COMMON`, `STORAGE_OPTIMAL` →
    `UNORDERED_ACCESS`, copy → `COPY_SOURCE`/`COPY_DEST`, and every layout
    but `UNDEFINED` to `COMMON` on copy queues ("Copy queues can only use
    COMMON or UNDEFINED layouts")
    (`drivers/d3d12/rendering_device_driver_d3d12.cpp:2373-2417`); a barrier
    from `UNDEFINED` sets `D3D12_TEXTURE_BARRIER_FLAG_DISCARD` (2496).
  - Metal reports `API_TRAIT_TEXTURES_REQUIRE_LAYOUT_TRANSITIONS` false
    (`drivers/metal/rendering_device_driver_metal.cpp:2774-2787`) and ignores
    the layout arguments of copies (1585-1587).
- SETTLED, NVRHI: a public state set, tracked by the engine from a
  caller-given start. `ResourceStates` flags (`Common`, `CopyDest`,
  `UnorderedAccess`, ...) (`include/nvrhi/nvrhi.h:349-379`);
  `TextureDesc::initialState` and `keepInitialState` (437-442).
  "By default, automatic barriers are enabled ... When automatic barriers are
  disabled, it is application's responsibility to set correct states"
  (3567-3570). `beginTrackingTextureState` "must be called ... before the
  first use of any textures that do not have the keepInitialState flag set"
  (3596); `setTextureState`, `setPermanentTextureState`, `commitBarriers`
  (3611-3644). The tracker errors on an unknown prior state
  (`src/common/state-tracking.cpp:178-180`). Vulkan maps `Common` to
  `eUndefined` and only `UnorderedAccess` to `eGeneral`
  (`src/vulkan/vulkan-constants.cpp:214-217`, 242-245); D3D12 uses enhanced
  layouts when `EnhancedBarriersSupported`, else legacy states
  (`src/d3d12/d3d12-constants.cpp:243-387`, `src/d3d12/d3d12-device.cpp:261`).
- SETTLED, Diligent: a public state set; the engine tracks per texture and
  each command takes a transition mode. `RESOURCE_STATE`
  (`Graphics/GraphicsEngine/interface/GraphicsTypes.h:4818-4922`);
  `RESOURCE_STATE_TRANSITION_MODE` `NONE`, `TRANSITION` ("Resources in
  unknown state are ignored"), `VERIFY`
  (`Graphics/GraphicsEngine/interface/DeviceContext.h:245-273`);
  `ITexture::SetState` "does not perform state transition, but resets the
  internal texture state" (`Graphics/GraphicsEngine/interface/Texture.h:519`).
  Vulkan maps `RESOURCE_STATE_COMMON` to `VK_IMAGE_LAYOUT_GENERAL`
  (`Graphics/GraphicsEngineVulkan/src/VulkanTypeConversions.cpp:1470`); D3D12
  uses legacy states only (`Graphics/GraphicsEngineD3D12/src/D3D12TypeConversions.cpp:366-399`;
  no `D3D12_BARRIER_LAYOUT` in `Graphics/`).
- SETTLED, The Forge: the caller tracks. `TextureBarrier` carries
  `mCurrentState` and `mNewState`
  (`Common_3/Graphics/Interfaces/IGraphics.h:651-666`), passed to
  `cmdResourceBarrier`; texture uploads take the caller's
  `TextureUpdateDesc::mCurrentState`
  (`Common_3/Resources/ResourceLoader/ResourceLoader.cpp:1156`). Vulkan maps
  `RESOURCE_STATE_COMMON` to `VK_IMAGE_LAYOUT_GENERAL`
  (`Common_3/Graphics/Vulkan/Vulkan.c:2317-2318`); D3D12 uses legacy states
  only (`Common_3/Graphics/Direct3D12/Direct3D12.c:1729`).
- SETTLED, O3DE: no state in the public API; callers declare per-scope
  attachment usage and access (`ScopeAttachmentUsage`,
  `ScopeAttachmentAccess`, `Gems/Atom/RHI/Code/Include/Atom/RHI.Reflect/AttachmentEnums.h:15-83`;
  `UseAttachment` and friends, `Gems/Atom/RHI/Code/Include/Atom/RHI/FrameGraphInterface.h:62-208`),
  and the frame graph compiler tracks per image (`Image::SetLayout`,
  `Gems/Atom/RHI/Vulkan/Code/Source/RHI/Image.h:164-165`). Vulkan falls back
  to `GENERAL` (`default: return VK_IMAGE_LAYOUT_GENERAL;`,
  `Gems/Atom/RHI/Vulkan/Code/Source/RHI/Conversion.cpp:481-482`); DX12 uses
  legacy states, `COMMON` on the copy queue
  (`Gems/Atom/RHI/DX12/Code/Source/RHI/FrameGraphCompiler.cpp:219-294`).
- SETTLED, Granite (Vulkan only): a per-image policy, `ImageLayout { Optimal,
  General }` (`vulkan/image.hpp:242-246`); `get_layout(optimal)` returns
  `GENERAL` for `General` images (451-454); `initial_layout` defaults to
  `VK_IMAGE_LAYOUT_GENERAL` (262). Command-buffer barriers take explicit old
  and new layouts (`vulkan/command_buffer.hpp:405`); the render graph tracks
  layouts and keeps storage images in `General`
  (`renderer/render_graph.cpp:2669-2670`).
- SETTLED (from the items above): Godot, NVRHI, Diligent and O3DE track
  layouts in the engine; The Forge and Granite's command buffer leave it to
  the caller. Diligent, The Forge, O3DE and Granite use
  `VK_IMAGE_LAYOUT_GENERAL` for a common or default state; NVRHI uses it
  only for storage, Godot for storage and its own `GENERAL` usage
  (`servers/rendering/rendering_device_graph.cpp:164-165`).

### Texture creation

**a. Fields.**

- SETTLED, Vulkan `VkImageCreateInfo`: `flags`, `imageType` (1D, 2D, 3D),
  `format`, `extent`, `mipLevels`, `arrayLayers`, `samples`, `tiling`,
  `usage`, `sharingMode`, `initialLayout`. (vk.xml 357)
- SETTLED, D3D12 `D3D12_RESOURCE_DESC`: `Dimension` (buffer, 1D, 2D, 3D),
  `Alignment`, `Width`, `Height`, `DepthOrArraySize` ("the depth of the
  resource, if it is 3D, or the array size if it is an array of 1D or 2D
  resources"), `MipLevels`, `Format`, `SampleDesc`, `Layout`, `Flags`.
  (Microsoft Learn, `D3D12_RESOURCE_DESC`)
- SETTLED, Metal `MTLTextureDescriptor`: `textureType`, `pixelFormat`,
  `width`, `height`, `depth`, `mipmapLevelCount`, `sampleCount`,
  `arrayLength`, `usage`, `storageMode`, `allowGPUOptimizedContents`,
  `compressionType`. (Apple, `MTLTextureDescriptor`) For cube textures "the
  property values describe one slice"; `arrayLength` of a cube array counts
  cubes. (Apple, `MTLTextureType`)

**b. Cube textures.**

- SETTLED, Vulkan: a flag on a 2D image: "If flags contains
  VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, imageType must be VK_IMAGE_TYPE_2D",
  width and height equal, "arrayLayers must be greater than or equal to 6".
  (VUID-VkImageCreateInfo-flags-00949, -08865, -08866) Cube-ness is chosen by
  the view type.
- SETTLED, D3D12: no cube dimension; `D3D12_RESOURCE_DIMENSION` has only
  `UNKNOWN`, `BUFFER`, `TEXTURE1D`, `TEXTURE2D`, `TEXTURE3D` (DirectX-Headers
  `d3d12.h:3108-3117`), and cube exists as a view dimension
  (`D3D12_SRV_DIMENSION_TEXTURECUBE`, `d3d12.h:3577`). (Godot maps cube to
  `D3D12_RESOURCE_DIMENSION_TEXTURE2D` with `DepthOrArraySize = depth *
  array_layers`, `drivers/d3d12/rendering_device_driver_d3d12.cpp:546-554`,
  1410)
- SETTLED, Metal: separate types `MTLTextureTypeCube` = 5 and
  `MTLTextureTypeCubeArray` = 6; multisampled 2D is also its own type
  (`TextureType2DMultisample` = 4, `TextureType2DMultisampleArray` = 8).
  (metal-cpp `MTLTexture.hpp:46-55`)

**c. Usage flags.**

- SETTLED, Vulkan: `VkImageUsageFlagBits` include `TRANSFER_SRC`,
  `TRANSFER_DST`, `SAMPLED`, `STORAGE`, `COLOR_ATTACHMENT`,
  `DEPTH_STENCIL_ATTACHMENT`, `TRANSIENT_ATTACHMENT`, `INPUT_ATTACHMENT`
  (vk.xml 357). Copies need them: "dstImage must have been created with the
  VK_IMAGE_USAGE_TRANSFER_DST_BIT usage flag set" (VUID-vkCmdCopyBufferToImage-dstImage-00177),
  "srcImage must have been created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT"
  (VUID-vkCmdCopyImageToBuffer-srcImage-00186), and the format must have
  `VK_FORMAT_FEATURE_TRANSFER_DST_BIT`/`_SRC_BIT` (-dstImage-01997,
  -srcImage-01998).
- SETTLED, D3D12: `D3D12_RESOURCE_FLAGS` include `ALLOW_RENDER_TARGET`,
  `ALLOW_DEPTH_STENCIL`, `ALLOW_UNORDERED_ACCESS`, `DENY_SHADER_RESOURCE`,
  `ALLOW_SIMULTANEOUS_ACCESS`; there is no copy flag. `ALLOW_RENDER_TARGET`
  "Can't be used with ... D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL";
  `ALLOW_UNORDERED_ACCESS` "Can't be used with MSAA textures";
  `DENY_SHADER_RESOURCE` "Must be used with ...
  D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL"; `ALLOW_DEPTH_STENCIL` "Precludes
  GPU copying of a subregion. CopyTextureRegion must copy a whole subresource
  to or from resources with this flag." (Microsoft Learn,
  `D3D12_RESOURCE_FLAGS`) "A SampleDesc.Count greater than 1 ... [is] only
  supported for TEXTURE2D and when either
  D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET or
  D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL are set." (`D3D12_RESOURCE_DESC`)
- SETTLED, Metal: `MTLTextureUsage` is `Unknown` = 0, `ShaderRead` = 1,
  `ShaderWrite` = 2, `RenderTarget` = 4, `PixelFormatView` = 16,
  `ShaderAtomic` = 32 (metal-cpp `MTLTexture.hpp:73-78`). No usage is
  stated as required for copies; the stated requirement is
  `isFramebufferOnly` false. (Apple, MTL4 copy method pages)
- SETTLED, Godot neutral usage (`TEXTURE_USAGE_*`) mapping:
  - Vulkan: `SAMPLING` → `SAMPLED`, `STORAGE` → `STORAGE`, color/depth
    attachment → attachment bits, `CAN_UPDATE` and `CAN_COPY_TO` →
    `TRANSFER_DST`, `CAN_COPY_FROM` → `TRANSFER_SRC`
    (`drivers/vulkan/rendering_device_driver_vulkan.cpp:2357-2386`).
  - D3D12: color attachment → `ALLOW_RENDER_TARGET`; otherwise
    `CAN_COPY_TO` on a UAV-capable format → `ALLOW_UNORDERED_ACCESS` "For
    clearing via UAV"; depth → `ALLOW_DEPTH_STENCIL`; storage →
    `ALLOW_UNORDERED_ACCESS`; storage forces one sample ("No MSAA in D3D12
    if storage") (`drivers/d3d12/rendering_device_driver_d3d12.cpp:1465-1493`).
  - Metal: `SAMPLING` → `ShaderRead`, `STORAGE` → `ShaderWrite`,
    attachments → `RenderTarget`, `CAN_COPY_FROM` "Covered by blits",
    shareable formats → `PixelFormatView`
    (`drivers/metal/rendering_device_driver_metal.cpp:358-400`).
- SETTLED, NVRHI: `TextureDesc` has `dimension` (cube and cube array and
  multisampled are dimension values, `include/nvrhi/nvrhi.h:328-340`) and
  bools `isShaderResource`, `isRenderTarget`, `isUAV`, `isTypeless`
  (407-471). Vulkan always adds `eTransferSrc | eTransferDst`
  (`src/vulkan/vulkan-texture.cpp:106-107`) and sets `eCubeCompatible` for
  cube dimensions (208-210); D3D12 maps cube and multisampled to `TEXTURE2D`
  (`src/d3d12/d3d12-texture.cpp:253-259`), `!isShaderResource` to
  `DENY_SHADER_RESOURCE` (272).
- SETTLED, other engines: cube is a dimension value in Diligent
  (`RESOURCE_DIM_TEX_CUBE`, `Graphics/GraphicsEngine/interface/GraphicsTypes.h:311-312`),
  a descriptor bit in The Forge (`DESCRIPTOR_TYPE_TEXTURE_CUBE`,
  `Common_3/Graphics/Vulkan/Vulkan.c:3235-3236`), a flag in O3DE
  (`m_isCubemap`, `Gems/Atom/RHI/Code/Include/Atom/RHI.Reflect/ImageDescriptor.h:105`)
  and the raw Vulkan flag in Granite. Diligent always adds
  `TRANSFER_SRC|DST` on Vulkan
  (`Graphics/GraphicsEngineVulkan/src/VulkanTypeConversions.cpp:2116`); O3DE
  always adds `TRANSFER_SRC` ("add transfer src usage for all images",
  `Gems/Atom/RHI/Vulkan/Code/Source/RHI/Device.cpp:1746-1811`).

**d. Memory: can a texture be CPU-visible?**

- SETTLED, D3D12: "textures (unlike buffers) can't be heap type UPLOAD or
  READBACK." (Microsoft Learn, `D3D12_HEAP_TYPE`) CPU access to a texture is
  through custom heaps and `WriteToSubresource`/`ReadFromSubresource`, which
  "enable near zero-copy optimizations for UMA adapters, but can
  prohibitively impair the efficiency of discrete/ NUMA adapters"; textures
  "must be in the D3D12_RESOURCE_STATE_COMMON state" for them. (Microsoft
  Learn, `ID3D12Resource::WriteToSubresource`) `D3D12_TEXTURE_LAYOUT_ROW_MAJOR`
  textures are limited to 2D, one mip, one slice, no MSAA, no depth, and "must
  be created on a heap with D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER". (Microsoft
  Learn, `D3D12_TEXTURE_LAYOUT`)
- SETTLED, Vulkan: a linear image always has host-visible coherent memory
  available: for "a linear image that was not created with the
  VK_IMAGE_CREATE_PROTECTED_BIT bit set ... the memoryTypeBits member always
  contains at least one bit set corresponding to a VkMemoryType with ...
  HOST_VISIBLE_BIT ... and ... HOST_COHERENT_BIT". (Vulkan spec, "Resource
  Memory Association") Linear images "may not be supported unless" they are
  2D, not depth/stencil, one mip, one layer, one sample, and usage only
  `TRANSFER_SRC`/`TRANSFER_DST`. (Vulkan spec, "Images", `VkImageCreateInfo`:
  "Creation of images with tiling VK_IMAGE_TILING_LINEAR may not be
  supported unless other parameters meet all of the constraints") Host access "is only well-defined for linear
  images" in `PREINITIALIZED` or `GENERAL`. (Vulkan spec, "Image Layouts")
- SETTLED, Metal: textures may be shared (CPU-visible) on Apple GPUs:
  shared is "the default for buffer and texture storage". (Apple, "Choosing a
  resource storage mode for Apple GPUs") `getBytes`: "Don't use this method
  for textures where storageMode is private." (Apple, `MTLTexture/getBytes`)
  A texture made from a buffer (`makeTexture(descriptor:offset:bytesPerRow:)`)
  must be 2D or texture buffer, depth 1, one mip, one slice, one sample, with
  `offset` and `bytesPerRow` aligned to `minimumLinearTextureAlignment(for:)`.
  (Apple, `MTLBuffer/makeTexture`)
- SETTLED (from the items above): the only texture memory all three APIs
  share without restriction is GPU-only memory. Each API has a CPU path for
  some textures (Metal `replace`/`getBytes`, Vulkan linear images, D3D12
  `WriteToSubresource` on custom heaps), but a buffer copy is the only path
  all three support for every texture.
- SETTLED, engines: Godot refuses CPU-readable textures on D3D12 ("CPU
  readable textures are unsupported on D3D12.",
  `drivers/d3d12/rendering_device_driver_d3d12.cpp:1479`), makes them
  linear-tiled and host-visible on Vulkan
  (`drivers/vulkan/rendering_device_driver_vulkan.cpp:2354`, 2397), and
  shared (buffer-backed linear when 2D, one mip, one layer, one sample, not
  depth or compressed) on Metal
  (`drivers/metal/rendering_device_driver_metal.cpp:241-249`, 344-349, 419).
- SETTLED, NVRHI and Diligent "staging textures" are buffers. NVRHI sizes a
  D3D12 buffer with `GetCopyableFootprints` and 512-aligned subresources
  (`src/d3d12/d3d12-texture.cpp:188-208`, 577-602) and a tightly packed
  Vulkan buffer (`src/vulkan/vulkan-staging-texture.cpp:47-98`, 111-136).
  Diligent uses an UPLOAD or READBACK buffer from `GetCopyableFootprints` on
  D3D12 (`Graphics/GraphicsEngineD3D12/src/TextureD3D12Impl.cpp:338-395`) and
  a tight `VkBuffer` on Vulkan (`Graphics/GraphicsEngineVulkan/src/TextureVkImpl.cpp:584-651`).
  The Forge and O3DE have no CPU-visible textures (DEFAULT heap, optimal
  tiling only: `Common_3/Graphics/Direct3D12/Direct3D12.c:3714`,
  `Common_3/Graphics/Vulkan/Vulkan.c:3229`). Granite has linear host images
  with `PREINITIALIZED`, falling back to an optimal image plus a host buffer
  when linear creation fails (`vulkan/device.cpp:256-307`, 4064-4069).

### Buffer and texture copies

**a. Vulkan `VkBufferImageCopy`.**

- SETTLED: `bufferRowLength` and `bufferImageHeight` "specify in texels a
  subregion of a larger two- or three-dimensional image in buffer memory ...
  If either of these values is zero, that aspect of the buffer memory is
  considered to be tightly packed according to the imageExtent." (Vulkan
  spec, `VkBufferImageCopy`) Each must be 0 or at least the extent
  (VUID-VkBufferImageCopy-bufferRowLength-09101, -bufferImageHeight-09102) and
  a multiple of the texel block extent
  (VUID-vkCmdCopyBufferToImage-bufferRowLength-09106, -bufferImageHeight-09107).
  So a row pitch in bytes is always a whole number of texel blocks.
- SETTLED: addressing: `rowExtent = ⌈max(bufferRowLength,
  imageExtent.width) / blockWidth⌉ × blockSize`; slices and layers follow
  tightly. (Vulkan spec, "Buffer and Image Addressing")
- SETTLED: offset: "bufferOffset must be a multiple of the texel block size"
  for formats that are not depth/stencil or multi-planar (-dstImage-07975);
  "If dstImage has a depth/stencil format, the bufferOffset member of any
  element of pRegions must be a multiple of 4" (-dstImage-07978); on a queue
  without graphics or compute and without `maintenance11`, a multiple of 4
  (-commandBuffer-07737). Same for `vkCmdCopyImageToBuffer` (-srcImage-07975,
  -07978, -commandBuffer-07746). (Vulkan-ValidationLayers 1.4.357)
- SETTLED: `optimalBufferCopyOffsetAlignment` and
  `optimalBufferCopyRowPitchAlignment` are advice: "The per texel alignment
  requirements are enforced, but applications should use the optimal
  alignment for optimal performance and power use. The value must be a power
  of two." The limits table lists both as "recommendation". (Vulkan spec,
  "Limits")
- SETTLED: one aspect per region: "The aspectMask member of
  imageSubresource must only have a single bit set"
  (VUID-VkBufferImageCopy-aspectMask-09103). Buffer data for depth or stencil
  is "laid out as separate planes rather than interleaved", using: D16 →
  D16; X8_D24 and D24_S8 depth → `X8_D24_UNORM_PACK32` (4 bytes); D32 and
  D32_S8 depth → `D32_SFLOAT`; stencil → `S8_UINT`. (Vulkan spec,
  "Depth/Stencil Aspect Copy" table) Depth and stencil copies are not allowed
  on queues without graphics unless `maintenance10`
  (-commandBuffer-07739).
- SETTLED: buffer/image copies need a single-sampled image: "dstImage must
  have a sample count equal to VK_SAMPLE_COUNT_1_BIT" (-dstImage-07973,
  -srcImage-07973).
- SETTLED: 3D images take one layer; 1D and 2D take `z = 0`, `depth = 1`
  (-07979, -07980, -07983). Copies must be outside a render pass
  (-renderpass).
- SETTLED: queues with graphics or compute report `minImageTransferGranularity`
  (1,1,1): "there are no additional restrictions on the granularity of image
  transfer operations for these queues." Other queues "are only required to
  support whole mip level transfers". (Vulkan spec, `VkQueueFamilyProperties`)

**b. D3D12 `CopyTextureRegion` with a placed footprint.**

- SETTLED: "The row pitch of every subresource must be aligned to a multiple
  of D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256)" and "the starting offset of
  every subresource must be aligned to a multiple of
  D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT (512) ... unless
  D3D12_FEATURE_DATA_D3D12_OPTIONS13::UnrestrictedBufferTextureCopyPitchSupported
  is TRUE." (Microsoft Learn, "Uploading texture data through buffers",
  "Copying") `RowPitch` "must be a multiple of
  D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256), and must be greater than or equal
  to the size of the data within a row." (Microsoft Learn,
  `D3D12_SUBRESOURCE_FOOTPRINT`) Constants: `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`
  256, `D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT` 512 (DirectX-Headers
  `d3d12.h:1413-1415`).
- SETTLED: with `UnrestrictedBufferTextureCopyPitchSupported`, "both offset
  and row-pitch MUST be aligned only to the whole unit size of the texture's
  format", e.g. 16 bytes for `R32G32B32A32_FLOAT`. It is a driver cap
  (`D3D12DDI_OPTIONS_DATA_0091`) tagged WDDM 3.1; the app reads it from
  `D3D12_FEATURE_DATA_D3D12_OPTIONS13`. (DirectX-Specs `VulkanOn12`,
  "Unrestricted Buffer Texture Copy Row Pitch and Offset"; DirectX-Headers
  `d3d12.h:2917`) Microsoft Learn gives the member no description.
- OPEN: the "whole unit size" for block-compressed formats under the
  relaxed rule. The `VulkanOn12` section gives only an uncompressed example.
- SETTLED: `GetCopyableFootprints` returns, per subresource, a placed
  footprint (offset, format, width, height, depth, row pitch), the row count,
  the unpadded row size and the total size; "examining pLayouts and getting
  the row pitch from that will give you 256 as it is aligned to
  D3D12_TEXTURE_DATA_PITCH_ALIGNMENT". "The resulting structures are GPU
  adapter-agnostic". (Microsoft Learn, `GetCopyableFootprints`) The d3dx12
  version aligns each subresource offset to 512
  (`d3dx12_resource_helpers.h:457`).
- SETTLED: depth/stencil: "for a depth-stencil buffer, the depth and stencil
  planes are separate subresources"; "If you use CopyTextureRegion with a
  depth-stencil buffer or a multisampled resource, you must copy the entire
  subresource rectangle", passing 0 for `DstX/Y/Z` and NULL for `pSrcBox`.
  (Microsoft Learn, `CopyTextureRegion`) Plane 0 is depth and plane 1
  stencil ("often plane 0 for Depth and plane 1 for Stencil"). (Microsoft
  Learn, "Subresources") Copyable footprints for `R24G8_TYPELESS` and
  `R32G8X24_TYPELESS` families use `R32_TYPELESS` for plane 0 and
  `R8_TYPELESS` for plane 1. (DirectX-Headers
  `src/d3dx12_property_format_table.cpp:2330-2341`) The d3dx12 helper
  aligns the row pitch of planar formats to 512 instead of 256 ("Formats
  with more than one plane choose a larger pitch alignment",
  `d3dx12_resource_helpers.h:482-490`), and marks `R24G8_TYPELESS` and
  `D24_UNORM_S8_UINT` planar (`src/d3dx12_property_format_table.cpp:706-707`).
- OPEN: whether the runtime `GetCopyableFootprints` also returns a
  512-aligned row pitch for depth/stencil formats, and whether
  `CopyTextureRegion` requires it. Microsoft Learn `GetCopyableFootprints`
  and `D3D12_SUBRESOURCE_FOOTPRINT` state only 256.
- SETTLED: coordinates: "if the resources are textures, all coordinates are
  in texels." (Microsoft Learn, `CopyTextureRegion`)

**c. Metal 4.**

- SETTLED: `MTL4ComputeCommandEncoder::copyFromBuffer(...toTexture...)`
  and `copyFromTexture(...toBuffer...)` exist with and without
  `MTL::BlitOption` (metal-cpp `MTL4ComputeCommandEncoder.hpp:67-68`,
  75-76). Rules on the MTL4 pages:
  - offset: "Set this value to a multiple of destinationTexture's pixel size,
    in bytes." (texture to buffer: "a multiple of sourceTexture's pixel size")
  - bytes per row: "a multiple of destinationTexture's pixel size, in bytes,
    and less than or equal to the product of destinationTexture's pixel size,
    in bytes, and the largest pixel width destinationTexture's type allows.
    If destinationTexture uses a compressed pixel format, set
    sourceBytesPerRow to the number of bytes between the starts of two row
    blocks."
  - bytes per image: "a multiple of destinationTexture's pixel size, in
    bytes, or 0 if sourceSize's depth value is 1" (texture to buffer: "Set
    this value to 0 if sourceSize's depth value is 1").
  - depth/stencil: "If destinationTexture's pixelFormat is a combined
    depth/stencil format, set options to either depthFromDepthStencil or
    stencilFromDepthStencil, but not both." PVRTC needs `rowLinearPVRTC`.
  (Apple, `MTL4ComputeCommandEncoder` copy method pages)
- OPEN: any alignment beyond the pixel size (for example 4 bytes, as the
  `MTLBlitCommandEncoder` buffer-to-buffer copy states on macOS; see "Metal
  buffer copy alignment"). The MTL4 copy pages state none.
- SETTLED: the options-taking buffer-to-texture page also says of
  `destinationTexture`: "set its isFramebufferOnly property to false and
  don't use a combined depth/stencil pixelFormat", while its `options`
  parameter describes combined depth/stencil formats. (Apple,
  `MTL4ComputeCommandEncoder/copy(sourceBuffer:...options:)`)
- SETTLED: `MTLBlitOption`: `None` = 0, `DepthFromDepthStencil` = 1,
  `StencilFromDepthStencil` = 2, `RowLinearPVRTC` = 4 (metal-cpp
  `MTLBlitCommandEncoder.hpp:43-46`).
- OPEN: the byte layout Metal uses for the depth plane of
  `Depth32Float_Stencil8` and the stencil plane in a buffer (bytes per depth
  texel, whether stencil rows are 1 byte per texel). Apple pages say only
  which option to pass. Godot subtracts one byte per texel from the row for
  depth and uses `width` bytes per row for stencil
  (`drivers/metal/metal3_objects.cpp:860-871`). Searched the MTL4 and
  `MTLBlitCommandEncoder` copy pages and `MTLBlitOption`.
- OPEN: whether a multisampled texture may be copied to or from a buffer.
  Not stated on the MTL4 or `MTLBlitCommandEncoder` copy pages.

**d. One rule for all three.**

- SETTLED (from a-c): row pitch rules:
  - Vulkan: a whole number of texel blocks, at least the region width.
  - D3D12: a multiple of 256 and at least the row size, unless
    `UnrestrictedBufferTextureCopyPitchSupported`, then a multiple of the
    format unit.
  - Metal: a multiple of the pixel size (bytes between block rows for
    compressed), at most pixel size × maximum width.
  - A pitch that is a multiple of both 256 and the texel block size satisfies
    all three. 256 alone does not: `R32G32B32_SFLOAT` has a 12-byte block
    (vk.xml `blockSize="12"`), and Vulkan cannot express a 256-byte pitch for
    it in texels.
- SETTLED (from a-c): offset rules: Vulkan, a multiple of the block size (4
  for depth/stencil); D3D12, 512 unless the OPTIONS13 cap; Metal, a multiple
  of the pixel size. A multiple of 512 and of the block size satisfies all
  three.
- SETTLED (from a-c): image height: Vulkan has `bufferImageHeight` in texels;
  D3D12's footprint `Height` with `RowPitch` gives the slice pitch; Metal has
  `bytesPerImage` in bytes, 0 when depth is 1.
- SETTLED, Godot: callers never pass a pitch. RenderingDevice packs rows to
  `API_TRAIT_TEXTURE_DATA_ROW_PITCH_STEP` (256 on D3D12, 1 elsewhere) and
  aligns offsets to `API_TRAIT_TEXTURE_TRANSFER_ALIGNMENT` (512 on D3D12;
  `max(16, optimalBufferCopyOffsetAlignment)` on Vulkan; 1 on Metal)
  (`servers/rendering/rendering_device.cpp:2164-2170`, 2255-2257,
  2391-2392; `drivers/d3d12/rendering_device_driver_d3d12.cpp:6085-6088`;
  `drivers/vulkan/rendering_device_driver_vulkan.cpp:7453-7454`;
  defaults `servers/rendering/rendering_device_driver.cpp:44-47`). The D3D12
  driver only asserts: "Row pitch must be aligned to 256 bytes. See
  API_TRAIT_TEXTURE_DATA_ROW_PITCH_STEP." (4300-4301, 4351-4352). It calls
  `GetCopyableFootprints` only for `texture_get_copyable_layout`
  (1899-1919). Vulkan converts the byte pitch to `bufferRowLength` in texels
  (`rendering_device_driver_vulkan.cpp:5134-5148`, 5280, 5303); Metal ignores
  the given pitch and recomputes a tight one
  (`drivers/metal/metal3_objects.cpp:854-855`). No use of
  `UnrestrictedBufferTextureCopyPitchSupported` (grepped `drivers`,
  `servers`).
- SETTLED, NVRHI: `writeTexture` takes the caller's row pitch
  (`include/nvrhi/nvrhi.h:3240-3251`) and repacks: on D3D12 into the 256-byte
  `GetCopyableFootprints` pitch at a 512-aligned offset
  (`src/d3d12/d3d12-texture.cpp:1441-1507`), on Vulkan into a tight pitch
  (`src/vulkan/vulkan-texture.cpp:474-546`). `mapStagingTexture` returns the
  row pitch to the caller (`nvrhi.h:3684`; D3D12 returns the footprint's
  `RowPitch`, `d3d12-texture.cpp:924`). There is no public
  buffer-to-texture copy (grepped `BufferToTexture|TextureToBuffer` in
  `include/`).
- SETTLED, Diligent: callers may give a buffer source with `SrcOffset` and
  `Stride` that must be multiples of `TextureUpdateOffsetAlignment` and
  `TextureUpdateStrideAlignment`
  (`Graphics/GraphicsEngine/interface/Texture.h:279-308`), which the D3D12
  backend sets to 512 and 256 and the Vulkan backend to the optimal copy
  limits (`Graphics/GraphicsEngineD3D12/src/EngineFactoryD3D12.cpp:1057-1058`;
  `Graphics/GraphicsEngineVulkan/src/EngineFactoryVk.cpp:187-188`). The check
  runs only in development builds (`Graphics/GraphicsEngine/src/TextureBase.cpp:401-414`);
  D3D12 passes the stride straight to the footprint
  (`DeviceContextD3D12Impl.cpp:2138`). CPU data is repacked to 256
  (2229-2234).
- SETTLED, The Forge: exposes the destination stride to callers:
  `getSubresourceUpdateDesc` returns `mDstRowStride = round_up(mSrcRowStride,
  util_get_texture_row_alignment(pRenderer))`
  (`Common_3/Resources/ResourceLoader/ResourceLoader.cpp:3728`), with row
  alignment 256 on D3D12, `optimalBufferCopyRowPitchAlignment` on Vulkan and
  1 on Metal (`Common_3/Graphics/Direct3D12/Direct3D12.c:1949-1951`;
  `Common_3/Graphics/Vulkan/Vulkan.c:2984-2986`;
  `Common_3/Graphics/Metal/MetalRenderer.mm:2088-2090`).
- SETTLED, O3DE: copy descriptors carry `m_sourceBytesPerRow` and
  `m_sourceBytesPerImage` (`Gems/Atom/RHI/Code/Include/Atom/RHI/DeviceCopyItem.h:36-64`),
  passed to D3D12 as `RowPitch` unchanged
  (`Gems/Atom/RHI/DX12/Code/Source/RHI/CommandList.cpp:229`); the DX12
  backend reports 256-aligned `m_bytesPerRow` in its subresource layouts
  (`Gems/Atom/RHI/DX12/Code/Source/RHI/Image.cpp:84-85`), and Vulkan divides
  bytes by the format size to get `bufferRowLength`
  (`Gems/Atom/RHI/Vulkan/Code/Source/RHI/CommandList.cpp:142-146`).
- SETTLED: no surveyed engine uses
  `UnrestrictedBufferTextureCopyPitchSupported` (grepped all six; it appears
  only in vendored `d3d12.h`/`d3dx12` headers in Godot and The Forge). Each either repacks rows to
  256 internally (Godot, NVRHI, Diligent for CPU data, O3DE uploads) or hands
  the backend's pitch alignment to the caller (Diligent buffer sources, The
  Forge, NVRHI staging maps, O3DE layouts).

**e. Compressed formats.**

- SETTLED, Vulkan: offsets must be multiples of the texel block extent
  (-dstImage-07274, -07275, -07276); an extent must be a block multiple
  unless it reaches the subresource edge ("if the sum of imageOffset.x and
  extent.width does not equal the width of the subresource ... extent.width
  must be a multiple of the texel block extent width", -dstImage-00207,
  -00208, -00209). Buffer addressing is per block. (Vulkan-ValidationLayers
  1.4.357; Vulkan spec, "Buffer and Image Addressing")
- SETTLED, Metal: "If destinationTexture uses a compressed pixel format, set
  sourceSize to a multiple of destinationTexture's pixelFormat block size. If
  the block extends outside the bounds of the texture, clamp sourceSize to
  the edge of the texture." (Apple, MTL4 copy method pages)
- SETTLED, D3D12: `GetCopyableFootprints` rounds width and height up to the
  format's block alignment (`D3DX12AlignAtLeast(..., WidthAlignment)`,
  `d3dx12_resource_helpers.h:440-466`). Godot builds footprints with
  `STEPIFY(w, block_w)`, `STEPIFY(h, block_h)` and copies whole small BC mips
  with a NULL box: "Some D3D12 drivers reject full-box copies on the
  smallest block-compressed mips"
  (`drivers/d3d12/rendering_device_driver_d3d12.cpp:4122-4125`).
- OPEN: a Microsoft Learn rule for `CopyTextureRegion` source boxes on
  block-compressed textures (block-aligned boxes, edge blocks). Searched
  `CopyTextureRegion`, `D3D12_BOX`, `D3D12_SUBRESOURCE_FOOTPRINT`,
  "Uploading texture data through buffers".

### Texture to texture copies

- SETTLED, Vulkan `vkCmdCopyImage`: layouts `TRANSFER_SRC/DST_OPTIMAL` or
  `GENERAL` (above). "The sample count of srcImage and dstImage must match"
  (-srcImage-00136). Formats must be "size-compatible" (-srcImage-01548):
  "Color formats with the same texel block size are considered
  size-compatible"; compressed formats "must have the same texel block extent"
  (-srcImage-09247). Depth/stencil aspects must match on both sides
  (-pRegions-12201). Offsets must be block multiples (-pRegions-07278 to
  -07283). (Vulkan-ValidationLayers 1.4.357; Vulkan spec, "Size
  Compatibility")
- SETTLED, D3D12: `CopyTextureRegion` needs "compatible DXGI_FORMATs
  (identical or from the same type group)"; depth-stencil or multisampled
  resources copy whole subresources and "should have identical sample count
  values". (Microsoft Learn, `CopyTextureRegion`) `CopyResource` copies the
  whole resource; the two "Must be the same type", "Must have identical
  dimensions ... or be a compatible Reinterpret Copy", "identical
  multi-sampled count and quality". Reinterpret copies cover e.g.
  `R32G32_UINT` ↔ `BC1` at a 1:4 width/height ratio. (Microsoft Learn,
  `CopyResource`) Copies between textures of different dimension count are
  formalised for drivers at DDI 0091. (DirectX-Specs `VulkanOn12`, "Texture
  Copy Between Inequal Dimension Textures")
- SETTLED, D3D12: a texture may be source and destination of one
  `CopyTextureRegion` in a common layout if the regions do not intersect.
  (Enhanced Barriers, "Self Resource Copy")
- SETTLED, Metal 4: the destination "has the same pixelFormat and
  sampleCount as sourceTexture"; for a region copy the destination level
  "needs to have the same size as the sourceTexture slice's mipmap at
  sourceLevel". (Apple, `MTL4ComputeCommandEncoder/copy(sourceTexture:...)`
  pages) Different formats are not stated as allowed; texture views
  ("a texture view pool") reinterpret formats. (Apple, "Understanding the
  Metal 4 core API")
- SETTLED (from the items above): a copy with the same format and the same
  sample count is valid on all three; Vulkan and D3D12 allow some
  format reinterpretation, Metal's copy does not.
- SETTLED, Godot: RenderingDevice checks copy usage bits and that both
  textures are colour or both depth (`servers/rendering/rendering_device.cpp:3021-3099`,
  3062-3063), not format or samples; the Metal driver checks sample count and
  bytes per block and copies through a view when formats differ
  (`drivers/metal/metal3_objects.cpp:747-759`); D3D12 uses
  `CopyTextureRegion` per layer, never `CopyResource`
  (`drivers/d3d12/rendering_device_driver_d3d12.cpp:4074-4136`).
- SETTLED, other engines: none checks format or sample count on
  texture-to-texture copies (NVRHI `src/validation/validation-commandlist.cpp:317-323`;
  Diligent `Graphics/GraphicsEngine/src/TextureBase.cpp:445-471`; Granite
  `vulkan/command_buffer.cpp:165`, 183-214; O3DE has no copy validation,
  grepped `ValidateCopy`). The Forge has no texture-to-texture copy (grepped
  `cmdCopy` in `IGraphics.h`, `vkCmdCopyImage(`).

### Queries

- SETTLED, texel block size and extent are facts of the format, not
  queries:
  - Vulkan: "The texel block size for each format is shown in the Compatible
    formats table" (Vulkan spec, "Formats"); vk.xml carries it as attributes
    (`VK_FORMAT_BC1_RGB_UNORM_BLOCK` `blockSize="8"`
    `blockExtent="4,4,1"`; `VK_FORMAT_D24_UNORM_S8_UINT` `blockSize="4"`).
    No command returns it (vk.xml 357 commands searched).
  - D3D12: `D3D12_FEATURE_DATA_FORMAT_INFO` returns only `PlaneCount`
    (DirectX-Headers `d3d12.h:2689-2693`). Bits per unit and block alignment
    live in the d3dx12 helper table `D3D12_PROPERTY_LAYOUT_FORMAT_TABLE`
    (`GetBitsPerUnit`, `GetWidthAlignment`,
    `d3dx12_property_format_table.h:79-82`), not in the runtime API.
  - Metal: metal-cpp has no bytes-per-pixel or block-size method
    (grepped `MTLDevice.hpp`, `MTLTexture.hpp`, `MTLPixelFormat.hpp`).
- SETTLED, copy alignment advice: Vulkan reports
  `optimalBufferCopyOffsetAlignment` and `optimalBufferCopyRowPitchAlignment`
  (`VkPhysicalDeviceLimits`, advice only). D3D12's 256/512 are constants, and
  `OPTIONS13.UnrestrictedBufferTextureCopyPitchSupported` is a query. Metal
  reports `minimumLinearTextureAlignmentForPixelFormat` and
  `minimumTextureBufferAlignmentForPixelFormat` (metal-cpp
  `MTLDevice.hpp:387-389`), which apply to textures made from buffers, not
  to copies.
- SETTLED, texture size: Vulkan `vkGetImageMemoryRequirements` (size,
  alignment, memory types); D3D12: "Applications must use
  ID3D12Device::GetResourceAllocationInfo to accurately understand their
  size" (Microsoft Learn, `D3D12_RESOURCE_DESC`); Metal
  `heapTextureSizeAndAlign` ("Returns the size and alignment, in bytes, of a
  texture if you create it from a heap") and `allocatedSize` ("The size of
  the resource, in bytes"). (Apple)
- SETTLED, copy footprint: D3D12 `GetCopyableFootprints` returns a whole
  staging layout. Vulkan `vkGetImageSubresourceLayout` returns a layout only
  for linear images ("Calling vkGetImageSubresourceLayout for a linear image
  returns a subresource layout mapping"; Vulkan spec, "Image Layouts"). Metal
  has no equivalent (metal-cpp searched).
- SETTLED: transfer support per format: Vulkan
  `VK_FORMAT_FEATURE_TRANSFER_SRC_BIT`/`_DST_BIT` from
  `vkGetPhysicalDeviceFormatProperties` (required by -dstImage-01997,
  -srcImage-01998).
- OPEN: a per-format copy capability on D3D12 or Metal. None is stated on
  the copy pages read.
- SETTLED, engines' format tables: every surveyed engine hard-codes block
  size and extent (Godot `servers/rendering/rendering_device_commons.cpp:288`,
  566, 658; NVRHI `src/common/format-info.cpp:28`; Diligent
  `Graphics/GraphicsAccessories/src/GraphicsAccessories.cpp:284`; The Forge
  `Common_3/Resources/ResourceLoader/ThirdParty/OpenSource/tinyimageformat/tinyimageformat_query.h:1163`; O3DE
  `Gems/Atom/RHI/Code/Source/RHI.Reflect/Format.cpp:14`; Granite
  `vulkan/texture/texture_format.cpp:42`, 110).

## Sources

- Godot: `servers/rendering/rendering_device.h` in
  [godotengine/godot](https://github.com/godotengine/godot)
- NVRHI: `include/nvrhi/nvrhi.h` in
  [NVIDIA-RTX/NVRHI](https://github.com/NVIDIA-RTX/NVRHI)
- The Forge: `Common_3/Graphics/Interfaces/IGraphics.h` in
  [ConfettiFX/The-Forge](https://github.com/ConfettiFX/The-Forge)
- D3D12 enhanced barriers: `d3d/D3D12EnhancedBarriers.md` in
  [microsoft/DirectX-Specs](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/D3D12EnhancedBarriers.md)
- D3D12 legacy states, promotion and decay:
  [Using resource barriers to synchronize resource states](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
- D3D12 heaps and readback:
  [D3D12_HEAP_TYPE](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type),
  [ID3D12Resource::Map](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12resource-map),
  [Readback data using heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/readback-data-using-heaps),
  [CopyBufferRegion](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copybufferregion)
- D3D12MA: `D3D12MemAlloc.h` in
  [GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator)
- Vulkan synchronization:
  [fences waiting](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization-fences-waiting),
  [host access types](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization-host-access-types),
  [host write ordering](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization-submission-host-writes),
  [semaphore signaling](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization-semaphores-signaling),
  [global memory barriers](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization-global-memory-barriers),
  [pipeline stages](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization-pipeline-stages),
  [domain operations](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#memory-model-vulkan-availability-visibility)
- Vulkan VUIDs: `layers/vulkan/generated/vk_validation_error_messages.cpp` in
  [KhronosGroup/Vulkan-ValidationLayers](https://github.com/KhronosGroup/Vulkan-ValidationLayers)
  1.4.357; `vk.xml` in
  [KhronosGroup/Vulkan-Docs](https://github.com/KhronosGroup/Vulkan-Docs)
- Metal 4:
  [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api),
  [Resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization),
  [Synchronizing stages within a pass](https://developer.apple.com/documentation/metal/synchronizing-stages-within-a-pass),
  [Synchronizing passes with consumer barriers](https://developer.apple.com/documentation/metal/synchronizing-passes-with-consumer-barriers),
  [Synchronizing passes with producer barriers](https://developer.apple.com/documentation/metal/synchronizing-passes-with-producer-barriers),
  [MTL4CommandEncoder](https://developer.apple.com/documentation/metal/mtl4commandencoder),
  [MTL4VisibilityOptions](https://developer.apple.com/documentation/metal/mtl4visibilityoptions),
  [MTLStages](https://developer.apple.com/documentation/metal/mtlstages),
  [MTL4ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtl4computecommandencoder),
  [MTL4CommandQueue addResidencySet](https://developer.apple.com/documentation/metal/mtl4commandqueue/addresidencyset(_:)),
  [MTL4CommandBuffer useResidencySet](https://developer.apple.com/documentation/metal/mtl4commandbuffer/useresidencyset(_:)),
  [MTLResidencySet](https://developer.apple.com/documentation/metal/mtlresidencyset),
  [Simplifying GPU resource management with residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets),
  [MTL4CommandQueue signalEvent](https://developer.apple.com/documentation/metal/mtl4commandqueue/signalevent(_:value:))
- Metal storage and copies:
  [MTLStorageMode.shared](https://developer.apple.com/documentation/metal/mtlstoragemode/shared),
  [MTLStorageMode.managed](https://developer.apple.com/documentation/metal/mtlstoragemode/managed),
  [Synchronizing a managed resource in macOS](https://developer.apple.com/documentation/metal/synchronizing-a-managed-resource-in-macos),
  [MTLSharedEvent](https://developer.apple.com/documentation/metal/mtlsharedevent),
  [MTLBlitCommandEncoder copy](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceoffset:to:destinationoffset:size:))
- WWDC25: [Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/),
  [Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)
- metal-cpp: [developer.apple.com/metal/cpp](https://developer.apple.com/metal/cpp/)
- Vulkan textures and copies:
  [image layouts](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#resources-image-layouts),
  [buffer and image addressing](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#copies-buffers-images-addressing),
  [depth/stencil aspect copy](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#copies-buffers-images-depth-stencil),
  [format compatibility classes](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#formats-compatibility-classes),
  [limits](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#limits),
  [resource memory association](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#resources-association),
  [VK_KHR_unified_image_layouts proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_KHR_unified_image_layouts.adoc)
- D3D12 textures and copies:
  [Uploading texture data through buffers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/upload-and-readback-of-texture-data),
  [CopyTextureRegion](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copytextureregion),
  [CopyResource](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copyresource),
  [D3D12_SUBRESOURCE_FOOTPRINT](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_subresource_footprint),
  [GetCopyableFootprints](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getcopyablefootprints),
  [D3D12_FEATURE_DATA_D3D12_OPTIONS13](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options13),
  [D3D12_RESOURCE_DESC](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_resource_desc),
  [D3D12_RESOURCE_FLAGS](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_resource_flags),
  [D3D12_TEXTURE_LAYOUT](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_texture_layout),
  [D3D12_BARRIER_LAYOUT](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_barrier_layout),
  [Subresources](https://learn.microsoft.com/en-us/windows/win32/direct3d12/subresources),
  [WriteToSubresource](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12resource-writetosubresource),
  [Vulkan on 12 (relaxed copy pitch and offset)](https://microsoft.github.io/DirectX-Specs/d3d/VulkanOn12.html),
  `src/d3dx12_property_format_table.cpp` in
  [microsoft/DirectX-Headers](https://github.com/microsoft/DirectX-Headers)
- Metal textures and copies:
  [MTL4ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtl4computecommandencoder) (copy and `optimizeContents(forGPUAccess:)` pages),
  [MTLBlitCommandEncoder](https://developer.apple.com/documentation/metal/mtlblitcommandencoder),
  [MTLBlitOption](https://developer.apple.com/documentation/metal/mtlblitoption),
  [MTLTextureDescriptor](https://developer.apple.com/documentation/metal/mtltexturedescriptor),
  [MTLTextureType](https://developer.apple.com/documentation/metal/mtltexturetype),
  [MTLTextureUsage](https://developer.apple.com/documentation/metal/mtltextureusage),
  [Optimizing texture data](https://developer.apple.com/documentation/metal/optimizing-texture-data),
  [Choosing a resource storage mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus),
  [MTLBuffer makeTexture](https://developer.apple.com/documentation/metal/mtlbuffer/maketexture(descriptor:offset:bytesperrow:)),
  [MTLTexture getBytes](https://developer.apple.com/documentation/metal/mtltexture/getbytes(_:bytesperrow:from:mipmaplevel:)),
  [heapTextureSizeAndAlign](https://developer.apple.com/documentation/metal/mtldevice/heaptexturesizeandalign(descriptor:)),
  [allocatedSize](https://developer.apple.com/documentation/metal/mtlresource/allocatedsize)
- Engines: [NVIDIA-RTX/NVRHI](https://github.com/NVIDIA-RTX/NVRHI),
  [DiligentGraphics/DiligentCore](https://github.com/DiligentGraphics/DiligentCore),
  [o3de/o3de](https://github.com/o3de/o3de),
  [Themaister/Granite](https://github.com/Themaister/Granite)
