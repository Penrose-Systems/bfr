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
