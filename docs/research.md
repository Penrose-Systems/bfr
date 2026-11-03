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

## Shaders and pipelines

Facts behind shader modules, the binding model, compute and graphics
pipelines, reflection, pipeline caches and pipeline lifetime: what roadmap
step 4 (draws and dispatches) needs from step 3. Findings only, except
"Options for the shader input". Each item is SETTLED (primary source named) or OPEN (no
primary source found; what was searched is named). Read on 2026-11-02:
Vulkan-Docs `chapters/shaders.adoc`, `pipelines.adoc`, `drawing.adoc`,
`commonvalidity/draw_common.adoc`, `commonvalidity/draw_vertex_binding.adoc`,
`fundamentals.adoc`, `features.adoc`, `limits.adoc`, `devsandqueues.adoc`
and `proposals/VK_EXT_shader_module_identifier.adoc`,
`VK_EXT_descriptor_heap.adoc` (main b94f8bd, 2026-10-30, vk.xml header
version 367); `vulkan_core.h` 1.4.357 for struct line numbers; DirectX-Specs
`d3d/HLSL_SM_6_6_DynamicResources.md`, `ShaderCache.md`, `MlirPrograms.md`,
`VulkanOn12.md` (master d5f41d5, 2026-10-30); hlsl-specs
`proposals/infra/INF-0004-validator-hashing.md` (main); DirectX-Headers
`include/directx/d3d12.h` (main faf928d, 2026-10-28); Microsoft Learn pages
from the MicrosoftDocs `win32` and `sdk-api` sources; the DXC v1.9.2609
Linux release package (`LICENSE-MS.txt`, `LICENSE-LLVM.txt`,
`ReleaseNotes.md`, `include/dxc/dxcapi.h`) and DXC main `LICENSE.TXT`,
`README.md`; SPIRV-Cross main 57e4efc (2026-10-28); Slang master 48802eb
(`LICENSE`, `docs/user-guide/08-compiling.md`, `09-targets.md`,
`a2-02-metal-target-specific.md`, `docs/command-line-slangc-reference.md`,
`docs/generated/design/target-pipelines/index.md`); Mesa main
`docs/license.rst`, `src/microsoft/spirv_to_dxil/spirv_to_dxil.h`,
`src/microsoft/compiler/nir_to_dxil.c`; Apple documentation as served on
2026-11-02 (pages listed under Sources) and the Metal shader converter
product page; metal-cpp 370.63.1 as vendored in
`build/metal-syntax/metal-cpp-stock`. Engines: Godot e7b12e7, NVRHI
8e8c36e, Diligent bb821b7, The Forge a00d6d3, O3DE 94790ad, Granite 7c59ad8,
Hazel c2050e08, SynapseEngine c1cc3e39 (AGPL, read only), Doriax 426c1b5.
Engine paths are relative to each engine's root (SynapseEngine's to
`SynapseEngine/SynapseEngine/`).

### Shader input

**a. Vulkan.**

- SETTLED: a shader module holds SPIR-V. "If the VK_NV_glsl_shader
  extension is not enabled, pCode must be a pointer to SPIR-V code"
  (VUID-VkShaderModuleCreateInfo-pCode-07912); "codeSize must be greater
  than 0" (-codeSize-01085). The module is named per stage with an entry
  point string `pName` and optional `pSpecializationInfo`
  (`VkPipelineShaderStageCreateInfo`, Vulkan spec "Pipelines").
- SETTLED: on Vulkan 1.4, `VK_KHR_maintenance5` or
  `VK_EXT_graphics_pipeline_library` (`pipelines.adoc:1110`), `module` may be
  `VK_NULL_HANDLE`: "If module is VK_NULL_HANDLE, the shader code is defined
  by the chained VkShaderModuleCreateInfo if present." On Vulkan 1.3
  without that extension, "The shader code used by the pipeline is defined
  by module." (Vulkan spec, `VkPipelineShaderStageCreateInfo`;
  `VK_KHR_maintenance5` `promotedto="VK_VERSION_1_4"`, vk.xml)
- SETTLED: "A shader module can be destroyed while pipelines created using
  its shaders are still in use." (Vulkan spec, `vkDestroyShaderModule`)
- SETTLED: `VK_EXT_shader_module_identifier` (ratified, depends on
  `VK_EXT_pipeline_creation_cache_control` or Vulkan 1.3, vk.xml) lets a
  stage name a module by identifier instead of SPIR-V; the lookup can
  fail: "If a pipeline is not found, pipeline compilation is not possible
  and the implementation must fail as specified by
  VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT", and
  "implementations may fail pipeline compilation with
  VK_PIPELINE_COMPILE_REQUIRED for any reason." The identifier algorithm is
  named by `shaderModuleIdentifierAlgorithmUUID`. (Vulkan spec "Pipelines"
  and "Limits") Its proposal states the purpose: "an implementation already
  has a pipeline blob in cache"; "this solution is intended to aid internal
  implementation caching" (`proposals/VK_EXT_shader_module_identifier.adoc`).

**b. D3D12.**

- SETTLED: a PSO takes each stage as `D3D12_SHADER_BYTECODE` (`VS`, `PS`,
  `DS`, `HS`, `GS` in `D3D12_GRAPHICS_PIPELINE_STATE_DESC`, `CS` in
  `D3D12_COMPUTE_PIPELINE_STATE_DESC`; `d3d12.h:2579-2611`). There is no
  separate shader object.
- SETTLED: the runtime accepts only signed DXIL: "D3D12 will only accept
  signed shaders. That means that if any patching or runtime optimizations
  are performed, such as constant folding, the shader must be re-validated
  and re-signed" (DirectX-Specs `ShaderCache.md`). The signature is a hash:
  "after the DXIL validator successfully validates a binary it computes a
  hash using a modified MD5 hash algorithm, which then gets encoded into the
  shader binary" (INF-0004); "Upon successful validation the validator will
  compute and store a hash in the 16-byte DX container digest field"
  (DirectX-Specs `MlirPrograms.md`).
- SETTLED, the current rules (INF-0004, "Detailed Design"): "Starting with
  AgilitySDK version 1.615", a shader whose hash is the `BYPASS` sentinel
  (`01010101010101010101010101010101`) runs "without validating the hash
  regardless of whether the machine is in developer mode as long as the
  shader targets a supported non-experimental shader model";
  `PREVIEW_BYPASS` runs only in developer mode with experimental shader
  models; "shaders with zero'd hash data will always be treated as invalid
  by the runtime and rejected", with a stated exception: "legacy zero'd
  hash shaders" are "treated as if they had the `PREVIEW_BYPASS` hash set"
  until runtimes supporting those preview shaders are no longer supported.
  The debug layer validates bypassed bytecode
  by default (`D3D12_DEBUG_DEVICE_BYTECODE_VALIDATE_WHEN_HASH_BYPASSED`) and
  "only reports validator error strings, but doesn't fail shader creation".
- SETTLED: the hash algorithm is public: "DXIL Validator Hash is open
  sourced" (DXC release notes, 1.8.2502), and "The compiler will now always
  use the internal validator instead of searching for an external
  DXIL.dll" (1.8.2505). The validator API remains:
  `CLSID_DxcValidator`, `IDxcValidator`, `DxcValidatorFlags_InPlaceEdit`
  (`dxcapi.h:1269` for `CLSID_DxcValidator`).
- SETTLED: SM 6.6 and resource binding tier 3 together guarantee heap
  indexing: "`ResourceDescriptorHeap`/`SamplerDescriptorHeap` ... must be
  supported on devices that support both `D3D12_RESOURCE_BINDING_TIER_3`
  and `D3D_SHADER_MODEL_6_6`" (DirectX-Specs
  `HLSL_SM_6_6_DynamicResources.md`, "Device Capability").

**c. Metal 4.**

- SETTLED: Metal 4 pipelines take functions through descriptors, not
  function objects: `MTL4LibraryFunctionDescriptor` names an `MTL::Library`
  and a function name (metal-cpp `MTL4LibraryFunctionDescriptor.hpp:41-54`);
  `MTL4ComputePipelineDescriptor::setComputeFunctionDescriptor`
  (`MTL4ComputePipeline.hpp:51`); `MTL4RenderPipelineDescriptor`
  `vertexFunctionDescriptor`, `fragmentFunctionDescriptor` (Apple,
  `MTL4RenderPipelineDescriptor`).
- SETTLED: a library comes from MSL source or from a compiled metallib.
  Source: `MTL4LibraryDescriptor::setSource` ("the source code of the shader
  language program to compile into a Metal library"; Apple and
  `MTL4LibraryDescriptor.hpp:55`) passed to `MTL4Compiler::newLibrary`
  (`MTL4Compiler.hpp:119`). Binary: `MTL::Device::newLibrary(dispatch_data_t)`
  "from a dispatch-data instance that contains the functions in a
  precompiled Metal library" (Apple `makeLibrary(data:)`;
  `MTLDevice.hpp:461`).
- SETTLED: `MTL4Compiler` builds libraries and pipeline states,
  synchronously or asynchronously (`newComputePipelineState`,
  `newRenderPipelineState`, each with an `NS::Error**` form and a
  completion-handler form returning `CompilerTask*`;
  `MTL4Compiler.hpp:103-134`). It is made with
  `MTL::Device::newCompiler` (`MTLDevice.hpp:419`).
- SETTLED: Slang's documentation states that MSL "source code is compiled
  directly to platform-specific binaries by Apple's compiler" and Slang's
  own `metallib` targets use the "Apple `metal` compiler" downstream (Slang
  `09-targets.md`; `docs/generated/design/target-pipelines/index.md:123`).

**d. DXC to SPIR-V and DXIL.**

- SETTLED: one HLSL front end emits both: "used to compile High-Level
  Shader Language (HLSL) programs into DirectX Intermediate Language (DXIL)
  for DirectX and the Standard Portable Intermediate Representation
  (SPIR-V) for Vulkan" (DXC `README.md`).
- SETTLED, licence: DXC's source is under the "University of
  Illinois/NCSA Open Source License" (DXC `LICENSE.TXT`), whose
  third-party section ends "THE FOLLOWING LICENSES ARE FOR BUILD AND TEST
  DEPENDENCIES ONLY", after which a GPL appears (`LICENSE.TXT:199`, 296).
  The binary release package states "LICENSE-MIT.txt d3d12shader.h;
  LICENSE-LLVM.txt all other files" (`ReleaseNotes.md`, "Licenses"); the
  same package also carries `LICENSE-MS.txt`, "MICROSOFT SOFTWARE LICENSE
  TERMS ... MICROSOFT DIRECTX SHADER COMPILER", granting use "solely for
  use on Windows".
- SETTLED: the hash is open source (1.8.2502) and the compiler always uses
  its internal validator instead of an external DXIL.dll (1.8.2505)
  (`ReleaseNotes.md:274`, `:314`). OPEN: neither note says the compiler
  signs its output; the Linux package still ships `lib/libdxil.so` and
  `bin/dxv`.
- OPEN: which licence governs Microsoft's prebuilt release binaries. The
  Linux tarball maps `LICENSE-LLVM.txt` to "all other files" yet also
  contains `LICENSE-MS.txt` ("solely for use on Windows"); read
  `ReleaseNotes.md`, `LICENSE-MS.txt`, the DXC README. Builds from the
  NCSA-licensed source are not affected by this question.
- SETTLED, platforms: releases are a Windows zip and a Linux x86_64 tarball
  (DXC GitHub release v1.9.2609 assets); "DXC still has no official
  release for macOS" (INF-0004). The Linux package ships `libdxcompiler.so`
  and `libdxil.so`.
- SETTLED, Metal: "When built from source DXC can utilize the Metal Shader
  Converter if it is available during build and configuration time. This
  allows DXC to generate Metal shader libraries directly using the `-metal`
  flag." (DXC `README.md`)

**e. SPIRV-Cross to MSL.**

- SETTLED: licence: repository `LICENSE` is Apache 2.0; the MSL backend
  files carry `SPDX-License-Identifier: Apache-2.0 OR MIT`
  (`spirv_msl.cpp:3`, `spirv_msl.hpp:3`, `spirv_cross.cpp:3`).
- SETTLED: it emits MSL source; options include `msl_version` (default
  `make_msl_version(1, 2)`), `argument_buffers` and `argument_buffers_tier`
  (`spirv_msl.hpp:306`, 350, 365). It runs inside an application at run
  time: Godot calls `CompilerMSL::compile()` when it loads a shader on
  macOS and iOS (`drivers/metal/rendering_shader_container_metal.cpp:665-686`).
  O3DE and Doriax run it offline (`spirv-cross` CLI; Doriax's
  `shadercompiler`).
- SETTLED: no Metal 4 argument-table option: `MTL4` and "argument table"
  do not occur in `spirv_msl.cpp`/`.hpp` (grepped).
- SETTLED: a metallib still needs Apple's compiler after it: Godot calls
  `/usr/bin/xcrun ... metal` only when exporting
  (`drivers/metal/rendering_shader_container_metal.cpp:149-184`, 721-725),
  O3DE runs `xcrun` "MslToAir" and "AirToMetallib"
  (`Gems/Atom/RHI/Metal/Code/Source/RHI.Builders/ShaderPlatformInterface.cpp:424`,
  433).

**f. Metal Shader Converter (DXIL to metallib).**

- SETTLED: "Metal shader converter converts shader intermediate
  representations in LLVM IR bytecode into bytecode suitable to be loaded
  into Metal. It's available as a command-line tool (metal-shaderconverter)
  and a dynamic library (libmetalirconverter)." "In its most basic form,
  Metal shader converter takes a DXIL file as input and produces a
  metallib." Requirements: "macOS 13 Ventura and Xcode 15, or later";
  "Metal shader converter for Windows requires Microsoft Windows 10 and
  Microsoft Visual Studio 2019, or later"; output "require[s] a device that
  supports Argument Buffers Tier 2, running macOS 14 Sonoma, iOS 17 or
  later". Supported shader models listed: SM6.0 to SM6.6 ("Dynamic
  resources" at 6.6). Current download: "Metal shader converter 4.0 beta".
  (Apple, Metal shader converter page)
- OPEN: the licence text. It ships in the download, which needs an Apple
  developer sign-in; the product page names no licence; no local tree
  vendors it (grepped all engines for `irconverter`, `metal shader
  converter`). Not quoted, because it was not read.
- OPEN: whether its output binds through Metal 4 argument tables or only
  Metal 3 argument buffers; its manual (PDF) is behind the same sign-in.
- SETTLED: no surveyed engine uses it (grepped `metal_irconverter`,
  `IRCompiler`, `MetalShaderConverter` in all nine).

**g. Slang.**

- SETTLED, licence: `SPDX-License-Identifier: Apache-2.0 WITH
  LLVM-exception` (Slang `LICENSE`).
- SETTLED, targets: SPIR-V directly ("EmitSpirvDirectly ... Slang's
  direct-to-SPIR-V backend", `08-compiling.md:1143`); DXIL through DXC
  ("HLSL ... (plus downstream `DXIL` ...) ... DXC (for `DXIL*`)",
  `target-pipelines/index.md:122`); MSL source and `metallib` through the
  "Apple `metal` compiler" (`:123`; targets `metallib_2_3` to
  `metallib_4_0`, `command-line-slangc-reference.md:1417-1422`). "Slang
  support for Metal is a work in progress." (`09-targets.md`)
- SETTLED, platforms: release v2026.20 ships Windows (x86_64, aarch64),
  Linux (x86_64, aarch64), macOS (x86_64, aarch64) and WebAssembly builds
  (GitHub release assets). It runs offline (`slangc`) or inside an
  application: "the Slang compilation API, which can be used to integrate
  Slang compilation into an application at runtime" (`08-compiling.md:12`).
- SETTLED: Slang's docs on DXIL: "the runtime API only accepts
  appropriately signed binaries (unless a developer mode is enabled on the
  host machine). A DXIL validator `dxil.dll` is included in SDK releases,
  and this validator can sign binaries that pass validation."
  (`09-targets.md`)
- SETTLED: SynapseEngine has an unregistered Slang compiler
  (`Engine/Shader/Compiler/Slang/SlangShaderCompiler.cpp:32` targets
  `SLANG_SPIRV`; no caller outside its directory). Granite's `slangmosh` is
  not Slang: it compiles GLSL with shaderc (`slangmosh/slangmosh.cpp:92`,
  105).

**h. Mesa spirv_to_dxil (Godot's D3D12 path).**

- SETTLED: Godot turns SPIR-V into DXIL at run time and at export through
  Mesa's NIR: `#include <nir_to_dxil.h>`, `<spirv_to_dxil.h>`
  (`drivers/d3d12/godot_nir.h:42-43`), `nir_to_dxil(...)`
  (`drivers/d3d12/rendering_shader_container_d3d12.cpp:485`), with
  `validator_version_max = NO_DXIL_VALIDATION` (:469) and a base of SM 6.2
  (`rendering_shader_container_d3d12.h:77`), raised to 6.6 only for buffer
  device address (`.cpp:474`).
- SETTLED: Godot signs the DXIL itself: `sign_bytecode` calls
  `compute_dxil_hash(w + 20, size - 20, w + 4)`
  (`rendering_shader_container_d3d12.cpp:243-245`), "Based on the patched
  public domain implementation released by Microsoft" in INF-0004
  (`drivers/d3d12/dxil_hash.cpp:31-32`). It patches specialization
  constants into DXIL at pipeline creation and re-signs
  (`rendering_device_driver_d3d12.cpp:3397`, 3408). No `dxil.dll`,
  `IDxcValidator` or `DxcCreateInstance` in Godot's drivers (grepped).
- SETTLED: Mesa is a prebuilt download, not vendored:
  `mesa_version = "25.3.1-3"` from `godotengine/godot-nir-static`
  (`misc/scripts/install_d3d12_sdk_windows.py:36`, 74). That repository's
  licence is MIT (GitHub repository metadata). Mesa: "Most of the Mesa code
  is licensed under MIT license, but individual files may have their own
  licenses" (`docs/license.rst`); `spirv_to_dxil.h` and `nir_to_dxil.c` carry
  "Copyright © Microsoft Corporation" with the MIT permission text.
- OPEN: the licence of every Mesa file linked into the godot-nir-static
  libraries; only the two files above and `docs/license.rst` were read.
- OPEN: platforms on which godot-nir-static builds. Godot's installer is a
  Windows script (`install_d3d12_sdk_windows.py`); nothing else read.

**i. Per-backend binaries, as NVRHI takes them.**

- SETTLED: NVRHI's `createShader(const ShaderDesc&, const void* binary,
  size_t binarySize)` takes one blob for the device's backend
  (`include/nvrhi/nvrhi.h:3701`). Vulkan makes a module and keeps no bytes
  (`src/vulkan/vulkan-shader.cpp:38-40`, 93); D3D12 copies the bytes
  (`src/d3d12/d3d12-shader.cpp:72-74`) into the PSO desc
  (`src/d3d12/d3d12-graphics.cpp:57`). NVRHI has no Metal backend (`ls src`:
  `common d3d11 d3d12 validation vulkan`). The guide: "applications
  typically cross-compile the shaders into SPIR-V using DXC and apply some
  offsets to the binding slots" (`doc/ProgrammingGuide.md:124`). No DXIL
  validation or signing in NVRHI (grepped `IDxcValidator`, `dxil.dll`).

**j. What each engine compiles with, and when.**

- SETTLED, Godot: GLSL to SPIR-V with glslang at run time
  (`modules/glslang/register_types.cpp:134`); Vulkan keeps SPIR-V
  (smolv-compressed, `drivers/vulkan/rendering_shader_container_vulkan.cpp:63`);
  D3D12 gets DXIL through Mesa (h above); Metal gets MSL from SPIRV-Cross
  at run time (`drivers/metal/rendering_shader_container_metal.cpp:665-686`)
  plus a metallib only when exported. Godot's Metal driver is Metal 3:
  `memnew(MTL3::RenderingDeviceDriverMetal(this))`
  (`drivers/metal/rendering_context_driver_metal.cpp:109`); no `MTL4Compiler`
  or `MTL4Archive` use outside `thirdparty` (grepped).
- SETTLED, Diligent: HLSL to SPIR-V through glslang or DXC (`-spirv`,
  `-fspv-target-env=vulkan1.2`, `Graphics/ShaderTools/src/DXCompiler.cpp:785`,
  794) at run time, or raw SPIR-V; D3D12 DXIL through DXC, validated and
  signed with `CLSID_DxcValidator` and `DxcValidatorFlags_InPlaceEdit`
  (`DXCompiler.cpp:425-428`). Metal implementation is not public
  (`Graphics/GraphicsEngineMetal/readme.md:4`).
- SETTLED, The Forge: offline FSL (Python) to GLSL then glslang for
  Vulkan, HLSL then `dxc` (`-Qstrip_reflect`, `-Qstrip_rootsignature`) for
  D3D12, MSL then `xcrun metal` for Metal (`Common_3/Tools/ForgeShadingLanguage/compilers.py:32-38`,
  160-193, 269-283, read with `git show HEAD:` since the checkout is
  sparse); Metal loads with `newLibraryWithData`
  (`Common_3/Graphics/Metal/MetalRenderer.mm:3333`). No DXIL validator call
  (grepped).
- SETTLED, O3DE: offline AZSL to HLSL (azslc), then `dxc` for DXIL (SM 6.2,
  `Gems/Atom/RHI/DX12/Code/Source/RHI.Builders/ShaderPlatformInterface.cpp:259`)
  and SPIR-V, then the `spirv-cross` CLI to MSL and `xcrun` to metallib
  (`Gems/Atom/RHI/Metal/Code/Source/RHI.Builders/ShaderPlatformInterface.cpp:233`,
  331). It patches DXIL and re-signs with code "from Renderdoc"
  (`Gems/Atom/RHI/DX12/Code/Source/RHI/ShaderUtils.cpp:95`, 123).
- SETTLED, Vulkan-only engines: Granite GLSL through shaderc, offline or at
  run time (`compiler/compiler.cpp:283`,
  `vulkan/managers/shader_manager.cpp:24`); Hazel GLSL through shaderc or HLSL
  through DXC at run time, and loads a cached SPIR-V when compilation fails
  (`Hazel/src/Hazel/Platform/Vulkan/ShaderCompiler/VulkanShaderCompiler.cpp:516`);
  SynapseEngine GLSL through shaderc to SPIR-V 1.6
  (`Engine/Shader/Compiler/Shaderc/ShadercShaderCompiler.cpp:27`). Doriax
  (sokol_gfx: Vulkan, D3D11, Metal, GL; no D3D12) compiles GLSL with glslang,
  then SPIRV-Cross to MSL 2.1 and HLSL SM 5.0 (`shadercompiler/src/spirvcross.cc:655-657`,
  `editor/shader/ShaderBuilder.cpp:302`, 308).
- SETTLED: no surveyed engine uses Metal 4 (`MTL4`) APIs outside vendored
  headers (grepped all nine).

### Resource binding

**a. Vulkan.**

- SETTLED: a pipeline is created with a `VkPipelineLayout`
  (`VkGraphicsPipelineCreateInfo::layout`, `vulkan_core.h:4401`;
  `VkComputePipelineCreateInfo::layout`, :4108), which holds set layouts
  and push-constant ranges.
- SETTLED: push constants: `maxPushConstantsSize` minimum is 128 bytes
  (Vulkan spec, "Required Limits" table).
- SETTLED: descriptor indexing is core 1.2; the `descriptorIndexing`
  feature "indicates whether the implementation supports the minimum set of
  descriptor indexing features"; "Enabling this feature ... does not imply
  the other minimum descriptor indexing features are also enabled. Those
  other descriptor indexing features must be enabled individually" (Vulkan
  spec, "Features"). Update-after-bind limits have required minimums of
  500000 (`maxUpdateAfterBindDescriptorsInAllPools`,
  `maxPerStageDescriptorUpdateAfterBindSampledImages` and others; "Required
  Limits").
- SETTLED: `VK_EXT_descriptor_buffer` is `deprecatedby="VK_EXT_descriptor_heap"`;
  `VK_EXT_descriptor_heap` is ratified and depends on Vulkan 1.4, or
  `VK_KHR_extended_flags`/`VK_KHR_maintenance5` with buffer device address
  (vk.xml 367). Its proposal: "Heaps can be accessed directly as arrays of
  data in the shader, and this is intended to be used to implement Shader
  Model 6.6 Resource Heaps in HLSL"; "This extension includes a new push
  interface for data, replacing both push constants and push descriptors
  ... (at least 256 bytes)"; it requires `VK_KHR_shader_untyped_pointers`
  (`proposals/VK_EXT_descriptor_heap.adoc`). Granite selects it when the
  device has it (`vulkan/descriptor_set.cpp:197`).

**b. D3D12.**

- SETTLED: "The maximum size of a root signature is 64 DWORDs." Descriptor
  tables cost 1 DWORD, root constants 1 DWORD each, root descriptors 2;
  static samplers cost nothing. Indirection cost: "zero for a root
  constant, 1 for a root descriptor, and 2 for a descriptor table".
  (Microsoft Learn, "Root Signature Limits")
- SETTLED: "Only one descriptor heap of each type can be set at one time,
  which means a maximum of 2 heaps (one sampler, one CBV/SRV/UAV)"
  (Learn, `SetDescriptorHeaps`); "switching the current descriptor heap may
  incur a wait for idle" (Learn, "Shader Visible Descriptor Heaps").
  Shader-visible sampler heap maximum 2048; CBV/SRV/UAV heap 1,000,000 at
  tiers 1-2 and "1,000,000+" at tier 3 (Learn, "Hardware Tiers").
- SETTLED, SM 6.6 dynamic resources: the root signature must carry
  `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` (0x400)
  and/or `SAMPLER_HEAP_DIRECTLY_INDEXED` (0x800); "SetDescriptorHeaps must
  be called, passing the corresponding heaps, before a call to
  SetGraphicsRootSignature or SetComputeRootSignature that uses either"
  flag; "SetDescriptorHeaps may not be called after
  SetGraphicsRootSignature or SetComputeRootSignature with different heap
  pointers before a Draw or Dispatch"; heap-indexed descriptors and data
  "must both be considered volatile". "Shader to Root Signature validation
  will fail if the shader creates resources from one of these heaps and
  the corresponding flag is not set." (DirectX-Specs
  `HLSL_SM_6_6_DynamicResources.md`)

**c. Metal 4.**

- SETTLED: Metal 4 encoders "don't have methods that bind individual
  buffers, textures, and heaps. Instead, you configure the resource
  bindings in an argument table and then bind that table to one or more
  pipeline stages" (Apple, "Understanding the Metal 4 core API").
  `MTL4RenderCommandEncoder.hpp` declares no `set*Bytes` (grepped `bytes`);
  bindings are `setArgumentTable(table, stages)` (:84) and
  `MTL4ComputeCommandEncoder::setArgumentTable` (:106).
- SETTLED: "Metal takes a snapshot of the resources in the argument table
  when you encode a draw, dispatch, or execute command." (Apple,
  `setArgumentTable(_:stages:)`) A table binds a GPU address
  (`setAddress(MTLGPUAddress, index)`), a texture or sampler by
  `MTLResourceID` (`setTexture`, `setSamplerState`), or any resource by
  `MTL::ResourceID` (`setResource`, `MTL4ArgumentTable.hpp:76`) (Apple
  `MTL4ArgumentTable`).
  `maxBufferBindCount`: "The maximum value of this parameter is 31."
  (Apple)
- SETTLED: bindless through Tier 2 argument buffers: "To directly encode
  the argument buffer resources on these Tier 2 devices, write the
  MTLBuffer.gpuAddress property — and for other resource types (samplers,
  textures, and acceleration structures), the gpuResourceID property —
  into the corresponding structure member" (iOS 16, macOS 13 and later);
  "In MSL, you access Tier 2 argument buffers by indexing a pointer."
  (Apple, "Improving CPU performance by using argument buffers")
- SETTLED: residency: "each MTL4CommandBuffer instance doesn't create
  strong references to resources" and "the framework considers all
  resources untracked" (Apple, "Understanding the Metal 4 core API").

**d. Engines: what is made ahead of time and what is recorded per draw.**

- SETTLED, Godot (layouts per shader, from reflection): Vulkan set layouts
  and one push-constant range per shader
  (`drivers/vulkan/rendering_device_driver_vulkan.cpp:4522`, 4542-4548),
  sets written at uniform-set creation (:4990, 5022), per draw
  `vkCmdBindDescriptorSets` (:5797) and `vkCmdPushConstants` (:5329).
  D3D12 root signature generated from reflection when the container is
  built, push constants as root constants at parameter 0, up to two tables
  per set, dynamic buffers as root CBV, SRV or UAV descriptors,
  `HEAP_DIRECTLY_INDEXED` only for buffer device address
  (`drivers/d3d12/rendering_shader_container_d3d12.cpp:536-540`, 561,
  675-690, 759-760); per draw `SetGraphicsRootDescriptorTable`,
  `SetGraphicsRoot32BitConstants` (`rendering_device_driver_d3d12.cpp:5091`,
  4424). Metal: Tier 2 argument buffers filled with `gpuResourceID` at set
  creation (`drivers/metal/rendering_device_driver_metal.cpp:1420`), bound
  per draw with `setVertexBuffer` (`drivers/metal/metal3_objects.cpp:1911`);
  push constants with `setVertexBytes` (:1047).
- SETTLED, NVRHI (caller-built layouts): `BindingLayout` makes a Vulkan set
  layout (`src/vulkan/vulkan-resource-bindings.cpp:244`) and D3D12 root
  parameters (`src/d3d12/d3d12-resource-bindings.cpp:415-598`); the
  pipeline layout or root signature is built from the pipeline's layouts
  (vulkan :1000-1125; d3d12 `buildRootSignature` :663-792, cached by hash
  :804). One push-constant block per pipeline, at most 128 bytes
  (`include/nvrhi/nvrhi.h:80`, 3320). `HEAP_DIRECTLY_INDEXED` when tier 3
  and SM 6.6 (`src/d3d12/d3d12-device.cpp:474`). Per draw it compares state
  and rebinds what changed ("State is cached by NVRHI", `nvrhi.h:3326`;
  `src/vulkan/vulkan-graphics.cpp:548`, 571).
- SETTLED, Diligent: two Vulkan sets per resource signature plus at most one
  push-constant range (`Graphics/GraphicsEngineVulkan/include/PipelineResourceSignatureVkImpl.hpp:108-111`,
  `src/PipelineLayoutVk.cpp:185`); dynamic set allocated and written per
  commit (`src/DeviceContextVkImpl.cpp:663`). D3D12 root signature 1.0 with
  root constants, dynamic descriptors copied into a GPU-visible ring per
  commit (`Graphics/GraphicsEngineD3D12/src/PipelineResourceSignatureD3D12Impl.cpp:644-656`).
  No `HEAP_DIRECTLY_INDEXED` (grepped).
- SETTLED, The Forge: four set frequencies fixed in shared headers
  (`Common_3/Graphics/FSL/defaults.h:27-30`); one D3D12 root signature for
  all graphics pipelines, compiled from an FSL string (`defaults.h:33-46`);
  Metal one argument buffer per descriptor set, bound per draw with
  `setVertexBuffer` (`Common_3/Graphics/Metal/MetalRenderer.mm:491`).
- SETTLED, O3DE: one Vulkan set per shader resource group and root
  constants as one push range (`Gems/Atom/RHI/Vulkan/Code/Source/RHI/PipelineLayout.cpp:255`,
  278); D3D12 root signature 1.0 with root constants and per-group tables
  (`Gems/Atom/RHI/DX12/Code/Source/RHI/PipelineLayout.cpp:122`, 392);
  Metal argument buffers plus `setVertexBytes` for root constants
  (`Gems/Atom/RHI/Metal/Code/Source/RHI/CommandList.cpp:235`).
- SETTLED, Vulkan-only engines: Granite picks descriptor heap, descriptor
  buffer, push descriptors or pooled sets by feature
  (`vulkan/descriptor_set.cpp:197`, 78, 256); SynapseEngine uses descriptor
  buffers and one push range of `maxPushConstantsSize` for all stages
  (`Engine/Vk/Shader/ShaderProgram.h:32`, `ShaderProgram.cpp:193`); sokol
  (Doriax) uses descriptor buffers with a fixed two-set layout and no push
  constants (`engine/libs/sokol/sokol_gfx.h:21266`, 21469).
- SETTLED: no surveyed engine binds through Metal 4 argument tables; Godot
  is the only one writing `gpuResourceID` (grepped all nine).

**e. Engines: allocation on the draw path.**

- SETTLED, allocates: Godot Vulkan `thread_local LocalVector` resized per
  bind, growing only past its high-water mark
  (`drivers/vulkan/rendering_device_driver_vulkan.cpp:5771-5773`;
  `core/templates/local_vector.h:82-89`); Godot Metal grows
  `uniform_sets` to a high-water mark (resized only when too small, as on
  Vulkan) and inserts into hash maps per bind
  (`drivers/metal/metal3_objects.cpp:921-923`, 1107-1117); NVRHI D3D12 makes
  a `std::make_shared<CommandListInstance>` per `open()` and grows
  `referencedResources` per draw (`src/d3d12/d3d12-commandlist.cpp:266`,
  `src/d3d12/d3d12-graphics.cpp:337`), and indexes an `unordered_map` with
  `operator[]` per draw (`d3d12-graphics.cpp:535`); Diligent Vulkan resizes
  a `std::vector` of dynamic offsets when a pipeline is set
  (`Graphics/GraphicsEngineVulkan/src/DeviceContextVkImpl.cpp:393`); O3DE
  Metal inserts into an `unordered_set` per draw
  (`Gems/Atom/RHI/Metal/Code/Source/RHI/CommandList.cpp:480`); Granite
  creates a descriptor pool when one runs out
  (`vulkan/descriptor_set.cpp:397`, 414) and builds pipelines on the draw
  path (Graphics pipelines, h); Hazel allocates a push-constant buffer per
  draw (`Hazel/src/Hazel/Platform/Vulkan/VulkanRenderer.cpp:407`).
- SETTLED, fixed storage: Godot D3D12 bind paths
  (`rendering_device_driver_d3d12.cpp:5057-5097`, 4415-4430); NVRHI
  `static_vector` binding arrays (`nvrhi.h:2651`); Diligent D3D12 placement
  new into stack memory (`PipelineResourceSignatureD3D12Impl.cpp:631-643`);
  The Forge fixed bound-set arrays (`Common_3/Graphics/Interfaces/IGraphics.h:1504`,
  1569); sokol bump allocation in a preallocated descriptor buffer, overflow
  an error (`sokol_gfx.h:20055-20057`).

### Compute pipelines

**a. Create-info.**

- SETTLED, Vulkan: `VkComputePipelineCreateInfo { flags, stage, layout,
  basePipelineHandle, basePipelineIndex }` (`vulkan_core.h:4103-4111`); the
  stage carries module, entry point and specialization constants.
- SETTLED, D3D12: `D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature, CS,
  NodeMask, CachedPSO, Flags }` (`d3d12.h:2604-2611`).
- SETTLED, Metal 4: `MTL4ComputePipelineDescriptor`:
  `computeFunctionDescriptor`, `maxTotalThreadsPerThreadgroup`,
  `requiredThreadsPerThreadgroup`, `threadGroupSizeIsMultipleOfThreadExecutionWidth`,
  `supportIndirectCommandBuffers`, `supportBinaryLinking`,
  `staticLinkingDescriptor` (`MTL4ComputePipeline.hpp:41-62`; Apple). No
  layout or argument-table shape is given at creation (none in the
  descriptor).

**b. Workgroup size.**

- SETTLED, Vulkan: the local workgroup size is in the shader, "set by
  assigning a value to the LocalSize or LocalSizeId execution mode or via
  an object decorated by the WorkgroupSize decoration" (Vulkan spec,
  "Compute Shaders"); "Support for LocalSizeId was added with
  VK_KHR_maintenance4 and promoted to core in Version 1.3" (Vulkan spec,
  `WorkgroupSize`). `vkCmdDispatch` takes "the number of local workgroups to
  dispatch" per dimension (Vulkan spec, "Dispatching Commands").
- SETTLED, D3D12: `Dispatch(ThreadGroupCountX, Y, Z)`, "The number of
  groups dispatched in the x direction", at most 65535 (Learn,
  `ID3D12GraphicsCommandList::Dispatch`); the group size is the shader's
  `numthreads`, compiled into the DXIL.
- SETTLED, Metal 4: the thread count per threadgroup is a dispatch
  argument of `dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size
  threadsPerThreadgroup)`, `dispatchThreadgroups(GPUAddress indirect,
  MTL::Size threadsPerThreadgroup)` and `dispatchThreads(MTL::Size
  threadsPerGrid, MTL::Size threadsPerThreadgroup)`
  (`MTL4ComputeCommandEncoder.hpp:80-83`); the indirect
  `dispatchThreads(MTL::GPUAddress indirectBuffer)` takes none from the CPU
  (`:84`).
  `requiredThreadsPerThreadgroup`: "When you set this value, you are
  responsible for ensuring that the threadsPerThreadgroup argument of any
  compute dispatch matches it. Setting this property is optional, except
  in cases where the pipeline uses CooperativeTensors. This property's
  default value is 0, which disables its effect." (Apple)
- SETTLED, engines, how Metal gets it: Godot reflects SPIR-V
  `local_size` into the container and passes it at dispatch
  (`servers/rendering/rendering_shader_container.cpp:340-342`;
  `drivers/metal/rendering_device_driver_metal.cpp:1326`;
  `drivers/metal/metal3_objects.cpp:1727`); The Forge parses `NUM_THREADS`
  offline into its binary header and uses it at dispatch
  (`Common_3/Graphics/Metal/MetalRenderer.mm:4412`); O3DE and Diligent take
  it from the caller (`Gems/Atom/RHI/Metal/Code/Source/RHI/CommandList.cpp:196`;
  Diligent `Graphics/GraphicsEngine/interface/DeviceContext.h:900-901`,
  "only used by Metal backend"). NVRHI's `dispatch` takes only group counts
  (`nvrhi.h:3391`) and it has no Metal backend.

**c. Fixed at creation, set at record.**

- SETTLED: Vulkan fixes the stage, layout and specialization constants;
  D3D12 fixes the root signature and CS; Metal 4 fixes the function and
  thread limits. At record: Vulkan binds sets and push constants, D3D12
  binds heaps, root signature and root arguments
  (`HLSL_SM_6_6_DynamicResources.md` ordering above), Metal 4 sets the
  argument table and passes threads per threadgroup to each direct dispatch
  (b).
- SETTLED: on D3D12 the root signature is set on the command list
  separately from the PSO (`SetComputeRootSignature`, ordering rule in
  Resource binding, b).
- OPEN: a primary source for the rule that the root signature set at
  record must match the PSO's `pRootSignature`; the Learn PSO desc page
  says only "A pointer to the ID3D12RootSignature object".

### Graphics pipelines

**a. What creation takes.**

- SETTLED, Vulkan: `VkGraphicsPipelineCreateInfo` holds stages, vertex
  input, input assembly, tessellation, viewport, rasterization,
  multisample, depth/stencil, colour blend and dynamic state, `layout`,
  `renderPass` and `subpass` (`vulkan_core.h:4386-4406`). With dynamic
  rendering, attachment formats come from `VkPipelineRenderingCreateInfo
  { viewMask, colorAttachmentCount, pColorAttachmentFormats,
  depthAttachmentFormat, stencilAttachmentFormat }` (:7924-7932).
- SETTLED, D3D12: `D3D12_GRAPHICS_PIPELINE_STATE_DESC { pRootSignature, VS,
  PS, DS, HS, GS, StreamOutput, BlendState, SampleMask, RasterizerState,
  DepthStencilState, InputLayout, IBStripCutValue, PrimitiveTopologyType,
  NumRenderTargets, RTVFormats[8], DSVFormat, SampleDesc, NodeMask,
  CachedPSO, Flags }` (`d3d12.h:2579-2602`).
- SETTLED, Metal 4: `MTL4RenderPipelineDescriptor`: vertex and fragment
  function descriptors, optional `vertexDescriptor`,
  `inputPrimitiveTopology` (`MTLPrimitiveTopologyClass`: unspecified,
  point, line, triangle; `MTLRenderPipeline.hpp:99-104`),
  `rasterSampleCount`, `isRasterizationEnabled`, `alphaToCoverageState`,
  `alphaToOneState`, `colorAttachments` (each: `pixelFormat`,
  `blendingState`, blend factors and operations, `writeMask`),
  `colorAttachmentMappingState`, `maxVertexAmplificationCount` (Apple;
  `MTL4RenderPipeline.hpp:53-185`). "Compared to MTLRenderPipelineDescriptor,
  this interface doesn't offer a mechanism to hint to Metal mutability of
  vertex and fragment buffers. Additionally, using this descriptor, you
  don't specify binary archives." (Apple)
- SETTLED: `MTL4RenderPipelineDescriptor` has no depth or stencil
  attachment pixel format: none in Apple's property list nor in metal-cpp
  (`depthAttachmentPixelFormat` occurs only in the Metal 3
  `MTLRenderPipeline.hpp:237`, 284).
- SETTLED: Metal depth and stencil test state is a separate object,
  `MTL::Device::newDepthStencilState` (`MTLDevice.hpp:438`), set with
  `setDepthStencilState` on the encoder (`MTL4RenderCommandEncoder.hpp:98`).
  Cull mode, front-facing winding, triangle fill mode, depth bias, depth
  clip mode, depth bounds, stencil reference, blend colour, viewports and
  scissors are encoder state (`MTL4RenderCommandEncoder.hpp:92-125`,
  `setViewport` at `:124-125`; Apple `MTL4RenderCommandEncoder` topics).
- SETTLED: Metal 4 colour formats may be left unspecialized and filled in
  later: "Configure the properties that vary across the various pipeline
  states to their unspecialized value, such as MTLPixelFormat.unspecialized
  for the pixelFormat property", then `newRenderPipelineStateBySpecialization`
  (Apple, "Using the Metal 4 compilation API"; `MTL4Compiler.hpp:132`).

**b. Topology.**

- SETTLED, Vulkan: topology is in the pipeline unless
  `VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY` (core 1.3); then, without
  `dynamicPrimitiveTopologyUnrestricted`, the set topology "must be of the
  same topology class as the pipeline
  VkPipelineInputAssemblyStateCreateInfo::topology state"
  (VUID-vkCmdDraw-dynamicPrimitiveTopologyUnrestricted-07500). Classes:
  point, line, triangle, patch (Vulkan spec, "Topology Class").
- SETTLED, D3D12: "the input-assembler primitive topology type (point,
  line, triangle, patch) is set within the PSO"; "The primitive adjacency
  and ordering (line list, line strip ...) is set from within a command
  list using IASetPrimitiveTopology" (Learn, "Managing Graphics Pipeline
  State in Direct3D 12").
- SETTLED, Metal 4: the pipeline holds a topology class
  (`inputPrimitiveTopology`); each draw names `MTL::PrimitiveType`
  (`drawIndexedPrimitives`, `MTL4RenderCommandEncoder.hpp:66-69`;
  `drawPrimitives`, `:76-79`).

**c. Dynamic state.**

- SETTLED, Vulkan 1.0 dynamic states: viewport, scissor, line width, depth
  bias, blend constants, depth bounds, stencil compare mask, write mask,
  reference. Vulkan 1.3 core adds cull mode, front face, primitive topology,
  viewport and scissor with count, vertex input binding stride, depth test,
  depth write, depth compare op, depth bounds test, stencil test, stencil
  op, rasterizer discard, depth bias enable, primitive restart enable
  (vk.xml 367, `VkDynamicState` and `VK_GRAPHICS_VERSION_1_3`).
  Polygon mode, sample count, sample mask, blend enable and equations and
  colour write masks are not core 1.3 dynamic states (same lists).
- SETTLED, D3D12: set on the command list: viewports, scissors, blend
  factor, stencil reference, primitive order (Learn, "Managing Graphics
  Pipeline State"); depth bounds (`OMSetDepthBounds`, `d3d12.h:7854`); front/back stencil
  reference (`OMSetFrontAndBackStencilRef`, `d3d12.h:39064`). Depth bias and
  strip cut are dynamic only if the PSO has
  `D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS` /
  `_DYNAMIC_INDEX_BUFFER_STRIP_CUT` (`d3d12.h:2572-2573`) and the device
  reports `DynamicDepthBiasSupported` (OPTIONS16) /
  `DynamicIndexBufferStripCutSupported` (OPTIONS15) (`d3d12.h:3261-3269`;
  DirectX-Specs `VulkanOn12.md`: "the assigned pipeline MUST have been
  created with D3D12DDI_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS").
- SETTLED, Metal 4: see a above; cull, winding, fill, depth bias, depth
  clip, depth/stencil state, stencil reference, blend colour, viewport and
  scissor are all encoder state.

**d. Attachments at draw time.**

- SETTLED, Vulkan: with `vkCmdBeginRendering`, the bound pipeline's
  `colorAttachmentCount` must equal the rendering's, and each attachment's
  format must equal the pipeline's format, or the pipeline's be
  `VK_FORMAT_UNDEFINED` where the view is null (VUID-vkCmdDraw-colorAttachmentCount-06179,
  -dynamicRenderingUnusedAttachments-08910, -08912); `viewMask` must match
  (-viewMask-06178).
- SETTLED, D3D12: "D3D12 currently requires/validates that
  DXGI_SAMPLE_DESC::Count matches the sample count of the bound render
  target(s) and depth stencil" (DirectX-Specs `VulkanOn12.md`).
- OPEN: a D3D12 primary source stating that bound RTV/DSV formats must
  equal the PSO's `RTVFormats`/`DSVFormat`; searched the Learn pages for
  the PSO desc, `OMSetRenderTargets` not fetched.

**e. Common to all three, and one-backend fields.**

- SETTLED, fixed at creation on all three: shader functions; colour
  attachment count and formats (Metal allows them unspecialized); sample
  count (`rasterizationSamples`, `SampleDesc.Count`, `rasterSampleCount`);
  per-attachment blend enable, factors, operations and write mask;
  alpha-to-coverage. (a above)
- SETTLED, topology class: fixed at creation on D3D12 (topology type) and
  on Vulkan (topology, unless `dynamicPrimitiveTopologyUnrestricted`). Metal
  may leave it open (`PrimitiveTopologyClassUnspecified = 0`,
  `MTLRenderPipeline.hpp:100`); OPEN: no source read says a Metal draw must
  match a class that was set.
- SETTLED, rasterizer discard: Vulkan `rasterizerDiscardEnable` (dynamic in
  1.3), Metal `isRasterizationEnabled` (pipeline). OPEN: a D3D12
  equivalent; not searched beyond the PSO desc fields, which have none.
- SETTLED, fixed on Vulkan and D3D12, encoder state on Metal: cull mode,
  front face, fill mode, depth bias, depth clip, depth test/write/compare,
  stencil test and ops (`D3D12_RASTERIZER_DESC` `FillMode`, `CullMode`,
  `FrontCounterClockwise`, `DepthBias`, `DepthClipEnable`, Learn; a, c
  above). On Vulkan 1.3 all but fill mode and depth clip may be made
  dynamic (c).
- SETTLED, only some backends:
  - depth/stencil attachment format: Vulkan (`depthAttachmentFormat`,
    `stencilAttachmentFormat`) and D3D12 (`DSVFormat`); Metal 4 has none;
  - binding layout at creation: Vulkan `layout`, D3D12 `pRootSignature`;
    Metal 4 none;
  - `SampleMask`: Vulkan `pSampleMask`, D3D12 `SampleMask`; Metal 4
    descriptor has none (Apple property list);
  - strip-cut / primitive restart: Vulkan `primitiveRestartEnable` (a bool),
    D3D12 `IBStripCutValue` (disabled, 0xFFFF or 0xFFFFFFFF; Learn
    `D3D12_INDEX_BUFFER_STRIP_CUT_VALUE`); Metal 4 descriptor has none;
  - `viewMask` (Vulkan) and `maxVertexAmplificationCount` (Metal);
  - colour attachment remapping (`colorAttachmentMappingState`, Metal 4 only);
  - `NodeMask` and `CachedPSO` (D3D12 only); `StreamOutput` (D3D12 only).
- SETTLED, vertex input: Vulkan `pVertexInputState`, D3D12 `InputLayout`
  and Metal `vertexDescriptor` (optional: "Configures an optional vertex
  descriptor for the vertex input", Apple) all describe fixed-function
  fetch. Godot's D3D12 root signature sets
  `ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT` only when the shader has vertex
  inputs (`drivers/d3d12/rendering_shader_container_d3d12.cpp:755`).
- OPEN: a cited rule, per API, that a pipeline with no vertex attributes
  (design.md's intent is vertex pulling) needs nothing more than an empty
  description; not searched.

**f. Engines.**

- SETTLED, attachment model: dynamic rendering in NVRHI
  (`src/vulkan/vulkan-graphics.cpp:412-419`), Diligent (with a render-pass
  fallback, `Graphics/GraphicsEngineVulkan/src/PipelineStateVkImpl.cpp:465-469`),
  The Forge (when supported, `Common_3/Graphics/Vulkan/Vulkan.c:7117`),
  SynapseEngine and sokol; render passes in Godot
  (`drivers/vulkan/rendering_device_driver_vulkan.cpp:6293`), O3DE, Granite
  and Hazel. NVRHI pipelines take a `FramebufferInfo` (colour formats,
  depth format, sample count) and "can then be used with any framebuffer
  which has the same FramebufferInfo" (`nvrhi.h:1315-1320`,
  `doc/ProgrammingGuide.md:150`).
- SETTLED, Vulkan dynamic states used: NVRHI viewport and scissor always,
  blend constants, stencil reference and shading rate when needed
  (`src/vulkan/vulkan-graphics.cpp:390-399`); Godot viewport and scissor
  always, seven more by flag (`rendering_device_driver_vulkan.cpp:6103-6136`);
  Diligent viewport, blend constants, stencil reference, scissor when
  enabled (`PipelineStateVkImpl.cpp:427-447`); The Forge viewport, scissor,
  blend constants, depth bounds, stencil reference
  and sample locations (`Vulkan.c:7357-7365`); SynapseEngine draws with
  shader objects and sets all state dynamically
  (`Engine/Vk/Rendering/RenderUtils.cpp:62-137`); its one `VkPipeline` is an
  AMDX execution-graph pipeline
  (`Engine/Render/Passes/Culling/Geometry/GeometryWorkGraphCullingPass.cpp:131`).
- SETTLED, Metal: Godot, The Forge and O3DE store cull, winding, fill,
  depth bias and the depth-stencil state object with the pipeline and set
  them on the encoder when the pipeline is bound
  (`drivers/metal/metal_objects_shared.h:1007-1043`;
  `Common_3/Graphics/Metal/MetalRenderer.mm:4041-4049`;
  `Gems/Atom/RHI/Metal/Code/Source/RHI/CommandList.cpp:702`, 793-797).
- SETTLED, D3D12 strip cut: Godot sets 0xFFFF with a TODO that 32-bit
  indices need another value (`rendering_device_driver_d3d12.cpp:5398-5399`);
  Diligent sets 0xFFFFFFFF for strip topologies (`Graphics/GraphicsEngineD3D12/src/PipelineStateD3D12Impl.cpp:699-703`);
  The Forge sets DISABLED (`Common_3/Graphics/Direct3D12/Direct3D12.c:4945-4965`);
  NVRHI and O3DE zero-initialise the desc and never set it
  (`src/d3d12/d3d12-graphics.cpp:52`;
  `Gems/Atom/RHI/DX12/Code/Source/RHI/PipelineState.cpp:51`), and
  `D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED` is 0 (`d3d12.h:2549`). All
  set `SampleMask` to all ones or the caller's.
- SETTLED, Granite creates pipelines lazily on the draw path, keyed by a
  state hash (`vulkan/command_buffer.cpp:1763-1765`), allocating on a miss
  (`vulkan/shader.cpp:1148`).

### Shader reflection

- SETTLED, Vulkan: needs the pipeline layout (set layouts, push-constant
  ranges) at pipeline creation (Resource binding, a). Nothing else; the
  workgroup size is in the SPIR-V.
- SETTLED, D3D12: needs a root signature at PSO creation and the root
  parameter indices at record; the workgroup size is in the DXIL. With a
  root signature embedded in the shader the compiler validates it ("This
  validation is run at shader compilation time if a root signature is
  attached to the entry point", `HLSL_SM_6_6_DynamicResources.md`).
- SETTLED, Metal 4: needs the threads per threadgroup at every direct
  dispatch (Compute pipelines, b) and the argument-table slot of each binding;
  pipeline reflection is opt-in (`MTL4::ShaderReflection`:
  `ShaderReflectionNone`, `BindingInfo`, `BufferTypeInfo`,
  `MTL4PipelineState.hpp:54-58`, set through `PipelineOptions`).
- OPEN: Metal shader converter's reflection JSON (fields, threadgroup
  size); its manual is behind the developer sign-in.
- SETTLED, engines: Godot extracts with SPIRV-Reflect and stores
  `compute_local_size`, `push_constant_size`, per-binding type, set and
  stage, spec constants and vertex/fragment masks
  (`servers/rendering/rendering_shader_container.h:58-93`); Vulkan uses the
  sets and push-constant size, D3D12 the bindings and push-constant size
  for the root signature, Metal the local size and push constants
  (`rendering_device_driver_metal.cpp:1326`, 1340-1342). NVRHI does no
  reflection: the caller supplies binding and input layouts (grepped
  `reflect`, `numthreads`, `LocalSize`). Diligent reflects SPIR-V with
  SPIRV-Cross and DXIL with `ID3D12ShaderReflection`
  (`Graphics/ShaderTools/src/SPIRVShaderResources.cpp:910`;
  `Graphics/GraphicsEngineD3D12/src/ShaderResourcesD3D12.cpp:64-73`). The
  Forge and O3DE reflect at build time only (FSL headers; azslc JSON).
  Granite, Hazel, SynapseEngine and Doriax reflect SPIR-V (SPIRV-Cross or
  spirv-reflect) for bindings and push-constant size; none of the four
  extracts the workgroup size (grepped `LocalSize`, `workgroup`).

### Pipeline caches

**a. Vulkan.**

- SETTLED: cache data begins with `VkPipelineCacheHeaderVersionOne
  { headerSize, headerVersion, vendorID, deviceID, pipelineCacheUUID }`,
  where `pipelineCacheUUID` is "the
  VkPhysicalDeviceProperties::pipelineCacheUUID of the implementation".
  "providing invalid pipeline cache data as input to any Vulkan API
  commands will result in the provided pipeline cache data being ignored."
  (Vulkan spec, "Pipeline Cache Header") The UUID is "used to identify a
  compatible device and driver combination to use when serializing and
  de-serializing pipeline state" (`devsandqueues.adoc:652-656`).
- SETTLED: "The use of the pipeline cache object in these commands is
  internally synchronized, and the same pipeline cache object can be used
  in multiple threads simultaneously", unless created with
  `VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT` (Vulkan spec,
  `vkCreatePipelineCache`).
- SETTLED: `VK_KHR_pipeline_binary` (ratified; vk.xml 367) adds
  `pipelineBinaryPrefersInternalCache`: "applications should not store
  pipeline binaries in their own on-disk caches" when set (Vulkan spec,
  "Limits"). Granite uses it with its own UUID check
  (`vulkan/pipeline_cache.cpp:294`).

**b. D3D12.**

- SETTLED: two mechanisms: `D3D12_SHADER_CACHE_SUPPORT_SINGLE_PSO`
  ("supports the CachedPSO member", "always supported") and `_LIBRARY`
  ("supports the ID3D12PipelineLibrary interface", a driver capability) (Learn,
  `D3D12_SHADER_CACHE_SUPPORT_FLAGS`).
- SETTLED: `CachedPSO` "contains data specific to the hardware, driver, and
  machine that it was retrieved from"; "If the driver has been upgraded
  ... you might see a D3D12_ERROR_DRIVER_VERSION_MISMATCH return code, or if
  you're running on a different GPU, the D3D12_ERROR_ADAPTER_NOT_FOUND
  return code"; "The rest of the data in the PSO needs to still be valid,
  and needs to match the cached PSO, otherwise E_INVALIDARG might be
  returned." (Learn, `D3D12_CACHED_PIPELINE_STATE`)
- SETTLED: `CreatePipelineLibrary` returns "E_INVALIDARG if the blob is
  corrupted or unrecognized, D3D12_ERROR_DRIVER_VERSION_MISMATCH if the
  provided data came from an old driver or runtime, and
  D3D12_ERROR_ADAPTER_NOT_FOUND if the data came from different hardware";
  the blob "must remain valid for the lifetime of the object returned. For
  efficiency reasons, the data is not copied." "The pipeline library is
  thread-safe to use ... with one exception: multiple threads loading the
  same PSO ... should synchronize themselves." (Learn,
  `ID3D12Device1::CreatePipelineLibrary`)
- SETTLED: for `CachedPSO` and `ID3D12PipelineLibrary`, compatibility is
  reported only by the errors above; no source read names an identifier to
  compare before loading them.
- SETTLED: D3D12 does report a shader-cache identity for Advanced Shader
  Delivery: `D3D12_FEATURE_SHADER_CACHE_ABI_SUPPORT` (`d3d12.h:2752`) fills
  `D3D12_FEATURE_DATA_SHADERCACHE_ABI_SUPPORT { szAdapterFamily[128],
  MinimumABISupportVersion, MaximumABISupportVersion, CompilerVersion,
  ApplicationProfileVersion }` (`d3d12.h:31608-31615`). A precompiled
  shader database "is usable by an adapter if the adapter family matches,
  the psdb abi version is between the supported min/max Abi version
  inclusive, and the top 32 bits of the application profile version
  matches" (DirectX-Specs `ShaderCacheRegistrationAPI.md:425-445`). It is a
  family and a version range, not one UUID, and it governs precompiled
  shader databases, not `CachedPSO` or pipeline libraries.

**c. Metal 4.**

- SETTLED: `MTL4Archive` is "A read-only container that stores pipeline
  states from a shader compiler", made with `MTL::Device::newArchive(url)`
  (`MTLDevice.hpp:396`); it can make pipeline states directly
  (`MTL4Archive.hpp`), or be listed in `MTL4CompilerTaskOptions::lookupArchives`:
  "Only add MTL4Archive instances to the array that are compatible with the
  Metal device." (Apple)
- SETTLED: archives are written by an `MTL4PipelineDataSetSerializer`
  attached to a compiler: `serializeAsArchiveAndFlushToURL` or
  `serializeAsPipelinesScript` for the offline `metal-tt` tool; "your
  program can't read data out of data set serializer instances" (Apple
  `MTL4PipelineDataSetSerializer`; `MTL4PipelineDataSetSerializer.hpp`).
  "both Metal 3 and 4 can load binary archives" (Apple, "Using the Metal 4
  compilation API").
- OPEN: how to tell whether an archive is compatible with a device before
  use, and what `newArchive` or a lookup does with an incompatible one;
  Apple's `makeArchive(url:)` page says only "or nil if the function
  failed".

**d. Engines.**

- SETTLED: Godot writes its own header for VkPipelineCache data with
  vendor, device, driver version, `pipelineCacheUUID` and pointer size and
  checks each on load (`drivers/vulkan/rendering_device_driver_vulkan.cpp:1631-1636`,
  5361-5366); its D3D12 and Metal caches return false
  (`rendering_device_driver_d3d12.cpp:4434-4435`;
  `rendering_device_driver_metal.cpp:1642-1648`, binary archive commented
  out). NVRHI keeps one empty VkPipelineCache, never saved, and no D3D12
  cache (`src/vulkan/vulkan-device.cpp:243-246`; grepped
  `ID3D12PipelineLibrary`, `CachedPSO`). Diligent and The Forge use
  `ID3D12PipelineLibrary` (Diligent always created empty,
  `Graphics/GraphicsEngineD3D12/src/PipelineStateCacheD3D12Impl.cpp:49`;
  The Forge from saved data, `Common_3/Graphics/Direct3D12/Direct3D12.c:5201`)
  and leave `CachedPSO` null. O3DE serialises VkPipelineCache,
  `ID3D12PipelineLibrary` and `MTLBinaryArchive`
  (`Gems/Atom/RHI/{Vulkan,DX12,Metal}/Code/Source/RHI/PipelineLibrary.cpp`).
  The Forge's Metal cache is an empty function
  (`MetalRenderer.mm:3655`). Granite checks `pipelineCacheUUID` and uses
  Fossilize (`vulkan/device.cpp:734`; `CMakeLists.txt:37`).

**e. BFR's `cache_uuid`.**

- SETTLED: `cache_uuid` ("A pipeline cache loads only on an adapter with
  the same cache_uuid", `api/bfr.yaml:676`) is filled from Vulkan's
  `pipelineCacheUUID` (`src/backends/vulkan/vulkan_caps.cpp:197`). The D3D12
  and Metal `infoOf` functions do not set `cacheUuid`
  (`src/backends/d3d12/d3d12_caps.cpp:67-79`,
  `src/backends/metal/metal_caps.cpp:36-42`), so it is all zeros there.
  Neither has a single UUID for `CachedPSO`, pipeline libraries or
  `MTL4Archive` (b, c above); D3D12's shader-cache ABI data is a family
  plus a version range (b above).
- SETTLED: Vulkan's own header check also uses `vendorID` and `deviceID`
  (a above); `cache_uuid` alone is not the whole check the header makes.
- SETTLED, so: the field carries meaning on Vulkan only. On D3D12 and Metal
  every adapter reports the same all-zero `cache_uuid`, so the rule in
  `api/bfr.yaml:676` would call a cache from any D3D12 or Metal adapter
  loadable on any other.

### Creation and destruction

- SETTLED, cost: "Avoid visual stutter by creating pipeline states at a
  noncritical time, such as during launch, because of the time it can take
  to make them" (Apple, `MTL4RenderCommandEncoder`); "graphics drivers may
  need time to evaluate and build each pipeline state" (Apple,
  `MTLRenderPipelineState`). D3D12 PSOs "are finalized upon creation"
  (Learn, "Important changes from Direct3D 11 to Direct3D 12"). Vulkan:
  linking the whole pipeline "allows the optimization of shaders based on
  their input/outputs and eliminates expensive draw time state validation"
  (Vulkan spec, "Pipelines"); the shader module identifier proposal says
  "Applications compile PSOs late -> stutter!". No spec gives a figure.
- SETTLED, threading: Vulkan pipeline cache use is internally synchronized
  (Pipeline caches, a). `MTL4Compiler` is declared `Sendable`, and it
  "applies multithreading in macOS and iOS by default" (Apple). D3D12
  pipeline libraries are thread-safe except loading one PSO from two
  threads (b above).
- OPEN: a D3D12 primary statement that `CreateGraphicsPipelineState` and
  `CreateComputePipelineState` may be called concurrently; the Learn
  `ID3D12Device` page has none (grepped "thread").
- SETTLED, Vulkan lifetime: "All submitted commands that refer to pipeline
  must have completed execution" before `vkDestroyPipeline`
  (VUID-vkDestroyPipeline-pipeline-00765). "Vulkan objects passed in during
  the creation or allocation of another object may be accessed by the
  implementation any time that the created/allocated object is accessed
  unless explicitly stated otherwise" (Vulkan spec, "Object Lifetime"), so
  shader modules are a stated exception (Shader input, a), and so is the
  pipeline layout on Vulkan 1.3: `VK_KHR_maintenance4` (promoted to 1.3,
  `vk.xml:28394`) lets the application "destroy their VkPipelineLayout
  object immediately after it was used to create another object. It is no
  longer necessary to keep its handle valid while the created object is in
  use." (`appendices/VK_KHR_maintenance4.adoc:31-34`)
- SETTLED, D3D12 lifetime: "All resource and description memory lifetimes
  are the sole responsibly of the app" (Learn, "Important changes"); a
  pipeline library's input blob must outlive the library (b above).
- SETTLED, Metal 4: command buffers hold no strong references to resources
  (Resource binding, c).
- OPEN: whether an `MTL4CommandBuffer` retains the pipeline states encoded
  into it; the Apple text names resources only.
- SETTLED, BFR today: destroying a buffer or texture retires its handle
  against `Submissions::mark()`, the submission ID covering every queue
  ("an object lives until the GPU has run every submit made before it was
  destroyed", `src/core/submissions.h:9-16`), through `Pool::retire`
  (`include/bfr/core/pool.h:114-125`), and collects completed ones
  (`src/backends/vulkan/vulkan_buffer.cpp:210-218`; the same in
  `vulkan_texture.cpp:189-192`, `d3d12_buffer.cpp:174-178`,
  `d3d12_texture.cpp:192-195`, `metal_buffer.cpp:156-160`,
  `metal_texture.cpp:140-143`).
- SETTLED, engines: Godot defers pipeline and shader frees to the frame
  slot's next turn (`servers/rendering/rendering_device.cpp:7844-7848`,
  8296-8322); Diligent and O3DE (Vulkan, D3D12) queue releases
  (`Graphics/GraphicsEngineVulkan/src/PipelineStateVkImpl.cpp:1194`;
  `Gems/Atom/RHI/Vulkan/Code/Source/RHI/PipelineState.cpp:98`); NVRHI holds
  references from command lists until their submission completes
  (`src/vulkan/vulkan-queue.cpp:349-351`); The Forge and O3DE Metal destroy
  at once (`Common_3/Graphics/Vulkan/Vulkan.c:7505`;
  `Gems/Atom/RHI/Metal/Code/Source/RHI/PipelineState.cpp:418-452`). Godot
  creates pipelines on worker threads, with the driver call outside its
  device lock (`servers/rendering/renderer_rd/pipeline_hash_map_rd.h:151`;
  `rendering_device.cpp:5072`); O3DE: "Pipeline states can be acquired at
  any time and from any thread" (`Gems/Atom/RHI/Code/Include/Atom/RHI/PipelineStateCache.h:89-91`).

### Options for the shader input

Facts per option, from the items above. Licences are as read; OPEN where
not read.

1. **Per-backend binaries from the caller (NVRHI's way).** BFR takes
   SPIR-V, DXIL and metallib (or MSL) and passes each to its API
   (Shader input, a-c). No compiler in BFR. The caller's toolchain must
   sign DXIL (b) and give Metal the threadgroup size per dispatch
   (Compute pipelines, b). design.md records the intent that a device
   rebuild uses "bytecode for every backend the app may switch to".
2. **DXC to SPIR-V and DXIL, SPIRV-Cross to MSL.** HLSL in. DXC: NCSA
   licence for source; release binaries Windows and Linux only; uses its
   internal validator since 1.8.2505 (whether it signs: OPEN, b); licence
   of the prebuilt binaries OPEN (d). SPIRV-Cross: Apache-2.0 OR MIT, a library Godot runs at run
   time on Apple platforms, emits MSL source with no Metal 4 argument-table
   option (e). The surveyed engines that make a metallib do it with
   `xcrun metal` on macOS (Godot at export, O3DE, The Forge). Used by O3DE
   (offline) and Diligent (run time, DXC to SPIR-V only).
3. **DXC to DXIL, Metal Shader Converter to metallib.** HLSL in. Converter:
   macOS and Windows, command line and library, SM 6.0-6.6, output for
   Argument Buffers Tier 2; licence OPEN (f). No surveyed engine uses it.
4. **Slang.** Slang source in. Apache-2.0 WITH LLVM-exception. Releases
   for Windows, Linux, macOS; offline or at run time through its
   compilation API. SPIR-V directly; DXIL through DXC; MSL directly and
   metallib through Apple's `metal`; Metal "a work in progress" (g). Used by
   no surveyed engine's live path.
5. **SPIR-V in, Mesa spirv_to_dxil for D3D12, SPIRV-Cross for Metal
   (Godot's way).** GLSL or any SPIR-V producer in. Mesa files read are
   MIT; godot-nir-static MIT; Godot signs DXIL with the public hash and
   runs it at run time and export (h). Godot's Mesa installer is a Windows
   script; other host platforms OPEN.

### Bindless layout across shader compilers

What a caller's compiled shader expects from one device-wide bindless
layout, per compiler and backend. Read on 2026-11-02: Vulkan-Docs main
b94f8bd (`xml/vk.xml`, `chapters/features.adoc`, `descriptorsets.adoc`,
`interfaces.adoc`, `limits.adoc`, `extensions.adoc`,
`appendices/VK_EXT_mutable_descriptor_type.adoc`,
`appendices/roadmap/Roadmap-2022.adoc`, `-2024.adoc`, `-2026.adoc`,
`proposals/VK_EXT_mutable_descriptor_type.adoc`); `vulkan_core.h`
1.4.357; Vulkan-Profiles main 96b1622 (`profiles/`); DXC main b9cd7c6
(`docs/SPIR-V.rst`, `include/dxc/Support/HLSLOptions.td`,
`tools/clang/lib/SPIRV/DeclResultIdMapper.cpp`,
`tools/clang/test/CodeGenSPIRV/`); Slang master 9b46bca
(`docs/user-guide/03-convenience-features.md`,
`a2-02-metal-target-specific.md`, `docs/command-line-slangc-reference.md`,
`source/slang/hlsl.meta.slang`, `slang-capabilities.capdef`); SPIRV-Cross
main 57e4efc (`spirv_msl.hpp`, `spirv_msl.cpp`, `main.cpp`,
`shaders-msl/`, `reference/shaders-msl/`); Mesa main b5b73e5
(`docs/features.txt` and the Vulkan drivers' physical-device files);
MoltenVK main 5189f8d (`Docs/MoltenVK_Runtime_UserGuide.md`); the NVIDIA
Vulkan driver page; gpuinfo.org (Vulkan Hardware Database) pages named
per item, fetched 2026-11-02; Apple Metal Feature Set Tables dated
2026-10-23, Metal Shading Language Specification 4.1 dated 2026-06-04, the
Metal shader converter page and the WWDC23 session 10124 transcript;
Microsoft Learn sources from MicrosoftDocs `win32` (docs ca8274d) and
`sdk-api` (docs 656e965). Engines at the commits in the section intro.

**a. `VK_EXT_mutable_descriptor_type`: status.**

- SETTLED: ratified, not promoted to any core version. vk.xml:
  `VK_EXT_mutable_descriptor_type` (number 495) has `ratified="vulkan"`,
  `specialuse="d3demulation"`, `depends="VK_KHR_maintenance3,VK_VERSION_1_1"`
  and no `promotedto` (vk.xml:29837); `VK_VALVE_mutable_descriptor_type`
  (number 352) has `promotedto="VK_EXT_mutable_descriptor_type"` and no
  `ratified` attribute (vk.xml:27576). Mesa lists it under "Khronos
  extensions that are not part of any Vulkan version"
  (`docs/features.txt:543`, 679).
- SETTLED: special use `d3demulation`: "Extension is intended to support
  D3D emulation layers, and applications ported from D3D, by adding
  functionality specific to D3D" (`extensions.adoc:982-984`); "This
  extension is primarily intended to be useful for API layering efforts"
  (`appendices/VK_EXT_mutable_descriptor_type.adoc`, Description).
- SETTLED: no Khronos profile requires it. Not named in the Roadmap 2022,
  2024 or 2026 appendices (grepped `mutable`). Not named in any profile
  definition in Vulkan-Profiles 96b1622 (`VP_KHR_roadmap`,
  `VP_ANDROID_baseline_2021`/`2022`, `VP_ANDROID_15`-`18` requirements,
  `VP_ANDROID_15`/`16` minimums, `VP_ANDROID_vulkan_profile_2021`/`2022`/`2025`,
  `VP_LUNARG_minimum_requirements`); the other matches are captured device
  reports (`profiles/LunarG/*/vp_gpuinfo_*.json`), test data and schema
  files. `VP_LUNARG_desktop_baseline_2026` is built from 23 reports, 21 of
  which have it, so its intersection leaves it out. Roadmap 2026 requires
  `VK_KHR_shader_untyped_pointers` but not `VK_EXT_descriptor_heap`
  (`Roadmap-2026.adoc:89-110`).
- SETTLED: MoltenVK's supported-extension list does not include it
  (`MoltenVK_Runtime_UserGuide.md:232` onward; it lists
  `VK_EXT_descriptor_indexing`, :339). MoltenVK is not a BFR target.

**b. `VK_EXT_mutable_descriptor_type`: what it requires, and its limits.**

- SETTLED: the feature struct has one member:
  `VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT { sType; pNext;
  VkBool32 mutableDescriptorType; }` (`vulkan_core.h:21054-21058`). There
  is no properties struct (grepped `MutableDescriptorTypeProperties`). A
  layout lists each mutable binding's types in
  `VkMutableDescriptorTypeCreateInfoEXT` and `VkMutableDescriptorTypeListEXT`
  (:21062-21074).
- SETTLED: guaranteed types: `mutableDescriptorType` means "any
  combination" of `SAMPLED_IMAGE`, `STORAGE_IMAGE`, `UNIFORM_TEXEL_BUFFER`,
  `STORAGE_TEXEL_BUFFER`, `UNIFORM_BUFFER`, `STORAGE_BUFFER` must be
  supported (`features.adoc:5950-5959`). `SAMPLER` is not in that list.
  The proposal: "Samplers live in separate heaps in DirectX 12, and do not
  need to be mutable like this"; anything beyond the minimum set is queried
  with `vkGetDescriptorSetLayoutSupport`
  (`proposals/VK_EXT_mutable_descriptor_type.adoc:100-117`).
- SETTLED: update-after-bind and non-uniform indexing narrow the guarantee:
  "`VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` with descriptorType of
  `VK_DESCRIPTOR_TYPE_MUTABLE_EXT` relaxes the list of required descriptor
  types to the descriptor types which have the corresponding
  update-after-bind feature enabled"; non-uniform indexing is supported
  only "if all descriptor types in a VkMutableDescriptorTypeListEXT ...
  have the corresponding non-uniform indexing features enabled"
  (`features.adoc:5961-5977`). The proposal names the case: "An example
  here would be `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` which may not be
  supported with update-after-bind on some implementations" (:113-115).
- SETTLED: BFR's Vulkan floor enables update-after-bind for sampled
  images, storage images and storage buffers, and non-uniform indexing
  for the same three (`src/backends/vulkan/vulkan_caps.cpp:39-68`). It
  does not enable `descriptorBindingUniformBufferUpdateAfterBind`, the
  two texel-buffer update-after-bind features, or non-uniform indexing for
  uniform buffers or texel buffers. Roadmap 2022 requires update-after-bind
  for sampled image, storage image, storage buffer, uniform texel buffer
  and storage texel buffer, not for uniform buffer
  (`Roadmap-2022.adoc:63-72`).
- SETTLED: gpuinfo coverage of the update-after-bind features BFR's floor
  lacks (`listfeaturescore12.php`, Recent (1y), Vulkan 1.3 and up, % of
  devices): `descriptorBindingUniformBufferUpdateAfterBind` Windows 81.86,
  Linux 88.47, Android 93.7; uniform and storage texel buffer
  update-after-bind Windows 99.07, Linux 97.74, Android 93.7. Of the 78
  Windows devices listed without uniform-buffer update-after-bind, 69 are
  NVIDIA (`listdevicescoverage.php?core=1.2&feature=descriptorBindingUniformBufferUpdateAfterBind&platform=windows&option=not`,
  same filters).
- SETTLED: limits. A mutable descriptor "counts as one individual resource
  and one for every unique resource limit per descriptor set type that is
  present in the associated binding's VkMutableDescriptorTypeListEXT"
  (`interfaces.adoc:1672-1680`). Required minimums:
  `maxPerStageDescriptorUpdateAfterBindUniformBuffers` 12,
  `maxDescriptorSetUpdateAfterBindUniformBuffers` 72 (n × 12, n the number
  of supported shader stages, footnote 8, `limits.adoc:7627-7632`; an
  update-after-bind limit may be 0 when its feature is unsupported,
  footnote 9, `:7635-7639`); the sampled-image,
  storage-image, storage-buffer and sampler update-after-bind limits are
  500000 (`limits.adoc:7135-7160`). gpuinfo value distribution of
  `maxPerStageDescriptorUpdateAfterBindUniformBuffers` on Windows
  (`displaycoreproperty.php?core=1.2&name=maxPerStageDescriptorUpdateAfterBindUniformBuffers&platform=windows`,
  Recent (1y), reports, no Vulkan 1.3 filter on this page): 1048576 in
  1782, 15 in 391, 4294967295 in 281, 33554432 in 232, 16777216 in 91,
  1000000 in 62, 12 in 3, 0 in 2, among others. OPEN: which vendors report 15 (the report list behind that
  bucket was not read in full).
- SETTLED: a mutable descriptor's size "is equal to the maximum size of any
  descriptor type included in the pDescriptorTypes list"
  (`limits.adoc:4600-4606`). A layout that exceeds any of the physical
  device limits (not only `maxPerSetDescriptors`) needs
  `vkGetDescriptorSetLayoutSupport` to have returned supported
  (VUID-vkCreateDescriptorSetLayout-support-09582,
  `descriptorsets.adoc:64-68`); a large mutable binding listing
  `UNIFORM_BUFFER` counts against the uniform-buffer limits.
- SETTLED, usage rules: the mutable binding needs the feature
  (VUID-VkDescriptorSetLayoutCreateInfo-mutableDescriptorType-04595), a
  type list in the pNext chain (-pBindings-07303) and no immutable samplers
  (-descriptorType-04594); the list must not contain `MUTABLE_EXT`, the
  two dynamic buffer types or inline uniform blocks
  (VUID-VkMutableDescriptorTypeListEXT-pDescriptorTypes-04600 to -04603);
  push-descriptor layouts cannot hold it (-flags-04591,
  `descriptorsets.adoc:162`) (`descriptorsets.adoc:166-205`, 320-345). The proposal: "For this to be
  well defined, `VK_DESCRIPTOR_BINDING_FLAG_PARTIALLY_BOUND_BIT` must be
  used on the mutable binding" (:269).
- SETTLED: when it is needed. "If multiple shader variables are declared
  with the same set and binding values, and with the same underlying
  descriptor type, they can all be statically used within the same
  shader"; with different types, "that shader can only be executed if the
  variables with the unsupported type are not statically used"
  (`interfaces.adoc:1718-1735`); that rule is about variables within one
  shader. For a layout shared by every pipeline: the layout binding "must
  be of a corresponding descriptor type" (`interfaces.adoc:1708`), and a
  write to a binding that is not `MUTABLE_EXT` must use that binding's
  type (`descriptorsets.adoc:3168-3171`). So a binding needs `MUTABLE_EXT`
  as soon as it holds more than one descriptor type across all shaders.

**c. `VK_EXT_mutable_descriptor_type`: device coverage.**

- SETTLED, gpuinfo coverage (`listextensions.php?platform=<p>&age=recent&apiversion=1.3`,
  "Device coverage", % of devices, Recent (1y), Vulkan 1.3 and up):

  | Extension | All | Windows | Linux | Android |
  |---|---|---|---|---|
  | `VK_EXT_mutable_descriptor_type` | 53.93 | 73.49 | 87.22 | 33.54 |
  | `VK_VALVE_mutable_descriptor_type` | 24.75 | 27.91 | 59.65 | 5.69 |
  | `VK_EXT_descriptor_heap` | 16.4 | 33.02 | 38.1 | 2.54 |
  | `VK_KHR_shader_untyped_pointers` | 25.87 | 40 | 63.41 | 4.36 |

  The same page with the default filter (Recent (1y), all Vulkan
  versions): `VK_EXT_mutable_descriptor_type` All 37.73, Windows 59.25,
  Linux 81.74, Android 19.76, macOS 42.11.
- SETTLED, which devices lack it (device lists
  `listdevicescoverage.php?extension=VK_EXT_mutable_descriptor_type&platform=<p>&age=recent&apiversion=1.3`,
  with and without `&option=not`; used for names and drivers, not for the
  percentages above):
  - NVIDIA proprietary driver: no Maxwell, Pascal or Volta device reports
    it on Windows or Linux, including reports on 580-582 drivers (GeForce
    840M to GTX 1080 Ti, MX110 to MX350, TITAN X, Xp and V, Quadro K, M
    and P, Tesla M40, P4, V100). Every NVIDIA device in the Windows
    supported list is Turing or later (GTX 1630, 1650, RTX). On Linux, Maxwell and Pascal GPUs report
    it under NVK. Why the proprietary driver omits those generations: OPEN
    (the NVIDIA Vulkan driver page gives only "September 27th, 2022 -
    Windows 517.55, Linux 515.49.18 New: VK_EXT_mutable_descriptor_type").
  - Intel: on Windows, Gen9 to Gen11 HD/UHD/Iris devices on 101.1692 to
    101.2141 drivers and Broadwell on 6400.x lack it; Arc, Xe and UHD 7xx
    on 101.4314 and later have it. On Linux, Ivy Bridge and Broadwell lack
    it; Skylake and later under anv have it.
  - AMD: on Windows, Radeon Pro V620 and Radeon Pro WX 8200 on driver
    2.0.233 lack it; on Linux no AMD device lacks it.
  - Android: Adreno 650, 710, 725, 730, 740, 750 and 8xx report it,
    mostly on 512.80 to 512.89 drivers; Adreno 611, 613, 619, 642L, 720,
    722, 732, 735 lack it, mostly on 512.67 and 512.76. Mali devices with
    driver major 53 to 56 have it and 38 to 51 lack it (most reports; three
    supported reports show 26.x and three lacking ones 15, 24 and 25); the
    lists split by driver, not GPU name (Mali-G68 is in both). No PowerVR (Rogue GE8300/GE8322, B, C, D series),
    Samsung Xclipse (530 to 960) or Huawei Maleoon device reports it.
    Mesa turnip on Adreno 6xx and 7xx reports it.
- SETTLED, Mesa drivers: radv (`src/amd/vulkan/radv_physical_device.c:902`, feature
  :1255), anv (`src/intel/vulkan/anv_physical_device.c:593`, :937), nvk
  (`src/nouveau/vulkan/nvk_physical_device.c:279`, :745), turnip (`src/freedreno/vulkan/tu_device.cc:371`, :825),
  honeykrisp (`src/asahi/vulkan/hk_physical_device.c:187`, :592) and panvk on arch 9 and
  later (`src/panfrost/vulkan/panvk_vX_physical_device.c:193`, :743) expose
  `VK_EXT_mutable_descriptor_type`; hasvk exposes only the VALVE name
  (`src/intel/vulkan_hasvk/anv_device.c:335`). `docs/features.txt:679`:
  "DONE (anv, hk, kk, lvp, nvk, panvk/v9+, radv, tu, vn)"; pvr (PowerVR),
  v3dv (Broadcom) and dzn are absent. `VK_EXT_descriptor_heap`: "DONE (anv,
  lvp, radv, vn)" (:636).

**d. DXC: the default emulated heap on Vulkan.**

- SETTLED: default lowering and its requirement: "By default, the SPIR-V
  backend supports SM6.6 resource heaps by emulating the heaps with
  descriptor-indexing runtime arrays, using 2 extensions:
  `SPV_EXT_descriptor_indexing`, `VK_EXT_mutable_descriptor_type`. This is
  also the behavior selected by `-fspv-use-emulated-heap`" (`SPIR-V.rst:1995-2004`).
- SETTLED, layout: "Each type loaded from a heap is considered to be an
  unbounded RuntimeArray bound to the descriptor set 0. Each heap uses at
  most 1 binding in that set. Meaning if 2 types are loaded from the same
  heap, DXC will generate 2 RuntimeArray, one for each type, and will bind
  them to the same binding/set. (This requires
  `VK_EXT_mutable_descriptor_type`)." At most three bindings: resource
  heap, sampler heap, UAV counters. "DXC takes the first 3 unused bindings
  in the set 0, and distributes them in that order: 1. Resource heap.
  2. Sampler heap. 3. Resouce heap counters. Bindings are lazily
  allocated" (:2006-2036). So the default binding numbers depend on what
  else the shader binds in set 0 (examples :2037-2072).
- SETTLED, explicit placement: `-fvk-bind-resource-heap <binding> <set>`,
  `-fvk-bind-sampler-heap`, `-fvk-bind-counter-heap` (:2074-2081;
  `HLSLOptions.td:451-456`). With `-fvk-bind-register`, the resource and
  sampler heap flags are required (`DeclResultIdMapper.cpp:2617-2632`).
  The resource and sampler heaps may be given the same binding (test
  `sm6_6.descriptorheap.aliasing.hlsl`: both at binding 4, set 7).
- SETTLED, Vulkan descriptor type per HLSL type loaded from the resource
  heap: `Texture*` sampled image; `RWTexture*` storage image; `Buffer<T>`
  uniform texel buffer; `RWBuffer<T>` storage texel buffer (:893-910);
  `ConstantBuffer` uniform buffer; `TextureBuffer`, `StructuredBuffer`,
  `RWStructuredBuffer`, `Append`/`ConsumeStructuredBuffer`,
  `ByteAddressBuffer`, `RWByteAddressBuffer` storage buffer (:1005-1018).
  The sampler heap holds only samplers.
- SETTLED, counters: a counter is a separate storage buffer of
  `struct type_ACSBuffer_counter { int; }` decorated `BufferBlock`
  (:1264-1283). Through the heap, "its associated counter is in its own
  binding, but shares the same index in the binding as its associated
  resource" (:1777-1790; "The index of a counter in the counters
  RuntimeArray matches the index of the associated ResourceDescriptorHeap
  RuntimeArray", :2023-2024). Not generated when unused (:2069-2072).
- SETTLED, consequence of b and d: of DXC's three bindings, only the
  resource heap holds more than one descriptor type; the sampler heap
  holds `SAMPLER` only and the counter heap `STORAGE_BUFFER` only, when
  each heap has its own binding. If the resource and sampler heaps share
  a binding (as the aliasing test above does), that binding holds
  `SAMPLER` too, which is outside the guaranteed mutable set
  (`features.adoc:5950-5959`).
- SETTLED, D3D12 has no separate counter array: a UAV's counter is given
  when the UAV descriptor is created (`pCounterResource`: "The
  ID3D12Resource for the counter (if any) associated with the UAV"; Learn,
  `CreateUnorderedAccessView`).

**e. DXC: other options for heaps on Vulkan.**

- SETTLED: DXC has two heap modes and the three binding flags; no other
  heap option (grepped `heap` in `HLSLOptions.td`: :432-442, 451-456).
- SETTLED, native heap: "When `-fspv-use-descriptor-heap` is specified, DXC
  lowers `ResourceDescriptorHeap` and `SamplerDescriptorHeap` through
  `SPV_EXT_descriptor_heap`"; it "also requires `SPV_KHR_untyped_pointers`
  and `-fspv-target-env=vulkan1.3`"; heaps are `OpUntypedVariableKHR`
  decorated `BuiltIn ResourceHeapEXT` / `SamplerHeapEXT`, so no descriptor
  set or binding (:2083-2100). Resource arrays share one stride,
  `max(sizeof(image_descriptor), sizeof(buffer_descriptor))`, from
  `OpConstantSizeOfEXT` (:2194-2203).
- SETTLED, native heap limits: "Append/consume structured buffers and UAV
  counter heap lowering are not supported by the native descriptor heap
  path yet"; `IncrementCounter`/`DecrementCounter` on a heap-loaded
  `RWStructuredBuffer` "emit a diagnostic"; a variable holding both a bound
  and a heap resource is rejected; buffer-type aliases cannot be passed to
  or returned from user functions (:2157-2192).
- SETTLED: the native path needs `VK_EXT_descriptor_heap` on the device;
  coverage in c (Windows 33.02, Linux 38.1, Android 2.54).

**f. Slang.**

- SETTLED: `DescriptorHandle<T>` (`T.Handle`) "translate[s] into a `uint2`"
  on HLSL, GLSL and SPIR-V; "On targets where resource handles are not
  opaque handles, `DescriptorHandle<T>` maps to `T`"
  (`03-convenience-features.md:668-674`). `ResourceDescriptorHeap[i]` and
  `SamplerDescriptorHeap[i]` are accepted as input on HLSL, SPIR-V, GLSL
  and WGSL; "On targets where that representation is unavailable (Metal,
  CUDA, CPU), using the syntax is diagnosed at compile time" (:706-740).
- SETTLED, HLSL target: handles become `ResourceDescriptorHeap[x]` and
  `SamplerDescriptorHeap[x]`; a combined sampler takes the texture from
  `.x` and the sampler from `.y` (:751-776).
- SETTLED, SPIR-V default: "Slang introduces a global array of descriptors
  and fetches from it. The descriptor set ID of the global descriptor array
  can be configured with the `-bindless-space-index`" option (:779-783;
  slangc reference :659-663). The default preset `VkMutable` uses
  binding 0 sampler, 1 combined image sampler, 2 sampled image, storage
  image, uniform and storage texel buffer, uniform buffer and storage
  buffer, 3 other (:791-806; `hlsl.meta.slang:29433-29446`, whose comment
  links the mutable-descriptor proposal). Binding 2 holds six Vulkan
  descriptor types (seven enum entries; `STORAGE_BUFFER` appears twice), so this preset needs `VK_EXT_mutable_descriptor_type`.
- SETTLED, without mutable: preset `None` ("Bind assuming regular binding
  model rules") uses one binding per type: 0 sampler, 1 combined, 2
  sampled image, 3 storage image, 4 uniform texel, 5 storage texel,
  6 uniform buffer, 7 storage buffer, 8 other (:908-931;
  `hlsl.meta.slang:29419-29431`, 29479-29483). It is selected by a user
  `getDescriptorFromHandle` that calls `defaultGetDescriptorFromHandle`
  with the option (:852-908). Both presets index every binding with the
  same `handle.x` (`hlsl.meta.slang:29521-29555`). No slangc option
  selects the preset or the binding numbers (slangc reference headings
  grepped; only `-bindless-space-index` and the three
  `-spirv-*-heap-stride` options).
- SETTLED, SPIR-V with `spvDescriptorHeapEXT`: handles map to
  `SPV_EXT_descriptor_heap` "without declaring any explicit descriptor
  sets"; stride options `-spirv-resource-heap-stride`,
  `-spirv-sampler-heap-stride`, `-spirv-unified-descriptor-heap-stride`
  (:810-850).
- SETTLED, Metal: `ParameterBlock` becomes an argument buffer whose
  resource fields are `T device*` or textures, bound as one `constant*`
  argument (`a2-02-metal-target-specific.md:170-200`); the
  `descriptor_handle` capability includes `metal`
  (`slang-capabilities.capdef:1478`), with `DescriptorHandle<T>` mapping to
  `T` there.
- OPEN: how Slang places UAV counters for `RWStructuredBuffer` or
  `AppendStructuredBuffer` handles on SPIR-V (searched
  `03-convenience-features.md` and the bindless code in
  `hlsl.meta.slang`).

**g. Typed arrays without mutable descriptors.**

- SETTLED, HLSL source: unbounded arrays are declared with an empty first
  dimension, e.g. `Texture2D terrain[] : register(t8)`; "Ranges declared in
  different register spaces never overlap"; indices that vary within a
  wave need `NonUniformResourceIndex` (Learn, "Resource binding in HLSL",
  `resource-binding-in-hlsl.md:44`, 65, 75-80, 292-294). Unbounded arrays
  "require hardware tier 2+ of resource binding support"
  (`example-root-signatures.md:74`).
- SETTLED, D3D12: a root signature may point several ranges at the same
  heap region: "the root signature mechanism makes it possible to alias
  (overlap) heap ranges of compatible types ... If such aliasing is
  desired, the shader must be compiled with D3D10_SHADER_RESOURCES_MAY_ALIAS"
  (`/res_may_alias` in FXC) (`resource-binding-in-hlsl.md:136`). DXC has
  `-res-may-alias`, "Assume that UAVs/SRVs may alias"
  (`HLSLOptions.td:575-576`).
- SETTLED, DXC to SPIR-V: `register(xX, spaceY)` becomes "descriptor set
  `Y` and binding number `X`, regardless of the register type `x`", so
  `t0` and `u0` in one space collide unless `-fvk-{b,s,t,u}-shift` is used
  (`SPIR-V.rst:1793-1807`). `[[vk::binding(X, Y)]]` sets them directly;
  "All C++11 attributes will only trigger warnings and be ignored if not
  compiling towards SPIR-V" (:417-419); `__spirv__` is defined when
  generating SPIR-V (:438-447).
- SETTLED: an unbounded `RWStructuredBuffer<uint> g_rwbuffer[] :
  register(u0, space2)` gets its data at set 2 binding 0 and a counter
  array at set 2 binding 1 (test
  `type.rwstructured-buffer.array.unbounded.counter.hlsl`).
- SETTLED: one HLSL source serves both backends this way. O3DE's
  `Bindless` group declares five typed unbounded arrays
  (`Gems/Atom/Feature/Common/Assets/ShaderLib/Atom/Features/Bindless.azsli:26-36`)
  compiled for D3D12, Vulkan and Metal. NVRHI's `Immutable` bindless
  layout is the same model (h).
- SETTLED, index space: on D3D12 the typed arrays overlap one heap region,
  so one index addresses any type ("All such ranges have the _same base
  offset_ and are completely overlapping", O3DE `Gems/Atom/RHI/Bindless.md:86-87`;
  NVRHI ranges at `OffsetInDescriptorsFromTableStart = 0`,
  `src/d3d12/d3d12-resource-bindings.cpp:647-651`). On Vulkan each type is
  its own binding. O3DE overlaps its D3D12 ranges without passing
  `-res-may-alias` (its DXC arguments are `-Zpr` and
  `-enable-16bit-types`, `Gems/Atom/Asset/Shader/Assets/Config/Shader/shader_build_options.settings`
  and `Platform/Windows/dx12/*.settings`; no `res_may_alias` in the
  checkout). O3DE gives Vulkan a separate allocator per type, so a
  view's index differs by backend ("the indices assigned to each static
  descriptor is given a distinct offset based on descriptor type",
  `Bindless.md:91-92`); NVRHI writes element `binding.slot` into the
  binding whose type matches, so the index is shared across backends
  (`src/vulkan/vulkan-resource-bindings.cpp:750`, 902-909).

**h. Engines.**

- SETTLED: only NVRHI uses `VK_DESCRIPTOR_TYPE_MUTABLE_EXT` (grepped
  `MUTABLE` in all nine engines). SynapseEngine enables the
  `mutableDescriptorType` feature (`Engine/Vk/Core/Device.cpp:57-58`) but
  no `MUTABLE_EXT` binding was found; its shaders are GLSL.
- SETTLED, NVRHI: `BindlessLayoutDesc::LayoutType` has `Immutable` and
  three mutable types, documented as matching DXC: "MutableSrvUavCbv,
  Corresponds to SPIR-V binding -fvk-bind-resource-heap",
  "MutableCounters ... -fvk-bind-counter-heap", "MutableSampler ...
  -fvk-bind-sampler-heap"; on D3D12 they set the `*_HEAP_DIRECTLY_INDEXED`
  flags (`include/nvrhi/nvrhi.h:2019-2049`;
  `src/d3d12/d3d12-resource-bindings.cpp:746`, 750). On Vulkan a mutable
  layout is one `eMutableEXT` binding at binding 0 and errors without the
  extension (`src/vulkan/vulkan-resource-bindings.cpp:122-140`); its type
  lists are the six CBV/SRV/UAV types, `eStorageBuffer` for counters and
  `eSampler` for samplers (:190-224); every bindless binding is
  `ePartiallyBound` (:184). `Immutable` makes one typed binding per
  register space (:141-160).
- SETTLED, O3DE: typed arrays (g). Vulkan: one set with bindings
  `SAMPLED_IMAGE`, `STORAGE_IMAGE`, `STORAGE_BUFFER` (read),
  `STORAGE_BUFFER` (read-write), `SAMPLED_IMAGE` (cube), each
  `UnboundedArraySize` = 100000, `PARTIALLY_BOUND | UPDATE_AFTER_BIND`
  (`Gems/Atom/RHI/Vulkan/Code/Source/RHI/BindlessDescriptorPool.cpp:48-89`;
  `Gems/Atom/RHI/Code/Include/Atom/RHI.Reflect/Limits.h:37`). Metal: one
  argument buffer with a region per type, 100000 each, because SPIRV-Cross
  lacked unbounded arrays for its second approach (`Bindless.md:104-113`;
  `UNBOUNDED_SIZE 100000` in the Mac and iOS
  `AzslcPlatformHeader.azsli:17`). O3DE runs SPIRV-Cross with no
  argument-buffer tier option, so it gets Tier 1, under which unsized
  arrays throw (SPIRV-Cross `spirv_msl.cpp:1569-1574`).
- SETTLED, Diligent: typed runtime arrays through
  `PIPELINE_RESOURCE_FLAG_RUNTIME_ARRAY` ("a run-time sized shader array",
  `Graphics/GraphicsEngine/interface/PipelineResourceSignature.h:141-142`).
- SETTLED, Granite (GLSL, Vulkan only): a bindless set has exactly one
  unsized binding (`vulkan/descriptor_set.cpp:190-192`).
- SETTLED: Godot (`drivers/`), Hazel, Doriax and The Forge's Vulkan
  backend have no partially bound, variable-count or update-after-bind
  bindings (grepped `PARTIALLY_BOUND`, `VARIABLE_DESCRIPTOR_COUNT`,
  `UPDATE_AFTER_BIND`, `runtimeDescriptorArray`).

**i. Metal: argument buffer tier and limits.**

- SETTLED: "Argument buffers tier 2 ... Metal 3 & 4 ... Apple6"; "Argument
  tables ... Metal 4 ... Apple7"; A14 and M1-series are Apple7 (Metal
  Feature Set Tables, 2026-10-23, pages 2, 4 and 5). So every Apple7 device has
  Tier 2 by this table. BFR's floor checks Apple7 and Metal 4
  (`src/backends/metal/metal_caps.cpp:86-91`) and does not read
  `argumentBuffersSupport` (grepped `src/`); metal-cpp declares it
  (`Metal/MTLDevice.hpp:328`, tiers :209-211).
- SETTLED, limits per stage from argument buffers (Feature Set Tables,
  page 8): buffers no limit from Apple6; textures 1M from Apple6;
  samplers 128 Apple6, 996 Apple7 and Apple8, 500K Apple9 to Apple11.
  Sampler argument-table entries per function: 16 on every family
  (page 7).
- SETTLED, unique samplers: "On iOS and tvOS, the maximum number of unique
  samplers per app is `96`, and for macOS, `1024` or greater"; query
  `maxArgumentBufferSamplerCount`; the limit applies to samplers with
  `supportArgumentBuffers` set (Apple, "Improving CPU performance by using
  argument buffers"). D3D12's shader-visible sampler heap holds 2048
  (Resource binding, b).
- SETTLED, MSL rules: "argument buffers cannot contain unions"; "The same
  index may not be assigned to more than one member of an argument
  buffer"; with Tier 2, "You can access argument buffers through pointer
  indexing", referring to "an array of consecutive, independently encoded
  argument buffers" (MSL Specification 4.1, sections 2.13 and 2.13.1,
  pages 54-56).

**j. Metal: mapping a heap or a bindless array.**

- SETTLED, Metal shader converter (public page): converts DXIL to
  metallib; supports SM 6.6 "Dynamic resources"; "Metal libraries built
  using Metal shader converter tools require a device that supports
  Argument Buffers Tier 2, running macOS 14 Sonoma, iOS 17 or later"
  (developer.apple.com/metal/shader-converter). WWDC23 session 10124: one
  "top level" argument buffer is bound per pipeline, with "two layout
  modes": automatic, or "explicitly defining a layout that matches your
  root signature. Use this mode ... if your game uses bindless resources."
- OPEN: the converter's descriptor heap layout (entry format, bind points,
  how SM 6.6 heap indices address it) and whether it emits for Metal 4
  argument tables. The manual (`Metal shader converter 4.0.pdf`) and
  packages are behind the developer download page, which redirects to
  Apple Account sign-in.
- SETTLED, SPIRV-Cross: `--msl-argument-buffers` and
  `--msl-argument-buffer-tier` (1 = Tier 2, matching
  `MTLArgumentBuffersTier`) (`spirv_msl.hpp:350-365`; `main.cpp:882-887`).
  Each descriptor set becomes a struct bound at `[[buffer(set)]]`, each
  binding a member `[[id(binding)]]`, and a runtime array a trailing
  `spvDescriptor<T> Ts [[id(2)]][1] /* unsized array hack */` read
  through `spvDescriptorArray<T>` (reference output of
  `argument-buffers-runtime-array.argument.device-argument-buffer.argument-tier-1.msl2.comp`).
  An unsized descriptor array throws "Unsized array of descriptors
  requires argument buffer tier 2" unless the tier option is Tier 2
  (`spirv_msl.cpp:1569-1574`).
- SETTLED, SPIRV-Cross with the DXC emulated-heap shape (several types at
  one binding): one member is declared and the others are listed as
  "Overlapping binding" comments; each type is read through a
  `reinterpret_cast` of the same storage (reference output of
  `overlapping-bindings.msl31.argument.argument-tier-1...comp:78-84`,
  160-165). The emitting function's comment: "To deal with buffer <->
  image aliasing, we need to perform an unholy UB ritual"
  (`spirv_msl.cpp:20357-20359`). `pad_argument_buffer_resources` and
  `force_active_argument_buffer_resources` keep one struct layout across
  shaders (`spirv_msl.hpp:396-411`).
- SETTLED: `runtime_array_rich_descriptor` makes a buffer entry
  `{ device T* value; int length; int padding; }` (`spirv_msl.cpp:8262-8272`).
  OPEN: its stride when buffers overlap 8-byte texture entries at one
  binding (no reference test combines the two; searched
  `reference/shaders-msl*` for `rich-descriptor` with overlapping
  bindings).
- SETTLED, what a hand-written MSL shader reads: Tier 2 argument buffers
  "match the memory layout of an equivalent C structure" (iOS 16, macOS
  13 and later), filled by writing `gpuAddress` for buffers and
  `gpuResourceID` for textures and samplers (Apple, "Improving CPU
  performance by using argument buffers"). Both are 64-bit: `MTL::ResourceID
  { uint64_t _impl; }` (`Metal/MTLTypes.hpp:92-95`), `GPUAddress =
  uint64_t` (`Metal/MTLGPUAddress.hpp:35`). Apple's MSL example reaches a
  Tier 2 array through a pointer to a struct of textures,
  `material.textures[4]->diffuse`. A Metal 4 argument table binds such a
  buffer by GPU address at a buffer index (`setAddress`, Resource binding,
  c).
- OPEN: whether MSL defines reading one array of 8-byte entries as
  different texture types or as buffer pointers. The MSL 4.1 specification
  forbids unions in argument buffers (i) and says nothing on reinterpreting
  resource handles (searched sections 2.13 and 2.23); SPIRV-Cross calls
  its own cast undefined behaviour.

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
- Vulkan shaders and pipelines:
  [shader modules](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shader-modules),
  [compute shaders](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shaders-compute),
  [pipelines](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#pipelines),
  [pipeline cache header](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#pipelines-cache-header),
  [topology class](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-primitive-topology-class),
  [object lifetime](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#fundamentals-objectmodel-lifetime),
  [VK_EXT_shader_module_identifier proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_shader_module_identifier.adoc),
  [VK_EXT_descriptor_heap proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_descriptor_heap.adoc);
  `vulkan_core.h` in
  [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)
- D3D12 shaders and pipelines:
  [HLSL SM 6.6 dynamic resources](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/HLSL_SM_6_6_DynamicResources.md),
  [D3D12 shader cache APIs](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/ShaderCache.md),
  [MLIR programs (bytecode validation)](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/MlirPrograms.md),
  [INF-0004 validator hashing](https://github.com/microsoft/hlsl-specs/blob/main/proposals/infra/INF-0004-validator-hashing.md),
  [Root signature limits](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits),
  [Shader visible descriptor heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shader-visible-descriptor-heaps),
  [Hardware tiers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support),
  [SetDescriptorHeaps](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-setdescriptorheaps),
  [Managing graphics pipeline state](https://learn.microsoft.com/en-us/windows/win32/direct3d12/managing-graphics-pipeline-state-in-direct3d-12),
  [Important changes from Direct3D 11](https://learn.microsoft.com/en-us/windows/win32/direct3d12/important-changes-from-directx-11-to-directx-12),
  [D3D12_GRAPHICS_PIPELINE_STATE_DESC](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_graphics_pipeline_state_desc),
  [D3D12_RASTERIZER_DESC](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_rasterizer_desc),
  [D3D12_CACHED_PIPELINE_STATE](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_cached_pipeline_state),
  [CreatePipelineLibrary](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device1-createpipelinelibrary),
  [D3D12_SHADER_CACHE_SUPPORT_FLAGS](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_shader_cache_support_flags),
  [Dispatch](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-dispatch);
  `include/directx/d3d12.h` in
  [microsoft/DirectX-Headers](https://github.com/microsoft/DirectX-Headers)
- DXC: `README.md`, `LICENSE.TXT` and release v1.9.2609 (`LICENSE-MS.txt`,
  `LICENSE-LLVM.txt`, `ReleaseNotes.md`, `include/dxc/dxcapi.h`) in
  [microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.9.2609)
- SPIRV-Cross: `LICENSE`, `spirv_msl.hpp`, `spirv_msl.cpp` in
  [KhronosGroup/SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)
- Slang: release v2026.20 assets, `LICENSE`, `docs/user-guide/08-compiling.md`, `09-targets.md`,
  `a2-02-metal-target-specific.md`, `docs/command-line-slangc-reference.md`,
  `docs/generated/design/target-pipelines/index.md` in
  [shader-slang/slang](https://github.com/shader-slang/slang)
- Mesa: `docs/license.rst`, `src/microsoft/spirv_to_dxil/spirv_to_dxil.h`,
  `src/microsoft/compiler/nir_to_dxil.c` in
  [mesa/mesa](https://gitlab.freedesktop.org/mesa/mesa);
  [godotengine/godot-nir-static](https://github.com/godotengine/godot-nir-static)
- Metal shader converter:
  [developer.apple.com/metal/shader-converter](https://developer.apple.com/metal/shader-converter/)
- Metal 4 shaders and pipelines:
  [Using the Metal 4 compilation API](https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api),
  [MTL4Compiler](https://developer.apple.com/documentation/metal/mtl4compiler),
  [MTL4LibraryDescriptor](https://developer.apple.com/documentation/metal/mtl4librarydescriptor),
  [MTL4LibraryFunctionDescriptor](https://developer.apple.com/documentation/metal/mtl4libraryfunctiondescriptor),
  [MTL4RenderPipelineDescriptor](https://developer.apple.com/documentation/metal/mtl4renderpipelinedescriptor),
  [MTL4ComputePipelineDescriptor](https://developer.apple.com/documentation/metal/mtl4computepipelinedescriptor),
  [requiredThreadsPerThreadgroup](https://developer.apple.com/documentation/metal/mtl4computepipelinedescriptor/requiredthreadsperthreadgroup),
  [MTL4ArgumentTable](https://developer.apple.com/documentation/metal/mtl4argumenttable),
  [MTL4ArgumentTableDescriptor](https://developer.apple.com/documentation/metal/mtl4argumenttabledescriptor),
  [setArgumentTable(_:stages:)](https://developer.apple.com/documentation/metal/mtl4rendercommandencoder/setargumenttable(_:stages:)),
  [MTL4RenderCommandEncoder](https://developer.apple.com/documentation/metal/mtl4rendercommandencoder),
  [MTL4Archive](https://developer.apple.com/documentation/metal/mtl4archive),
  [MTL4CompilerTaskOptions lookupArchives](https://developer.apple.com/documentation/metal/mtl4compilertaskoptions/lookuparchives),
  [MTL4PipelineDataSetSerializer](https://developer.apple.com/documentation/metal/mtl4pipelinedatasetserializer),
  [makeLibrary(data:)](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(data:)),
  [MTLRenderPipelineState](https://developer.apple.com/documentation/metal/mtlrenderpipelinestate),
  [Improving CPU performance by using argument buffers](https://developer.apple.com/documentation/metal/improving-cpu-performance-by-using-argument-buffers)
- Engines (shaders and pipelines): [Themaister/Granite](https://github.com/Themaister/Granite);
  Hazel, SynapseEngine and Doriax at the commits named in the section intro
- Bindless layout across shader compilers:
  [VK_EXT_mutable_descriptor_type proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_mutable_descriptor_type.adoc),
  [VK_EXT_mutable_descriptor_type](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_mutable_descriptor_type.html),
  [mutableDescriptorType feature](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#features-mutableDescriptorType),
  [descriptor set layouts](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#descriptorsets-setlayout),
  [shader resource interface](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#interfaces-resources),
  [Roadmap 2022](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#roadmap-2022),
  [Roadmap 2024](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#roadmap-2024),
  [Roadmap 2026](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#roadmap-2026),
  [extension special use](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#extendingvulkan-specialuse-table);
  `profiles/` in [KhronosGroup/Vulkan-Profiles](https://github.com/KhronosGroup/Vulkan-Profiles)
- DXC heaps on Vulkan: `docs/SPIR-V.rst`
  ([ResourceDescriptorHeaps & SamplerDescriptorHeaps](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#resourcedescriptorheaps-samplerdescriptorheaps)),
  `include/dxc/Support/HLSLOptions.td`,
  `tools/clang/lib/SPIRV/DeclResultIdMapper.cpp` and
  `tools/clang/test/CodeGenSPIRV/` in
  [microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler)
- Slang bindless: `docs/user-guide/03-convenience-features.md`
  ([DescriptorHandle](https://github.com/shader-slang/slang/blob/master/docs/user-guide/03-convenience-features.md)),
  `docs/user-guide/a2-02-metal-target-specific.md`,
  `docs/command-line-slangc-reference.md`, `source/slang/hlsl.meta.slang`,
  `source/slang/slang-capabilities.capdef` in
  [shader-slang/slang](https://github.com/shader-slang/slang)
- SPIRV-Cross argument buffers: `spirv_msl.hpp`, `spirv_msl.cpp`,
  `main.cpp`, `shaders-msl/comp/` and `reference/shaders-msl/comp/` in
  [KhronosGroup/SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)
- Device coverage: gpuinfo.org
  [extension coverage](https://vulkan.gpuinfo.org/listextensions.php?platform=windows&age=recent&apiversion=1.3)
  (and `platform=all`, `linux`, `android`),
  [devices without VK_EXT_mutable_descriptor_type](https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_EXT_mutable_descriptor_type&age=recent&platform=windows&apiversion=1.3&option=not),
  [Vulkan 1.2 feature coverage](https://vulkan.gpuinfo.org/listfeaturescore12.php?platform=windows&age=recent&apiversion=1.3),
  [maxPerStageDescriptorUpdateAfterBindUniformBuffers](https://vulkan.gpuinfo.org/displaycoreproperty.php?core=1.2&name=maxPerStageDescriptorUpdateAfterBindUniformBuffers&platform=windows);
  [NVIDIA Vulkan driver support](https://developer.nvidia.com/vulkan-driver);
  `docs/features.txt` and `src/{amd,intel,nouveau,freedreno,asahi,panfrost}/vulkan*/`
  in [mesa/mesa](https://gitlab.freedesktop.org/mesa/mesa);
  `Docs/MoltenVK_Runtime_UserGuide.md` in
  [KhronosGroup/MoltenVK](https://github.com/KhronosGroup/MoltenVK)
- D3D12 typed arrays and counters:
  [Resource binding in HLSL](https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl),
  [Example root signatures](https://learn.microsoft.com/en-us/windows/win32/direct3d12/example-root-signatures),
  [CreateUnorderedAccessView](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createunorderedaccessview)
- Metal bindless:
  [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf),
  [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
  [Metal shader converter](https://developer.apple.com/metal/shader-converter/),
  [WWDC23: Bring your game to Mac, Part 2: Compile your shaders](https://developer.apple.com/videos/play/wwdc2023/10124/)
- Engines (bindless): O3DE `Gems/Atom/RHI/Bindless.md` in
  [o3de/o3de](https://github.com/o3de/o3de); the other engines at the
  commits named in the Shaders and pipelines intro
