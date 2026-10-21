# BFR respec

Working notes for the top-down respec, started 2026-10-20. Each decision moves
into its owning doc when it lands; this file is deleted when it is empty.

## Decided

- BFR (Babel Fish RHI) is its own public project; Vesper consumes it.
- Backends: Vulkan, D3D12, Metal. WebGPU named, then removed: no explicit
  barriers, no bindless, no push constants, one queue.
- Platforms: Windows, Linux, macOS, Android (Vulkan), iOS (Metal).
- Ownership points down only: an object never holds a pointer or reference to
  its owner or creator. What it needs from above is given at creation, by
  value.
- Namespaces: `bfr` for the API, `bfr_core` for the base (result, handle,
  pool, allocator, log hook, window handles). No nested namespaces.

## Decision order

1. Scope. Platforms and backends decided above. Still open: translation only;
   translation plus a frame graph; translation plus automatic barrier tracking.
2. Shader contract: what each backend accepts, cross-compilation inside or
   outside BFR.
3. Object model: instance, adapter, device, queues, swapchain, resources,
   pipelines, commands, sync; who creates whom. Includes window at device
   creation and which queues to open.
4. References: handles with generations versus pointers; how a handle names
   its device.
5. Failure model: expected with an error struct, device loss, validation.
6. Threading contract.
7. Host hooks, one row of the table below at a time. The CPU allocator sits
   here.
8. Repo, build, tests; how Vesper consumes BFR; CI on lavapipe.

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

## Sources

- Godot: `servers/rendering/rendering_device.h` in
  [godotengine/godot](https://github.com/godotengine/godot)
- NVRHI: `include/nvrhi/nvrhi.h` in
  [NVIDIA-RTX/NVRHI](https://github.com/NVIDIA-RTX/NVRHI)
- The Forge: `Common_3/Graphics/Interfaces/IGraphics.h` in
  [ConfettiFX/The-Forge](https://github.com/ConfettiFX/The-Forge)
