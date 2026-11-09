# Roadmap

What the library still has to do, in order. BFR is a babelfish: every feature
a modern engine or any visual simulation would use from Vulkan, D3D12 or Metal
gets a BFR form, exposed where the backend has it and reported through caps
where it does not. Steps keep coming until nothing usable is left unexposed.
The library comes first; samples follow it. A step is done when it works on
every backend.

Done: instance and adapters, device, swapchain and present, command pools and
lists, clear, submit and wait, buffers (create, map, destroy), buffer copies,
textures and texture copies, shaders and pipelines, draws and dispatches,
samplers, timings (GPU timestamps, tick rates, CPU-GPU calibration, CPU time
in each call), sparse and tiled resources (partly resident textures and
buffers, sparse memory, tile mapping, sparse caps), indirect draws and
dispatches (counts and strides), exact topologies and primitive restart,
depth-stencil and multisampled attachments in use (depth and stencil ops
apart, read-only depth-stencil, sample mask, alpha to one, sample shading),
resolves (color and depth-stencil as rendering ends, into a swapchain's
image, and by resolveTexture), compute and copy queues in use (what each
kind of queue runs, the layouts each kind moves a texture between, work
handed across queues by waits), pipeline cache blobs (caches made from bytes
and read back as bytes, which pipelines are looked for in and added to).
Vulkan only so far: the caller's API version, uncapped, with BFR's minimum
requirements (Vulkan 1.2 with dynamic rendering and synchronization2).

D3D12 and Metal are paused while Vulkan takes shape (owner, 2026-11-08); they
still type-check.

12. The caller's API version and BFR's minimum requirements on D3D12 and
    Metal: the instance and devices are made at the version the caller asks
    for, below the minimum fails, above it nothing is capped.
13. Mesh shaders and ray tracing, exposed where the backend has them.
14. Queries (occlusion, statistics), debug labels, memory heaps and placed
    resources, formatted buffer views, variable rate shading, programmable
    sample positions, HDR and present timing, and whatever each backend's
    caps turn up as we go.

Next: 13 on Vulkan; 12 on D3D12 and Metal when they resume.
