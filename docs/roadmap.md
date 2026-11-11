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
requirements (Vulkan 1.2 with dynamic rendering and synchronization2); mesh
and task shaders (mesh pipelines, direct, indirect and counted mesh draws,
mesh limits, task and mesh timestamps); acceleration structures and inline
ray queries (builds of triangles, boxes and instances, updates, copies,
instance records, the acceleration structure table); ray tracing pipelines
(shader objects, shader groups and their identifiers, binding tables, direct
and indirect traces, buffer addresses); compaction, serialization and
acceleration structure properties, stack sizes, position fetch, invocation
reorder caps; pipeline libraries, capture-replay group identifiers, indirect
builds; opacity micromaps; motion (vertex, matrix and SRT instance keyframes);
partitioned top structures (instances written, updated and moved by partition
from records the GPU reads, their own table); cluster acceleration structures
(clusters, templates and cluster bottom structures built, instantiated and
moved from records the GPU reads, instances naming bottom structures by
reference).

D3D12 and Metal are paused while Vulkan takes shape (owner, 2026-11-08); they
still type-check.

12. The caller's API version and BFR's minimum requirements on D3D12 and
    Metal: the instance and devices are made at the version the caller asks
    for, below the minimum fails, above it nothing is capped.
13. Ray tracing, exposed where the backend has it; mesh shaders on D3D12 and
    Metal, which report none until then.
14. Queries (occlusion, statistics), debug labels, memory heaps and placed
    resources, formatted buffer views, variable rate shading, programmable
    sample positions, HDR and present timing, and whatever each backend's
    caps turn up as we go.

Next: 13 on Vulkan; 12 on D3D12 and Metal when they resume.
