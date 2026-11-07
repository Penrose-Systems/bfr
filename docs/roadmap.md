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
samplers.

6. Timings: everything the backends can time, requestable by the client. GPU
   time per pass, draw, dispatch and copy; submit-to-complete, acquire and
   present waits; CPU time inside each BFR call.
7. Sparse and tiled resources: partly resident textures and buffers, tile
   mapping, and every sparse cap each backend reports.
8. Indirect draws and dispatches; exact topologies.
9. Depth-stencil and multisampled attachments in use; resolves.
10. Compute and copy queues in use: cross-queue waits.
11. Pipeline cache blobs.
12. Mesh shaders and ray tracing, exposed where the backend has them.
13. Queries (occlusion, statistics), debug labels, memory heaps and placed
    resources, formatted buffer views, variable rate shading, HDR and
    present timing, and whatever each backend's caps turn up as we go.

Next: 6.
