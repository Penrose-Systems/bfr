# BFR

BFR is one library that puts one API over several native graphics APIs, so
that the same caller code runs on every backend, operating system and window
system. These are the words BFR uses for its own concepts; general graphics
or programming terms are not listed.

## Structure

**Backend**:
One native graphics API that BFR drives: Vulkan, D3D12 or Metal. An app never
sees a backend's own types or words through BFR.
_Avoid_: driver, platform, API (ambiguous with BFR's own API)

**Instance**:
One backend, opened in this process. Several may exist at once, including of
different backends.

**Adapter**:
A GPU an instance found, described but not opened. Nothing of it exists on the
device side until a device is created on it.
_Avoid_: GPU, physical device

**Device**:
An open adapter. It creates and owns every handle made on it.
_Avoid_: context, logical device

**Null backend**:
A backend for the tests alone: it lists one adapter that reports nothing and
draws nothing. It is not part of the library an app links.

**Wrapper**:
A language binding over the C API (C++, C#, Rust, Python) that this project
writes and generates from the API description.
_Avoid_: binding, FFI layer

## Capabilities

**Floor**:
What every adapter BFR lists must have. An adapter missing any of it is not
listed, so the floor is never queried.
_Avoid_: baseline, minimum spec, required features

**Optional feature**:
Something an adapter may or may not have beyond the floor. It is reported by
the adapter and asked for when the device is created.
_Avoid_: extension, capability bit

**Caps**:
Everything about an adapter that stays the same while it runs: info, limits,
optional features, memory, queues. Readable before any device exists. A limit
for a feature the adapter lacks is 0.
_Avoid_: limits (one part of caps), properties, device info

**Budget**:
How much memory the device's process may use and uses now. It changes while
the app runs, so it is not part of caps.

## Presentation

**Swapchain**:
The images presented to one window, in turn. Created from a device and a
window; any number per device.

**Swapchain caps**:
What a swapchain for one window may be on one adapter: image count, size,
present modes, alpha modes. Asked of the window system each time, since the
window changes; not part of caps.
_Avoid_: surface caps, surface capabilities

**Swapchain format**:
A format and colour space a swapchain for one window may use on one adapter.
_Avoid_: surface format

**Display rotation**:
How a window's display is turned relative to its natural orientation, as the
window system reports it now.
_Avoid_: surface rotation, surface transform, pre-transform

## Handles and lifetime

**Object**:
An instance or a device: something an app holds as an opaque pointer.

**Handle**:
A value naming one thing a device owns, safe against stale use: a wrong or
destroyed handle is caught, never dereferenced. Each handle type is its own
C type. A handle belongs to one device and every call names that device.
_Avoid_: reference, id, pointer

**Generation**:
The part of a handle that tells one occupant of a slot from the next, so a
handle to a destroyed thing never names its replacement.

**Deferred destruction**:
How BFR frees what the GPU may still use: the handle is invalid at once, and
the memory is freed only once every queue has passed the work submitted before
the destroy call. Swapchains are the exception and are destroyed at once.

**Submission ID**:
The number a handle table retires an object against: the count of submits a
queue has made, which the GPU passes in order. Internal; a caller never sees
one. Not a completion token.
_Avoid_: serial, token

**Retire**:
To end a handle's life while its underlying object waits for the GPU to pass
the work that may use it.

**Device loss**:
The backend reports the device unusable. It is sticky: every later call
reports it until the device is rebuilt.

**Rebuild**:
Recreating every object a device owns on a new device (another backend,
another adapter, or the same one after device loss) so that the app's handles
stay the same.

## Work and synchronization

**Queue**:
Where submitted work runs, in order. Types: graphics, compute, copy. A device's
queues are numbered: graphics is 0, then the compute queues, then the copy
queues it was opened with.
_Avoid_: queue family (a backend's grouping)

**Timeline**:
A queue's counter, which the GPU advances as work finishes. The one primitive
for all queue progress.
_Avoid_: fence, semaphore, event (backend words)

**Completion token**:
A point on a queue's timeline, `{queue, value}`, that a submit returns, later
submits may wait on, and the CPU waits on or polls. An all-zero token has
already completed.
_Avoid_: fence value, sync point, submission ID

**Command pool**:
Where command lists are recorded from: the memory they record into, given back
to the pool by a reset once the GPU has run them. Used by one thread at a
time, its lists included.
_Avoid_: command allocator, pool (alone, which also names the handle table)

**Command list**:
Commands recorded for the GPU, from begin until submit sends them. Comes from
one command pool; recorded by one thread at a time.
_Avoid_: command buffer, encoder

**Submit**:
Hand recorded command lists to one queue, with the tokens the GPU must pass
first, and receive the completion token they signal. Lists run in the order
given, whatever order threads recorded them in.
_Avoid_: execute, commit, flush

**Barrier**:
A statement by the caller of a resource's access before and after a point in
a command list, with stage, access and texture layout. In the core the
caller states every barrier; a tracker that infers them is a separate
authority.
_Avoid_: transition, resource state

**Rendering scope**:
A span of a command list between begin rendering and end rendering, with its
attachments and their load and store actions.
_Avoid_: render pass (a backend object BFR does not have)

**Frame slot**:
One of the frames the CPU may have in flight at once. A slot is reused once
the GPU has finished the frame that last used it.
_Avoid_: frame index, ring index

## Resources and binding

**Resource**:
A buffer, texture, sampler or acceleration structure.

**View**:
How a shader sees a resource. Creating one takes a bindless slot; destroying
it gives the slot back under deferred destruction.
_Avoid_: descriptor, SRV, UAV

**Bindless table**:
The one table of views, and the one table of samplers, that every pipeline
shares. Shaders reach resources by index into it, never by binding.
_Avoid_: descriptor set, descriptor heap, argument buffer

**Bindless slot**:
One entry of a bindless table; its index is what a shader uses.

**Push constants**:
The small block of per-draw values every pipeline takes. The only per-draw
binding there is.
_Avoid_: root constants, setBytes

**Heap**:
Memory the caller controls, into which resources are placed.

**Placed resource**:
A resource created into a heap at an offset the caller chose, rather than
through BFR's own allocator.

**Residency**:
Whether a resource is reachable by shaders right now. BFR keeps every live
bindless resource resident.

## Pipelines

**Pipeline**:
Graphics, mesh or compute: shader bytecode for each backend plus fixed state.
There is one layout for all pipelines, so a pipeline has no layout of its own.

**Pending / ready / failed**:
The three states of a pipeline, whose creation is asynchronous. Recording with
one that is not ready is a usage error.

**Vertex pulling**:
Shaders fetch vertices by index from buffers; there are no vertex buffers or
input layouts.

## Render graph

**Render graph**:
The layer that ships with BFR for building a frame from passes and their reads
and writes: it orders passes, emits barriers and pools transient textures. It
uses only the public core API.
_Avoid_: frame graph

**Core**:
The lower public layer: devices, resources, pipelines, commands, explicit
barriers and GPU lifetime. Backends implement only the core.

**Authority**:
Whoever currently states a resource's barriers: the graph, the caller's code or
native interop. One authority per resource at a time.

**Import / export**:
Handing a resource into the graph with its current state, and out of it with
its final state.

**Transient**:
A texture that lives only within one graph execution and may share memory
with other transients.
_Avoid_: temporary, scratch

## API shape

**Desc**:
A description struct a caller fills to create something. The caller owns what
it passes; BFR copies what it keeps.
_Avoid_: create info, request, params

**Chain**:
The way a desc grows after 1.0: new fields arrive as extension structs linked
from the desc, never as changes to the desc itself.
_Avoid_: pNext

**Result**:
What a call did. Errors are negative; `incomplete` means the caller's array
was too small and is not an error.

**Message callback**:
The one callback BFR has: it receives BFR's messages and validation's, on any
thread.
_Avoid_: logger, host callback, debug callback

**Validation**:
The backend's own checking layers: Vulkan's validation layers, the D3D12 debug
layer, Metal's debug layer. Slow; turned on at instance creation, only for
debugging, and never otherwise.
_Avoid_: debug layer (one backend's word), usage check

**Usage check**:
A cheap test BFR itself makes of a call against its contract, such as
rendering already open or a token no submit returned. Always compiled in;
refused with a result, or reported through the message callback where the
call returns nothing.
_Avoid_: validation

**Usage error**:
A call that breaks BFR's contract. A usage check refuses or reports it.
