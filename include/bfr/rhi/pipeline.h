#pragma once

#include "bfr/bfr.h"
#include "bfr/core/handle.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace bfr::rhi {

/// A compute or graphics pipeline: shaders with the fixed state they run
/// under. Every pipeline shares one resource layout, which api/bfr.yaml
/// states under bfr_pipeline and the shaders declare. Only named, by a
/// PipelineHandle.
struct Pipeline;

using PipelineHandle = core::Handle<Pipeline>;

/// Pipelines the backend has compiled, which it reads back as bytes for a
/// later run to start from. Pipelines made with one look in it first and add
/// themselves on a miss. Only named, by a PipelineCacheHandle.
struct PipelineCache;

using PipelineCacheHandle = core::Handle<PipelineCache>;

/// How to create a pipeline cache: bytes Device::pipelineCacheData gave
/// before, read during the call only. Empty is an empty cache.
struct PipelineCacheDesc {
    std::span<const std::byte> data;
    /// What a debugger or profiler shows it as, where Features::objectNames;
    /// read during the call only. Empty is none. The create call fails with
    /// RhiUnsupported for a name where Features::objectNames is false,
    /// RhiInvalidCall for one holding a '\0', which the backend reads as its
    /// end, and RhiDeviceFailed when the backend refuses it.
    std::string_view name{};
};

/// What Device::pipelineCacheData wrote: how many bytes, and whether the
/// backend says that is all of them.
struct PipelineCacheWrite {
    std::size_t written = 0;
    bool complete = false;
};

/// The most color targets a graphics pipeline renders to.
inline constexpr std::uint32_t kMaxColorTargets = BFR_MAX_COLOR_TARGETS;

/// One shader stage: bytecode for the backend the device runs on, and the
/// name of the function in it to run. Both are borrowed for the call.
struct ShaderCode {
    std::span<const std::byte> code;
    std::string_view entryPoint;
};

/// Code compiled once, as ShaderCode gives it, and named by the ray tracing
/// pipelines made from it, which keep what they need of it: it may be
/// destroyed once they are made. Only named, by a ShaderHandle.
struct Shader;

using ShaderHandle = core::Handle<Shader>;

/// How to create a compute pipeline. workgroupSize is the threads in each
/// workgroup, as the shader declares them.
struct ComputePipelineDesc {
    ShaderCode compute;
    std::array<std::uint32_t, kWorkgroupDimensions> workgroupSize{};
    /// Where the pipeline is looked for, and added on a miss; null is none.
    PipelineCacheHandle cache{};
    /// What a debugger or profiler shows it as, where Features::objectNames;
    /// read during the call only. Empty is none. The create call fails with
    /// RhiUnsupported for a name where Features::objectNames is false,
    /// RhiInvalidCall for one holding a '\0', which the backend reads as its
    /// end, and RhiDeviceFailed when the backend refuses it.
    std::string_view name{};
};

/// What draws assemble vertices into. Lists take each primitive's vertices
/// apart; strips share each one with the next; a fan shares the first vertex
/// with every triangle, where Features::triangleFans. The values are the C
/// API's.
enum class Topology : std::uint8_t {
    TriangleList = 0,
    TriangleStrip = 1,
    TriangleFan = 2,
    LineList = 3,
    LineStrip = 4,
    PointList = 5,
};

/// Whether an index of all ones ends a strip and starts the next, in indexed
/// draws, and the index type it is all ones of: D3D12 fixes that value in the
/// pipeline. Restart on a list topology needs Features::listRestart. The
/// values are the C API's.
enum class PrimitiveRestart : std::uint8_t {
    Off = 0,
    Index16 = 1,
    Index32 = 2,
};

/// Which triangles a graphics pipeline discards by the way they face. The
/// values are the C API's.
enum class CullMode : std::uint8_t {
    None = 0,
    Front = 1,
    Back = 2,
};

/// Which winding of a triangle's vertices on screen faces the viewer. The
/// values are the C API's.
enum class FrontFace : std::uint8_t {
    CounterClockwise = 0,
    Clockwise = 1,
};

/// How a graphics pipeline fills triangles. The values are the C API's.
enum class FillMode : std::uint8_t {
    Solid = 0,
    Wireframe = 1,
};

/// How a depth or stencil test compares the new value against the stored
/// one, or a compare sampler a reference against each texel; the test passes
/// when the comparison holds. The values are the C API's.
enum class CompareOp : std::uint8_t {
    Always = 0,
    Never = 1,
    Less = 2,
    Equal = 3,
    LessEqual = 4,
    Greater = 5,
    NotEqual = 6,
    GreaterEqual = 7,
};

/// What a stencil test does to the stored stencil value. The values are the
/// C API's.
enum class StencilOp : std::uint8_t {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementClamp = 3,
    DecrementClamp = 4,
    Invert = 5,
    IncrementWrap = 6,
    DecrementWrap = 7,
};

/// What a blend multiplies a color or alpha by. The values are the C API's.
enum class BlendFactor : std::uint8_t {
    Zero = 0,
    One = 1,
    SrcColor = 2,
    OneMinusSrcColor = 3,
    DstColor = 4,
    OneMinusDstColor = 5,
    SrcAlpha = 6,
    OneMinusSrcAlpha = 7,
    DstAlpha = 8,
    OneMinusDstAlpha = 9,
    ConstantColor = 10,
    OneMinusConstantColor = 11,
    ConstantAlpha = 12,
    OneMinusConstantAlpha = 13,
    SrcAlphaSaturate = 14,
};

/// How a blend combines the source and destination once each is multiplied
/// by its factor. The values are the C API's.
enum class BlendOp : std::uint8_t {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
};

/// How a graphics pipeline turns primitives into fragments.
struct RasterState {
    CullMode cull = CullMode::None;
    FrontFace frontFace = FrontFace::CounterClockwise;
    FillMode fill = FillMode::Solid;
    /// Clamps depth to the viewport's range instead of clipping to it.
    bool depthClamp = false;
    /// Added to each fragment's depth, in units of the depth format's
    /// smallest step. D3D12 takes whole numbers only, so there a fraction
    /// fails with RhiInvalidCall.
    float depthBias = 0;
    /// Added to each fragment's depth, times its depth slope.
    float depthBiasSlope = 0;
    /// The most the bias may add; 0 is no limit.
    float depthBiasClamp = 0;
};

/// The stencil test for triangles facing one way. The masks' low 8 bits
/// count.
struct StencilFace {
    StencilOp fail = StencilOp::Keep;
    StencilOp pass = StencilOp::Keep;
    StencilOp depthFail = StencilOp::Keep;
    CompareOp compare = CompareOp::Always;
    std::uint32_t readMask = 0;
    std::uint32_t writeMask = 0;
};

/// How a graphics pipeline tests depth and stencil. Off by default; the
/// pipeline then needs no depth-stencil target.
struct DepthStencilState {
    bool depthTest = false;
    bool depthWrite = false;
    CompareOp depthCompare = CompareOp::Always;
    bool stencilTest = false;
    StencilFace front;
    StencilFace back;
};

/// The bits of BlendState::writeMask, one per channel.
inline constexpr std::uint32_t kWriteRed = 1U << 0U;
inline constexpr std::uint32_t kWriteGreen = 1U << 1U;
inline constexpr std::uint32_t kWriteBlue = 1U << 2U;
inline constexpr std::uint32_t kWriteAlpha = 1U << 3U;

/// How a graphics pipeline blends a fragment into one color target. With
/// enable false the fragment's color replaces the target's, in the channels
/// writeMask names: the kWrite bits, 0 for none; bits above 3 are ignored.
struct BlendState {
    bool enable = false;
    BlendFactor srcColor = BlendFactor::Zero;
    BlendFactor dstColor = BlendFactor::Zero;
    BlendOp colorOp = BlendOp::Add;
    BlendFactor srcAlpha = BlendFactor::Zero;
    BlendFactor dstAlpha = BlendFactor::Zero;
    BlendOp alphaOp = BlendOp::Add;
    std::uint32_t writeMask = 0;
};

/// One color target a graphics pipeline renders to.
struct ColorTarget {
    Format format = Format::Undefined;
    BlendState blend;
};

/// How to create a graphics pipeline. Vertices are pulled: the vertex shader
/// reads them by index from buffers, and the pipeline takes no vertex layout.
/// An empty fragment.code is no fragment shader. The first colorTargetCount
/// of colorTargets are rendered to, in attachment order.
struct GraphicsPipelineDesc {
    ShaderCode vertex;
    ShaderCode fragment;
    /// Task and mesh shaders in place of vertices: with mesh.code given, the
    /// pipeline draws by drawMesh, vertex.code is empty, and topology and
    /// primitiveRestart are unused. It needs Features::meshShader, and a task
    /// shader Features::taskShader; an empty task.code is none. With a task
    /// shader, the mesh shader reads no draw index. The workgroup sizes are
    /// the threads in each workgroup, as the shaders declare them.
    ShaderCode task{};
    std::array<std::uint32_t, kWorkgroupDimensions> taskWorkgroupSize{};
    ShaderCode mesh{};
    std::array<std::uint32_t, kWorkgroupDimensions> meshWorkgroupSize{};
    Topology topology = Topology::TriangleList;
    PrimitiveRestart primitiveRestart = PrimitiveRestart::Off;
    RasterState raster;
    DepthStencilState depthStencil;
    std::uint32_t colorTargetCount = 0;
    std::array<ColorTarget, kMaxColorTargets> colorTargets{};
    /// Undefined is no depth-stencil target.
    Format depthStencilFormat = Format::Undefined;
    /// Samples per pixel of every target; 1 is no multisampling.
    std::uint32_t samples = 0;
    /// The fragment's alpha sets how many samples it covers.
    bool alphaToCoverage = false;
    /// The samples written, bit i for sample i; bits from samples up are
    /// ignored. Leaving one of the low samples bits clear needs
    /// Features::sampleMask.
    std::uint64_t sampleMask = 0;
    /// The fragment's alpha becomes 1 after coverage is taken from it, where
    /// Features::alphaToOne.
    bool alphaToOne = false;
    /// The fraction of samples, 0 to 1, shaded apart; above 0 shades per
    /// sample, where Features::sampleRateShading.
    float minSampleShading = 0;
    /// Where the pipeline is looked for, and added on a miss; null is none.
    PipelineCacheHandle cache{};
    /// What a debugger or profiler shows it as, where Features::objectNames;
    /// read during the call only. Empty is none. The create call fails with
    /// RhiUnsupported for a name where Features::objectNames is false,
    /// RhiInvalidCall for one holding a '\0', which the backend reads as its
    /// end, and RhiDeviceFailed when the backend refuses it.
    std::string_view name{};
};

/// The stage a ray tracing pipeline's shader runs at. The values are the C
/// API's.
enum class RayTracingStage : std::uint8_t {
    RayGeneration = 0,
    Miss = 1,
    ClosestHit = 2,
    AnyHit = 3,
    Intersection = 4,
    Callable = 5,
};

/// One shader of a ray tracing pipeline, run at stage.
struct RayTracingShader {
    RayTracingStage stage = RayTracingStage::RayGeneration;
    ShaderHandle shader{};
};

/// What a ray tracing pipeline's group holds. The values are the C API's.
enum class RayTracingGroupType : std::uint8_t {
    /// general: one ray generation, miss or callable shader.
    General = 0,
    /// closestHit and anyHit, either kNoShader, for triangles.
    TriangleHit = 1,
    /// intersection, and closestHit and anyHit, either kNoShader, for boxes.
    BoxHit = 2,
};

/// No shader in a RayTracingGroup's slot.
inline constexpr std::uint32_t kNoShader = BFR_NO_SHADER;

/// Shaders, by index into RayTracingPipelineDesc::shaders, that a binding
/// table record names by its group's identifier.
struct RayTracingGroup {
    RayTracingGroupType type = RayTracingGroupType::General;
    std::uint32_t general = kNoShader;
    std::uint32_t closestHit = kNoShader;
    std::uint32_t anyHit = kNoShader;
    std::uint32_t intersection = kNoShader;
};

/// The shader of a group whose stack size Device::rayTracingStackSize reads.
/// The values are the C API's.
enum class RayTracingGroupShader : std::uint8_t {
    General = 0,
    ClosestHit = 1,
    AnyHit = 2,
    Intersection = 3,
};

/// How to create a ray tracing pipeline: at least one ray generation shader,
/// its own or a library's. maxPayloadSize and maxAttributeSize are the largest
/// payload and hit attributes its shaders declare, passed to backends that
/// take them. Rays trace recursively to maxRecursionDepth, at most
/// RayTracingLimits'; at 0 ray generation traces none. The spans are read
/// during the call only.
///
/// With library, it is a pipeline library, where
/// Features::rayTracingPipelineLibrary: never bound, only linked into
/// pipelines that name it in libraries. A linked library's shaders and groups
/// follow the pipeline's own, in the order libraries names them, so group
/// indices count on past its groups; each library was made with the same
/// maxRecursionDepth, maxPayloadSize and maxAttributeSize, and with
/// captureReplay where the pipeline is.
struct RayTracingPipelineDesc {
    std::span<const RayTracingShader> shaders{};
    std::span<const RayTracingGroup> groups{};
    std::uint32_t maxRecursionDepth = 0;
    std::uint32_t maxPayloadSize = 0;
    std::uint32_t maxAttributeSize = 0;
    /// Where the pipeline is looked for, and added on a miss; null is none.
    PipelineCacheHandle cache{};
    /// The bytes of stack its traces take, large enough for every trace it
    /// makes (Device::rayTracingStackSize gives each shader's); 0 leaves the
    /// backend's own size.
    std::uint64_t stackSize = 0;
    bool library = false;
    std::span<const PipelineHandle> libraries{};
    /// Its groups' identifiers may be captured, by
    /// Device::captureReplayIdentifiers, and replayed, where
    /// Features::rayTracingCaptureReplay.
    bool captureReplay = false;
    /// Empty, or the identifiers captureReplayIdentifiers gave for every group
    /// of groups, in order, RayTracingLimits::captureReplayIdentifierSize bytes
    /// each, of a pipeline made from the same desc: its groups then have the
    /// identifiers that pipeline's had. With captureReplay. Where
    /// Features::rayTracingCaptureReplayMixed is false, a device makes none
    /// with them unless its first ray tracing pipeline had them, and none with
    /// them while it makes one without.
    std::span<const std::byte> replayIdentifiers{};
    /// Its traces take the opacity micromaps structures hold, where
    /// Features::opacityMicromap.
    bool allowOpacityMicromaps = false;
    /// Its traces are at a time, of structures with motion, where
    /// Features::rayTracingMotion.
    bool allowMotion = false;
    /// Its traces may reach cluster bottom structures, which they must be
    /// made with to, where Features::clusterAccelerationStructure.
    bool allowClusters = false;
    /// What a debugger or profiler shows it as, where Features::objectNames;
    /// read during the call only. Empty is none. The create call fails with
    /// RhiUnsupported for a name where Features::objectNames is false,
    /// RhiInvalidCall for one holding a '\0', which the backend reads as its
    /// end, and RhiDeviceFailed when the backend refuses it.
    std::string_view name{};
};

} // namespace bfr::rhi
