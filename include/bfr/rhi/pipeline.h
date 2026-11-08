#pragma once

#include "bfr/bfr.h"
#include "bfr/core/handle.h"
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

/// The most color targets a graphics pipeline renders to.
inline constexpr std::uint32_t kMaxColorTargets = BFR_MAX_COLOR_TARGETS;

/// Dimensions of a workgroup.
inline constexpr std::size_t kWorkgroupDimensions = 3;

/// One shader stage: bytecode for the backend the device runs on, and the
/// name of the function in it to run. Both are borrowed for the call.
struct ShaderCode {
    std::span<const std::byte> code;
    std::string_view entryPoint;
};

/// How to create a compute pipeline. workgroupSize is the threads in each
/// workgroup, as the shader declares them.
struct ComputePipelineDesc {
    ShaderCode compute;
    std::array<std::uint32_t, kWorkgroupDimensions> workgroupSize{};
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
};

} // namespace bfr::rhi
