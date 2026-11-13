#include "d3d12/d3d12_pipeline.h"

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_formats.h"
#include "d3d12/d3d12_pipeline_cache.h"
#include "d3d12/d3d12_pipeline_hash.h"
#include "d3d12/d3d12_result.h"
#include "rhi/pipeline_check.h"

#include <d3d12.h>
#include <windows.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

constexpr std::uint32_t kResourceRangeCount = 3;
constexpr std::uint32_t kBaseRegister = 0;
constexpr std::uint32_t kUnbounded = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kTableStart = 0;
// The descriptors are written while the GPU may read the table, as every
// bindless table is.
constexpr D3D12_DESCRIPTOR_RANGE_FLAGS kTableFlags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                                     D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
constexpr UINT8 kStencilMaskBits = std::numeric_limits<UINT8>::max();

// A range at register 0 of space, whose index 0 is descriptor offset of the
// table: its region's start.
D3D12_DESCRIPTOR_RANGE1 rangeFor(D3D12_DESCRIPTOR_RANGE_TYPE type, std::uint32_t space, std::uint32_t offset) {
    return {.RangeType = type,
            .NumDescriptors = kUnbounded,
            .BaseShaderRegister = kBaseRegister,
            .RegisterSpace = space,
            .Flags = kTableFlags,
            .OffsetInDescriptorsFromTableStart = offset};
}

core::Result<ComPtr<ID3D12DescriptorHeap>> createHeap(ID3D12Device& device, D3D12_DESCRIPTOR_HEAP_TYPE type,
                                                      std::uint32_t count) {
    const D3D12_DESCRIPTOR_HEAP_DESC desc{.Type = type,
                                          .NumDescriptors = count,
                                          .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                                          .NodeMask = 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (const HRESULT result = device.CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12Device::CreateDescriptorHeap", .result = result});
    }
    return heap;
}

D3D12_SHADER_BYTECODE bytecodeFor(const ShaderCode& code) {
    return {.pShaderBytecode = code.code.data(), .BytecodeLength = code.code.size()};
}

// The class of topology, which the PSO fixes; IASetPrimitiveTopology sets the
// exact one. A fan is a triangle topology (D3D_PRIMITIVE_TOPOLOGY_TRIANGLEFAN
// draws triangles, d3dcommon.h:134); NVRHI calls fans unsupported
// (d3d12-constants.cpp:186-188).
D3D12_PRIMITIVE_TOPOLOGY_TYPE topologyTypeFor(Topology topology) {
    switch (topology) {
        case Topology::PointList:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
        case Topology::LineList:
        case Topology::LineStrip:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        case Topology::TriangleList:
        case Topology::TriangleStrip:
        case Topology::TriangleFan:
        default:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }
}

// D3D12 fixes the all-ones index that restarts a strip in the PSO, by index
// type.
D3D12_INDEX_BUFFER_STRIP_CUT_VALUE stripCutFor(PrimitiveRestart restart) {
    switch (restart) {
        case PrimitiveRestart::Index16:
            return D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
        case PrimitiveRestart::Index32:
            return D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
        case PrimitiveRestart::Off:
        default:
            return D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
    }
}

D3D12_CULL_MODE cullFor(CullMode cull) {
    switch (cull) {
        case CullMode::Front:
            return D3D12_CULL_MODE_FRONT;
        case CullMode::Back:
            return D3D12_CULL_MODE_BACK;
        case CullMode::None:
        default:
            return D3D12_CULL_MODE_NONE;
    }
}

} // namespace

D3D12_COMPARISON_FUNC compareFor(CompareOp op) {
    switch (op) {
        case CompareOp::Never:
            return D3D12_COMPARISON_FUNC_NEVER;
        case CompareOp::Less:
            return D3D12_COMPARISON_FUNC_LESS;
        case CompareOp::Equal:
            return D3D12_COMPARISON_FUNC_EQUAL;
        case CompareOp::LessEqual:
            return D3D12_COMPARISON_FUNC_LESS_EQUAL;
        case CompareOp::Greater:
            return D3D12_COMPARISON_FUNC_GREATER;
        case CompareOp::NotEqual:
            return D3D12_COMPARISON_FUNC_NOT_EQUAL;
        case CompareOp::GreaterEqual:
            return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        case CompareOp::Always:
        default:
            return D3D12_COMPARISON_FUNC_ALWAYS;
    }
}

namespace {

D3D12_STENCIL_OP stencilFor(StencilOp op) {
    switch (op) {
        case StencilOp::Zero:
            return D3D12_STENCIL_OP_ZERO;
        case StencilOp::Replace:
            return D3D12_STENCIL_OP_REPLACE;
        case StencilOp::IncrementClamp:
            return D3D12_STENCIL_OP_INCR_SAT;
        case StencilOp::DecrementClamp:
            return D3D12_STENCIL_OP_DECR_SAT;
        case StencilOp::Invert:
            return D3D12_STENCIL_OP_INVERT;
        case StencilOp::IncrementWrap:
            return D3D12_STENCIL_OP_INCR;
        case StencilOp::DecrementWrap:
            return D3D12_STENCIL_OP_DECR;
        case StencilOp::Keep:
        default:
            return D3D12_STENCIL_OP_KEEP;
    }
}

D3D12_DEPTH_STENCILOP_DESC stencilFaceFor(const StencilFace& face) {
    return {.StencilFailOp = stencilFor(face.fail),
            .StencilDepthFailOp = stencilFor(face.depthFail),
            .StencilPassOp = stencilFor(face.pass),
            .StencilFunc = compareFor(face.compare)};
}

D3D12_BLEND blendFactorFor(BlendFactor factor) {
    switch (factor) {
        case BlendFactor::One:
            return D3D12_BLEND_ONE;
        case BlendFactor::SrcColor:
            return D3D12_BLEND_SRC_COLOR;
        case BlendFactor::OneMinusSrcColor:
            return D3D12_BLEND_INV_SRC_COLOR;
        case BlendFactor::DstColor:
            return D3D12_BLEND_DEST_COLOR;
        case BlendFactor::OneMinusDstColor:
            return D3D12_BLEND_INV_DEST_COLOR;
        case BlendFactor::SrcAlpha:
            return D3D12_BLEND_SRC_ALPHA;
        case BlendFactor::OneMinusSrcAlpha:
            return D3D12_BLEND_INV_SRC_ALPHA;
        case BlendFactor::DstAlpha:
            return D3D12_BLEND_DEST_ALPHA;
        case BlendFactor::OneMinusDstAlpha:
            return D3D12_BLEND_INV_DEST_ALPHA;
        case BlendFactor::ConstantColor:
            return D3D12_BLEND_BLEND_FACTOR;
        case BlendFactor::OneMinusConstantColor:
            return D3D12_BLEND_INV_BLEND_FACTOR;
        case BlendFactor::ConstantAlpha:
            return D3D12_BLEND_ALPHA_FACTOR;
        case BlendFactor::OneMinusConstantAlpha:
            return D3D12_BLEND_INV_ALPHA_FACTOR;
        case BlendFactor::SrcAlphaSaturate:
            return D3D12_BLEND_SRC_ALPHA_SAT;
        case BlendFactor::Zero:
        default:
            return D3D12_BLEND_ZERO;
    }
}

D3D12_BLEND_OP blendOpFor(BlendOp op) {
    switch (op) {
        case BlendOp::Subtract:
            return D3D12_BLEND_OP_SUBTRACT;
        case BlendOp::ReverseSubtract:
            return D3D12_BLEND_OP_REV_SUBTRACT;
        case BlendOp::Min:
            return D3D12_BLEND_OP_MIN;
        case BlendOp::Max:
            return D3D12_BLEND_OP_MAX;
        case BlendOp::Add:
        default:
            return D3D12_BLEND_OP_ADD;
    }
}

// The write mask's bits are the API's, so it passes through.
static_assert(D3D12_COLOR_WRITE_ENABLE_RED == kWriteRed && D3D12_COLOR_WRITE_ENABLE_GREEN == kWriteGreen &&
              D3D12_COLOR_WRITE_ENABLE_BLUE == kWriteBlue && D3D12_COLOR_WRITE_ENABLE_ALPHA == kWriteAlpha);

D3D12_RENDER_TARGET_BLEND_DESC blendFor(const BlendState& blend) {
    return {.BlendEnable = blend.enable ? TRUE : FALSE,
            .LogicOpEnable = FALSE,
            .SrcBlend = blendFactorFor(blend.srcColor),
            .DestBlend = blendFactorFor(blend.dstColor),
            .BlendOp = blendOpFor(blend.colorOp),
            .SrcBlendAlpha = blendFactorFor(blend.srcAlpha),
            .DestBlendAlpha = blendFactorFor(blend.dstAlpha),
            .BlendOpAlpha = blendOpFor(blend.alphaOp),
            .LogicOp = D3D12_LOGIC_OP_NOOP,
            .RenderTargetWriteMask = static_cast<UINT8>(blend.writeMask & D3D12_COLOR_WRITE_ENABLE_ALL)};
}

} // namespace

core::Result<D3D12BindlessLayout> createD3D12BindlessLayout(ID3D12Device& device, const DeviceDesc& desc,
                                                            std::uint32_t viewHeapSize) {
    D3D12BindlessLayout layout;
    layout.storageTextures = desc.maxBindlessTextures;
    layout.storageBuffers = desc.maxBindlessTextures + desc.maxBindlessStorageTextures;
    layout.descriptorBytes = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    layout.samplerBytes = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    const std::array<D3D12_DESCRIPTOR_RANGE1, kResourceRangeCount> resourceRanges{
        rangeFor(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kSampledTextureSpace, kTableStart),
        rangeFor(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kStorageTextureSpace, layout.storageTextures),
        rangeFor(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kStorageBufferSpace, layout.storageBuffers),
    };
    const D3D12_DESCRIPTOR_RANGE1 samplerRange = rangeFor(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, kSamplerSpace,
                                                          kTableStart);
    std::array<D3D12_ROOT_PARAMETER1, kRootParameterCount> parameters{};
    parameters.at(kPushConstantsParameter).ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters.at(kPushConstantsParameter).Constants = {.ShaderRegister = kBaseRegister,
                                                        .RegisterSpace = kPushConstantsSpace,
                                                        .Num32BitValues = kPushConstantValues};
    parameters.at(kPushConstantsParameter).ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters.at(kResourceTableParameter).ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters.at(kResourceTableParameter).DescriptorTable = {.NumDescriptorRanges = kResourceRangeCount,
                                                              .pDescriptorRanges = resourceRanges.data()};
    parameters.at(kResourceTableParameter).ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters.at(kSamplerTableParameter).ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters.at(kSamplerTableParameter).DescriptorTable = {.NumDescriptorRanges = 1,
                                                             .pDescriptorRanges = &samplerRange};
    parameters.at(kSamplerTableParameter).ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC signature{};
    signature.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    signature.Desc_1_1 = {.NumParameters = kRootParameterCount,
                          .pParameters = parameters.data(),
                          .NumStaticSamplers = 0,
                          .pStaticSamplers = nullptr,
                          .Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> errors;
    if (const HRESULT result = D3D12SerializeVersionedRootSignature(&signature, &serialized, &errors); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "D3D12SerializeVersionedRootSignature", .result = result});
    }
    if (const HRESULT result = device.CreateRootSignature(0, serialized->GetBufferPointer(),
                                                          serialized->GetBufferSize(),
                                                          IID_PPV_ARGS(&layout.rootSignature));
        FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12Device::CreateRootSignature", .result = result});
    }
    PipelineHash signatureHash;
    signatureHash.add(
        std::span{static_cast<const std::byte*>(serialized->GetBufferPointer()), serialized->GetBufferSize()});
    layout.signatureHash = signatureHash.value();
    // The three resource tables share one heap, which has a size of its own
    // beyond each table's.
    const std::uint64_t resourceCount = std::uint64_t{desc.maxBindlessTextures} + desc.maxBindlessStorageTextures +
                                        desc.maxBindlessStorageBuffers;
    if (resourceCount > viewHeapSize) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the bindless tables hold {} textures, storage textures and storage buffers together; "
                             "the device's shader-visible heap holds at most {}",
                             resourceCount, viewHeapSize);
    }
    // Tables of 0 slots are none, so a heap with no descriptors is not made.
    if (resourceCount != 0) {
        auto resources = createHeap(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                    static_cast<std::uint32_t>(resourceCount));
        if (!resources) {
            return std::unexpected(std::move(resources.error()));
        }
        layout.resources = std::move(*resources);
    }
    if (desc.maxBindlessSamplers != 0) {
        auto samplers = createHeap(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, desc.maxBindlessSamplers);
        if (!samplers) {
            return std::unexpected(std::move(samplers.error()));
        }
        layout.samplers = std::move(*samplers);
    }
    return layout;
}

core::Result<D3D12Pipeline> createD3D12ComputePipeline(ID3D12Device& device, const D3D12BindlessLayout& layout,
                                                       std::shared_ptr<D3D12PipelineLibrary> library,
                                                       const ComputePipelineDesc& desc) {
    const D3D12_COMPUTE_PIPELINE_STATE_DESC info{.pRootSignature = layout.rootSignature.Get(),
                                                 .CS = bytecodeFor(desc.compute),
                                                 .NodeMask = 0,
                                                 .CachedPSO = {},
                                                 .Flags = D3D12_PIPELINE_STATE_FLAG_NONE};
    auto state = pipelineStateFor(device, library.get(), info, layout.signatureHash);
    if (!state) {
        return std::unexpected(std::move(state.error()));
    }
    return D3D12Pipeline{.library = std::move(library),
                         .state = std::move(*state),
                         .compute = true,
                         .topology = Topology::TriangleList};
}

core::Result<D3D12Pipeline> createD3D12GraphicsPipeline(ID3D12Device& device, const D3D12BindlessLayout& layout,
                                                        std::shared_ptr<D3D12PipelineLibrary> library,
                                                        const GraphicsPipelineDesc& desc) {
    const StencilFace& front = desc.depthStencil.front;
    const StencilFace& back = desc.depthStencil.back;
    // Only the masks' low 8 bits count, so only they need be the same.
    const bool sameMasks = (front.readMask & kStencilMaskBits) == (back.readMask & kStencilMaskBits) &&
                           (front.writeMask & kStencilMaskBits) == (back.writeMask & kStencilMaskBits);
    if (desc.depthStencil.stencilTest && !sameMasks) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the adapter masks both stencil faces the same way");
    }
    // D3D12_RASTERIZER_DESC holds the bias as an INT, so a fraction, or a
    // value past an INT, would be cut off. D3D12_RASTERIZER_DESC1 holds it as
    // a FLOAT, but only in a pipeline state stream (a RASTERIZER1 subobject),
    // on devices whose Options16 DynamicDepthBiasSupported is true
    // (d3dx12_pipeline_state_stream.h). Moving to it would take that call and
    // a caps feature for it; until then a fraction is refused.
    const auto depthBias = static_cast<double>(desc.raster.depthBias);
    if (std::trunc(depthBias) != depthBias || depthBias < std::numeric_limits<INT>::min() ||
        depthBias > std::numeric_limits<INT>::max()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "depth bias {} is not a whole number, which D3D12's rasterizer state holds", depthBias);
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC info{};
    info.pRootSignature = layout.rootSignature.Get();
    info.VS = bytecodeFor(desc.vertex);
    info.PS = bytecodeFor(desc.fragment);
    info.BlendState.AlphaToCoverageEnable = desc.alphaToCoverage ? TRUE : FALSE;
    info.BlendState.IndependentBlendEnable = TRUE;
    const std::span targets{desc.colorTargets.data(), desc.colorTargetCount};
    const std::span blends{info.BlendState.RenderTarget};
    const std::span formats{info.RTVFormats};
    for (std::size_t i = 0; i < targets.size(); ++i) {
        blends[i] = blendFor(targets[i].blend);
        formats[i] = dxgiFormatsFor(targets[i].format).texture;
    }
    // A UINT (d3d12.h:2289) holds every sample: D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT is 32 (d3d12.h:1141).
    info.SampleMask = static_cast<UINT>(desc.sampleMask);
    info.RasterizerState = {.FillMode = desc.raster.fill == FillMode::Wireframe ? D3D12_FILL_MODE_WIREFRAME
                                                                                : D3D12_FILL_MODE_SOLID,
                            .CullMode = cullFor(desc.raster.cull),
                            .FrontCounterClockwise = desc.raster.frontFace == FrontFace::CounterClockwise ? TRUE
                                                                                                          : FALSE,
                            .DepthBias = static_cast<INT>(depthBias),
                            .DepthBiasClamp = desc.raster.depthBiasClamp,
                            .SlopeScaledDepthBias = desc.raster.depthBiasSlope,
                            .DepthClipEnable = desc.raster.depthClamp ? FALSE : TRUE,
                            .MultisampleEnable = desc.samples > 1 ? TRUE : FALSE,
                            .AntialiasedLineEnable = FALSE,
                            .ForcedSampleCount = 0,
                            .ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF};
    info.DepthStencilState = {.DepthEnable = desc.depthStencil.depthTest ? TRUE : FALSE,
                              .DepthWriteMask = desc.depthStencil.depthWrite ? D3D12_DEPTH_WRITE_MASK_ALL
                                                                             : D3D12_DEPTH_WRITE_MASK_ZERO,
                              .DepthFunc = compareFor(desc.depthStencil.depthCompare),
                              .StencilEnable = desc.depthStencil.stencilTest ? TRUE : FALSE,
                              .StencilReadMask = static_cast<UINT8>(front.readMask & kStencilMaskBits),
                              .StencilWriteMask = static_cast<UINT8>(front.writeMask & kStencilMaskBits),
                              .FrontFace = stencilFaceFor(front),
                              .BackFace = stencilFaceFor(back)};
    // Vertices are pulled, so there is no input layout.
    info.InputLayout = {.pInputElementDescs = nullptr, .NumElements = 0};
    info.IBStripCutValue = stripCutFor(desc.primitiveRestart);
    info.PrimitiveTopologyType = topologyTypeFor(desc.topology);
    info.NumRenderTargets = desc.colorTargetCount;
    if (desc.depthStencilFormat != Format::Undefined) {
        info.DSVFormat = dxgiFormatsFor(desc.depthStencilFormat).depth;
    }
    info.SampleDesc = {.Count = desc.samples, .Quality = 0};
    auto state = pipelineStateFor(device, library.get(), info, layout.signatureHash);
    if (!state) {
        return std::unexpected(std::move(state.error()));
    }
    return D3D12Pipeline{.library = std::move(library),
                         .state = std::move(*state),
                         .compute = false,
                         .topology = desc.topology};
}

// The device's pipeline calls, beside the pipelines they make.

core::Result<D3D12Pipeline*> D3D12Device::pipelineFor(PipelineHandle pipeline) {
    D3D12Pipeline* found = pipelines_.find(pipeline);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    return found;
}

core::Result<PipelineHandle> D3D12Device::createComputePipeline(const ComputePipelineDesc& desc) {
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    auto library = pipelineLibraryFor(desc.cache);
    if (!library) {
        return std::unexpected(std::move(library.error()));
    }
    auto made = createD3D12ComputePipeline(*device_.Get(), layout_, std::move(*library), desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines",
                             desc_.maxPipelines);
    }
    return pipeline;
}

core::Result<PipelineHandle> D3D12Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkGraphicsPipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkMultisample(desc, features_); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkMeshStages(desc, features_); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Features::meshShader and taskShader are false until BFR has D3D12
    // mesh pipelines, so checkMeshStages refuses every mesh desc.
    collectRetired();
    auto library = pipelineLibraryFor(desc.cache);
    if (!library) {
        return std::unexpected(std::move(library.error()));
    }
    auto made = createD3D12GraphicsPipeline(*device_.Get(), layout_, std::move(*library), desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines",
                             desc_.maxPipelines);
    }
    return pipeline;
}

core::Result<void> D3D12Device::destroyPipeline(PipelineHandle pipeline) {
    if (pipeline.isNull()) {
        return {};
    }
    if (pipelines_.find(pipeline) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!pipelines_.retire(pipeline, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    collectRetired();
    return {};
}

} // namespace bfr::rhi::d3d12
