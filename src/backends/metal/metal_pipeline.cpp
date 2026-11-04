#include "metal/metal_pipeline.h"

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "metal/metal_buffer.h"
#include "metal/metal_device.h"
#include "metal/metal_formats.h"
#include "metal/metal_util.h"
#include "rhi/pipeline_check.h"

#include <dispatch/dispatch.h>
#include <Foundation/NSError.hpp>
#include <Foundation/NSString.hpp>
#include <Metal/MTL4ArgumentTable.hpp>
#include <Metal/MTL4Compiler.hpp>
#include <Metal/MTL4ComputePipeline.hpp>
#include <Metal/MTL4LibraryFunctionDescriptor.hpp>
#include <Metal/MTL4PipelineState.hpp>
#include <Metal/MTL4RenderPipeline.hpp>
#include <Metal/MTLComputePipeline.hpp>
#include <Metal/MTLDepthStencil.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLGPUAddress.hpp>
#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLRenderPipeline.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <Metal/MTLResource.hpp>
#include <Metal/MTLTypes.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>

namespace bfr::rhi::metal {

namespace {

// What error said, for a message; Metal may give none.
const char* reasonOf(const NS::Error* error) {
    return error != nullptr ? error->localizedDescription()->utf8String() : "no reason given";
}

// The functions of one stage: the library its code makes, and the descriptor
// naming the entry point in it, which a pipeline descriptor points at.
struct Stage {
    Ref<MTL::Library> library;
    Ref<MTL4::LibraryFunctionDescriptor> function;
};

// The stage code names. Fails with RhiDeviceFailed when Metal refuses the
// library or has no function of the entry point's name.
core::Result<Stage> stageFor(MTL::Device& device, const ShaderCode& code) {
    // dispatch copies the bytes, so the caller's are read during the call only.
    dispatch_data_t data = dispatch_data_create(code.code.data(), code.code.size(), nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error* error = nullptr;
    MTL::Library* library = device.newLibrary(data, &error);
    dispatch_release(data);
    if (library == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal refused the shader library: {}", reasonOf(error));
    }
    Stage stage{.library = adopt(library), .function = {}};
    const std::string entryPoint{code.entryPoint};
    NS::String* name = NS::String::string(entryPoint.c_str(), NS::UTF8StringEncoding);
    const Ref<MTL::Function> found = adopt(library->newFunction(name));
    if (!found) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "the shader library has no function {}", entryPoint);
    }
    stage.function = adopt(MTL4::LibraryFunctionDescriptor::alloc()->init());
    stage.function->setLibrary(library);
    stage.function->setName(name);
    return stage;
}

MTL::PrimitiveTopologyClass topologyFor(TopologyClass topology) {
    switch (topology) {
        case TopologyClass::Point:
            return MTL::PrimitiveTopologyClassPoint;
        case TopologyClass::Line:
            return MTL::PrimitiveTopologyClassLine;
        case TopologyClass::Triangle:
        default:
            return MTL::PrimitiveTopologyClassTriangle;
    }
}

MTL::CompareFunction compareFor(CompareOp op) {
    switch (op) {
        case CompareOp::Never:
            return MTL::CompareFunctionNever;
        case CompareOp::Less:
            return MTL::CompareFunctionLess;
        case CompareOp::Equal:
            return MTL::CompareFunctionEqual;
        case CompareOp::LessEqual:
            return MTL::CompareFunctionLessEqual;
        case CompareOp::Greater:
            return MTL::CompareFunctionGreater;
        case CompareOp::NotEqual:
            return MTL::CompareFunctionNotEqual;
        case CompareOp::GreaterEqual:
            return MTL::CompareFunctionGreaterEqual;
        case CompareOp::Always:
        default:
            return MTL::CompareFunctionAlways;
    }
}

MTL::StencilOperation stencilFor(StencilOp op) {
    switch (op) {
        case StencilOp::Zero:
            return MTL::StencilOperationZero;
        case StencilOp::Replace:
            return MTL::StencilOperationReplace;
        case StencilOp::IncrementClamp:
            return MTL::StencilOperationIncrementClamp;
        case StencilOp::DecrementClamp:
            return MTL::StencilOperationDecrementClamp;
        case StencilOp::Invert:
            return MTL::StencilOperationInvert;
        case StencilOp::IncrementWrap:
            return MTL::StencilOperationIncrementWrap;
        case StencilOp::DecrementWrap:
            return MTL::StencilOperationDecrementWrap;
        case StencilOp::Keep:
        default:
            return MTL::StencilOperationKeep;
    }
}

Ref<MTL::StencilDescriptor> stencilFaceFor(const StencilFace& face) {
    Ref<MTL::StencilDescriptor> stencil = adopt(MTL::StencilDescriptor::alloc()->init());
    stencil->setStencilFailureOperation(stencilFor(face.fail));
    stencil->setDepthStencilPassOperation(stencilFor(face.pass));
    stencil->setDepthFailureOperation(stencilFor(face.depthFail));
    stencil->setStencilCompareFunction(compareFor(face.compare));
    stencil->setReadMask(face.readMask);
    stencil->setWriteMask(face.writeMask);
    return stencil;
}

// Metal sets depth and stencil testing on the encoder, as one state object.
// Fails with RhiDeviceFailed when Metal gives none.
core::Result<Ref<MTL::DepthStencilState>> depthStencilFor(MTL::Device& device, const DepthStencilState& state) {
    const Ref<MTL::DepthStencilDescriptor> desc = adopt(MTL::DepthStencilDescriptor::alloc()->init());
    desc->setDepthCompareFunction(state.depthTest ? compareFor(state.depthCompare) : MTL::CompareFunctionAlways);
    desc->setDepthWriteEnabled(state.depthTest && state.depthWrite);
    if (state.stencilTest) {
        desc->setFrontFaceStencil(stencilFaceFor(state.front).get());
        desc->setBackFaceStencil(stencilFaceFor(state.back).get());
    }
    MTL::DepthStencilState* made = device.newDepthStencilState(desc.get());
    if (made == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no depth-stencil state");
    }
    return adopt(made);
}

MTL::BlendFactor blendFactorFor(BlendFactor factor) {
    switch (factor) {
        case BlendFactor::One:
            return MTL::BlendFactorOne;
        case BlendFactor::SrcColor:
            return MTL::BlendFactorSourceColor;
        case BlendFactor::OneMinusSrcColor:
            return MTL::BlendFactorOneMinusSourceColor;
        case BlendFactor::DstColor:
            return MTL::BlendFactorDestinationColor;
        case BlendFactor::OneMinusDstColor:
            return MTL::BlendFactorOneMinusDestinationColor;
        case BlendFactor::SrcAlpha:
            return MTL::BlendFactorSourceAlpha;
        case BlendFactor::OneMinusSrcAlpha:
            return MTL::BlendFactorOneMinusSourceAlpha;
        case BlendFactor::DstAlpha:
            return MTL::BlendFactorDestinationAlpha;
        case BlendFactor::OneMinusDstAlpha:
            return MTL::BlendFactorOneMinusDestinationAlpha;
        case BlendFactor::ConstantColor:
            return MTL::BlendFactorBlendColor;
        case BlendFactor::OneMinusConstantColor:
            return MTL::BlendFactorOneMinusBlendColor;
        case BlendFactor::ConstantAlpha:
            return MTL::BlendFactorBlendAlpha;
        case BlendFactor::OneMinusConstantAlpha:
            return MTL::BlendFactorOneMinusBlendAlpha;
        case BlendFactor::SrcAlphaSaturate:
            return MTL::BlendFactorSourceAlphaSaturated;
        case BlendFactor::Zero:
        default:
            return MTL::BlendFactorZero;
    }
}

MTL::BlendOperation blendOpFor(BlendOp op) {
    switch (op) {
        case BlendOp::Subtract:
            return MTL::BlendOperationSubtract;
        case BlendOp::ReverseSubtract:
            return MTL::BlendOperationReverseSubtract;
        case BlendOp::Min:
            return MTL::BlendOperationMin;
        case BlendOp::Max:
            return MTL::BlendOperationMax;
        case BlendOp::Add:
        default:
            return MTL::BlendOperationAdd;
    }
}

// The API's write mask runs red to alpha from bit 0; Metal's runs the other way.
MTL::ColorWriteMask writeMaskFor(std::uint32_t mask) {
    const auto bit = [mask](std::uint32_t ours, MTL::ColorWriteMask theirs) {
        return (mask & ours) != 0 ? theirs : MTL::ColorWriteMaskNone;
    };
    return bit(kWriteRed, MTL::ColorWriteMaskRed) | bit(kWriteGreen, MTL::ColorWriteMaskGreen) |
           bit(kWriteBlue, MTL::ColorWriteMaskBlue) | bit(kWriteAlpha, MTL::ColorWriteMaskAlpha);
}

void describeTarget(MTL4::RenderPipelineColorAttachmentDescriptor& attachment, const ColorTarget& target) {
    const BlendState& blend = target.blend;
    attachment.setPixelFormat(pixelFormatFor(target.format));
    attachment.setBlendingState(blend.enable ? MTL4::BlendStateEnabled : MTL4::BlendStateDisabled);
    attachment.setSourceRGBBlendFactor(blendFactorFor(blend.srcColor));
    attachment.setDestinationRGBBlendFactor(blendFactorFor(blend.dstColor));
    attachment.setRgbBlendOperation(blendOpFor(blend.colorOp));
    attachment.setSourceAlphaBlendFactor(blendFactorFor(blend.srcAlpha));
    attachment.setDestinationAlphaBlendFactor(blendFactorFor(blend.dstAlpha));
    attachment.setAlphaBlendOperation(blendOpFor(blend.alphaOp));
    attachment.setWriteMask(writeMaskFor(blend.writeMask));
}

} // namespace

core::Result<MetalBindlessTables> createMetalBindlessTables(MTL::Device& device, const DeviceDesc& desc,
                                                            Ref<MTL::ResidencySet> residency) {
    const std::uint32_t storageTextures = desc.maxBindlessTextures;
    const std::uint32_t storageBuffers = storageTextures + desc.maxBindlessStorageTextures;
    const std::uint32_t samplers = storageBuffers + desc.maxBindlessStorageBuffers;
    const std::uint64_t entries = std::uint64_t{samplers} + desc.maxBindlessSamplers;
    // Shared, so the CPU writes entries as views are made.
    MTL::Buffer* buffer = device.newBuffer(entries * kTableEntryBytes, MTL::ResourceStorageModeShared);
    if (buffer == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no buffer for the bindless tables");
    }
    return MetalBindlessTables{.buffer = adopt(buffer),
                               .storageTextures = storageTextures,
                               .storageBuffers = storageBuffers,
                               .samplers = samplers,
                               .resident = Resident{std::move(residency), buffer}};
}

namespace {

// Where entry index of the tables is, in the shared buffer's memory.
void writeEntry(const MetalBindlessTables& tables, std::uint32_t index, std::uint64_t value) {
    auto* entries = static_cast<std::uint64_t*>(tables.buffer->contents());
    std::span{entries, tables.samplers}[index] = value;
}

} // namespace

void MetalBindlessTables::writeStorageBuffer(std::uint32_t slot, MTL::GPUAddress address) const {
    writeEntry(*this, storageBuffers + slot, address);
}

void MetalBindlessTables::writeSampledTexture(std::uint32_t slot, MTL::ResourceID texture) const {
    writeEntry(*this, slot, texture._impl);
}

void MetalBindlessTables::writeStorageTexture(std::uint32_t slot, MTL::ResourceID texture) const {
    writeEntry(*this, storageTextures + slot, texture._impl);
}

core::Result<Ref<MTL4::ArgumentTable>> createArgumentTable(MTL::Device& device) {
    const Ref<MTL4::ArgumentTableDescriptor> descriptor = adopt(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(kArgumentBufferBindings);
    NS::Error* error = nullptr;
    MTL4::ArgumentTable* table = device.newArgumentTable(descriptor.get(), &error);
    if (table == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no argument table: {}",
                             error != nullptr ? error->localizedDescription()->utf8String() : "no reason given");
    }
    return adopt(table);
}

core::Result<MetalPipeline> createMetalComputePipeline(MTL::Device& device, MTL4::Compiler& compiler,
                                                       const ComputePipelineDesc& desc) {
    const AutoreleasePool pool;
    auto stage = stageFor(device, desc.compute);
    if (!stage) {
        return std::unexpected(std::move(stage.error()));
    }
    const Ref<MTL4::ComputePipelineDescriptor> info = adopt(MTL4::ComputePipelineDescriptor::alloc()->init());
    info->setComputeFunctionDescriptor(stage->function.get());
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* state = compiler.newComputePipelineState(info.get(), nullptr, &error);
    if (state == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal refused the compute pipeline: {}",
                             reasonOf(error));
    }
    return MetalPipeline{.compute = adopt(state),
                         .render = {},
                         .depthStencil = {},
                         .raster = {},
                         .topology = TopologyClass::Triangle,
                         .workgroupSize = desc.workgroupSize};
}

core::Result<MetalPipeline> createMetalGraphicsPipeline(MTL::Device& device, MTL4::Compiler& compiler,
                                                        const GraphicsPipelineDesc& desc) {
    const AutoreleasePool pool;
    auto vertex = stageFor(device, desc.vertex);
    if (!vertex) {
        return std::unexpected(std::move(vertex.error()));
    }
    Stage fragment;
    if (!desc.fragment.code.empty()) {
        auto made = stageFor(device, desc.fragment);
        if (!made) {
            return std::unexpected(std::move(made.error()));
        }
        fragment = std::move(*made);
    }
    const Ref<MTL4::RenderPipelineDescriptor> info = adopt(MTL4::RenderPipelineDescriptor::alloc()->init());
    info->setVertexFunctionDescriptor(vertex->function.get());
    if (fragment.function) {
        info->setFragmentFunctionDescriptor(fragment.function.get());
    }
    // Vertices are pulled, so there is no vertex descriptor.
    info->setInputPrimitiveTopology(topologyFor(desc.topology));
    info->setRasterSampleCount(desc.samples);
    info->setAlphaToCoverageState(desc.alphaToCoverage ? MTL4::AlphaToCoverageStateEnabled
                                                       : MTL4::AlphaToCoverageStateDisabled);
    const std::span targets{desc.colorTargets.data(), desc.colorTargetCount};
    for (std::size_t i = 0; i < targets.size(); ++i) {
        describeTarget(*info->colorAttachments()->object(i), targets[i]);
    }
    auto depthStencil = depthStencilFor(device, desc.depthStencil);
    if (!depthStencil) {
        return std::unexpected(std::move(depthStencil.error()));
    }
    NS::Error* error = nullptr;
    MTL::RenderPipelineState* state = compiler.newRenderPipelineState(info.get(), nullptr, &error);
    if (state == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal refused the render pipeline: {}",
                             reasonOf(error));
    }
    return MetalPipeline{.compute = {},
                         .render = adopt(state),
                         .depthStencil = std::move(*depthStencil),
                         .raster = desc.raster,
                         .topology = desc.topology,
                         .workgroupSize = {}};
}

// The device's pipeline calls, beside the pipelines they make.

core::Result<MetalPipeline*> MetalDevice::pipelineFor(PipelineHandle pipeline) {
    MetalPipeline* found = pipelines_.find(pipeline);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline handle is stale");
    }
    return found;
}

core::Result<PipelineHandle> MetalDevice::createComputePipeline(const ComputePipelineDesc& desc) {
    if (auto checked = checkComputePipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectBuffers();
    auto made = createMetalComputePipeline(*device_.get(), *compiler_.get(), desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines", kBufferLimit);
    }
    return pipeline;
}

core::Result<PipelineHandle> MetalDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkGraphicsPipelineDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    collectBuffers();
    auto made = createMetalGraphicsPipeline(*device_.get(), *compiler_.get(), desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineHandle pipeline = pipelines_.emplace(std::move(*made));
    if (pipeline.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipelines", kBufferLimit);
    }
    return pipeline;
}

core::Result<void> MetalDevice::destroyPipeline(PipelineHandle pipeline) {
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
    collectBuffers();
    return {};
}

} // namespace bfr::rhi::metal
