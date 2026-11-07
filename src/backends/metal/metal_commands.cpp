#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "metal/metal_device.h"
#include "metal/metal_pipeline.h"
#include "metal/metal_util.h"
#include "rhi/command_check.h"
#include "rhi/enum_check.h"

#include <Foundation/NSTypes.hpp>
#include <Metal/MTL4ArgumentTable.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTL4RenderCommandEncoder.hpp>
#include <Metal/MTLArgument.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLGPUAddress.hpp>
#include <Metal/MTLRenderCommandEncoder.hpp>
#include <Metal/MTLTypes.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <utility>

namespace bfr::rhi::metal {

namespace {

// The list primitive of a pipeline's class, which the pipeline fixes.
MTL::PrimitiveType primitiveFor(TopologyClass topology) {
    switch (topology) {
        case TopologyClass::Point:
            return MTL::PrimitiveTypePoint;
        case TopologyClass::Line:
            return MTL::PrimitiveTypeLine;
        case TopologyClass::Triangle:
        default:
            return MTL::PrimitiveTypeTriangle;
    }
}

MTL::CullMode cullFor(CullMode cull) {
    switch (cull) {
        case CullMode::Front:
            return MTL::CullModeFront;
        case CullMode::Back:
            return MTL::CullModeBack;
        case CullMode::None:
        default:
            return MTL::CullModeNone;
    }
}

} // namespace

// The device's recording commands between beginRendering and a submit: the
// state draws and dispatches take, and the draws and dispatches. Each makes
// the usage checks every backend makes, then records on the encoder. Metal
// sets pipelines on encoders, so a graphics pipeline bound outside rendering
// is set on the encoder at the first draw after it.

core::Result<const MetalPipeline*> MetalDevice::applyGraphics(MetalCommandList& list) {
    // Looked up every draw: the pipeline may have been destroyed since the last.
    auto found = pipelineFor(list.graphics);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (list.encoderGraphics == list.graphics) {
        return *found;
    }
    const MetalPipeline& pipeline = **found;
    MTL4::RenderCommandEncoder& encoder = *list.encoder.get();
    encoder.setRenderPipelineState(pipeline.render.get());
    encoder.setDepthStencilState(pipeline.depthStencil.get());
    encoder.setCullMode(cullFor(pipeline.raster.cull));
    encoder.setFrontFacingWinding(pipeline.raster.frontFace == FrontFace::Clockwise ? MTL::WindingClockwise
                                                                                    : MTL::WindingCounterClockwise);
    encoder.setTriangleFillMode(pipeline.raster.fill == FillMode::Wireframe ? MTL::TriangleFillModeLines
                                                                            : MTL::TriangleFillModeFill);
    encoder.setDepthBias(pipeline.raster.depthBias, pipeline.raster.depthBiasSlope, pipeline.raster.depthBiasClamp);
    encoder.setDepthClipMode(pipeline.raster.depthClamp ? MTL::DepthClipModeClamp : MTL::DepthClipModeClip);
    list.encoderGraphics = list.graphics;
    return *found;
}

core::Result<void> MetalDevice::setViewport(CommandListHandle commandList, const Viewport& viewport) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    (*list)->encoder->setViewport({.originX = static_cast<double>(viewport.x),
                                   .originY = static_cast<double>(viewport.y),
                                   .width = static_cast<double>(viewport.width),
                                   .height = static_cast<double>(viewport.height),
                                   .znear = static_cast<double>(viewport.minDepth),
                                   .zfar = static_cast<double>(viewport.maxDepth)});
    return {};
}

core::Result<void> MetalDevice::setScissor(CommandListHandle commandList, const Rect& scissor) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    (*list)->encoder->setScissorRect(
        {.x = scissor.x, .y = scissor.y, .width = scissor.width, .height = scissor.height});
    return {};
}

core::Result<void> MetalDevice::setBlendConstants(CommandListHandle commandList, const std::array<float, 4>& color) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    const auto [red, green, blue, alpha] = color;
    (*list)->encoder->setBlendColor(red, green, blue, alpha);
    return {};
}

core::Result<void> MetalDevice::setStencilReference(CommandListHandle commandList, std::uint32_t reference) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    (*list)->encoder->setStencilReferenceValue(reference);
    return {};
}

core::Result<void> MetalDevice::bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = pipelineFor(pipeline);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*found)->compute) {
        (*list)->compute = pipeline;
    }
    else {
        (*list)->graphics = pipeline;
    }
    return {};
}

core::Result<void> MetalDevice::pushConstants(CommandListHandle commandList, std::span<const std::byte> data) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkPushConstants(data, kPushConstantsSize); !checked) {
        return checked;
    }
    MetalCommandPool* pool = commandPools_.find((*list)->pool);
    if (pool == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    const std::uint32_t entry = pool->pushed;
    if (auto counted = countPushConstants(pool->pushed, pool->pushConstantUpdates); !counted) {
        return counted;
    }
    // The next entry of the pool's ring takes the list's last push, then the
    // bytes, and the argument table points at it for the draws and
    // dispatches after this.
    const MetalListSlot& first = lists_[pool->first];
    const std::uint64_t offset = std::uint64_t{entry} * kPushConstantsSize;
    auto* const ring = static_cast<std::byte*>(first.pushRing->contents());
    if ((*list)->pushedAt) {
        std::memcpy(ring + offset, ring + *(*list)->pushedAt, kPushConstantsSize);
    }
    std::memcpy(ring + offset, data.data(), data.size());
    (*list)->pushedAt = offset;
    lists_[(*list)->slot].arguments->setAddress(first.pushRing->gpuAddress() + offset, kPushConstantsBufferIndex);
    return {};
}

core::Result<void> MetalDevice::bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
                                                std::uint64_t offset, IndexType type) {
    if (auto checked = checkKnown(type, "index type"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // Metal takes the bytes from offset to the end, which there are none of
    // past it.
    if (offset > (*found)->buffer->length()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "index offset {} is past the buffer's {} bytes", offset,
                             (*found)->buffer->length());
    }
    (*list)->indexType = type == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;
    (*list)->indexBuffer = (*found)->buffer->gpuAddress() + offset;
    (*list)->indexBytes = (*found)->buffer->length() - offset;
    return {};
}

core::Result<void> MetalDevice::draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                     std::uint32_t instanceCount, std::uint32_t firstVertex,
                                     std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    const auto pipeline = applyGraphics(**list);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    (*list)->encoder->drawPrimitives(primitiveFor((*pipeline)->topology), firstVertex, vertexCount, instanceCount,
                                     firstInstance);
    return {};
}

core::Result<void> MetalDevice::drawIndexed(CommandListHandle commandList, std::uint32_t indexCount,
                                            std::uint32_t instanceCount, std::uint32_t firstIndex,
                                            std::int32_t vertexOffset, std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    // Metal takes the first index as an offset into the buffer.
    const NS::UInteger indexBytes = (*list)->indexType == MTL::IndexTypeUInt16 ? sizeof(std::uint16_t)
                                                                               : sizeof(std::uint32_t);
    const NS::UInteger skipped = NS::UInteger{firstIndex} * indexBytes;
    if (skipped > (*list)->indexBytes) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "first index {} is past the index buffer's end",
                             firstIndex);
    }
    const auto pipeline = applyGraphics(**list);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    (*list)->encoder->drawIndexedPrimitives(primitiveFor((*pipeline)->topology), indexCount, (*list)->indexType,
                                            (*list)->indexBuffer + skipped, (*list)->indexBytes - skipped,
                                            instanceCount, vertexOffset, firstInstance);
    return {};
}

core::Result<void> MetalDevice::dispatch(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                         std::uint32_t z) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto pipeline = pipelineFor((*list)->compute);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    // One encoder per dispatch, which the pool drains the autoreleased
    // reference of; Metal 4 takes the threadgroup size at the dispatch.
    const AutoreleasePool pool;
    MTL4::ComputeCommandEncoder* encoder = (*list)->buffer->computeCommandEncoder();
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compute command encoder");
    }
    encoder->setArgumentTable(lists_[(*list)->slot].arguments.get());
    encoder->setComputePipelineState((*pipeline)->compute.get());
    const auto& workgroup = (*pipeline)->workgroupSize;
    encoder->dispatchThreadgroups(MTL::Size::Make(x, y, z), MTL::Size::Make(workgroup[0], workgroup[1], workgroup[2]));
    encoder->endEncoding();
    return {};
}

} // namespace bfr::rhi::metal
