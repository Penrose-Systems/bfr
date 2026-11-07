#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "metal/metal_buffer.h"
#include "metal/metal_util.h"

#include <Metal/Metal.hpp>

#include <array>
#include <cstdint>

namespace bfr::rhi::metal {

/// Where the shaders find the tables and the push constants, which
/// api/bfr.yaml states under bfr_pipeline: the argument buffer holding the
/// four tables is at buffer(0), the push constants at buffer(1).
inline constexpr std::uint32_t kTablesBufferIndex = 0;
inline constexpr std::uint32_t kPushConstantsBufferIndex = 1;
/// Bytes of each push constant entry in a pool's ring. Metal has no limit to
/// query; 128 is what NVRHI and Godot use.
inline constexpr std::uint32_t kPushConstantsSize = 128;

/// Bytes each table entry takes: a resource ID or a GPU address.
inline constexpr std::uint64_t kTableEntryBytes = 8;

/// op as Metal's, for depth and stencil tests and compare samplers.
[[nodiscard]] MTL::CompareFunction compareFor(CompareOp op);

/// The one resource layout every pipeline shares: one argument buffer holding
/// the sampled textures, storage textures, storage buffers and samplers, in
/// that order, each table sized as the DeviceDesc asked, resident while the
/// device lives. Metal 4 binds it per encoder, so no pipeline is made on it.
struct MetalBindlessTables {
    Ref<MTL::Buffer> buffer;
    /// Where each table starts in the buffer, in entries.
    std::uint32_t storageTextures = 0;
    std::uint32_t storageBuffers = 0;
    std::uint32_t samplers = 0;
    /// Entries in the buffer, every table's together.
    std::uint32_t entries = 0;
    // Declared after buffer, so it leaves the set before it is released.
    Resident resident;

    /// Points slot of the storage buffer table at address, the texture table
    /// at texture, and the storage texture table at texture: the buffer is
    /// shared, so the CPU writes the entry.
    void writeStorageBuffer(std::uint32_t slot, MTL::GPUAddress address) const;
    void writeSampledTexture(std::uint32_t slot, MTL::ResourceID texture) const;
    void writeStorageTexture(std::uint32_t slot, MTL::ResourceID texture) const;
    /// Points slot of the sampler table at sampler.
    void writeSampler(std::uint32_t slot, MTL::ResourceID sampler) const;
};

/// The buffer bindings an argument table holds: the tables and the push
/// constants.
inline constexpr std::uint32_t kArgumentBufferBindings = 2;

/// An argument table of kArgumentBufferBindings buffers, which each list binds
/// the tables and its push constants through; made with each list slot. Fails
/// with RhiDeviceFailed when Metal gives none, with what it said.
[[nodiscard]] core::Result<Ref<MTL4::ArgumentTable>> createArgumentTable(MTL::Device& device);

/// The tables on device as desc sizes them, added to residency, which the
/// caller commits under its lock. Fails with RhiDeviceFailed when Metal gives
/// no buffer.
[[nodiscard]] core::Result<MetalBindlessTables> createMetalBindlessTables(MTL::Device& device, const DeviceDesc& desc,
                                                                          Ref<MTL::ResidencySet> residency);

/// A pipeline: its Metal state, and the state Metal sets on the encoder
/// instead, kept to set when the pipeline is bound. For compute, the
/// workgroup size its desc gave, which a dispatch passes to Metal.
struct MetalPipeline {
    Ref<MTL::ComputePipelineState> compute;
    Ref<MTL::RenderPipelineState> render;
    Ref<MTL::DepthStencilState> depthStencil;
    RasterState raster;
    TopologyClass topology = TopologyClass::Triangle;
    std::array<std::uint32_t, kWorkgroupDimensions> workgroupSize{};
};

/// A compute pipeline through compiler as desc says, whose code is a Metal
/// library. Fails with RhiDeviceFailed when Metal refuses the library, has no
/// function of the entry point's name, or refuses the pipeline, with what it
/// said.
[[nodiscard]] core::Result<MetalPipeline> createMetalComputePipeline(MTL::Device& device, MTL4::Compiler& compiler,
                                                                     const ComputePipelineDesc& desc);

/// A graphics pipeline through compiler as desc says, whose code is a Metal
/// library, with no vertex descriptor: vertices are pulled. Metal takes the
/// depth-stencil format from the render pass, not the pipeline. Fails as
/// createMetalComputePipeline does, and with RhiDeviceFailed when Metal gives
/// no depth-stencil state.
[[nodiscard]] core::Result<MetalPipeline> createMetalGraphicsPipeline(MTL::Device& device, MTL4::Compiler& compiler,
                                                                      const GraphicsPipelineDesc& desc);

} // namespace bfr::rhi::metal
