#pragma once

#include "bfr/bfr.h"
#include "bfr/bfr_enums.hpp" // IWYU pragma: export
#include "bfr/rhi/texture.h"

#include <span>

namespace bfr::rhi {

// A barrier's stages and accesses are laid out as bfr.h's, each a bfr_bool
// read as set when it is not BFR_FALSE, so the C API passes the caller's
// barriers to Device::barrier as they are, copying and allocating nothing;
// C writes any non-zero value for true, which a C++ bool may not hold.

/// The work a barrier waits on, as its before scope, or holds back, as its
/// after scope, as api/bfr.yaml states under bfr_stages: how each maps to each
/// backend, and which fail where. None set is no work.
struct Stages {
    bfr_bool all = BFR_FALSE;
    bfr_bool drawIndirect = BFR_FALSE;
    bfr_bool indexInput = BFR_FALSE;
    bfr_bool vertexShader = BFR_FALSE;
    bfr_bool taskShader = BFR_FALSE;
    bfr_bool meshShader = BFR_FALSE;
    bfr_bool fragmentShader = BFR_FALSE;
    bfr_bool earlyFragmentTests = BFR_FALSE;
    bfr_bool lateFragmentTests = BFR_FALSE;
    bfr_bool colorOutput = BFR_FALSE;
    bfr_bool computeShader = BFR_FALSE;
    bfr_bool copy = BFR_FALSE;
    bfr_bool resolve = BFR_FALSE;
    bfr_bool rayTracingShader = BFR_FALSE;
    bfr_bool accelerationStructureBuild = BFR_FALSE;
    bfr_bool accelerationStructureCopy = BFR_FALSE;
    bfr_bool accelerationStructureProperties = BFR_FALSE;
};

/// The memory accesses of a barrier's stages, as api/bfr.yaml states under
/// bfr_accesses: before it, those whose writes it makes visible; after it,
/// those that see them. Each needs one of its stages in the same scope.
struct Accesses {
    bfr_bool indirectRead = BFR_FALSE;
    bfr_bool indexRead = BFR_FALSE;
    bfr_bool uniformRead = BFR_FALSE;
    bfr_bool sampledRead = BFR_FALSE;
    bfr_bool storageRead = BFR_FALSE;
    bfr_bool storageWrite = BFR_FALSE;
    bfr_bool colorTargetRead = BFR_FALSE;
    bfr_bool colorTargetWrite = BFR_FALSE;
    bfr_bool depthStencilRead = BFR_FALSE;
    bfr_bool depthStencilWrite = BFR_FALSE;
    bfr_bool copyRead = BFR_FALSE;
    bfr_bool copyWrite = BFR_FALSE;
    bfr_bool resolveRead = BFR_FALSE;
    bfr_bool resolveWrite = BFR_FALSE;
    bfr_bool accelerationStructureInputRead = BFR_FALSE;
    bfr_bool accelerationStructureRead = BFR_FALSE;
    bfr_bool accelerationStructureWrite = BFR_FALSE;
    bfr_bool shaderBindingTableRead = BFR_FALSE;
};

/// One side of a barrier.
struct BarrierScope {
    Stages stages;
    Accesses accesses;
};

/// A barrier over all memory, buffers and textures alike, that moves no
/// texture's layout.
struct GlobalBarrier {
    BarrierScope before;
    BarrierScope after;
};

/// A barrier over range's mip levels and layers of texture that moves them
/// from layoutBefore to layoutAfter; the rest stay as they are. What each
/// backend refuses of it is in api/bfr.yaml under bfr_texture_barrier.
struct TextureBarrier {
    TextureHandle texture;
    BarrierScope before;
    BarrierScope after;
    TextureLayout layoutBefore = TextureLayout::Undefined;
    TextureLayout layoutAfter = TextureLayout::Undefined;
    TextureRange range;
};

/// The barriers one Device::barrier records, read during the call only.
struct BarrierDesc {
    std::span<const GlobalBarrier> globals;
    std::span<const TextureBarrier> textures;
};

} // namespace bfr::rhi
