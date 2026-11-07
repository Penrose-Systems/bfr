#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "metal/metal_buffer.h"
#include "metal/metal_util.h"

#include <Metal/MTLDevice.hpp>
#include <Metal/MTLHeap.hpp>
#include <Metal/MTLResource.hpp>

#include <array>
#include <cstdint>

namespace bfr::rhi::metal {

/// Every page size Metal makes a sparse resource with (MTLResource.hpp:59-63).
inline constexpr std::array kSparsePageSizes{MTL::SparsePageSize16, MTL::SparsePageSize64, MTL::SparsePageSize256};
static_assert(kSparsePageSizes.size() == kMaxSparseTileSizes);

/// The page size whose tiles device says are tileBytes bytes, as
/// sparseTileSizeInBytes (MTLDevice.hpp:539) gives them. Fails with
/// RhiUnsupported when none is, 0 included.
[[nodiscard]] core::Result<MTL::SparsePageSize> pageSizeFor(MTL::Device& device, std::uint32_t tileBytes);

/// Fails with RhiUnsupported when a resource made at page can't be mapped to
/// a heap whose largest compatible page is heapPage
/// (MTLHeapDescriptor, MTLHeap.hpp:56). The name says the heap takes pages up to its own; that
/// smaller ones fit, by SparsePageSize's order (MTLResource.hpp:59-63), is
/// inferred and unverified on hardware.
[[nodiscard]] core::Result<void> checkPageFits(MTL::SparsePageSize page, MTL::SparsePageSize heapPage);

/// A placement heap sparse textures' and buffers' tiles are mapped into,
/// released when the last Ref to it goes, and resident while it lives, as a
/// buffer is. page is the largest compatible page it was made with, which
/// the heap itself does not report.
struct MetalSparseMemory {
    Ref<MTL::Heap> heap;
    MTL::SparsePageSize page = MTL::SparsePageSize16;
    // Declared after heap, so it leaves the set before it is released.
    Resident resident;
};

} // namespace bfr::rhi::metal
