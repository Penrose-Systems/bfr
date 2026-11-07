#pragma once

#include "bfr/core/error.h"
#include "core/fixed_vector.h"

#include <d3d12.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>

namespace bfr::rhi::d3d12 {

/// A heap sparse resources' tiles are mapped into, released when it goes.
/// flags are the heap's: D3D12_HEAP_FLAG_NONE where ResourceHeapTier 2 lets
/// it hold any resource, else the one D3D12_HEAP_FLAG_ALLOW_ONLY_* kind of
/// resource ResourceHeapTier 1 lets it hold.
struct D3D12SparseMemory {
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;
    D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;

    /// Whether a resource whose tier 1 heap kind is kind maps to this heap.
    [[nodiscard]] bool takes(D3D12_HEAP_FLAGS kind) const { return flags == D3D12_HEAP_FLAG_NONE || flags == kind; }
};

/// One resource's regions for ID3D12CommandQueue::UpdateTileMappings, in the
/// arrays it takes: each region is one range of the same tile count. Made at
/// the device's maxTileMappings, so mapping allocates nothing.
struct D3D12TileMappings {
    core::FixedVector<D3D12_TILED_RESOURCE_COORDINATE> coordinates;
    core::FixedVector<D3D12_TILE_REGION_SIZE> sizes;
    core::FixedVector<D3D12_TILE_RANGE_FLAGS> flags;
    core::FixedVector<UINT> starts;
    core::FixedVector<UINT> counts;

    explicit D3D12TileMappings(std::size_t capacity = 0)
        : coordinates(capacity), sizes(capacity), flags(capacity), starts(capacity), counts(capacity) {}

    void clear() {
        coordinates.clear();
        sizes.clear();
        flags.clear();
        starts.clear();
        counts.clear();
    }

    /// Adds a region from coordinate of size, mapped from heap tile start, or
    /// to no memory where map is false, when start is unused. The caller
    /// keeps to the capacity. Fails with RhiInvalidCall for a start past the
    /// UINT D3D12 takes.
    [[nodiscard]] core::Result<void> add(const D3D12_TILED_RESOURCE_COORDINATE& coordinate,
                                         const D3D12_TILE_REGION_SIZE& size, bool map, std::uint64_t start);
};

} // namespace bfr::rhi::d3d12
