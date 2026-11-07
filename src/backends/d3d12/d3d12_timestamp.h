#pragma once

#include <d3d12.h>
#include <wrl/client.h>

namespace bfr::rhi::d3d12 {

/// A query heap of timestamps, released when this goes.
struct D3D12TimestampPool {
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> heap;
};

} // namespace bfr::rhi::d3d12
