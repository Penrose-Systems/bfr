#pragma once

#include "bfr/core/error.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace bfr::rhi::d3d12 {

/// An ID3D12PipelineLibrary with the bytes it was made from, which "must
/// remain valid for the lifetime of the object returned ... the data is not
/// copied" (ID3D12Device1::CreatePipelineLibrary, docs/research.md:2258-2266).
/// lock is held across each look-up and each store: "multiple threads loading
/// the same PSO ... should synchronize themselves" (the same page), and
/// StorePipeline fails "if the name already exists"
/// (ID3D12PipelineLibrary::StorePipeline).
struct D3D12PipelineLibrary {
    std::vector<std::byte> bytes;
    Microsoft::WRL::ComPtr<ID3D12PipelineLibrary> library;
    std::mutex lock;
};

/// A pipeline cache. The library is shared with every pipeline loaded from
/// it, since no D3D12 source read says whether such a PSO keeps its library
/// alive, so destroying the cache never pulls it from under one.
struct D3D12PipelineCache {
    std::shared_ptr<D3D12PipelineLibrary> library;
};

/// A cache that starts with a copy of data. Fails with RhiUnsupported when
/// D3D12 answers DXGI_ERROR_UNSUPPORTED, and RhiDeviceFailed when it refuses
/// the bytes, as it does for another driver's or adapter's.
[[nodiscard]] core::Result<D3D12PipelineCache> createD3D12PipelineCache(ID3D12Device1& device,
                                                                        std::span<const std::byte> data);

/// The PSO info says, loaded from library where a pipeline of the same name
/// is in it, else created and stored in it; created alone where library is
/// null. signature names the root signature info is on. The create runs
/// outside the library's lock, so pipelines compile on several threads at
/// once. Fails with RhiDeviceFailed, with what D3D12 said, when it refuses
/// the create, a load other than a miss, or the store, which a name already
/// stored for another desc fails.
[[nodiscard]] core::Result<Microsoft::WRL::ComPtr<ID3D12PipelineState>> pipelineStateFor(
    ID3D12Device& device, D3D12PipelineLibrary* library, const D3D12_COMPUTE_PIPELINE_STATE_DESC& info,
    std::uint64_t signature);
[[nodiscard]] core::Result<Microsoft::WRL::ComPtr<ID3D12PipelineState>> pipelineStateFor(
    ID3D12Device& device, D3D12PipelineLibrary* library, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& info,
    std::uint64_t signature);

} // namespace bfr::rhi::d3d12
