#include "d3d12/d3d12_pipeline_cache.h"

#include "bfr/core/error.h"
#include "bfr/rhi/pipeline.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_pipeline_hash.h"
#include "d3d12/d3d12_result.h"
#include "rhi/name_check.h"

#include <d3d12.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bfr::rhi::d3d12 {

using Microsoft::WRL::ComPtr;

namespace {

// Hex digits of a 64-bit hash, which a pipeline's name holds in full.
constexpr int kNameDigits = 16;

std::span<const std::byte> bytesOf(const D3D12_SHADER_BYTECODE& code) {
    return {static_cast<const std::byte*>(code.pShaderBytecode), code.BytecodeLength};
}

// Field by field, since the structs' padding holds no set value.
void addStencilFace(PipelineHash& hash, const D3D12_DEPTH_STENCILOP_DESC& face) {
    hash.add(face.StencilFailOp);
    hash.add(face.StencilDepthFailOp);
    hash.add(face.StencilPassOp);
    hash.add(face.StencilFunc);
}

std::wstring nameOf(const D3D12_COMPUTE_PIPELINE_STATE_DESC& info, std::uint64_t signature) {
    PipelineHash hash;
    hash.add(signature);
    hash.add(info.CS.BytecodeLength);
    hash.add(bytesOf(info.CS));
    hash.add(info.NodeMask);
    hash.add(info.Flags);
    return std::format(L"bfr-compute-{:0{}x}", hash.value(), kNameDigits);
}

std::wstring nameOf(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& info, std::uint64_t signature) {
    PipelineHash hash;
    hash.add(signature);
    for (const D3D12_SHADER_BYTECODE& code : {info.VS, info.PS, info.DS, info.HS, info.GS}) {
        // Each length first, so stages cut at another point hash apart.
        hash.add(code.BytecodeLength);
        hash.add(bytesOf(code));
    }
    hash.add(info.BlendState.AlphaToCoverageEnable);
    hash.add(info.BlendState.IndependentBlendEnable);
    for (const D3D12_RENDER_TARGET_BLEND_DESC& target : std::span{info.BlendState.RenderTarget}) {
        hash.add(target.BlendEnable);
        hash.add(target.LogicOpEnable);
        hash.add(target.SrcBlend);
        hash.add(target.DestBlend);
        hash.add(target.BlendOp);
        hash.add(target.SrcBlendAlpha);
        hash.add(target.DestBlendAlpha);
        hash.add(target.BlendOpAlpha);
        hash.add(target.LogicOp);
        hash.add(target.RenderTargetWriteMask);
    }
    hash.add(info.SampleMask);
    const D3D12_RASTERIZER_DESC& raster = info.RasterizerState;
    hash.add(raster.FillMode);
    hash.add(raster.CullMode);
    hash.add(raster.FrontCounterClockwise);
    hash.add(raster.DepthBias);
    hash.add(raster.DepthBiasClamp);
    hash.add(raster.SlopeScaledDepthBias);
    hash.add(raster.DepthClipEnable);
    hash.add(raster.MultisampleEnable);
    hash.add(raster.AntialiasedLineEnable);
    hash.add(raster.ForcedSampleCount);
    hash.add(raster.ConservativeRaster);
    const D3D12_DEPTH_STENCIL_DESC& depth = info.DepthStencilState;
    hash.add(depth.DepthEnable);
    hash.add(depth.DepthWriteMask);
    hash.add(depth.DepthFunc);
    hash.add(depth.StencilEnable);
    hash.add(depth.StencilReadMask);
    hash.add(depth.StencilWriteMask);
    addStencilFace(hash, depth.FrontFace);
    addStencilFace(hash, depth.BackFace);
    hash.add(info.IBStripCutValue);
    hash.add(info.PrimitiveTopologyType);
    hash.add(info.NumRenderTargets);
    for (const DXGI_FORMAT format : std::span{info.RTVFormats}) {
        hash.add(format);
    }
    hash.add(info.DSVFormat);
    hash.add(info.SampleDesc.Count);
    hash.add(info.SampleDesc.Quality);
    hash.add(info.NodeMask);
    hash.add(info.Flags);
    return std::format(L"bfr-graphics-{:0{}x}", hash.value(), kNameDigits);
}

// Loads info's PSO from library by name, or creates it and stores it there;
// an E_INVALIDARG load is a miss: no pipeline of that name, or one of another
// desc (ID3D12PipelineLibrary::LoadGraphicsPipeline).
template <class Info, class Load, class Create>
core::Result<ComPtr<ID3D12PipelineState>> loadedOrCreated(D3D12PipelineLibrary* library, const Info& info,
                                                          std::uint64_t signature, std::string_view loadCall,
                                                          std::string_view createCall, const Load& load,
                                                          const Create& create) {
    ComPtr<ID3D12PipelineState> state;
    if (library == nullptr) {
        if (const HRESULT result = create(state); FAILED(result)) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = createCall, .result = result});
        }
        return state;
    }
    const std::wstring name = nameOf(info, signature);
    // True when the library held it; false on a miss.
    const auto loadedFrom = [&]() -> core::Result<bool> {
        const HRESULT loaded = load(*library->library.Get(), name.c_str(), state);
        if (loaded == E_INVALIDARG) {
            return false;
        }
        if (FAILED(loaded)) {
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = loadCall, .result = loaded});
        }
        return true;
    };
    {
        const std::scoped_lock lock{library->lock};
        auto found = loadedFrom();
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        if (*found) {
            return state;
        }
    }
    if (const HRESULT result = create(state); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = createCall, .result = result});
    }
    const std::scoped_lock lock{library->lock};
    // Another thread may have stored it while this one compiled; its PSO is
    // the one the library holds.
    ComPtr<ID3D12PipelineState> created = std::move(state);
    auto found = loadedFrom();
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (*found) {
        return state;
    }
    if (const HRESULT result = library->library->StorePipeline(name.c_str(), created.Get()); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12PipelineLibrary::StorePipeline", .result = result});
    }
    return created;
}

} // namespace

core::Result<D3D12PipelineCache> createD3D12PipelineCache(ID3D12Device1& device, std::span<const std::byte> data) {
    auto library = std::make_shared<D3D12PipelineLibrary>();
    library->bytes.assign(data.begin(), data.end());
    if (const HRESULT result = device.CreatePipelineLibrary(library->bytes.data(), library->bytes.size(),
                                                            IID_PPV_ARGS(&library->library));
        FAILED(result)) {
        return failure(result == DXGI_ERROR_UNSUPPORTED ? core::ErrorCode::RhiUnsupported
                                                        : core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12Device1::CreatePipelineLibrary", .result = result});
    }
    return D3D12PipelineCache{.library = std::move(library)};
}

core::Result<ComPtr<ID3D12PipelineState>> pipelineStateFor(ID3D12Device& device, D3D12PipelineLibrary* library,
                                                           const D3D12_COMPUTE_PIPELINE_STATE_DESC& info,
                                                           std::uint64_t signature) {
    return loadedOrCreated(
        library, info, signature, "ID3D12PipelineLibrary::LoadComputePipeline",
        "ID3D12Device::CreateComputePipelineState",
        [&](ID3D12PipelineLibrary& from, const wchar_t* name, ComPtr<ID3D12PipelineState>& state) {
            return from.LoadComputePipeline(name, &info, IID_PPV_ARGS(state.ReleaseAndGetAddressOf()));
        },
        [&](ComPtr<ID3D12PipelineState>& state) {
            return device.CreateComputePipelineState(&info, IID_PPV_ARGS(state.ReleaseAndGetAddressOf()));
        });
}

core::Result<ComPtr<ID3D12PipelineState>> pipelineStateFor(ID3D12Device& device, D3D12PipelineLibrary* library,
                                                           const D3D12_GRAPHICS_PIPELINE_STATE_DESC& info,
                                                           std::uint64_t signature) {
    return loadedOrCreated(
        library, info, signature, "ID3D12PipelineLibrary::LoadGraphicsPipeline",
        "ID3D12Device::CreateGraphicsPipelineState",
        [&](ID3D12PipelineLibrary& from, const wchar_t* name, ComPtr<ID3D12PipelineState>& state) {
            return from.LoadGraphicsPipeline(name, &info, IID_PPV_ARGS(state.ReleaseAndGetAddressOf()));
        },
        [&](ComPtr<ID3D12PipelineState>& state) {
            return device.CreateGraphicsPipelineState(&info, IID_PPV_ARGS(state.ReleaseAndGetAddressOf()));
        });
}

// The device's pipeline cache calls.

core::Result<PipelineCacheHandle> D3D12Device::createPipelineCache(const PipelineCacheDesc& desc) {
    if (auto checked = checkObjectName(features_, desc.name); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (device1_.Get() == nullptr) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the runtime has no ID3D12Device1");
    }
    auto made = createD3D12PipelineCache(*device1_.Get(), desc.data);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const PipelineCacheHandle cache = pipelineCaches_.emplace(std::move(*made));
    if (cache.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} pipeline caches",
                             desc_.maxPipelineCaches);
    }
    return cache;
}

core::Result<std::size_t> D3D12Device::pipelineCacheSize(PipelineCacheHandle cache) {
    const D3D12PipelineCache* held = pipelineCaches_.find(cache);
    if (held == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    const std::scoped_lock lock{held->library->lock};
    return held->library->library->GetSerializedSize();
}

core::Result<PipelineCacheWrite> D3D12Device::pipelineCacheData(PipelineCacheHandle cache, std::span<std::byte> data) {
    const D3D12PipelineCache* held = pipelineCaches_.find(cache);
    if (held == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    // Held so no store lands between the size and the write.
    const std::scoped_lock lock{held->library->lock};
    const std::size_t size = held->library->library->GetSerializedSize();
    if (const HRESULT result = held->library->library->Serialize(data.data(), data.size()); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "ID3D12PipelineLibrary::Serialize", .result = result});
    }
    // Serialize writes the whole library or fails.
    return PipelineCacheWrite{.written = size, .complete = true};
}

core::Result<void> D3D12Device::destroyPipelineCache(PipelineCacheHandle cache) {
    if (cache.isNull()) {
        return {};
    }
    // Pipelines loaded from the library share it and its bytes, so the cache
    // goes at once.
    if (!pipelineCaches_.erase(cache)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    return {};
}

core::Result<std::shared_ptr<D3D12PipelineLibrary>> D3D12Device::pipelineLibraryFor(PipelineCacheHandle cache) {
    if (cache.isNull()) {
        return nullptr;
    }
    const D3D12PipelineCache* held = pipelineCaches_.find(cache);
    if (held == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the pipeline cache handle is stale");
    }
    return held->library;
}

} // namespace bfr::rhi::d3d12
