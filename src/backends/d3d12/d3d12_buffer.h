#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"

#include <d3d12.h>
#include <D3D12MemAlloc.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstddef>
#include <span>

namespace bfr::rhi::d3d12 {

/// The allocator a device's buffers take their memory from, over device on
/// adapter. Release it once every buffer is gone. Fails with RhiDeviceFailed
/// when D3D12MA refuses.
[[nodiscard]] core::Result<Microsoft::WRL::ComPtr<D3D12MA::Allocator>> createAllocator(ID3D12Device* device,
                                                                                       IDXGIAdapter* adapter);

/// A buffer resource and the memory D3D12MA placed it in, both released when
/// it goes: before the allocator is. mapped is where the CPU reaches upload
/// and readback memory, empty for device memory.
struct D3D12Buffer {
    // The allocation is declared first so that the resource is released first.
    Microsoft::WRL::ComPtr<D3D12MA::Allocation> allocation;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    BufferMemory memory = BufferMemory::Device;
    std::span<std::byte> mapped;
};

/// A buffer as desc says, mapped for good when it is upload or readback
/// memory. Fails with RhiOutOfMemory when memory runs out and RhiDeviceFailed
/// when D3D12MA or D3D12 refuses.
[[nodiscard]] core::Result<D3D12Buffer> createD3D12Buffer(D3D12MA::Allocator& allocator, const BufferDesc& desc);

} // namespace bfr::rhi::d3d12
