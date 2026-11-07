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
/// and readback memory, empty for device memory. slot is its place in the
/// storage buffer table, kEnd without storage usage; the device gives it back
/// when the buffer is collected. A sparse one is a reserved resource with no
/// allocation.
struct D3D12Buffer {
    // The allocation is declared first so that the resource is released first.
    Microsoft::WRL::ComPtr<D3D12MA::Allocation> allocation;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    BufferMemory memory = BufferMemory::Device;
    std::uint64_t size = 0;
    std::span<std::byte> mapped;
    std::uint32_t slot = 0;
    bool sparse = false;
};

/// A buffer as desc says, mapped for good when it is upload or readback
/// memory, or a reserved resource on device with no memory when it is sparse,
/// holding slot. Fails with RhiInvalidCall for a sparse buffer not in device
/// memory, RhiUnsupported for a sparse tileBytes other than D3D12's tile
/// size, RhiOutOfMemory when memory runs out and RhiDeviceFailed when D3D12MA
/// or D3D12 refuses.
[[nodiscard]] core::Result<D3D12Buffer> createD3D12Buffer(D3D12MA::Allocator& allocator, ID3D12Device& device,
                                                          const BufferDesc& desc, std::uint32_t slot);

} // namespace bfr::rhi::d3d12
