#pragma once

#include "bfr/rhi/caps.h"
#include "bfr/rhi/instance.h"

#include <d3d12.h>
#include <dxgi1_6.h>

#include <string>

namespace bfr::rhi::d3d12 {

/// Why device is below the floor, empty when it meets it. The floor is
/// bindless resource binding, shader model 6.6 and enhanced barriers.
[[nodiscard]] std::string belowFloor(ID3D12Device* device);

/// What kind of adapter desc describes.
[[nodiscard]] AdapterType typeOf(const DXGI_ADAPTER_DESC1& desc);

/// The adapter's name, in UTF-8.
[[nodiscard]] std::string nameOf(const DXGI_ADAPTER_DESC1& desc);

/// Everything fixed about the adapter desc describes, as device, made on it,
/// answers. A question the device won't answer leaves its fields at 0 or false.
[[nodiscard]] Caps readCaps(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device);

} // namespace bfr::rhi::d3d12
