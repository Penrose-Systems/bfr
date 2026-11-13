#include "bfr/core/error.h"
#include "bfr/rhi/debug_label.h"
#include "d3d12/d3d12_device.h"

#include <cstdint>

namespace bfr::rhi::d3d12 {

// D3D12's debug labels are not made yet, so Features::debugLabels and
// queueDebugLabels are false.

core::Result<void> D3D12Device::beginDebugLabel(CommandListHandle /*commandList*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "D3D12 records no debug labels yet (Features::debugLabels)");
}

core::Result<void> D3D12Device::endDebugLabel(CommandListHandle /*commandList*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "D3D12 records no debug labels yet (Features::debugLabels)");
}

core::Result<void> D3D12Device::insertDebugLabel(CommandListHandle /*commandList*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "D3D12 records no debug labels yet (Features::debugLabels)");
}

core::Result<void> D3D12Device::beginQueueDebugLabel(std::uint32_t /*queue*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported,
                         "D3D12 labels no queue's work yet (Features::queueDebugLabels)");
}

core::Result<void> D3D12Device::endQueueDebugLabel(std::uint32_t /*queue*/) {
    return core::failure(core::ErrorCode::RhiUnsupported,
                         "D3D12 labels no queue's work yet (Features::queueDebugLabels)");
}

core::Result<void> D3D12Device::insertQueueDebugLabel(std::uint32_t /*queue*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported,
                         "D3D12 labels no queue's work yet (Features::queueDebugLabels)");
}

} // namespace bfr::rhi::d3d12
