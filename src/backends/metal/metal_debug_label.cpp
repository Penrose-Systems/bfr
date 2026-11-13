#include "bfr/core/error.h"
#include "bfr/rhi/debug_label.h"
#include "metal/metal_device.h"

#include <cstdint>

namespace bfr::rhi::metal {

// Metal's debug labels are not made yet, so Features::debugLabels and
// queueDebugLabels are false.

core::Result<void> MetalDevice::beginDebugLabel(CommandListHandle /*commandList*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "Metal records no debug labels yet (Features::debugLabels)");
}

core::Result<void> MetalDevice::endDebugLabel(CommandListHandle /*commandList*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "Metal records no debug labels yet (Features::debugLabels)");
}

core::Result<void> MetalDevice::insertDebugLabel(CommandListHandle /*commandList*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "Metal records no debug labels yet (Features::debugLabels)");
}

core::Result<void> MetalDevice::beginQueueDebugLabel(std::uint32_t /*queue*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported,
                         "Metal labels no queue's work yet (Features::queueDebugLabels)");
}

core::Result<void> MetalDevice::endQueueDebugLabel(std::uint32_t /*queue*/) {
    return core::failure(core::ErrorCode::RhiUnsupported,
                         "Metal labels no queue's work yet (Features::queueDebugLabels)");
}

core::Result<void> MetalDevice::insertQueueDebugLabel(std::uint32_t /*queue*/, const DebugLabel& /*label*/) {
    return core::failure(core::ErrorCode::RhiUnsupported,
                         "Metal labels no queue's work yet (Features::queueDebugLabels)");
}

} // namespace bfr::rhi::metal
