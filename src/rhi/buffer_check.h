#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "rhi/enum_check.h"

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc asks for a buffer no backend can make:
/// one of no bytes, of memory BFR doesn't have, or one nothing uses. Each backend's createBuffer calls it
/// first, so every backend answers such a desc the same way.
[[nodiscard]] inline core::Result<void> checkBufferDesc(const BufferDesc& desc) {
    if (desc.size == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer has no bytes; size must be 1 or more");
    }
    if (auto checked = checkEnums(desc); !checked) {
        return checked;
    }
    const BufferUsage& usage = desc.usage;
    if (!(usage.copySrc || usage.copyDst || usage.uniform || usage.storage || usage.index || usage.indirect ||
          usage.accelerationStructureInput || usage.accelerationStructureScratch || usage.shaderBindingTable)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer has no usage; set every use it will have");
    }
    return {};
}

} // namespace bfr::rhi
