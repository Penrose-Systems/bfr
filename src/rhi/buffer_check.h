#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "rhi/enum_check.h"

namespace bfr::rhi {

/// Fails with RhiInvalidCall for memory BFR doesn't have. Each backend's
/// createBuffer calls it first.
[[nodiscard]] inline core::Result<void> checkBufferDesc(const BufferDesc& desc) {
    return checkEnums(desc);
}

} // namespace bfr::rhi
