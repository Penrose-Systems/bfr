#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/enum_check.h"

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc holds what BFR can't pass on: an enum
/// value or format BFR doesn't have, or more than kMaxColorTargets targets.
/// A color target of format Undefined is an unused one. Each backend's
/// createGraphicsPipeline calls it first.
[[nodiscard]] inline core::Result<void> checkGraphicsPipelineDesc(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return checked;
    }
    if (desc.colorTargetCount > kMaxColorTargets) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} color targets; a pipeline renders to at most {}",
                             desc.colorTargetCount, kMaxColorTargets);
    }
    return {};
}

} // namespace bfr::rhi
