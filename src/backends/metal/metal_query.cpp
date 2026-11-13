#include "bfr/core/error.h"
#include "bfr/rhi/query.h"
#include "metal/metal_device.h"

#include <cstdint>

namespace bfr::rhi::metal {

// Metal's queries are not made yet, so Features::queries is false: no pool is
// made, and every handle a call is given is one this device never made.

core::Result<QueryPoolHandle> MetalDevice::createQueryPool(const QueryPoolDesc& /*desc*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "Metal makes no query pools yet (Features::queries)");
}

core::Result<void> MetalDevice::destroyQueryPool(QueryPoolHandle pool) {
    if (pool.isNull()) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<QueryLayout> MetalDevice::queryLayout(QueryPoolHandle /*pool*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> MetalDevice::resetQueries(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                             std::uint32_t /*first*/, std::uint32_t /*count*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> MetalDevice::beginQuery(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                           std::uint32_t /*index*/, bool /*precise*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> MetalDevice::endQuery(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                         std::uint32_t /*index*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> MetalDevice::resolveQueries(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                               std::uint32_t /*first*/, std::uint32_t /*count*/,
                                               BufferHandle /*buffer*/, std::uint64_t /*offset*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

} // namespace bfr::rhi::metal
