#include "bfr/core/error.h"
#include "bfr/rhi/query.h"
#include "d3d12/d3d12_device.h"

#include <cstdint>

namespace bfr::rhi::d3d12 {

// D3D12's queries are not made yet, so Features::queries is false: no pool is
// made, and every handle a call is given is one this device never made.

core::Result<QueryPoolHandle> D3D12Device::createQueryPool(const QueryPoolDesc& /*desc*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "D3D12 makes no query pools yet (Features::queries)");
}

core::Result<void> D3D12Device::destroyQueryPool(QueryPoolHandle pool) {
    if (pool.isNull()) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<QueryLayout> D3D12Device::queryLayout(QueryPoolHandle /*pool*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> D3D12Device::resetQueries(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                             std::uint32_t /*first*/, std::uint32_t /*count*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> D3D12Device::beginQuery(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                           std::uint32_t /*index*/, bool /*precise*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> D3D12Device::endQuery(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                         std::uint32_t /*index*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

core::Result<void> D3D12Device::resolveQueries(CommandListHandle /*commandList*/, QueryPoolHandle /*pool*/,
                                               std::uint32_t /*first*/, std::uint32_t /*count*/,
                                               BufferHandle /*buffer*/, std::uint64_t /*offset*/) {
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the query pool handle is stale");
}

} // namespace bfr::rhi::d3d12
