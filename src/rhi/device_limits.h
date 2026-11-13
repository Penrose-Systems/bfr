#pragma once

#include "bfr/core/error.h"
#include "bfr/core/handle.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "rhi/enum_check.h"

#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when a maximum in desc is more than a handle can
/// name, so a device holds exactly what it was asked for, 0 being none, when
/// its graphics, compute and copy queues together are more than a uint32_t
/// counts, or when a queue's level is not one BFR has. Each backend's
/// createDevice calls it first.
[[nodiscard]] inline core::Result<void> checkLimits(const DeviceDesc& desc) {
    constexpr std::uint32_t kMost = core::Handle<void>::kMaxIndices;
    for (const auto& [name, value] :
         {std::pair{"maxCommandLists", desc.maxCommandLists},
          std::pair{"maxCommandPools", desc.maxCommandPools},
          std::pair{"maxSwapchains", desc.maxSwapchains},
          std::pair{"maxBindlessTextures", desc.maxBindlessTextures},
          std::pair{"maxBindlessStorageTextures", desc.maxBindlessStorageTextures},
          std::pair{"maxBindlessStorageBuffers", desc.maxBindlessStorageBuffers},
          std::pair{"maxBindlessSamplers", desc.maxBindlessSamplers},
          std::pair{"maxBindlessUniformBuffers", desc.maxBindlessUniformBuffers},
          std::pair{"maxTimestampPools", desc.maxTimestampPools},
          std::pair{"maxQueryPools", desc.maxQueryPools},
          std::pair{"maxSparseMemories", desc.maxSparseMemories},
          std::pair{"maxTileMappings", desc.maxTileMappings},
          std::pair{"maxBindlessAccelerationStructures", desc.maxBindlessAccelerationStructures},
          std::pair{"maxBindlessPartitionedTops", desc.maxBindlessPartitionedTops},
          std::pair{"maxBuildGeometries", desc.maxBuildGeometries},
          std::pair{"maxAccelerationStructurePropertyPools", desc.maxAccelerationStructurePropertyPools},
          std::pair{"maxBuffers", desc.maxBuffers},
          std::pair{"maxTextures", desc.maxTextures},
          std::pair{"maxShaders", desc.maxShaders},
          std::pair{"maxPipelines", desc.maxPipelines},
          std::pair{"maxPipelineCaches", desc.maxPipelineCaches},
          std::pair{"maxAccelerationStructures", desc.maxAccelerationStructures}}) {
        if (value > kMost) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "{} is {}; it is at most {}", name, value, kMost);
        }
    }
    const std::uint64_t queues = std::uint64_t{desc.graphics.size()} + desc.compute.size() + desc.copy.size();
    if (!std::in_range<std::uint32_t>(queues)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} graphics, {} compute and {} copy queues are more than BFR counts",
                             desc.graphics.size(), desc.compute.size(), desc.copy.size());
    }
    for (const std::span<const QueueDesc> kind : {desc.graphics, desc.compute, desc.copy}) {
        for (const QueueDesc& queue : kind) {
            if (auto checked = checkKnown(queue.level, "queue level"); !checked) {
                return checked;
            }
        }
    }
    return {};
}

/// Fails with RhiLimitReached when desc asks for acceleration structure,
/// uniform buffer or partitioned top structure slots on an adapter with no
/// such table, as caps
/// say: the backend has nothing to pass them to. Each backend's createDevice
/// calls it.
[[nodiscard]] inline core::Result<void> checkTables(const DeviceDesc& desc, const Caps& caps) {
    if (desc.maxBindlessAccelerationStructures != 0 && caps.bindless.maxAccelerationStructures == 0) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the adapter holds no acceleration structure table (BindlessLimits::"
                             "maxAccelerationStructures is 0); {} slots were asked for",
                             desc.maxBindlessAccelerationStructures);
    }
    if (desc.maxBindlessUniformBuffers != 0 && caps.bindless.maxUniformBuffers == 0) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the adapter holds no uniform buffer table (BindlessLimits::maxUniformBuffers is 0); {} "
                             "slots were asked for",
                             desc.maxBindlessUniformBuffers);
    }
    if (desc.maxBindlessPartitionedTops != 0 && !caps.features.partitionedTop) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the adapter holds no partitioned top structure table (Features::partitionedTop is "
                             "false); {} slots were asked for",
                             desc.maxBindlessPartitionedTops);
    }
    return {};
}

/// How many queues desc opens, which checkLimits has seen a uint32_t counts.
[[nodiscard]] inline std::uint32_t queueCount(const DeviceDesc& desc) {
    return static_cast<std::uint32_t>(desc.graphics.size() + desc.compute.size() + desc.copy.size());
}

/// The part of desc a device keeps: its spans point into the caller's memory,
/// which lives only for createDevice.
[[nodiscard]] inline DeviceDesc heldPart(DeviceDesc desc) {
    desc.graphics = {};
    desc.compute = {};
    desc.copy = {};
    desc.indirectStrides = {};
    return desc;
}

/// Fails with RhiUnsupported, naming backend, when desc's queues have
/// different ranks, for a backend that doesn't rank a device's queues.
[[nodiscard]] inline core::Result<void> checkUnranked(const DeviceDesc& desc, std::string_view backend) {
    std::optional<float> first;
    for (const std::span<const QueueDesc> kind : {desc.graphics, desc.compute, desc.copy}) {
        for (const QueueDesc& queue : kind) {
            if (!first) {
                first = queue.rank;
            }
            if (queue.rank != *first) {
                return core::failure(core::ErrorCode::RhiUnsupported,
                                     "{} doesn't rank a device's queues; they were given ranks {} and {}", backend,
                                     *first, queue.rank);
            }
        }
    }
    return {};
}

/// Fails with RhiUnsupported, saying why, when a queue in desc asks for a
/// level other than normal, for a backend with no level to set.
[[nodiscard]] inline core::Result<void> checkNormalLevels(const DeviceDesc& desc, std::string_view why) {
    for (const std::span<const QueueDesc> kind : {desc.graphics, desc.compute, desc.copy}) {
        for (const QueueDesc& queue : kind) {
            if (queue.level != QueueLevel::Normal) {
                return core::failure(core::ErrorCode::RhiUnsupported, "a queue asks for level {}, but {}",
                                     std::to_underlying(queue.level), why);
            }
        }
    }
    return {};
}

} // namespace bfr::rhi
