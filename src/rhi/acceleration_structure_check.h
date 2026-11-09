#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "rhi/enum_check.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Fails with RhiUnsupported where the device has no acceleration structures,
/// as Features::accelerationStructure says.
[[nodiscard]] inline core::Result<void> checkAccelerationStructures(bool accelerationStructure) {
    if (!accelerationStructure) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no acceleration structures (Features::accelerationStructure)");
    }
    return {};
}

/// Fails with RhiInvalidCall when inputs hold an enum value BFR doesn't have,
/// or a bottom structure of both triangles and boxes.
[[nodiscard]] inline core::Result<void> checkInputs(const AccelerationStructureInputs& inputs) {
    if (auto checked = checkEnums(inputs); !checked) {
        return checked;
    }
    if (inputs.type == AccelerationStructureType::Bottom && !inputs.triangles.empty() && !inputs.boxes.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a bottom acceleration structure holds triangles or boxes, not both");
    }
    return {};
}

/// The geometries a structure built from inputs holds: a top one's instances
/// are one.
[[nodiscard]] inline std::size_t geometriesOf(const AccelerationStructureInputs& inputs) {
    return inputs.type == AccelerationStructureType::Top ? 1 : inputs.triangles.size() + inputs.boxes.size();
}

/// Fails with RhiLimitReached when inputs' geometries are more than
/// maxBuildGeometries (DeviceDesc).
[[nodiscard]] inline core::Result<void> countGeometries(const AccelerationStructureInputs& inputs,
                                                        std::uint32_t maxBuildGeometries) {
    if (geometriesOf(inputs) > maxBuildGeometries) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "a build of {} geometries; the device takes at most {} in one", geometriesOf(inputs),
                             maxBuildGeometries);
    }
    return {};
}

/// Fails with RhiInvalidCall when out holds fewer than instances' records of
/// instanceSize bytes, or an instance has an index or offset
/// kInstanceIndexLimit or above or an opacity BFR doesn't have.
[[nodiscard]] inline core::Result<void> checkInstances(std::span<const AccelerationStructureInstance> instances,
                                                       std::span<const std::byte> out, std::uint32_t instanceSize) {
    if (out.size() / instanceSize < instances.size()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes hold fewer than {} instance records of {}",
                             out.size(), instances.size(), instanceSize);
    }
    for (const AccelerationStructureInstance& instance : instances) {
        if (instance.customIndex >= kInstanceIndexLimit || instance.hitGroupOffset >= kInstanceIndexLimit) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "an instance's custom index {} and hit group offset {} are below {}",
                                 instance.customIndex, instance.hitGroupOffset, kInstanceIndexLimit);
        }
        if (auto checked = checkKnown(instance.opacity, "instance opacity"); !checked) {
            return checked;
        }
    }
    return {};
}

} // namespace bfr::rhi
