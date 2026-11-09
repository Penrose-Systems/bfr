#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/caps.h"
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

/// Fails with RhiInvalidCall for a property BFR doesn't have, and
/// RhiUnsupported for one whose feature the device lacks.
[[nodiscard]] inline core::Result<void> checkProperty(AccelerationStructureProperty property,
                                                      const Features& features) {
    if (auto checked = checkKnown(property, "acceleration structure property"); !checked) {
        return checked;
    }
    const bool has =
        property == AccelerationStructureProperty::CompactedSize ||
        (property == AccelerationStructureProperty::SerializedSize && features.accelerationStructureSerialization) ||
        (property == AccelerationStructureProperty::SerializedBottomCount && features.serializedBottomCount) ||
        (property == AccelerationStructureProperty::CurrentSize && features.accelerationStructureCurrentSize);
    if (!has) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device writes no acceleration structure property {}, as its Features say",
                             static_cast<std::uint32_t>(property));
    }
    return {};
}

/// Fails with RhiUnsupported where the device builds no acceleration
/// structures from counts the GPU reads, as
/// Features::accelerationStructureIndirectBuild says.
[[nodiscard]] inline core::Result<void> checkIndirectBuilds(bool indirectBuild) {
    if (!indirectBuild) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no indirect acceleration structure builds "
                             "(Features::accelerationStructureIndirectBuild)");
    }
    return {};
}

/// Fails with RhiUnsupported where the device serializes no acceleration
/// structures, as Features::accelerationStructureSerialization says.
[[nodiscard]] inline core::Result<void> checkSerialization(bool serialization) {
    if (!serialization) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device serializes no acceleration structures "
                                                              "(Features::accelerationStructureSerialization)");
    }
    return {};
}

/// Fails with RhiInvalidCall when structures from first are more than the
/// pool's count entries hold.
[[nodiscard]] inline core::Result<void> checkPropertyRange(std::uint32_t first, std::size_t structures,
                                                           std::uint32_t count) {
    if (first > count || structures > count - first) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} structures from entry {} are more than a pool of {} holds", structures, first, count);
    }
    return {};
}

/// Fails with RhiUnsupported for an opacity micromap, or triangles that
/// take their opacity from one, where Features::opacityMicromap is false.
[[nodiscard]] inline core::Result<void> checkOpacityMicromaps(bool opacityMicromap, bool named) {
    if (named && !opacityMicromap) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no opacity micromaps (Features::opacityMicromap)");
    }
    return {};
}

/// Whether triangles take their opacity from a micromap.
[[nodiscard]] inline bool hasOpacityMicromap(const TriangleGeometry& triangles) {
    return !triangles.opacityMicromap.micromap.isNull() || !triangles.opacityMicromap.indices.isNull();
}

/// Fails with RhiInvalidCall when inputs hold an enum value BFR doesn't have,
/// a bottom structure of both triangles and boxes, or triangles naming
/// micromap usages past micromapUsages, which BFR reads; and with
/// RhiUnsupported for opacity micromaps where Features::opacityMicromap is
/// false.
[[nodiscard]] inline core::Result<void> checkInputs(const AccelerationStructureInputs& inputs, bool opacityMicromap) {
    if (auto checked = checkEnums(inputs); !checked) {
        return checked;
    }
    if (inputs.type == AccelerationStructureType::Bottom && !inputs.triangles.empty() && !inputs.boxes.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a bottom acceleration structure holds triangles or boxes, not both");
    }
    // The micromap build flags are the extension's (VUID-VkAccelerationStructureBuildGeometryInfoKHR-flags-11710).
    const AccelerationStructureFlags& flags = inputs.flags;
    const bool micromapFlags = flags.allowOpacityMicromapUpdate || flags.allowDisableOpacityMicromaps ||
                               flags.allowOpacityMicromapDataUpdate;
    if (auto checked = checkOpacityMicromaps(opacityMicromap,
                                             inputs.type == AccelerationStructureType::OpacityMicromap ||
                                                 micromapFlags);
        !checked) {
        return checked;
    }
    for (const TriangleGeometry& triangles : inputs.triangles) {
        if (!hasOpacityMicromap(triangles)) {
            continue;
        }
        if (auto checked = checkOpacityMicromaps(opacityMicromap, true); !checked) {
            return checked;
        }
        const TriangleOpacityMicromap& named = triangles.opacityMicromap;
        if (named.firstUsage > inputs.micromapUsages.size() ||
            named.usageCount > inputs.micromapUsages.size() - named.firstUsage) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "triangles name {} micromap usages from {}, past the inputs' {}", named.usageCount,
                                 named.firstUsage, inputs.micromapUsages.size());
        }
    }
    return {};
}

/// The geometries a structure built from inputs holds: a top one's instances
/// are one.
[[nodiscard]] inline std::size_t geometriesOf(const AccelerationStructureInputs& inputs) {
    switch (inputs.type) {
        case AccelerationStructureType::Top:
            return 1;
        case AccelerationStructureType::OpacityMicromap:
            return 0;
        case AccelerationStructureType::Bottom:
        default:
            return inputs.triangles.size() + inputs.boxes.size();
    }
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

/// Fails with RhiInvalidCall when an instance has an index or offset
/// kInstanceIndexLimit or above or an opacity BFR doesn't have, and
/// RhiUnsupported for its opacity micromap flags where
/// Features::opacityMicromap is false.
[[nodiscard]] inline core::Result<void> checkInstance(const AccelerationStructureInstance& instance,
                                                      bool opacityMicromap) {
    if (instance.customIndex >= kInstanceIndexLimit || instance.hitGroupOffset >= kInstanceIndexLimit) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "an instance's custom index {} and hit group offset {} are below {}", instance.customIndex,
                             instance.hitGroupOffset, kInstanceIndexLimit);
    }
    if (auto checked = checkOpacityMicromaps(opacityMicromap,
                                             instance.forceOpacityMicromap2State || instance.disableOpacityMicromaps);
        !checked) {
        return checked;
    }
    return checkKnown(instance.opacity, "instance opacity");
}

/// Fails with RhiInvalidCall when out holds fewer than instances' records of
/// instanceSize bytes, and as checkInstance does.
[[nodiscard]] inline core::Result<void> checkInstances(std::span<const AccelerationStructureInstance> instances,
                                                       std::span<const std::byte> out, std::uint32_t instanceSize,
                                                       bool opacityMicromap) {
    if (out.size() / instanceSize < instances.size()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes hold fewer than {} instance records of {}",
                             out.size(), instances.size(), instanceSize);
    }
    for (const AccelerationStructureInstance& instance : instances) {
        if (auto checked = checkInstance(instance, opacityMicromap); !checked) {
            return checked;
        }
    }
    return {};
}

/// Fails with RhiUnsupported where the device has no motion, as
/// Features::rayTracingMotion says.
[[nodiscard]] inline core::Result<void> checkRayTracingMotion(bool rayTracingMotion) {
    if (!rayTracingMotion) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no structures with motion (Features::rayTracingMotion)");
    }
    return {};
}

/// Fails as checkRayTracingMotion does for motionKeyframes other than 0.
[[nodiscard]] inline core::Result<void> checkMotion(bool rayTracingMotion, std::uint32_t motionKeyframes) {
    return motionKeyframes == 0 ? core::Result<void>{} : checkRayTracingMotion(rayTracingMotion);
}

/// Fails as checkInstances does, with RhiUnsupported where
/// Features::rayTracingMotion is false, and RhiInvalidCall for out too small
/// for the records stride bytes apart, the last only recordSize bytes, or for
/// a motion BFR doesn't have or, for a moving instance, keyframes past
/// keyframes, which BFR reads.
[[nodiscard]] inline core::Result<void> checkMotionInstances(
    bool rayTracingMotion, std::span<const AccelerationStructureMotionInstance> instances,
    std::span<const MotionKeyframe> keyframes, std::span<const std::byte> out, std::uint32_t stride,
    std::size_t recordSize, bool opacityMicromap) {
    if (auto checked = checkRayTracingMotion(rayTracingMotion); !checked) {
        return checked;
    }
    if (!instances.empty() && (out.size() < recordSize || (out.size() - recordSize) / stride < instances.size() - 1)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} bytes hold fewer than {} motion instance records {} apart, the last of {}", out.size(),
                             instances.size(), stride, recordSize);
    }
    for (const AccelerationStructureMotionInstance& moving : instances) {
        if (auto checked = checkInstance(moving.instance, opacityMicromap); !checked) {
            return checked;
        }
        if (auto checked = checkKnown(moving.motion, "instance motion"); !checked) {
            return checked;
        }
        if (moving.motion != InstanceMotion::None && (moving.firstKeyframe > keyframes.size() ||
                                                      moving.keyframeCount > keyframes.size() - moving.firstKeyframe)) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "{} keyframes from {} run past the {} given",
                                 moving.keyframeCount, moving.firstKeyframe, keyframes.size());
        }
    }
    return {};
}

} // namespace bfr::rhi
