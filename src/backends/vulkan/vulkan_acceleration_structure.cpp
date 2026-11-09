#include "vulkan/vulkan_acceleration_structure.h"

#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "core/slot_chains.h"
#include "rhi/acceleration_structure_check.h"
#include "rhi/bindless_slots.h"
#include "rhi/enum_check.h"
#include "vulkan/vulkan_buffer.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <mutex>
#include <optional>
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

VulkanAccelerationStructure::VulkanAccelerationStructure(const VolkDeviceTable& table, VkDevice device,
                                                         VulkanBuffer storage, VkAccelerationStructureKHR structure,
                                                         VkMicromapEXT micromap, const AccelerationStructureDesc& desc,
                                                         std::uint32_t slot, VkDeviceAddress reference)
    : table_(&table),
      device_(device),
      storage_(std::move(storage)),
      structure_(structure),
      micromap_(micromap),
      type_(desc.type),
      size_(desc.size),
      slot_(slot),
      reference_(reference) {}

VulkanAccelerationStructure::VulkanAccelerationStructure(VulkanAccelerationStructure&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      storage_(std::move(other.storage_)),
      structure_(std::exchange(other.structure_, VK_NULL_HANDLE)),
      micromap_(std::exchange(other.micromap_, VK_NULL_HANDLE)),
      type_(other.type_),
      size_(other.size_),
      slot_(other.slot_),
      reference_(other.reference_) {}

VulkanAccelerationStructure::~VulkanAccelerationStructure() {
    if (structure_ != VK_NULL_HANDLE) {
        table_->vkDestroyAccelerationStructureKHR(device_, structure_, nullptr);
    }
    if (micromap_ != VK_NULL_HANDLE) {
        table_->vkDestroyMicromapEXT(device_, micromap_, nullptr);
    }
}

namespace {

// What the size query is given for a buffer that is named: it reads only
// whether a transform is null (resources.adoc:8821-8826), and no address.
constexpr VkDeviceAddress kNamedAddress = 1;

// The bits an instance's custom index and hit group offset take.
constexpr std::uint32_t kIndexBits = kInstanceIndexLimit - 1;

VkAccelerationStructureTypeKHR typeFor(AccelerationStructureType type) {
    return type == AccelerationStructureType::Top ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                                                  : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
}

VkBuildAccelerationStructureFlagsKHR flagsFor(const AccelerationStructureFlags& flags) {
    VkBuildAccelerationStructureFlagsKHR vulkan = 0;
    if (flags.preference == BuildPreference::FastTrace) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    }
    else if (flags.preference == BuildPreference::FastBuild) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    }
    if (flags.allowUpdate) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    if (flags.allowCompaction) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }
    if (flags.allowDataAccess) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_BIT_KHR;
    }
    if (flags.minimizeMemory) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
    }
    if (flags.allowOpacityMicromapUpdate) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_UPDATE_BIT_EXT;
    }
    if (flags.allowDisableOpacityMicromaps) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISABLE_OPACITY_MICROMAPS_BIT_EXT;
    }
    if (flags.allowOpacityMicromapDataUpdate) {
        vulkan |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_DATA_UPDATE_BIT_EXT;
    }
    return vulkan;
}

// A micromap build takes the preference and compaction of flags; the rest
// are a bottom structure's.
VkBuildMicromapFlagsEXT micromapFlagsFor(const AccelerationStructureFlags& flags) {
    VkBuildMicromapFlagsEXT vulkan = 0;
    if (flags.preference == BuildPreference::FastTrace) {
        vulkan |= VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
    }
    else if (flags.preference == BuildPreference::FastBuild) {
        vulkan |= VK_BUILD_MICROMAP_PREFER_FAST_BUILD_BIT_EXT;
    }
    if (flags.allowCompaction) {
        vulkan |= VK_BUILD_MICROMAP_ALLOW_COMPACTION_BIT_EXT;
    }
    return vulkan;
}

// usages as Vulkan reads them, in place: the layouts and format values are
// the same.
const VkMicromapUsageEXT* usagesOf(std::span<const OpacityMicromapUsage> usages) {
    static_assert(sizeof(OpacityMicromapUsage) == sizeof(VkMicromapUsageEXT) &&
                  offsetof(OpacityMicromapUsage, subdivisionLevel) == offsetof(VkMicromapUsageEXT, subdivisionLevel) &&
                  offsetof(OpacityMicromapUsage, format) == offsetof(VkMicromapUsageEXT, format));
    static_assert(std::to_underlying(OpacityMicromapFormat::TwoState) == VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT &&
                  std::to_underlying(OpacityMicromapFormat::FourState) == VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT);
    return usages.empty() ? nullptr : reinterpret_cast<const VkMicromapUsageEXT*>(usages.data());
}

// An opacity micromap's build of inputs; the caller sets its micromap,
// data, records and scratch.
VkMicromapBuildInfoEXT micromapBuildOf(const AccelerationStructureInputs& inputs) {
    // A record is read as Vulkan's (VkMicromapTriangleEXT).
    static_assert(sizeof(OpacityMicromapRecord) == sizeof(VkMicromapTriangleEXT) &&
                  offsetof(OpacityMicromapRecord, subdivisionLevel) ==
                      offsetof(VkMicromapTriangleEXT, subdivisionLevel) &&
                  offsetof(OpacityMicromapRecord, format) == offsetof(VkMicromapTriangleEXT, format));
    VkMicromapBuildInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT;
    info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    info.flags = micromapFlagsFor(inputs.flags);
    info.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
    info.usageCountsCount = static_cast<std::uint32_t>(inputs.micromapUsages.size());
    info.pUsageCounts = usagesOf(inputs.micromapUsages);
    info.triangleArrayStride = inputs.micromapRecordStride;
    return info;
}

VkGeometryFlagsKHR geometryFlagsFor(bool opaque, bool noDuplicateAnyHit) {
    return (opaque ? VkGeometryFlagsKHR{VK_GEOMETRY_OPAQUE_BIT_KHR} : 0U) |
           (noDuplicateAnyHit ? VkGeometryFlagsKHR{VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR} : 0U);
}

VkGeometryInstanceFlagsKHR instanceFlagsFor(const AccelerationStructureInstance& instance) {
    VkGeometryInstanceFlagsKHR flags = 0;
    if (instance.cullDisable) {
        flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }
    if (instance.frontCounterClockwise) {
        flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
    }
    if (instance.opacity == InstanceOpacity::ForceOpaque) {
        flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    }
    else if (instance.opacity == InstanceOpacity::ForceNonOpaque) {
        flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    }
    if (instance.forceOpacityMicromap2State) {
        flags |= VK_GEOMETRY_INSTANCE_FORCE_OPACITY_MICROMAP_2_STATE_BIT_EXT;
    }
    if (instance.disableOpacityMicromaps) {
        flags |= VK_GEOMETRY_INSTANCE_DISABLE_OPACITY_MICROMAPS_BIT_EXT;
    }
    return flags;
}

VkAccelerationStructureGeometryKHR geometryOf(VkGeometryTypeKHR type, VkGeometryFlagsKHR flags) {
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = type;
    geometry.flags = flags;
    return geometry;
}

// Adds inputs' geometries to arrays, with a range and a primitive count for
// each, and returns the build info naming them, a build of its type and
// flags; the caller sets its mode, structures and scratch. addressOf gives a
// buffer's device address, offset bytes in, and micromapOf a structure's
// VkMicromapEXT. The arrays have room, as countGeometries has seen.
template <class AddressOf, class MicromapOf>
core::Result<VkAccelerationStructureBuildGeometryInfoKHR> addGeometries(const AccelerationStructureInputs& inputs,
                                                                        VulkanBuildArrays& arrays, AddressOf addressOf,
                                                                        MicromapOf micromapOf) {
    const std::size_t first = arrays.geometries.size();
    const auto add = [&arrays](const VkAccelerationStructureGeometryKHR& geometry, std::uint32_t primitives) {
        std::ignore = arrays.geometries.push(geometry);
        std::ignore = arrays.ranges.push(
            {.primitiveCount = primitives, .primitiveOffset = 0, .firstVertex = 0, .transformOffset = 0});
        std::ignore = arrays.primitiveCounts.push(primitives);
    };
    if (inputs.type == AccelerationStructureType::Top) {
        const auto instances = addressOf(inputs.instances, inputs.instanceOffset);
        if (!instances) {
            return std::unexpected(instances.error());
        }
        VkAccelerationStructureGeometryKHR geometry = geometryOf(VK_GEOMETRY_TYPE_INSTANCES_KHR, 0);
        geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometry.geometry.instances.arrayOfPointers = VK_FALSE;
        geometry.geometry.instances.data.deviceAddress = *instances;
        add(geometry, inputs.instanceCount);
    }
    else {
        for (const TriangleGeometry& triangles : inputs.triangles) {
            const auto vertices = addressOf(triangles.vertices, triangles.vertexOffset);
            const auto indices = addressOf(triangles.indices, triangles.indexOffset);
            const auto transform = addressOf(triangles.transform, triangles.transformOffset);
            for (const auto* address : {&vertices, &indices, &transform}) {
                if (!*address) {
                    return std::unexpected(address->error());
                }
            }
            VkAccelerationStructureGeometryKHR geometry = geometryOf(VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                                                                     geometryFlagsFor(triangles.opaque,
                                                                                      triangles.noDuplicateAnyHit));
            VkAccelerationStructureGeometryTrianglesDataKHR& data = geometry.geometry.triangles;
            data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            data.vertexFormat = vulkanFormat(triangles.vertexFormat);
            data.vertexData.deviceAddress = *vertices;
            data.vertexStride = triangles.vertexStride;
            // The highest vertex index (accelstructures.adoc:1152).
            data.maxVertex = triangles.vertexCount == 0 ? 0 : triangles.vertexCount - 1;
            data.indexType = triangles.indices.isNull()                 ? VK_INDEX_TYPE_NONE_KHR
                             : triangles.indexType == IndexType::Uint16 ? VK_INDEX_TYPE_UINT16
                                                                        : VK_INDEX_TYPE_UINT32;
            data.indexData.deviceAddress = *indices;
            data.transformData.deviceAddress = *transform;
            if (hasOpacityMicromap(triangles)) {
                const TriangleOpacityMicromap& named = triangles.opacityMicromap;
                const auto micromap = micromapOf(named.micromap);
                if (!micromap) {
                    return std::unexpected(micromap.error());
                }
                const auto micromapIndices = addressOf(named.indices, named.indexOffset);
                if (!micromapIndices) {
                    return std::unexpected(micromapIndices.error());
                }
                VkAccelerationStructureTrianglesOpacityMicromapEXT linked{};
                linked.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT;
                linked.indexType = named.indices.isNull()                 ? VK_INDEX_TYPE_NONE_KHR
                                   : named.indexType == IndexType::Uint16 ? VK_INDEX_TYPE_UINT16
                                                                          : VK_INDEX_TYPE_UINT32;
                linked.indexBuffer.deviceAddress = *micromapIndices;
                linked.indexStride = named.indexStride;
                linked.baseTriangle = named.baseTriangle;
                linked.usageCountsCount = named.usageCount;
                linked.pUsageCounts = usagesOf(inputs.micromapUsages.subspan(named.firstUsage, named.usageCount));
                linked.micromap = *micromap;
                std::ignore = arrays.micromaps.push(linked);
                data.pNext = &arrays.micromaps[arrays.micromaps.size() - 1];
            }
            add(geometry, triangles.triangleCount);
        }
        for (const BoxGeometry& boxes : inputs.boxes) {
            const auto data = addressOf(boxes.boxes, boxes.offset);
            if (!data) {
                return std::unexpected(data.error());
            }
            VkAccelerationStructureGeometryKHR geometry = geometryOf(VK_GEOMETRY_TYPE_AABBS_KHR,
                                                                     geometryFlagsFor(boxes.opaque,
                                                                                      boxes.noDuplicateAnyHit));
            geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
            geometry.geometry.aabbs.data.deviceAddress = *data;
            geometry.geometry.aabbs.stride = boxes.stride;
            add(geometry, boxes.count);
        }
    }
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = typeFor(inputs.type);
    info.flags = flagsFor(inputs.flags);
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = static_cast<std::uint32_t>(arrays.geometries.size() - first);
    info.pGeometries = arrays.geometries.data() + first;
    return info;
}

} // namespace

// The device's acceleration structure calls, beside the structures they make.

core::Result<VulkanAccelerationStructure*> VulkanDevice::accelerationStructureFor(
    AccelerationStructureHandle structure) {
    VulkanAccelerationStructure* found = accelerationStructures_.find(structure);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the acceleration structure handle is stale");
    }
    return found;
}

core::Result<VkMicromapEXT> VulkanDevice::micromapOf(AccelerationStructureHandle structure) {
    if (structure.isNull()) {
        return VkMicromapEXT{VK_NULL_HANDLE};
    }
    auto found = accelerationStructureFor(structure);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (!(*found)->isMicromap()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "triangles name a structure that is not an opacity "
                                                              "micromap as their micromap");
    }
    return (*found)->micromap();
}

core::Result<void> VulkanDevice::checkRunsCompute(const VulkanCommandList& list) {
    const VulkanCommandPool* pool = commandPools_.find(list.pool);
    if (pool == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    if (!queues_[pool->queue]->compute) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the list's queue runs no compute, which acceleration structure work needs");
    }
    return {};
}

core::Result<VkDeviceAddress> VulkanDevice::addressOf(BufferHandle buffer, std::uint64_t offset) {
    if (buffer.isNull()) {
        return VkDeviceAddress{0};
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return (*found)->address() + offset;
}

core::Result<AccelerationStructureSizes> VulkanDevice::accelerationStructureSizes(
    const AccelerationStructureInputs& inputs) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkInputs(inputs, features_.opacityMicromap); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto counted = countGeometries(inputs, desc_.maxBuildGeometries); !counted) {
        return std::unexpected(std::move(counted.error()));
    }
    if (inputs.type == AccelerationStructureType::OpacityMicromap) {
        const VkMicromapBuildInfoEXT info = micromapBuildOf(inputs);
        VkMicromapBuildSizesInfoEXT sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT;
        table_.vkGetMicromapBuildSizesEXT(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &sizes);
        return AccelerationStructureSizes{.size = sizes.micromapSize,
                                          .buildScratch = sizes.buildScratchSize,
                                          .updateScratch = 0,
                                          .discardable = sizes.discardable == VK_TRUE};
    }
    const std::scoped_lock lock{buildLock_};
    builds_.clear();
    const auto info = addGeometries(
        inputs, builds_,
        [](BufferHandle buffer, std::uint64_t /*offset*/) {
            return core::Result<VkDeviceAddress>{buffer.isNull() ? 0 : kNamedAddress};
        },
        [this](AccelerationStructureHandle structure) { return micromapOf(structure); });
    if (!info) {
        return std::unexpected(std::move(info.error()));
    }
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    table_.vkGetAccelerationStructureBuildSizesKHR(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &*info,
                                                   builds_.primitiveCounts.data(), &sizes);
    return AccelerationStructureSizes{.size = sizes.accelerationStructureSize,
                                      .buildScratch = sizes.buildScratchSize,
                                      .updateScratch = sizes.updateScratchSize};
}

core::Result<AccelerationStructureHandle> VulkanDevice::createAccelerationStructure(
    const AccelerationStructureDesc& desc) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkKnown(desc.type, "acceleration structure type"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkOpacityMicromaps(features_.opacityMicromap,
                                             desc.type == AccelerationStructureType::OpacityMicromap);
        !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (desc.size == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the acceleration structure has no bytes");
    }
    // Cold, so structures the GPU is done with are freed here.
    collectBuffers();
    auto slot = bindless_.takeAccelerationStructure(desc.type);
    if (!slot) {
        return std::unexpected(std::move(slot.error()));
    }
    const auto giveBack = [this, &slot] {
        bindless_.giveAccelerationStructure(*slot);
    };
    // The structure is at offset 0 of a buffer of its own, which a multiple
    // of 256 is (resources.adoc:8756), as a micromap's must be too.
    const bool micromap = desc.type == AccelerationStructureType::OpacityMicromap;
    auto storage = placeVulkanBuffer(table_, device_, allocator_, desc.size,
                                     micromap
                                         ? VkBufferUsageFlags{VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT}
                                         : VkBufferUsageFlags{VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT},
                                     BufferMemory::Device, families_, 0, core::SlotChains::kEnd);
    if (!storage) {
        giveBack();
        return std::unexpected(std::move(storage.error()));
    }
    if (micromap) {
        VkMicromapCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT;
        info.buffer = storage->buffer();
        info.size = desc.size;
        info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
        VkMicromapEXT made = VK_NULL_HANDLE;
        if (const VkResult result = table_.vkCreateMicromapEXT(device_, &info, nullptr, &made); result != VK_SUCCESS) {
            giveBack();
            return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateMicromapEXT", .result = result});
        }
        const AccelerationStructureHandle named = accelerationStructures_.emplace(
            VulkanAccelerationStructure{table_, device_, std::move(*storage), VK_NULL_HANDLE, made, desc, *slot, 0});
        if (named.isNull()) {
            giveBack();
            return core::failure(core::ErrorCode::RhiLimitReached,
                                 "the device holds at most {} acceleration structures", kBufferLimit);
        }
        return named;
    }
    VkAccelerationStructureCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    info.buffer = storage->buffer();
    info.offset = 0;
    info.size = desc.size;
    info.type = typeFor(desc.type);
    VkAccelerationStructureKHR structure = VK_NULL_HANDLE;
    if (const VkResult result = table_.vkCreateAccelerationStructureKHR(device_, &info, nullptr, &structure);
        result != VK_SUCCESS) {
        giveBack();
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "vkCreateAccelerationStructureKHR", .result = result});
    }
    VkAccelerationStructureDeviceAddressInfoKHR named{};
    named.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    named.accelerationStructure = structure;
    const VkDeviceAddress reference = table_.vkGetAccelerationStructureDeviceAddressKHR(device_, &named);
    const AccelerationStructureHandle made = accelerationStructures_.emplace(
        VulkanAccelerationStructure{table_, device_, std::move(*storage), structure, VK_NULL_HANDLE, desc, *slot,
                                    reference});
    if (made.isNull()) {
        giveBack();
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} acceleration structures",
                             kBufferLimit);
    }
    if (*slot != core::SlotChains::kEnd) {
        layout_.writeAccelerationStructure(*slot, structure);
    }
    return made;
}

core::Result<std::uint32_t> VulkanDevice::accelerationStructureSlot(AccelerationStructureHandle structure) {
    auto found = accelerationStructureFor(structure);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reportedStructure((*found)->slot());
}

core::Result<std::uint64_t> VulkanDevice::accelerationStructureReference(AccelerationStructureHandle structure) {
    auto found = accelerationStructureFor(structure);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // Vulkan gives a micromap no address (VK_EXT_opacity_micromap).
    if ((*found)->isMicromap()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "an opacity micromap has no reference");
    }
    return (*found)->reference();
}

core::Result<void> VulkanDevice::destroyAccelerationStructure(AccelerationStructureHandle structure) {
    if (structure.isNull()) {
        return {};
    }
    if (accelerationStructures_.find(structure) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the acceleration structure handle is stale");
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!accelerationStructures_.retire(structure, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the acceleration structure handle is stale");
    }
    collectBuffers();
    return {};
}

core::Result<void> VulkanDevice::writeInstances(std::span<const AccelerationStructureInstance> instances,
                                                std::span<std::byte> out) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return checked;
    }
    if (auto checked = checkInstances(instances, out, rayTracing_.instanceSize); !checked) {
        return checked;
    }
    for (std::size_t i = 0; i < instances.size(); ++i) {
        const AccelerationStructureInstance& instance = instances[i];
        VkDeviceAddress reference = 0;
        if (!instance.bottomStructure.isNull()) {
            auto found = accelerationStructureFor(instance.bottomStructure);
            if (!found) {
                return std::unexpected(std::move(found.error()));
            }
            if ((*found)->type() != AccelerationStructureType::Bottom) {
                return core::failure(core::ErrorCode::RhiInvalidCall,
                                     "instance {} names a structure that is not a bottom one; instances are of bottom "
                                     "ones",
                                     i);
            }
            reference = (*found)->reference();
        }
        VkAccelerationStructureInstanceKHR record{};
        static_assert(sizeof(record.transform) == sizeof(instance.transform));
        std::memcpy(&record.transform, instance.transform.data(), sizeof(record.transform));
        // checkInstances has seen the indices fit their 24 bits, and the
        // flags are 8 bits' worth.
        record.instanceCustomIndex = instance.customIndex & kIndexBits;
        record.mask = instance.mask;
        record.instanceShaderBindingTableRecordOffset = instance.hitGroupOffset & kIndexBits;
        record.flags = static_cast<std::uint8_t>(instanceFlagsFor(instance));
        record.accelerationStructureReference = reference;
        std::memcpy(out.subspan(i * sizeof(record)).data(), &record, sizeof(record));
    }
    return {};
}

core::Result<void> VulkanDevice::buildAccelerationStructure(CommandListHandle commandList,
                                                            const AccelerationStructureBuild& build) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return checked;
    }
    return recordBuild(commandList, build, nullptr);
}

core::Result<void> VulkanDevice::buildAccelerationStructureIndirect(CommandListHandle commandList,
                                                                    const AccelerationStructureBuild& build,
                                                                    BufferHandle ranges, std::uint64_t offset,
                                                                    std::uint32_t stride) {
    if (auto checked = checkIndirectBuilds(features_.accelerationStructureIndirectBuild); !checked) {
        return checked;
    }
    auto found = bufferFor(ranges);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const std::pair<VkDeviceAddress, std::uint32_t> read{(*found)->address() + offset, stride};
    return recordBuild(commandList, build, &read);
}

core::Result<void> VulkanDevice::recordBuild(CommandListHandle commandList, const AccelerationStructureBuild& build,
                                             const std::pair<VkDeviceAddress, std::uint32_t>* ranges) {
    // An indirect build's ranges are read in place on the GPU.
    static_assert(sizeof(AccelerationStructureBuildRange) == sizeof(VkAccelerationStructureBuildRangeInfoKHR) &&
                  offsetof(AccelerationStructureBuildRange, primitiveOffset) ==
                      offsetof(VkAccelerationStructureBuildRangeInfoKHR, primitiveOffset) &&
                  offsetof(AccelerationStructureBuildRange, firstVertex) ==
                      offsetof(VkAccelerationStructureBuildRangeInfoKHR, firstVertex) &&
                  offsetof(AccelerationStructureBuildRange, transformOffset) ==
                      offsetof(VkAccelerationStructureBuildRangeInfoKHR, transformOffset));
    if (auto checked = checkInputs(build.inputs, features_.opacityMicromap); !checked) {
        return checked;
    }
    if (auto counted = countGeometries(build.inputs, desc_.maxBuildGeometries); !counted) {
        return counted;
    }
    // Vulkan builds a micromap whole, from counts the CPU gives
    // (VkBuildMicromapModeEXT, vkCmdBuildMicromapsEXT).
    if (build.inputs.type == AccelerationStructureType::OpacityMicromap && (build.update || ranges != nullptr)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "an opacity micromap is built whole, never updated or built from GPU counts");
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRunsCompute(**list); !checked) {
        return checked;
    }
    auto dst = accelerationStructureFor(build.dst);
    if (!dst) {
        return std::unexpected(std::move(dst.error()));
    }
    if ((*dst)->type() != build.inputs.type) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a build's inputs are of another type than the structure it builds");
    }
    VkAccelerationStructureKHR src = VK_NULL_HANDLE;
    if (build.update) {
        auto from = accelerationStructureFor(build.src);
        if (!from) {
            return std::unexpected(std::move(from.error()));
        }
        src = (*from)->structure();
    }
    const auto scratch = addressOf(build.scratch, build.scratchOffset);
    if (!scratch) {
        return std::unexpected(scratch.error());
    }
    if (build.inputs.type == AccelerationStructureType::OpacityMicromap) {
        const auto data = addressOf(build.inputs.micromapData, build.inputs.micromapDataOffset);
        const auto records = addressOf(build.inputs.micromapRecords, build.inputs.micromapRecordOffset);
        for (const auto* address : {&data, &records}) {
            if (!*address) {
                return std::unexpected(address->error());
            }
        }
        VkMicromapBuildInfoEXT info = micromapBuildOf(build.inputs);
        info.dstMicromap = (*dst)->micromap();
        info.data.deviceAddress = *data;
        info.triangleArray.deviceAddress = *records;
        info.scratchData.deviceAddress = *scratch;
        table_.vkCmdBuildMicromapsEXT((*list)->buffer, 1, &info);
        return {};
    }
    const std::scoped_lock lock{buildLock_};
    builds_.clear();
    auto info = addGeometries(
        build.inputs, builds_, [this](BufferHandle buffer, std::uint64_t offset) { return addressOf(buffer, offset); },
        [this](AccelerationStructureHandle structure) { return micromapOf(structure); });
    if (!info) {
        return std::unexpected(std::move(info.error()));
    }
    info->mode = build.update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                              : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info->srcAccelerationStructure = src;
    info->dstAccelerationStructure = (*dst)->structure();
    info->scratchData.deviceAddress = *scratch;
    if (ranges != nullptr) {
        const std::uint32_t* maxCounts = builds_.primitiveCounts.data();
        table_.vkCmdBuildAccelerationStructuresIndirectKHR((*list)->buffer, 1, &*info, &ranges->first, &ranges->second,
                                                           &maxCounts);
        return {};
    }
    const VkAccelerationStructureBuildRangeInfoKHR* counts = builds_.ranges.data();
    table_.vkCmdBuildAccelerationStructuresKHR((*list)->buffer, 1, &*info, &counts);
    return {};
}

core::Result<void> VulkanDevice::copyStructure(CommandListHandle commandList, AccelerationStructureHandle dst,
                                               AccelerationStructureHandle src,
                                               VkCopyAccelerationStructureModeKHR mode) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRunsCompute(**list); !checked) {
        return checked;
    }
    auto to = accelerationStructureFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    auto from = accelerationStructureFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    if ((*from)->isMicromap() || (*to)->isMicromap()) {
        // Vulkan copies micromaps only into micromaps (VkCopyMicromapInfoEXT).
        if (!(*from)->isMicromap() || !(*to)->isMicromap()) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "an opacity micromap is copied only into an opacity micromap");
        }
        VkCopyMicromapInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_COPY_MICROMAP_INFO_EXT;
        info.src = (*from)->micromap();
        info.dst = (*to)->micromap();
        info.mode = mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR ? VK_COPY_MICROMAP_MODE_COMPACT_EXT
                                                                            : VK_COPY_MICROMAP_MODE_CLONE_EXT;
        table_.vkCmdCopyMicromapEXT((*list)->buffer, &info);
        return {};
    }
    // The destination's size is the caller's to keep, as the API states it
    // (copy_acceleration_structure_common.adoc, VUID-...-mode-12418, -12419).
    VkCopyAccelerationStructureInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
    info.src = (*from)->structure();
    info.dst = (*to)->structure();
    info.mode = mode;
    table_.vkCmdCopyAccelerationStructureKHR((*list)->buffer, &info);
    return {};
}

core::Result<void> VulkanDevice::copyAccelerationStructure(CommandListHandle commandList,
                                                           AccelerationStructureHandle dst,
                                                           AccelerationStructureHandle src) {
    return copyStructure(commandList, dst, src, VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR);
}

core::Result<void> VulkanDevice::compactAccelerationStructure(CommandListHandle commandList,
                                                              AccelerationStructureHandle dst,
                                                              AccelerationStructureHandle src) {
    return copyStructure(commandList, dst, src, VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR);
}

VulkanPropertyPool::VulkanPropertyPool(const VolkDeviceTable& table, VkDevice device, VkQueryPool pool,
                                       VkQueryType type, std::uint32_t count, bool micromaps)
    : table_(&table),
      device_(device),
      pool_(pool),
      type_(type),
      count_(count),
      micromaps_(micromaps),
      structures_(micromaps ? 0 : count),
      micromapList_(micromaps ? count : 0) {}

VulkanPropertyPool::VulkanPropertyPool(VulkanPropertyPool&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      pool_(std::exchange(other.pool_, VK_NULL_HANDLE)),
      type_(other.type_),
      count_(other.count_),
      micromaps_(other.micromaps_),
      structures_(std::move(other.structures_)),
      micromapList_(std::move(other.micromapList_)) {}

VulkanPropertyPool::~VulkanPropertyPool() {
    if (pool_ != VK_NULL_HANDLE) {
        table_->vkDestroyQueryPool(device_, pool_, nullptr);
    }
}

namespace {

// A micromap's query type for property; Vulkan has none for the others
// (VkQueryType, VK_EXT_opacity_micromap).
std::optional<VkQueryType> micromapQueryTypeFor(AccelerationStructureProperty property) {
    switch (property) {
        case AccelerationStructureProperty::CompactedSize:
            return VK_QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT;
        case AccelerationStructureProperty::SerializedSize:
            return VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT;
        case AccelerationStructureProperty::SerializedBottomCount:
        case AccelerationStructureProperty::CurrentSize:
        default:
            return std::nullopt;
    }
}

VkQueryType queryTypeFor(AccelerationStructureProperty property) {
    switch (property) {
        case AccelerationStructureProperty::SerializedSize:
            return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
        case AccelerationStructureProperty::SerializedBottomCount:
            return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR;
        case AccelerationStructureProperty::CurrentSize:
            return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR;
        case AccelerationStructureProperty::CompactedSize:
        default:
            return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    }
}

} // namespace

core::Result<AccelerationStructurePropertyPoolHandle> VulkanDevice::createAccelerationStructurePropertyPool(
    const AccelerationStructurePropertyPoolDesc& desc) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkProperty(desc.property, features_); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkOpacityMicromaps(features_.opacityMicromap, desc.opacityMicromaps); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const std::optional<VkQueryType> micromapType = micromapQueryTypeFor(desc.property);
    if (desc.opacityMicromaps && !micromapType) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "Vulkan writes no property {} of opacity micromaps, only compacted and serialized sizes",
                             static_cast<std::uint32_t>(desc.property));
    }
    collectBuffers();
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = desc.opacityMicromaps ? *micromapType : queryTypeFor(desc.property);
    info.queryCount = desc.count;
    VkQueryPool pool = VK_NULL_HANDLE;
    if (const VkResult result = table_.vkCreateQueryPool(device_, &info, nullptr, &pool); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateQueryPool", .result = result});
    }
    const AccelerationStructurePropertyPoolHandle made = propertyPools_.emplace(
        VulkanPropertyPool{table_, device_, pool, info.queryType, desc.count, desc.opacityMicromaps});
    if (made.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             "the device holds at most {} acceleration structure property pools",
                             desc_.maxAccelerationStructurePropertyPools);
    }
    return made;
}

core::Result<void> VulkanDevice::destroyAccelerationStructurePropertyPool(
    AccelerationStructurePropertyPoolHandle pool) {
    if (pool.isNull()) {
        return {};
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!propertyPools_.retire(pool, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the property pool handle is stale");
    }
    collectBuffers();
    return {};
}

core::Result<std::pair<VulkanCommandList*, VulkanPropertyPool*>> VulkanDevice::propertyCall(
    CommandListHandle commandList, AccelerationStructurePropertyPoolHandle pool) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    // Reset, write and resolve take a graphics or compute queue (vkdocs
    // xml/vk.xml:15872, 15886, 17426), which a compute-capable one is.
    if (auto checked = checkRunsCompute(**list); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    VulkanPropertyPool* found = propertyPools_.find(pool);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the property pool handle is stale");
    }
    return std::pair{*list, found};
}

core::Result<void> VulkanDevice::resetAccelerationStructureProperties(CommandListHandle commandList,
                                                                      AccelerationStructurePropertyPoolHandle pool,
                                                                      std::uint32_t first, std::uint32_t count) {
    const auto call = propertyCall(commandList, pool);
    if (!call) {
        return std::unexpected(call.error());
    }
    table_.vkCmdResetQueryPool(call->first->buffer, call->second->pool(), first, count);
    return {};
}

core::Result<void> VulkanDevice::writeAccelerationStructureProperties(
    CommandListHandle commandList, AccelerationStructurePropertyPoolHandle pool, std::uint32_t first,
    std::span<const AccelerationStructureHandle> structures) {
    const auto call = propertyCall(commandList, pool);
    if (!call) {
        return std::unexpected(call.error());
    }
    VulkanPropertyPool& found = *call->second;
    if (auto checked = checkPropertyRange(first, structures.size(), found.count()); !checked) {
        return checked;
    }
    // The pool's room holds count, which checkPropertyRange has seen is enough.
    found.structures().clear();
    found.micromapList().clear();
    for (const AccelerationStructureHandle structure : structures) {
        auto named = accelerationStructureFor(structure);
        if (!named) {
            return std::unexpected(std::move(named.error()));
        }
        // A query pool is of one query type, a micromap's or a structure's.
        if ((*named)->isMicromap() != found.micromaps()) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 found.micromaps() ? "a pool of opacity micromaps' properties names another structure"
                                                   : "a pool of structures' properties names an opacity micromap");
        }
        if (found.micromaps()) {
            std::ignore = found.micromapList().push((*named)->micromap());
        }
        else {
            std::ignore = found.structures().push((*named)->structure());
        }
    }
    if (found.micromaps()) {
        table_.vkCmdWriteMicromapsPropertiesEXT(call->first->buffer,
                                                static_cast<std::uint32_t>(found.micromapList().size()),
                                                found.micromapList().data(), found.type(), found.pool(), first);
        return {};
    }
    table_.vkCmdWriteAccelerationStructuresPropertiesKHR(call->first->buffer,
                                                         static_cast<std::uint32_t>(found.structures().size()),
                                                         found.structures().data(), found.type(), found.pool(), first);
    return {};
}

core::Result<void> VulkanDevice::resolveAccelerationStructureProperties(CommandListHandle commandList,
                                                                        AccelerationStructurePropertyPoolHandle pool,
                                                                        std::uint32_t first, std::uint32_t count,
                                                                        BufferHandle buffer, std::uint64_t offset) {
    const auto call = propertyCall(commandList, pool);
    if (!call) {
        return std::unexpected(call.error());
    }
    auto target = bufferFor(buffer);
    if (!target) {
        return std::unexpected(std::move(target.error()));
    }
    // As resolveTimestamps does: waiting for each entry, 64 bits each, the
    // caps' propertyEntryBytes apart.
    table_.vkCmdCopyQueryPoolResults(call->first->buffer, call->second->pool(), first, count, (*target)->buffer(),
                                     offset, rayTracing_.propertyEntryBytes,
                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return {};
}

core::Result<void> VulkanDevice::serializeAccelerationStructure(CommandListHandle commandList,
                                                                AccelerationStructureHandle src, BufferHandle buffer,
                                                                std::uint64_t offset) {
    if (auto checked = checkSerialization(features_.accelerationStructureSerialization); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRunsCompute(**list); !checked) {
        return checked;
    }
    auto from = accelerationStructureFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    const auto address = addressOf(buffer, offset);
    if (!address) {
        return std::unexpected(address.error());
    }
    if ((*from)->isMicromap()) {
        VkCopyMicromapToMemoryInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_COPY_MICROMAP_TO_MEMORY_INFO_EXT;
        info.src = (*from)->micromap();
        info.dst.deviceAddress = *address;
        info.mode = VK_COPY_MICROMAP_MODE_SERIALIZE_EXT;
        table_.vkCmdCopyMicromapToMemoryEXT((*list)->buffer, &info);
        return {};
    }
    VkCopyAccelerationStructureToMemoryInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR;
    info.src = (*from)->structure();
    info.dst.deviceAddress = *address;
    info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR;
    table_.vkCmdCopyAccelerationStructureToMemoryKHR((*list)->buffer, &info);
    return {};
}

core::Result<void> VulkanDevice::deserializeAccelerationStructure(CommandListHandle commandList, BufferHandle buffer,
                                                                  std::uint64_t offset,
                                                                  AccelerationStructureHandle dst) {
    if (auto checked = checkSerialization(features_.accelerationStructureSerialization); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRunsCompute(**list); !checked) {
        return checked;
    }
    auto to = accelerationStructureFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    const auto address = addressOf(buffer, offset);
    if (!address) {
        return std::unexpected(address.error());
    }
    if ((*to)->isMicromap()) {
        VkCopyMemoryToMicromapInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_MICROMAP_INFO_EXT;
        info.src.deviceAddress = *address;
        info.dst = (*to)->micromap();
        info.mode = VK_COPY_MICROMAP_MODE_DESERIALIZE_EXT;
        table_.vkCmdCopyMemoryToMicromapEXT((*list)->buffer, &info);
        return {};
    }
    VkCopyMemoryToAccelerationStructureInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR;
    info.src.deviceAddress = *address;
    info.dst = (*to)->structure();
    info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;
    table_.vkCmdCopyMemoryToAccelerationStructureKHR((*list)->buffer, &info);
    return {};
}

core::Result<bool> VulkanDevice::serializedCompatible(
    AccelerationStructureType type, std::span<const std::byte, kSerializedIdentifierSize> identifier) const {
    if (auto checked = checkSerialization(features_.accelerationStructureSerialization); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkKnown(type, "acceleration structure type"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Two VK_UUID_SIZE identifiers (accelstructures.adoc:3393), a micromap's
    // too (VkMicromapVersionInfoEXT).
    static_assert(kSerializedIdentifierSize == 2 * VK_UUID_SIZE);
    if (type == AccelerationStructureType::OpacityMicromap) {
        if (auto checked = checkOpacityMicromaps(features_.opacityMicromap, true); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
        VkMicromapVersionInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_MICROMAP_VERSION_INFO_EXT;
        info.pVersionData = reinterpret_cast<const std::uint8_t*>(identifier.data());
        VkAccelerationStructureCompatibilityKHR compatibility =
            VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
        table_.vkGetDeviceMicromapCompatibilityEXT(device_, &info, &compatibility);
        return compatibility == VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR;
    }
    VkAccelerationStructureVersionInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR;
    info.pVersionData = reinterpret_cast<const std::uint8_t*>(identifier.data());
    VkAccelerationStructureCompatibilityKHR compatibility = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
    table_.vkGetDeviceAccelerationStructureCompatibilityKHR(device_, &info, &compatibility);
    return compatibility == VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR;
}

} // namespace bfr::rhi::vulkan
