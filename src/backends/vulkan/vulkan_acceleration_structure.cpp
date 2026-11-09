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
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

VulkanAccelerationStructure::VulkanAccelerationStructure(const VolkDeviceTable& table, VkDevice device,
                                                         VulkanBuffer storage, VkAccelerationStructureKHR structure,
                                                         const AccelerationStructureDesc& desc, std::uint32_t slot,
                                                         VkDeviceAddress reference)
    : table_(&table),
      device_(device),
      storage_(std::move(storage)),
      structure_(structure),
      type_(desc.type),
      size_(desc.size),
      slot_(slot),
      reference_(reference) {}

VulkanAccelerationStructure::VulkanAccelerationStructure(VulkanAccelerationStructure&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      storage_(std::move(other.storage_)),
      structure_(std::exchange(other.structure_, VK_NULL_HANDLE)),
      type_(other.type_),
      size_(other.size_),
      slot_(other.slot_),
      reference_(other.reference_) {}

VulkanAccelerationStructure::~VulkanAccelerationStructure() {
    if (structure_ != VK_NULL_HANDLE) {
        table_->vkDestroyAccelerationStructureKHR(device_, structure_, nullptr);
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
    return vulkan;
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
// buffer's device address, offset bytes in. The arrays have room, as
// countGeometries has seen.
template <class AddressOf>
core::Result<VkAccelerationStructureBuildGeometryInfoKHR> addGeometries(const AccelerationStructureInputs& inputs,
                                                                        VulkanBuildArrays& arrays,
                                                                        AddressOf addressOf) {
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
    if (auto checked = checkInputs(inputs); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto counted = countGeometries(inputs, desc_.maxBuildGeometries); !counted) {
        return std::unexpected(std::move(counted.error()));
    }
    const std::scoped_lock lock{buildLock_};
    builds_.clear();
    const auto info = addGeometries(inputs, builds_, [](BufferHandle buffer, std::uint64_t /*offset*/) {
        return core::Result<VkDeviceAddress>{buffer.isNull() ? 0 : kNamedAddress};
    });
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
    // of 256 is (resources.adoc:8756).
    auto storage = placeVulkanBuffer(table_, device_, allocator_, desc.size,
                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                     BufferMemory::Device, families_, 0, core::SlotChains::kEnd);
    if (!storage) {
        giveBack();
        return std::unexpected(std::move(storage.error()));
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
        VulkanAccelerationStructure{table_, device_, std::move(*storage), structure, desc, *slot, reference});
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
                                     "instance {} names a top acceleration structure; instances are of bottom ones", i);
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
    if (auto checked = checkInputs(build.inputs); !checked) {
        return checked;
    }
    if (auto counted = countGeometries(build.inputs, desc_.maxBuildGeometries); !counted) {
        return counted;
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
    const std::scoped_lock lock{buildLock_};
    builds_.clear();
    auto info = addGeometries(build.inputs, builds_,
                              [this](BufferHandle buffer, std::uint64_t offset) { return addressOf(buffer, offset); });
    if (!info) {
        return std::unexpected(std::move(info.error()));
    }
    info->mode = build.update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                              : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info->srcAccelerationStructure = src;
    info->dstAccelerationStructure = (*dst)->structure();
    info->scratchData.deviceAddress = *scratch;
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = builds_.ranges.data();
    table_.vkCmdBuildAccelerationStructuresKHR((*list)->buffer, 1, &*info, &ranges);
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
                                       VkQueryType type, std::uint32_t count)
    : table_(&table), device_(device), pool_(pool), type_(type), count_(count), structures_(count) {}

VulkanPropertyPool::VulkanPropertyPool(VulkanPropertyPool&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      pool_(std::exchange(other.pool_, VK_NULL_HANDLE)),
      type_(other.type_),
      count_(other.count_),
      structures_(std::move(other.structures_)) {}

VulkanPropertyPool::~VulkanPropertyPool() {
    if (pool_ != VK_NULL_HANDLE) {
        table_->vkDestroyQueryPool(device_, pool_, nullptr);
    }
}

namespace {

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
    collectBuffers();
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = queryTypeFor(desc.property);
    info.queryCount = desc.count;
    VkQueryPool pool = VK_NULL_HANDLE;
    if (const VkResult result = table_.vkCreateQueryPool(device_, &info, nullptr, &pool); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateQueryPool", .result = result});
    }
    const AccelerationStructurePropertyPoolHandle made = propertyPools_.emplace(
        VulkanPropertyPool{table_, device_, pool, info.queryType, desc.count});
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
    for (const AccelerationStructureHandle structure : structures) {
        auto named = accelerationStructureFor(structure);
        if (!named) {
            return std::unexpected(std::move(named.error()));
        }
        std::ignore = found.structures().push((*named)->structure());
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
    VkCopyMemoryToAccelerationStructureInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR;
    info.src.deviceAddress = *address;
    info.dst = (*to)->structure();
    info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;
    table_.vkCmdCopyMemoryToAccelerationStructureKHR((*list)->buffer, &info);
    return {};
}

core::Result<bool> VulkanDevice::serializedCompatible(
    std::span<const std::byte, kSerializedIdentifierSize> identifier) const {
    if (auto checked = checkSerialization(features_.accelerationStructureSerialization); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Two VK_UUID_SIZE identifiers (accelstructures.adoc:3393).
    static_assert(kSerializedIdentifierSize == 2 * VK_UUID_SIZE);
    VkAccelerationStructureVersionInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR;
    info.pVersionData = reinterpret_cast<const std::uint8_t*>(identifier.data());
    VkAccelerationStructureCompatibilityKHR compatibility = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
    table_.vkGetDeviceAccelerationStructureCompatibilityKHR(device_, &info, &compatibility);
    return compatibility == VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR;
}

} // namespace bfr::rhi::vulkan
