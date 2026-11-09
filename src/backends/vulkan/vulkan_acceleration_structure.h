#pragma once

#include "bfr/rhi/acceleration_structure.h"
#include "core/fixed_vector.h"
#include "vulkan/vulkan_buffer.h"

#include <volk.h>

#include <cstdint>

namespace bfr::rhi::vulkan {

/// A VkAccelerationStructureKHR at the start of a buffer of its own, which
/// it destroys when it goes, before the buffer. slot is its place in the
/// acceleration structure table, kEnd for a bottom one; the device gives it
/// back when the structure is collected. reference is its device address,
/// which instance records name it by.
class VulkanAccelerationStructure {
public:
    VulkanAccelerationStructure(const VolkDeviceTable& table, VkDevice device, VulkanBuffer storage,
                                VkAccelerationStructureKHR structure, const AccelerationStructureDesc& desc,
                                std::uint32_t slot, VkDeviceAddress reference);
    ~VulkanAccelerationStructure();
    VulkanAccelerationStructure(const VulkanAccelerationStructure&) = delete;
    VulkanAccelerationStructure& operator=(const VulkanAccelerationStructure&) = delete;
    VulkanAccelerationStructure(VulkanAccelerationStructure&& other) noexcept;
    VulkanAccelerationStructure& operator=(VulkanAccelerationStructure&&) = delete;

    [[nodiscard]] VkAccelerationStructureKHR structure() const { return structure_; }
    [[nodiscard]] AccelerationStructureType type() const { return type_; }
    [[nodiscard]] std::uint64_t size() const { return size_; }
    [[nodiscard]] std::uint32_t slot() const { return slot_; }
    [[nodiscard]] VkDeviceAddress reference() const { return reference_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VulkanBuffer storage_;
    VkAccelerationStructureKHR structure_;
    AccelerationStructureType type_;
    std::uint64_t size_;
    std::uint32_t slot_;
    VkDeviceAddress reference_;
};

/// What one buildAccelerationStructure or accelerationStructureSizes call
/// fills: a geometry, a range and a primitive count for each of its
/// geometries. Made with the device at DeviceDesc::maxBuildGeometries, so no
/// call allocates; used by one call at a time, under the device's lock.
struct VulkanBuildArrays {
    explicit VulkanBuildArrays(std::uint32_t maxBuildGeometries)
        : geometries(maxBuildGeometries), ranges(maxBuildGeometries), primitiveCounts(maxBuildGeometries) {}

    void clear() {
        geometries.clear();
        ranges.clear();
        primitiveCounts.clear();
    }

    core::FixedVector<VkAccelerationStructureGeometryKHR> geometries;
    core::FixedVector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    core::FixedVector<std::uint32_t> primitiveCounts;
};

} // namespace bfr::rhi::vulkan
