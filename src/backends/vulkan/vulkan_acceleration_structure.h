#pragma once

#include "bfr/rhi/acceleration_structure.h"
#include "core/fixed_vector.h"
#include "vulkan/vulkan_buffer.h"

#include <volk.h>

#include <cstdint>

namespace bfr::rhi::vulkan {

/// The alignment of a partitioned top structure's address and its build's
/// scratch (VUID-vkCmdBuildPartitionedAccelerationStructuresNV-pBuildInfo-10542, -10545).
inline constexpr VkDeviceSize kPartitionedAlignment = 256;

/// A VkAccelerationStructureKHR, or for an opacity micromap a VkMicromapEXT,
/// at the start of a buffer of its own, which it destroys when it goes,
/// before the buffer. slot is its place in the acceleration structure table,
/// or a partitioned top one's in its own, kEnd for others; the device gives
/// it back when the structure
/// is collected. reference is a structure's device address, which instance
/// records name it by; a micromap has none. A partitioned top structure is
/// only its buffer, named by reference, with neither object.
class VulkanAccelerationStructure {
public:
    VulkanAccelerationStructure(const VolkDeviceTable& table, VkDevice device, VulkanBuffer storage,
                                VkAccelerationStructureKHR structure, VkMicromapEXT micromap,
                                const AccelerationStructureDesc& desc, std::uint32_t slot, VkDeviceAddress reference,
                                const AccelerationStructureCapture& captured);
    ~VulkanAccelerationStructure();
    VulkanAccelerationStructure(const VulkanAccelerationStructure&) = delete;
    VulkanAccelerationStructure& operator=(const VulkanAccelerationStructure&) = delete;
    VulkanAccelerationStructure(VulkanAccelerationStructure&& other) noexcept;
    VulkanAccelerationStructure& operator=(VulkanAccelerationStructure&&) = delete;

    [[nodiscard]] VkAccelerationStructureKHR structure() const { return structure_; }
    [[nodiscard]] VkMicromapEXT micromap() const { return micromap_; }
    [[nodiscard]] AccelerationStructureType type() const { return type_; }
    [[nodiscard]] bool isMicromap() const { return type_ == AccelerationStructureType::OpacityMicromap; }
    [[nodiscard]] std::uint64_t size() const { return size_; }
    [[nodiscard]] std::uint32_t slot() const { return slot_; }
    [[nodiscard]] VkDeviceAddress reference() const { return reference_; }
    /// What makes it again at its reference; all zero for one made without
    /// captureReplay.
    [[nodiscard]] const AccelerationStructureCapture& captured() const { return captured_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VulkanBuffer storage_;
    VkAccelerationStructureKHR structure_;
    VkMicromapEXT micromap_;
    AccelerationStructureType type_;
    std::uint64_t size_;
    std::uint32_t slot_;
    VkDeviceAddress reference_;
    AccelerationStructureCapture captured_;
};

/// A VkQueryPool of one acceleration structure property, of opacity
/// micromaps or of other structures, destroyed when this goes, with room made
/// with it for the structures a write names, so no write allocates.
class VulkanPropertyPool {
public:
    VulkanPropertyPool(const VolkDeviceTable& table, VkDevice device, VkQueryPool pool, VkQueryType type,
                       std::uint32_t count, bool micromaps);
    ~VulkanPropertyPool();
    VulkanPropertyPool(const VulkanPropertyPool&) = delete;
    VulkanPropertyPool& operator=(const VulkanPropertyPool&) = delete;
    VulkanPropertyPool(VulkanPropertyPool&& other) noexcept;
    VulkanPropertyPool& operator=(VulkanPropertyPool&&) = delete;

    [[nodiscard]] VkQueryPool pool() const { return pool_; }
    [[nodiscard]] VkQueryType type() const { return type_; }
    [[nodiscard]] std::uint32_t count() const { return count_; }
    [[nodiscard]] bool micromaps() const { return micromaps_; }
    /// Where a write's structures, or micromaps, go, count of them.
    [[nodiscard]] core::FixedVector<VkAccelerationStructureKHR>& structures() { return structures_; }
    [[nodiscard]] core::FixedVector<VkMicromapEXT>& micromapList() { return micromapList_; }

private:
    const VolkDeviceTable* table_;
    VkDevice device_;
    VkQueryPool pool_;
    VkQueryType type_;
    std::uint32_t count_;
    bool micromaps_;
    core::FixedVector<VkAccelerationStructureKHR> structures_;
    core::FixedVector<VkMicromapEXT> micromapList_;
};

/// What one buildAccelerationStructure or accelerationStructureSizes call
/// fills: a geometry, a range and a primitive count for each of its
/// geometries, and the opacity micromap and motion vertices its triangles
/// chain where they have them. Made with the device at DeviceDesc::maxBuildGeometries, so no call
/// allocates, nor moves what a geometry points at; used by one call at a
/// time, under the device's lock.
struct VulkanBuildArrays {
    explicit VulkanBuildArrays(std::uint32_t maxBuildGeometries)
        : geometries(maxBuildGeometries),
          ranges(maxBuildGeometries),
          primitiveCounts(maxBuildGeometries),
          micromaps(maxBuildGeometries),
          motions(maxBuildGeometries) {}

    void clear() {
        geometries.clear();
        ranges.clear();
        primitiveCounts.clear();
        micromaps.clear();
        motions.clear();
    }

    core::FixedVector<VkAccelerationStructureGeometryKHR> geometries;
    core::FixedVector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    core::FixedVector<std::uint32_t> primitiveCounts;
    core::FixedVector<VkAccelerationStructureTrianglesOpacityMicromapEXT> micromaps;
    core::FixedVector<VkAccelerationStructureGeometryMotionTrianglesDataNV> motions;
};

} // namespace bfr::rhi::vulkan
