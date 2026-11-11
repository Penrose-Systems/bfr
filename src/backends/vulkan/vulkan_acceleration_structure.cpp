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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
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
      motion_(desc.motionKeyframes != 0),
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
      motion_(other.motion_),
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

// Fails as checkMotion does, and with RhiInvalidCall for motion keyframes
// other than the two Vulkan's motion structures hold, or any on an opacity
// micromap, whose create info has no motion (VkMicromapCreateInfoEXT;
// VUID-VkAccelerationStructureCreateInfo2KHR-createFlags-11611).
core::Result<void> checkVulkanMotion(bool rayTracingMotion, AccelerationStructureType type,
                                     std::uint32_t motionKeyframes) {
    if (auto checked = checkMotion(rayTracingMotion, motionKeyframes); !checked) {
        return checked;
    }
    if (motionKeyframes != 0 && type == AccelerationStructureType::OpacityMicromap) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "Vulkan's opacity micromaps have no motion");
    }
    // VkPartitionedAccelerationStructureInstancesInputNV has no motion.
    if (motionKeyframes != 0 && type == AccelerationStructureType::PartitionedTop) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "Vulkan's partitioned top structures have no motion");
    }
    if (motionKeyframes != 0 && motionKeyframes != kVulkanMotionKeyframes) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "Vulkan's structures with motion have {} keyframes, not {}", kVulkanMotionKeyframes,
                             motionKeyframes);
    }
    return {};
}

// Fails with RhiInvalidCall for a partitioned top structure, which Vulkan
// holds as memory with no VkAccelerationStructureKHR, so it has no copy,
// compaction, serialization or property write of one
// (VK_NV_partitioned_acceleration_structure).
core::Result<void> checkNotPartitioned(AccelerationStructureType type, std::string_view call) {
    if (type == AccelerationStructureType::PartitionedTop) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "Vulkan {} no partitioned top structure", call);
    }
    return {};
}

// The records a partitioned top structure's build reads in place: BFR's
// layouts are Vulkan's.
static_assert(sizeof(PartitionedTopOp) == sizeof(VkBuildPartitionedAccelerationStructureIndirectCommandNV) &&
              offsetof(PartitionedTopOp, count) ==
                  offsetof(VkBuildPartitionedAccelerationStructureIndirectCommandNV, argCount) &&
              offsetof(PartitionedTopOp, address) ==
                  offsetof(VkBuildPartitionedAccelerationStructureIndirectCommandNV, argData.startAddress) &&
              offsetof(PartitionedTopOp, stride) ==
                  offsetof(VkBuildPartitionedAccelerationStructureIndirectCommandNV, argData.strideInBytes));
static_assert(std::to_underlying(PartitionedTopOpType::WriteInstance) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV &&
              std::to_underlying(PartitionedTopOpType::UpdateInstance) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_UPDATE_INSTANCE_NV &&
              std::to_underlying(PartitionedTopOpType::WritePartitionTranslation) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_PARTITION_TRANSLATION_NV);
static_assert(sizeof(PartitionedInstanceRecord) == sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV) &&
              offsetof(PartitionedInstanceRecord, bounds) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, explicitAABB) &&
              offsetof(PartitionedInstanceRecord, customIndex) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, instanceID) &&
              offsetof(PartitionedInstanceRecord, mask) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, instanceMask) &&
              offsetof(PartitionedInstanceRecord, hitGroupOffset) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV,
                           instanceContributionToHitGroupIndex) &&
              offsetof(PartitionedInstanceRecord, flags) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, instanceFlags) &&
              offsetof(PartitionedInstanceRecord, instanceIndex) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, instanceIndex) &&
              offsetof(PartitionedInstanceRecord, partitionIndex) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, partitionIndex) &&
              offsetof(PartitionedInstanceRecord, bottomReference) ==
                  offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, accelerationStructure));
static_assert(sizeof(PartitionedInstanceUpdateRecord) ==
                  sizeof(VkPartitionedAccelerationStructureUpdateInstanceDataNV) &&
              offsetof(PartitionedInstanceUpdateRecord, hitGroupOffset) ==
                  offsetof(VkPartitionedAccelerationStructureUpdateInstanceDataNV,
                           instanceContributionToHitGroupIndex) &&
              offsetof(PartitionedInstanceUpdateRecord, bottomReference) ==
                  offsetof(VkPartitionedAccelerationStructureUpdateInstanceDataNV, accelerationStructure));
static_assert(sizeof(PartitionTranslationRecord) ==
                  sizeof(VkPartitionedAccelerationStructureWritePartitionTranslationDataNV) &&
              offsetof(PartitionTranslationRecord, translation) ==
                  offsetof(VkPartitionedAccelerationStructureWritePartitionTranslationDataNV, partitionTranslation));
static_assert(kGlobalPartition == VK_PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV);
static_assert(std::to_underlying(PartitionedInstanceFlag::CullDisable) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE_BIT_NV &&
              std::to_underlying(PartitionedInstanceFlag::FrontCounterClockwise) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FLIP_FACING_BIT_NV &&
              std::to_underlying(PartitionedInstanceFlag::ForceOpaque) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_OPAQUE_BIT_NV &&
              std::to_underlying(PartitionedInstanceFlag::ForceNonOpaque) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_NO_OPAQUE_BIT_NV &&
              std::to_underlying(PartitionedInstanceFlag::ExplicitBounds) ==
                  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_ENABLE_EXPLICIT_BOUNDING_BOX_NV);

// A partitioned top structure's counts and flags as Vulkan takes them,
// chaining translation, which the caller keeps while the input is read.
VkPartitionedAccelerationStructureInstancesInputNV partitionedInputOf(
    const AccelerationStructureInputs& inputs, VkPartitionedAccelerationStructureFlagsNV& translation) {
    translation = {};
    translation.sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV;
    translation.enablePartitionTranslation = inputs.partitioned.partitionTranslation ? VK_TRUE : VK_FALSE;
    VkPartitionedAccelerationStructureInstancesInputNV input{};
    input.sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV;
    input.pNext = &translation;
    input.flags = flagsFor(inputs.flags);
    input.instanceCount = inputs.partitioned.instanceCount;
    input.maxInstancePerPartitionCount = inputs.partitioned.maxPartitionInstances;
    input.partitionCount = inputs.partitioned.partitionCount;
    input.maxInstanceInGlobalPartitionCount = inputs.partitioned.maxGlobalInstances;
    return input;
}

// Cluster records are read in place on the GPU, so BFR's are Vulkan's; a
// word Vulkan splits into bit-fields is one uint32_t here.
static_assert(sizeof(ClusterMoveRecord) == sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV));
static_assert(sizeof(ClusterBottomRecord) == sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) &&
              offsetof(ClusterBottomRecord, clusterStride) ==
                  offsetof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV, clusterReferencesStride) &&
              offsetof(ClusterBottomRecord, clusters) ==
                  offsetof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV, clusterReferences));
static_assert(sizeof(ClusterTriangleRecord) == sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV) &&
              offsetof(ClusterTriangleRecord, clusterFlags) ==
                  offsetof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV, clusterFlags) &&
              offsetof(ClusterTriangleRecord, baseGeometry) ==
                  offsetof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV,
                           baseGeometryIndexAndGeometryFlags) &&
              offsetof(ClusterTriangleRecord, indexStride) ==
                  offsetof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV, indexBufferStride) &&
              offsetof(ClusterTriangleRecord, micromapIndexStride) ==
                  offsetof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV,
                           opacityMicromapIndexBufferStride) &&
              offsetof(ClusterTriangleRecord, indices) ==
                  offsetof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV, indexBuffer) &&
              offsetof(ClusterTriangleRecord, micromapIndices) ==
                  offsetof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV, opacityMicromapIndexBuffer));
static_assert(sizeof(ClusterTemplateRecord) ==
                  sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV) &&
              offsetof(ClusterTemplateRecord, boundsLimit) ==
                  offsetof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV,
                           instantiationBoundingBoxLimit));
static_assert(sizeof(ClusterInstantiateRecord) == sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) &&
              offsetof(ClusterInstantiateRecord, clusterTemplate) ==
                  offsetof(VkClusterAccelerationStructureInstantiateClusterInfoNV, clusterTemplateAddress) &&
              offsetof(ClusterInstantiateRecord, vertices) ==
                  offsetof(VkClusterAccelerationStructureInstantiateClusterInfoNV, vertexBuffer));
static_assert(sizeof(ClusterTemplateIndicesRecord) == sizeof(VkClusterAccelerationStructureGetTemplateIndicesInfoNV));
static_assert(std::to_underlying(ClusterOperation::Move) == VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV &&
              std::to_underlying(ClusterOperation::BuildBottoms) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV &&
              std::to_underlying(ClusterOperation::BuildClusters) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV &&
              std::to_underlying(ClusterOperation::BuildTemplates) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV &&
              std::to_underlying(ClusterOperation::InstantiateTemplates) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV &&
              std::to_underlying(ClusterOperation::TemplateIndices) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_GET_CLUSTER_TEMPLATE_INDICES_NV);
static_assert(std::to_underlying(ClusterMode::ImplicitDestinations) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV &&
              std::to_underlying(ClusterMode::ExplicitDestinations) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV &&
              std::to_underlying(ClusterMode::ComputeSizes) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV);
static_assert(std::to_underlying(ClusterStructureType::Bottom) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_CLUSTERS_BOTTOM_LEVEL_NV &&
              std::to_underlying(ClusterStructureType::Cluster) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV &&
              std::to_underlying(ClusterStructureType::Template) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_TEMPLATE_NV);
static_assert(std::to_underlying(ClusterIndexFormat::Uint8) == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV &&
              std::to_underlying(ClusterIndexFormat::Uint16) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV &&
              std::to_underlying(ClusterIndexFormat::Uint32) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV);
static_assert(std::to_underlying(ClusterGeometryFlag::CullDisable) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_CULL_DISABLE_BIT_NV &&
              std::to_underlying(ClusterGeometryFlag::NoDuplicateAnyHit) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_NO_DUPLICATE_ANYHIT_INVOCATION_BIT_NV &&
              std::to_underlying(ClusterGeometryFlag::Opaque) ==
                  VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV);
static_assert(std::to_underlying(ClusterFlag::AllowDisableOpacityMicromaps) ==
              VK_CLUSTER_ACCELERATION_STRUCTURE_CLUSTER_ALLOW_DISABLE_OPACITY_MICROMAPS_NV);

// A cluster operation's kind, mode and bounds as Vulkan takes them; info
// points at the bound the operation reads, held beside it.
struct VulkanClusterInput {
    VkClusterAccelerationStructureInputInfoNV info{};
    VkClusterAccelerationStructureMoveObjectsInputNV move{};
    VkClusterAccelerationStructureClustersBottomLevelInputNV bottoms{};
    VkClusterAccelerationStructureTriangleClusterInputNV triangles{};

    explicit VulkanClusterInput(const ClusterInputs& inputs) {
        info.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
        info.maxAccelerationStructureCount = inputs.maxCount;
        info.flags =
            flagsFor(inputs.flags) |
            (inputs.allowOpacityMicromaps
                 ? VkBuildAccelerationStructureFlagsKHR{VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_CLUSTER_OPACITY_MICROMAPS_BIT_NV}
                 : 0U);
        info.opType = static_cast<VkClusterAccelerationStructureOpTypeNV>(inputs.operation);
        info.opMode = static_cast<VkClusterAccelerationStructureOpModeNV>(inputs.mode);
        switch (inputs.operation) {
            case ClusterOperation::Move:
                move.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV;
                move.type = static_cast<VkClusterAccelerationStructureTypeNV>(inputs.move.type);
                move.noMoveOverlap = inputs.move.noOverlap ? VK_TRUE : VK_FALSE;
                move.maxMovedBytes = inputs.move.maxBytes;
                info.opInput.pMoveObjects = &move;
                break;
            case ClusterOperation::BuildBottoms:
                bottoms.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
                bottoms.maxTotalClusterCount = inputs.bottoms.maxTotalClusters;
                bottoms.maxClusterCountPerAccelerationStructure = inputs.bottoms.maxClustersPerBottom;
                info.opInput.pClustersBottomLevel = &bottoms;
                break;
            case ClusterOperation::BuildClusters:
            case ClusterOperation::BuildTemplates:
            case ClusterOperation::InstantiateTemplates:
            case ClusterOperation::TemplateIndices:
                triangles.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
                triangles.vertexFormat = vulkanFormat(inputs.triangles.vertexFormat);
                triangles.maxGeometryIndexValue = inputs.triangles.maxGeometryIndex;
                triangles.maxClusterUniqueGeometryCount = inputs.triangles.maxUniqueGeometries;
                triangles.maxClusterTriangleCount = inputs.triangles.maxTriangles;
                triangles.maxClusterVertexCount = inputs.triangles.maxVertices;
                triangles.maxTotalTriangleCount = inputs.triangles.maxTotalTriangles;
                triangles.maxTotalVertexCount = inputs.triangles.maxTotalVertices;
                triangles.minPositionTruncateBitCount = inputs.triangles.minTruncateBits;
                info.opInput.pTriangleClusters = &triangles;
                break;
        }
    }
    VulkanClusterInput(const VulkanClusterInput&) = delete;
    VulkanClusterInput& operator=(const VulkanClusterInput&) = delete;
    VulkanClusterInput(VulkanClusterInput&&) = delete;
    VulkanClusterInput& operator=(VulkanClusterInput&&) = delete;
    ~VulkanClusterInput() = default;
};

VkClusterAccelerationStructureAddressResolutionFlagsNV resolutionFor(const ClusterIndirection& indirection) {
    const auto flag = [](bool set, VkClusterAccelerationStructureAddressResolutionFlagBitsNV bit) {
        return set ? VkClusterAccelerationStructureAddressResolutionFlagsNV{bit} : 0U;
    };
    return flag(indirection.destination,
                VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_IMPLICIT_DATA_BIT_NV) |
           flag(indirection.scratch,
                VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SCRATCH_DATA_BIT_NV) |
           flag(indirection.addresses,
                VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_ADDRESS_ARRAY_BIT_NV) |
           flag(indirection.sizes,
                VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_SIZES_ARRAY_BIT_NV) |
           flag(indirection.records,
                VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SRC_INFOS_ARRAY_BIT_NV) |
           flag(indirection.count,
                VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SRC_INFOS_COUNT_BIT_NV);
}

// usages as Vulkan reads them, in place: the layouts and format values are
// the same.
const VkMicromapUsageEXT* usagesOf(std::span<const OpacityMicromapUsage> usages) {
    static_assert(sizeof(OpacityMicromapUsage) == sizeof(VkMicromapUsageEXT) &&
                  offsetof(OpacityMicromapUsage, subdivisionLevel) == offsetof(VkMicromapUsageEXT, subdivisionLevel) &&
                  offsetof(OpacityMicromapUsage, format) == offsetof(VkMicromapUsageEXT, format));
    static_assert(std::to_underlying(OpacityMicromapFormat::TwoState) == VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT &&
                  std::to_underlying(OpacityMicromapFormat::FourState) == VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT);
    // Special indices are written into index buffers as they are.
    static_assert(std::to_underlying(OpacityMicromapSpecialIndex::FullyTransparent) ==
                      VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT_EXT &&
                  std::to_underlying(OpacityMicromapSpecialIndex::FullyOpaque) ==
                      VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT &&
                  std::to_underlying(OpacityMicromapSpecialIndex::FullyUnknownTransparent) ==
                      VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_TRANSPARENT_EXT &&
                  std::to_underlying(OpacityMicromapSpecialIndex::FullyUnknownOpaque) ==
                      VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT);
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
            // Vulkan's second keyframe, at time 1, is the only one after the
            // first.
            if (inputs.motionKeyframes != 0 && !triangles.motionVertices.isNull()) {
                const auto moved = addressOf(triangles.motionVertices, triangles.motionVertexOffset);
                if (!moved) {
                    return std::unexpected(moved.error());
                }
                VkAccelerationStructureGeometryMotionTrianglesDataNV motion{};
                motion.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV;
                motion.pNext = data.pNext;
                motion.vertexData.deviceAddress = *moved;
                std::ignore = arrays.motions.push(motion);
                data.pNext = &arrays.motions[arrays.motions.size() - 1];
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
    info.flags = flagsFor(inputs.flags) |
                 (inputs.motionKeyframes != 0
                      ? VkBuildAccelerationStructureFlagsKHR{VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV}
                      : 0U);
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
    if ((*found)->address() == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a buffer read or written by address has no usage that gives it one");
    }
    return (*found)->address() + offset;
}

core::Result<VkStridedDeviceAddressRegionKHR> VulkanDevice::regionOf(BufferHandle buffer, std::uint64_t offset,
                                                                     std::uint64_t stride) {
    if (buffer.isNull()) {
        return VkStridedDeviceAddressRegionKHR{};
    }
    const auto address = addressOf(buffer, offset);
    if (!address) {
        return std::unexpected(address.error());
    }
    // The region is the buffer's bytes from offset, as the caller made it.
    const VkDeviceSize size = (*bufferFor(buffer))->size();
    return VkStridedDeviceAddressRegionKHR{.deviceAddress = *address,
                                           .stride = stride,
                                           .size = size > offset ? size - offset : 0};
}

core::Result<AccelerationStructureSizes> VulkanDevice::accelerationStructureSizes(
    const AccelerationStructureInputs& inputs) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkInputs(inputs, features_); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkVulkanMotion(features_.rayTracingMotion, inputs.type, inputs.motionKeyframes); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto counted = countGeometries(inputs, desc_.maxBuildGeometries); !counted) {
        return std::unexpected(std::move(counted.error()));
    }
    if (inputs.type == AccelerationStructureType::PartitionedTop) {
        VkPartitionedAccelerationStructureFlagsNV translation{};
        const VkPartitionedAccelerationStructureInstancesInputNV input = partitionedInputOf(inputs, translation);
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        table_.vkGetPartitionedAccelerationStructuresBuildSizesNV(device_, &input, &sizes);
        return AccelerationStructureSizes{.size = sizes.accelerationStructureSize,
                                          .buildScratch = sizes.buildScratchSize,
                                          .updateScratch = sizes.updateScratchSize};
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
    if (auto checked = checkPartitionedTops(features_.partitionedTop, desc.type); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkVulkanMotion(features_.rayTracingMotion, desc.type, desc.motionKeyframes); !checked) {
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
    const auto giveBack = [this, &slot, &desc] {
        bindless_.giveAccelerationStructure(desc.type, *slot);
    };
    if (desc.type == AccelerationStructureType::PartitionedTop) {
        // A partitioned top structure is the memory at an address, a multiple
        // of 256 (VUID-vkCmdBuildPartitionedAccelerationStructuresNV-pBuildInfo-10544, -10545,
        // -10552), with no object of its own.
        auto storage = placeVulkanBuffer(table_, device_, allocator_, desc.size,
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         BufferMemory::Device, families_, kPartitionedAlignment,
                                         core::SlotChains::kEnd);
        if (!storage) {
            giveBack();
            return std::unexpected(std::move(storage.error()));
        }
        const VkDeviceAddress address = storage->address();
        const AccelerationStructureHandle made = accelerationStructures_.emplace(
            VulkanAccelerationStructure{table_, device_, std::move(*storage), VK_NULL_HANDLE, VK_NULL_HANDLE, desc,
                                        *slot, address});
        if (made.isNull()) {
            giveBack();
            return core::failure(core::ErrorCode::RhiLimitReached,
                                 "the device holds at most {} acceleration structures", kBufferLimit);
        }
        if (*slot != core::SlotChains::kEnd) {
            layout_.writePartitionedTop(*slot, address);
        }
        return made;
    }
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
    VkAccelerationStructureMotionInfoNV motion{};
    motion.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV;
    motion.maxInstances = desc.motionInstances;
    if (desc.motionKeyframes != 0) {
        info.createFlags |= VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
        info.pNext = &motion;
    }
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
    if (auto checked = checkInstances(instances, out, rayTracing_.instanceSize, features_.opacityMicromap); !checked) {
        return checked;
    }
    for (std::size_t i = 0; i < instances.size(); ++i) {
        const auto record = instanceRecord(instances[i], i);
        if (!record) {
            return std::unexpected(record.error());
        }
        std::memcpy(out.subspan(i * sizeof(*record)).data(), &*record, sizeof(*record));
    }
    return {};
}

core::Result<VkAccelerationStructureInstanceKHR> VulkanDevice::instanceRecord(
    const AccelerationStructureInstance& instance, std::size_t index) {
    VkDeviceAddress reference = instance.bottomReference;
    if (!instance.bottomStructure.isNull()) {
        auto found = accelerationStructureFor(instance.bottomStructure);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        if ((*found)->type() != AccelerationStructureType::Bottom) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "instance {} names a structure that is not a bottom one; instances are of bottom ones",
                                 index);
        }
        reference = (*found)->reference();
    }
    VkAccelerationStructureInstanceKHR record{};
    static_assert(sizeof(record.transform) == sizeof(instance.transform));
    std::memcpy(&record.transform, instance.transform.data(), sizeof(record.transform));
    // checkInstance has seen the indices fit their 24 bits, and the flags are
    // 8 bits' worth.
    record.instanceCustomIndex = instance.customIndex & kIndexBits;
    record.mask = instance.mask;
    record.instanceShaderBindingTableRecordOffset = instance.hitGroupOffset & kIndexBits;
    record.flags = static_cast<std::uint8_t>(instanceFlagsFor(instance));
    record.accelerationStructureReference = reference;
    return record;
}

namespace {

VkSRTDataNV srtFor(const SrtTransform& srt) {
    return {.sx = srt.scale[0],
            .a = srt.shear[0],
            .b = srt.shear[1],
            .pvx = srt.pivot[0],
            .sy = srt.scale[1],
            .c = srt.shear[2],
            .pvy = srt.pivot[1],
            .sz = srt.scale[2],
            .pvz = srt.pivot[2],
            .qx = srt.rotation[0],
            .qy = srt.rotation[1],
            .qz = srt.rotation[2],
            .qw = srt.rotation[3],
            .tx = srt.translation[0],
            .ty = srt.translation[1],
            .tz = srt.translation[2]};
}

// The fields a motion record shares with a static one, from record.
template <class Motion>
void copyFields(const VkAccelerationStructureInstanceKHR& record, Motion& motion) {
    motion.instanceCustomIndex = record.instanceCustomIndex;
    motion.mask = record.mask;
    motion.instanceShaderBindingTableRecordOffset = record.instanceShaderBindingTableRecordOffset;
    motion.flags = record.flags;
    motion.accelerationStructureReference = record.accelerationStructureReference;
}

} // namespace

core::Result<void> VulkanDevice::writeMotionInstances(std::span<const AccelerationStructureMotionInstance> instances,
                                                      std::span<const MotionKeyframe> keyframes,
                                                      std::span<std::byte> out) {
    if (auto checked = checkAccelerationStructures(features_.accelerationStructure); !checked) {
        return checked;
    }
    // The last record takes only its own bytes, short of the stride
    // (accelstructures.adoc:1796-1797, 1981-1984).
    if (auto checked = checkMotionInstances(features_.rayTracingMotion, instances, keyframes, out,
                                            rayTracing_.motionInstanceSize,
                                            sizeof(VkAccelerationStructureMotionInstanceNV), features_.opacityMicromap);
        !checked) {
        return checked;
    }
    static_assert(sizeof(VkAccelerationStructureMotionInstanceNV) <= kVulkanMotionInstanceStride);
    for (std::size_t i = 0; i < instances.size(); ++i) {
        const AccelerationStructureMotionInstance& moving = instances[i];
        if (moving.motion != InstanceMotion::None && moving.keyframeCount != kVulkanMotionKeyframes) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "motion instance {} has {} keyframes; Vulkan's record holds {}", i,
                                 moving.keyframeCount, kVulkanMotionKeyframes);
        }
        const auto record = instanceRecord(moving.instance, i);
        if (!record) {
            return std::unexpected(record.error());
        }
        VkAccelerationStructureMotionInstanceNV motion{};
        switch (moving.motion) {
            case InstanceMotion::Matrix: {
                motion.type = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV;
                VkAccelerationStructureMatrixMotionInstanceNV& matrix = motion.data.matrixMotionInstance;
                const MotionKeyframe& first = keyframes[moving.firstKeyframe];
                const MotionKeyframe& last = keyframes[moving.firstKeyframe + 1];
                static_assert(sizeof(matrix.transformT0) == sizeof(first.matrix));
                std::memcpy(&matrix.transformT0, first.matrix.data(), sizeof(matrix.transformT0));
                std::memcpy(&matrix.transformT1, last.matrix.data(), sizeof(matrix.transformT1));
                copyFields(*record, matrix);
                break;
            }
            case InstanceMotion::Srt: {
                motion.type = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV;
                VkAccelerationStructureSRTMotionInstanceNV& srt = motion.data.srtMotionInstance;
                srt.transformT0 = srtFor(keyframes[moving.firstKeyframe].srt);
                srt.transformT1 = srtFor(keyframes[moving.firstKeyframe + 1].srt);
                copyFields(*record, srt);
                break;
            }
            case InstanceMotion::None:
            default:
                motion.type = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV;
                motion.data.staticInstance = *record;
                break;
        }
        // The stride's padding past the record is left as out holds it.
        std::memcpy(out.subspan(i * kVulkanMotionInstanceStride).data(), &motion, sizeof(motion));
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
    if (auto checked = checkInputs(build.inputs, features_); !checked) {
        return checked;
    }
    if (auto checked = checkVulkanMotion(features_.rayTracingMotion, build.inputs.type, build.inputs.motionKeyframes);
        !checked) {
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
    // A structure made with motion is built with it, and one made without,
    // without (VUID-VkAccelerationStructureBuildGeometryInfoKHR-dstAccelerationStructure-04927, -flags-04928).
    if ((*dst)->motion() != (build.inputs.motionKeyframes != 0)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a build's inputs have motion unlike the structure it builds");
    }
    const VulkanAccelerationStructure* from = nullptr;
    if (build.update) {
        auto found = accelerationStructureFor(build.src);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        from = *found;
        // An update starts from a structure Vulkan names as dst's kind is
        // named: a partitioned top one by address, the others by object.
        if ((from->type() == AccelerationStructureType::PartitionedTop) !=
            (build.inputs.type == AccelerationStructureType::PartitionedTop)) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "a partitioned top structure updates only from a partitioned top one, and only it "
                                 "from one");
        }
    }
    const auto scratch = addressOf(build.scratch, build.scratchOffset);
    if (!scratch) {
        return std::unexpected(scratch.error());
    }
    if (build.inputs.type == AccelerationStructureType::PartitionedTop) {
        return recordPartitionedBuild((*list)->buffer, build, **dst, from, *scratch, ranges != nullptr);
    }
    const VkAccelerationStructureKHR src = from != nullptr ? from->structure() : VK_NULL_HANDLE;
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

core::Result<void> VulkanDevice::recordPartitionedBuild(VkCommandBuffer buffer, const AccelerationStructureBuild& build,
                                                        const VulkanAccelerationStructure& dst,
                                                        const VulkanAccelerationStructure* src, VkDeviceAddress scratch,
                                                        bool indirect) {
    // Every build reads its counts on the GPU; there is no other indirect
    // form (vkCmdBuildPartitionedAccelerationStructuresNV).
    if (indirect) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a partitioned top structure's build reads its counts on the GPU already; it has no "
                             "indirect build");
    }
    const PartitionedTopInputs& partitioned = build.inputs.partitioned;
    const auto ops = addressOf(partitioned.ops, partitioned.opOffset);
    if (!ops) {
        return std::unexpected(ops.error());
    }
    const auto opCount = addressOf(partitioned.opCount, partitioned.opCountOffset);
    if (!opCount) {
        return std::unexpected(opCount.error());
    }
    VkPartitionedAccelerationStructureFlagsNV translation{};
    VkBuildPartitionedAccelerationStructureInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV;
    info.input = partitionedInputOf(build.inputs, translation);
    info.srcAccelerationStructureData = src != nullptr ? src->reference() : 0;
    info.dstAccelerationStructureData = dst.reference();
    info.scratchData = scratch;
    info.srcInfos = *ops;
    info.srcInfosCount = *opCount;
    table_.vkCmdBuildPartitionedAccelerationStructuresNV(buffer, &info);
    return {};
}

core::Result<AccelerationStructureSizes> VulkanDevice::clusterSizes(const ClusterInputs& inputs) {
    if (auto checked = checkClusterInputs(inputs, features_); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const VulkanClusterInput input{inputs};
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    table_.vkGetClusterAccelerationStructureBuildSizesNV(device_, &input.info, &sizes);
    return AccelerationStructureSizes{.size = sizes.accelerationStructureSize,
                                      .buildScratch = sizes.buildScratchSize,
                                      .updateScratch = sizes.updateScratchSize};
}

core::Result<void> VulkanDevice::buildClusters(CommandListHandle commandList, const ClusterBuild& build) {
    if (auto checked = checkClusterInputs(build.inputs, features_); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRunsCompute(**list); !checked) {
        return checked;
    }
    const auto destination = addressOf(build.destination, build.destinationOffset);
    const auto scratch = addressOf(build.scratch, build.scratchOffset);
    const auto count = addressOf(build.count, build.countOffset);
    const auto addresses = regionOf(build.addresses, build.addressOffset, build.addressStride);
    const auto sizes = regionOf(build.sizes, build.sizeOffset, build.sizeStride);
    const auto records = regionOf(build.records, build.recordOffset, build.recordStride);
    for (const auto* address : {&destination, &scratch, &count}) {
        if (!*address) {
            return std::unexpected(address->error());
        }
    }
    for (const auto* region : {&addresses, &sizes, &records}) {
        if (!*region) {
            return std::unexpected(region->error());
        }
    }
    const VulkanClusterInput input{build.inputs};
    VkClusterAccelerationStructureCommandsInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    info.input = input.info;
    info.dstImplicitData = *destination;
    info.scratchData = *scratch;
    info.dstAddressesArray = *addresses;
    info.dstSizesArray = *sizes;
    info.srcInfosArray = *records;
    info.srcInfosCount = *count;
    info.addressResolutionFlags = resolutionFor(build.indirection);
    table_.vkCmdBuildClusterAccelerationStructureIndirectNV((*list)->buffer, &info);
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
    for (const VulkanAccelerationStructure* named : {*from, *to}) {
        if (auto checked = checkNotPartitioned(named->type(), "copies"); !checked) {
            return checked;
        }
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
        if (auto checked = checkNotPartitioned((*named)->type(), "writes properties of"); !checked) {
            return checked;
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
    if (auto checked = checkNotPartitioned((*from)->type(), "serializes"); !checked) {
        return checked;
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
    if (auto checked = checkNotPartitioned((*to)->type(), "deserializes"); !checked) {
        return checked;
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
    if (auto checked = checkNotPartitioned(type, "serializes"); !checked) {
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
