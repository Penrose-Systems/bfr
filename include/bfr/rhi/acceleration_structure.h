#pragma once

#include "bfr/core/handle.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Geometry or instances arranged for rays to be traced against, in memory
/// the device allocates. A bottom structure holds triangles or boxes; a top
/// one holds instances of bottom ones, and is what shaders trace rays
/// against. An opacity micromap holds micromaps, which say which parts of a
/// bottom structure's triangles are opaque. A partitioned top one holds
/// instances in partitions, which builds write, update and move from records
/// the GPU reads, rebuilding only the partitions they touch. Only named, by an
/// AccelerationStructureHandle.
struct AccelerationStructure;

using AccelerationStructureHandle = core::Handle<AccelerationStructure>;

enum class AccelerationStructureType : std::uint8_t {
    Bottom,
    Top,
    /// Where Features::opacityMicromap.
    OpacityMicromap,
    /// Where Features::partitionedTop: traced as a top one is, from its own
    /// table.
    PartitionedTop,
};

/// The states a micromap's subtriangles take: 1 bit each for two, opaque or
/// transparent, or 2 bits for four, adding unknown-opaque and
/// unknown-transparent, which any-hit shaders decide. The values are the
/// C API's.
enum class OpacityMicromapFormat : std::uint32_t {
    TwoState = 1,
    FourState = 2,
};

/// Micromap indices that name no micromap: the whole triangle in one state.
/// Written as they are into an index of TriangleOpacityMicromap::indexType.
enum class OpacityMicromapSpecialIndex : std::int32_t {
    FullyTransparent = -1,
    FullyOpaque = -2,
    FullyUnknownTransparent = -3,
    FullyUnknownOpaque = -4,
};

/// count micromaps, or triangles naming micromaps, of subdivisionLevel and
/// format.
struct OpacityMicromapUsage {
    std::uint32_t count = 0;
    std::uint32_t subdivisionLevel = 0;
    OpacityMicromapFormat format{};
};

/// One micromap, as a buffer holds it for an opacity micromap's build: its
/// states from dataOffset bytes into the build's data, of subdivisionLevel
/// and format (an OpacityMicromapFormat's value). The layout every backend
/// reads.
struct OpacityMicromapRecord {
    std::uint32_t dataOffset = 0;
    std::uint16_t subdivisionLevel = 0;
    std::uint16_t format = 0;
};

/// The micromaps a triangle geometry's triangles take their opacity from:
/// triangle i's is the micromap at the index read from indices (a null
/// indices is i), indexStride bytes apart from indexOffset, plus
/// baseTriangle, or an OpacityMicromapSpecialIndex. A null micromap is none,
/// where every index is special. usageCount usages from firstUsage in the
/// inputs' micromapUsages count the geometry's triangles of each subdivision
/// level and format, passed to backends that take them. None where micromap
/// and indices are both null.
struct TriangleOpacityMicromap {
    AccelerationStructureHandle micromap{};
    BufferHandle indices{};
    std::uint64_t indexOffset = 0;
    IndexType indexType = IndexType::Uint32;
    std::uint32_t indexStride = 0;
    std::uint32_t baseTriangle = 0;
    std::uint32_t firstUsage = 0;
    std::uint32_t usageCount = 0;
};

/// What a build favours; passed to the backend as given.
enum class BuildPreference : std::uint8_t {
    None,
    FastTrace,
    FastBuild,
};

struct AccelerationStructureFlags {
    BuildPreference preference = BuildPreference::None;
    /// Later builds may update the structure.
    bool allowUpdate = false;
    bool minimizeMemory = false;
    /// compactAccelerationStructure may copy it, and its CompactedSize be
    /// written.
    bool allowCompaction = false;
    /// Shaders read its triangles' positions, where Features::rayPositionFetch.
    bool allowDataAccess = false;
    /// Of a bottom structure with opacity micromaps, where
    /// Features::opacityMicromap: updates may name other micromaps, instances
    /// may disable them, and updates may follow their micromaps' data.
    bool allowOpacityMicromapUpdate = false;
    bool allowDisableOpacityMicromaps = false;
    bool allowOpacityMicromapDataUpdate = false;
};

/// Triangles in a bottom structure. A null indices is none: triangleCount * 3
/// vertices are read in order. A null transform is none; else a 3x4
/// row-major float matrix at transformOffset, a multiple of 16. vertexFormat
/// is one FormatCaps::accelerationStructureVertices is true for; vertexOffset
/// and vertexStride are multiples of its component size, indexOffset of the
/// index size. The buffers have accelerationStructureInput usage.
struct TriangleGeometry {
    BufferHandle vertices{};
    std::uint64_t vertexOffset = 0;
    Format vertexFormat = Format::Undefined;
    std::uint32_t vertexStride = 0;
    std::uint32_t vertexCount = 0;
    BufferHandle indices{};
    std::uint64_t indexOffset = 0;
    IndexType indexType = IndexType::Uint32;
    std::uint32_t triangleCount = 0;
    BufferHandle transform{};
    std::uint64_t transformOffset = 0;
    bool opaque = false;
    bool noDuplicateAnyHit = false;
    /// Where Features::opacityMicromap.
    TriangleOpacityMicromap opacityMicromap{};
    /// In a structure with motion, the vertices at each keyframe after the
    /// first, which vertices holds: keyframe k's from
    /// motionVertexOffset + (k - 1) * motionKeyframeStride in motionVertices,
    /// laid out as vertices are. Read only where the inputs' motionKeyframes
    /// is not 0. The buffer has accelerationStructureInput usage.
    BufferHandle motionVertices{};
    std::uint64_t motionVertexOffset = 0;
    std::uint64_t motionKeyframeStride = 0;
};

/// Boxes in a bottom structure: six floats each, min xyz then max xyz, min
/// at most max, stride bytes apart; offset and stride multiples of 8. The
/// buffer has accelerationStructureInput usage.
struct BoxGeometry {
    BufferHandle boxes{};
    std::uint64_t offset = 0;
    std::uint32_t stride = 0;
    std::uint32_t count = 0;
    bool opaque = false;
    bool noDuplicateAnyHit = false;
};

/// The floats of a 3x4 row-major transform.
inline constexpr std::size_t kTransformFloats = 12;

/// The partition index that names a partitioned top structure's global
/// partition, which holds instances apart from the others.
inline constexpr std::uint32_t kGlobalPartition = 0xFFFFFFFF;

/// What a PartitionedTopOp record does: writes instances, updates them, or
/// moves partitions. The values are the records'.
enum class PartitionedTopOpType : std::uint32_t {
    WriteInstance = 0,
    UpdateInstance = 1,
    WritePartitionTranslation = 2,
};

/// One kind of work in a partitioned top structure's build, as a buffer
/// holds it: count records of type (a PartitionedTopOpType's value), stride
/// bytes apart from address, a Device::bufferAddress plus an offset. No two
/// of a build's ops are of one type, and no instance is named twice in one.
/// The layout the backend reads.
struct PartitionedTopOp {
    std::uint32_t type = 0;
    std::uint32_t count = 0;
    std::uint64_t address = 0;
    std::uint64_t stride = 0;
};

/// The bits of a PartitionedInstanceRecord's flags. The values are the
/// records'.
enum class PartitionedInstanceFlag : std::uint32_t {
    CullDisable = 1,
    FrontCounterClockwise = 2,
    ForceOpaque = 4,
    ForceNonOpaque = 8,
    /// The instance's bounds are read, and later UpdateInstance records may
    /// name it.
    ExplicitBounds = 16,
};

/// The floats of a box: min xyz, then max xyz.
inline constexpr std::size_t kBoxFloats = 6;

/// A WriteInstance record: instance instanceIndex, below the structure's
/// instanceCount, of bottomReference (Device::accelerationStructureReference;
/// 0 is an instance no ray hits, which Vulkan may keep so), in partition
/// partitionIndex, below its partitionCount, or kGlobalPartition. transform
/// is a 3x4 row-major object-to-world transform; bounds, read with
/// ExplicitBounds, hold the bottom structures it will name, transformed,
/// before any partition translation. hitGroupOffset is below
/// kInstanceIndexLimit and mask is 8 bits. flags ORs PartitionedInstanceFlag
/// values. The layout the backend reads.
struct PartitionedInstanceRecord {
    std::array<float, kTransformFloats> transform{};
    std::array<float, kBoxFloats> bounds{};
    std::uint32_t customIndex = 0;
    std::uint32_t mask = 0;
    std::uint32_t hitGroupOffset = 0;
    std::uint32_t flags = 0;
    std::uint32_t instanceIndex = 0;
    std::uint32_t partitionIndex = 0;
    std::uint64_t bottomReference = 0;
};

/// An UpdateInstance record: instance instanceIndex, written with
/// ExplicitBounds or with a bottomReference of 0, takes bottomReference,
/// within its bounds, or 0, which disables it, and hitGroupOffset, below
/// kInstanceIndexLimit. One written with 0 may stay disabled, as Vulkan
/// says. The layout the backend reads.
struct PartitionedInstanceUpdateRecord {
    std::uint32_t instanceIndex = 0;
    std::uint32_t hitGroupOffset = 0;
    std::uint64_t bottomReference = 0;
};

/// A WritePartitionTranslation record: partition partitionIndex, below its
/// partitionCount, or kGlobalPartition, of a structure built with partitionTranslation, is
/// traced as if translation were added to its instances' transforms. The
/// layout the backend reads.
struct PartitionTranslationRecord {
    std::uint32_t partitionIndex = 0;
    std::array<float, 3> translation{};
};

/// A partitioned top structure's counts and work: instanceCount instances, in
/// partitionCount partitions of at most maxPartitionInstances each and the
/// global partition of at most maxGlobalInstances, partitionCount plus
/// maxGlobalInstances at most RayTracingLimits::maxPartitions; with
/// partitionTranslation, partitions may be moved. A build runs the
/// PartitionedTopOp records from opOffset in ops, as many as the GPU reads
/// from the uint32_t at opCountOffset, a multiple of 4, in opCount; both
/// buffers are given, with accelerationStructureInput usage. The size query reads the
/// flags, the counts and partitionTranslation.
struct PartitionedTopInputs {
    std::uint32_t instanceCount = 0;
    std::uint32_t maxPartitionInstances = 0;
    std::uint32_t partitionCount = 0;
    std::uint32_t maxGlobalInstances = 0;
    bool partitionTranslation = false;
    BufferHandle ops{};
    std::uint64_t opOffset = 0;
    BufferHandle opCount{};
    std::uint64_t opCountOffset = 0;
};

/// What a structure is built from. A bottom one holds triangles or boxes,
/// not both; a geometry's index in its span is the geometry index a hit
/// group is chosen by. A top one holds instanceCount instance records from
/// instanceOffset, a multiple of 16, in instances, which has
/// accelerationStructureInput usage, laid out as writeInstances writes them
/// (bfr_write_instances in api/bfr.yaml). A bottom one's triangles name
/// their micromaps' usages in micromapUsages. An opacity micromap holds
/// micromapUsages' micromaps: OpacityMicromapRecords, micromapRecordStride
/// bytes apart from micromapRecordOffset in
/// micromapRecords, whose states micromapData holds from micromapDataOffset;
/// both buffers have accelerationStructureInput usage, both offsets
/// multiples of 256. A partitioned top one is built from partitioned. The
/// size query reads the counts, formats, flags, usages and whether a
/// transform is given.
struct AccelerationStructureInputs {
    AccelerationStructureType type = AccelerationStructureType::Bottom;
    AccelerationStructureFlags flags{};
    std::span<const TriangleGeometry> triangles{};
    std::span<const BoxGeometry> boxes{};
    BufferHandle instances{};
    std::uint64_t instanceOffset = 0;
    std::uint32_t instanceCount = 0;
    std::span<const OpacityMicromapUsage> micromapUsages{};
    BufferHandle micromapData{};
    std::uint64_t micromapDataOffset = 0;
    BufferHandle micromapRecords{};
    std::uint64_t micromapRecordOffset = 0;
    std::uint32_t micromapRecordStride = 0;
    /// The structure's AccelerationStructureDesc::motionKeyframes: its
    /// triangles' motion vertices are read, or its instance records are
    /// writeMotionInstances'.
    std::uint32_t motionKeyframes = 0;
    PartitionedTopInputs partitioned{};
};

/// The bytes a structure takes, and the scratch bytes its build and its
/// update take. discardable: the backend says an opacity micromap may be
/// destroyed once the structures built with it are built.
struct AccelerationStructureSizes {
    std::uint64_t size = 0;
    std::uint64_t buildScratch = 0;
    std::uint64_t updateScratch = 0;
    bool discardable = false;
};

/// A structure of size bytes, in memory the device allocates. With
/// motionKeyframes, at most RayTracingLimits::maxMotionKeyframes,
/// where Features::rayTracingMotion, its builds have motion: a bottom one's
/// triangles move between keyframes, a top one holds motion instances, at
/// most motionInstances of them, passed to backends that take it. 0 is none.
struct AccelerationStructureDesc {
    AccelerationStructureType type = AccelerationStructureType::Bottom;
    std::uint64_t size = 0;
    std::uint32_t motionKeyframes = 0;
    std::uint32_t motionInstances = 0;
};

/// A build of dst, of inputs.type, from inputs; or, with update, an update
/// of src into dst (they may be the same), src built with allowUpdate and
/// inputs as its last build gave them but for the data the buffers hold. A
/// partitioned top one's update starts from src, a partitioned top one built
/// before, with no allowUpdate; its build starts from none. Scratch is
/// scratchOffset bytes into scratch, which has accelerationStructureScratch
/// usage, a multiple of RayTracingLimits::scratchAlignment, and of 256 for a
/// partitioned top one, at least the build's or update's scratch bytes, the
/// build's for both of a partitioned top one, and overlaps neither
/// structure.
struct AccelerationStructureBuild {
    AccelerationStructureHandle dst{};
    AccelerationStructureHandle src{};
    bool update = false;
    AccelerationStructureInputs inputs{};
    BufferHandle scratch{};
    std::uint64_t scratchOffset = 0;
};

/// One geometry's counts, as a buffer holds them for
/// Device::buildAccelerationStructureIndirect: primitiveCount triangles,
/// boxes or instances, from primitiveOffset bytes into the geometry's
/// indices, or its vertices, boxes or instances where it has none. A
/// triangle's vertices are firstVertex on from its indices, or from
/// primitiveOffset; its transform is transformOffset bytes in. The layout
/// every backend reads.
struct AccelerationStructureBuildRange {
    std::uint32_t primitiveCount = 0;
    std::uint32_t primitiveOffset = 0;
    std::uint32_t firstVertex = 0;
    std::uint32_t transformOffset = 0;
};

/// What a property pool's entries hold, one per structure a write names, of
/// RayTracingLimits::propertyEntryBytes each.
enum class AccelerationStructureProperty : std::uint8_t {
    /// The bytes a compact copy takes, of a structure built with
    /// allowCompaction.
    CompactedSize = 0,
    /// The bytes serializeAccelerationStructure writes, where
    /// Features::accelerationStructureSerialization.
    SerializedSize = 1,
    /// How many bottom structure references a serialized top structure
    /// holds, where Features::serializedBottomCount.
    SerializedBottomCount = 2,
    /// The bytes the structure takes now, where
    /// Features::accelerationStructureCurrentSize.
    CurrentSize = 3,
};

/// Entries of one AccelerationStructureProperty, written by lists and
/// resolved into a buffer. Calls on one pool must not overlap. Only named, by
/// an AccelerationStructurePropertyPoolHandle.
struct AccelerationStructurePropertyPool;

using AccelerationStructurePropertyPoolHandle = core::Handle<AccelerationStructurePropertyPool>;

/// count entries of property, at least 1, of opacity micromaps where
/// opacityMicromaps, and of other structures where not.
struct AccelerationStructurePropertyPoolDesc {
    AccelerationStructureProperty property = AccelerationStructureProperty::CompactedSize;
    std::uint32_t count = 0;
    bool opacityMicromaps = false;
};

/// The bytes of the identifier a serialized structure begins with, which
/// Device::serializedCompatible reads: two 16-byte identifiers, on every
/// backend that serializes.
inline constexpr std::size_t kSerializedIdentifierSize = 32;

/// Instance custom indices and hit group offsets are below this: every
/// backend's record holds them in 24 bits.
inline constexpr std::uint32_t kInstanceIndexLimit = 1U << 24U;

/// How an instance's geometry counts as opaque: as each geometry says, or
/// all opaque, or none.
enum class InstanceOpacity : std::uint8_t {
    AsGeometry,
    ForceOpaque,
    ForceNonOpaque,
};

/// One instance of a bottom structure in a top one. A null bottomStructure
/// is an instance no ray hits.
struct AccelerationStructureInstance {
    /// 3x4 row-major object-to-world transform.
    std::array<float, kTransformFloats> transform{};
    /// What the shader reads as the instance's custom index; below
    /// kInstanceIndexLimit.
    std::uint32_t customIndex = 0;
    /// Rays whose mask ANDs to 0 with it miss the instance.
    std::uint8_t mask = 0;
    /// The instance's offset, in records, into the hit region; below
    /// kInstanceIndexLimit.
    std::uint32_t hitGroupOffset = 0;
    bool cullDisable = false;
    bool frontCounterClockwise = false;
    InstanceOpacity opacity = InstanceOpacity::AsGeometry;
    /// Its opacity micromaps' unknown states are taken as known, or the
    /// micromaps ignored, of a bottom structure built with
    /// allowDisableOpacityMicromaps; where Features::opacityMicromap.
    bool forceOpacityMicromap2State = false;
    bool disableOpacityMicromaps = false;
    AccelerationStructureHandle bottomStructure{};
};

/// How a motion instance moves: not at all, by its keyframes' matrices, or
/// by their SrtTransforms. The values are the C API's.
enum class InstanceMotion : std::uint8_t {
    None = 0,
    Matrix = 1,
    Srt = 2,
};

/// A transform as its parts: scale, the shears a (xy), b (xz) and c (yz), and
/// a pivot added after them, then a rotation quaternion (x, y, z, w) about
/// the origin and a translation. Keyframes between are blended part by part.
struct SrtTransform {
    std::array<float, 3> scale{};
    std::array<float, 3> shear{};
    std::array<float, 3> pivot{};
    std::array<float, 4> rotation{};
    std::array<float, 3> translation{};
};

/// One keyframe of a motion instance's transform: matrix, a 3x4 row-major
/// transform, for Matrix motion; srt for Srt motion.
struct MotionKeyframe {
    std::array<float, kTransformFloats> matrix{};
    SrtTransform srt{};
};

/// An instance in a top structure with motion: instance, whose transform is
/// the instance's where motion is None, and keyframeCount keyframes from
/// firstKeyframe in the keyframes writeMotionInstances is given, the first
/// at time 0 and the last at 1, where it moves.
struct AccelerationStructureMotionInstance {
    AccelerationStructureInstance instance{};
    InstanceMotion motion = InstanceMotion::None;
    std::uint32_t firstKeyframe = 0;
    std::uint32_t keyframeCount = 0;
};

} // namespace bfr::rhi
