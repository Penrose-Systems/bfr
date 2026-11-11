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

/// The words of an AccelerationStructureCapture.
inline constexpr std::size_t kAccelerationStructureCaptureWords = 3;

/// What a backend needs to make a structure again at the reference it had,
/// as Device::captureAccelerationStructure gives it: opaque, the same on
/// every backend. All zero is none.
struct AccelerationStructureCapture {
    std::array<std::uint64_t, kAccelerationStructureCaptureWords> words{};
};

/// A structure of size bytes, in memory the device allocates. With
/// motionKeyframes, at most RayTracingLimits::maxMotionKeyframes,
/// where Features::rayTracingMotion, its builds have motion: a bottom one's
/// triangles move between keyframes, a top one holds motion instances, at
/// most motionInstances of them, passed to backends that take it. 0 is none.
/// With captureReplay, where Features::accelerationStructureCaptureReplay,
/// its reference may be captured, by Device::captureAccelerationStructure,
/// and replayed: with replay, what that gave for a structure made with this
/// desc but replay, on a device of the same adapter opened with the same
/// queues, it is made at the reference that one had, or fails where the
/// backend cannot give it. An opacity micromap has none where the backend
/// gives it no reference (Vulkan).
struct AccelerationStructureDesc {
    AccelerationStructureType type = AccelerationStructureType::Bottom;
    std::uint64_t size = 0;
    std::uint32_t motionKeyframes = 0;
    std::uint32_t motionInstances = 0;
    bool captureReplay = false;
    AccelerationStructureCapture replay{};
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
    /// Where bottomStructure is null, what the instance names its bottom
    /// structure by, for one BFR holds no handle to, such as a cluster bottom
    /// structure (Device::buildClusters); 0, with no bottomStructure, is an
    /// instance no ray hits.
    std::uint64_t bottomReference = 0;
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

/// What a cluster operation does: moves structures, builds cluster bottom
/// structures from clusters, builds clusters or cluster templates from
/// triangles, instantiates templates into clusters, or fetches templates'
/// indices, which is all Vulkan states of it. A cluster holds at most RayTracingLimits::maxClusterTriangles
/// triangles; a cluster bottom structure holds clusters and is what an
/// instance names; a template is a cluster built without its positions. The
/// values are Vulkan's.
enum class ClusterOperation : std::uint8_t {
    Move = 0,
    BuildBottoms = 1,
    BuildClusters = 2,
    BuildTemplates = 3,
    InstantiateTemplates = 4,
    TemplateIndices = 5,
};

/// Where a cluster operation's structures go: packed from one destination,
/// their addresses and sizes written; at addresses the GPU reads, their
/// sizes read; or nowhere, only their sizes written. The values are
/// Vulkan's.
enum class ClusterMode : std::uint8_t {
    ImplicitDestinations = 0,
    ExplicitDestinations = 1,
    ComputeSizes = 2,
};

/// What a Move moves. The values are Vulkan's.
enum class ClusterStructureType : std::uint8_t {
    Bottom = 0,
    Cluster = 1,
    Template = 2,
};

/// The upper bounds a cluster operation of clusters or templates is sized
/// for: vertices of vertexFormat, geometry indices to maxGeometryIndex, at
/// most maxUniqueGeometries of them, maxTriangles triangles and maxVertices
/// vertices in each, at most RayTracingLimits::maxClusterTriangles and
/// maxClusterVertices, maxTotalTriangles and maxTotalVertices in all, and no
/// record truncating fewer than minTruncateBits position bits, at most 32.
struct ClusterTriangleLimits {
    Format vertexFormat = Format::Undefined;
    std::uint32_t maxGeometryIndex = 0;
    std::uint32_t maxUniqueGeometries = 0;
    std::uint32_t maxTriangles = 0;
    std::uint32_t maxVertices = 0;
    std::uint32_t maxTotalTriangles = 0;
    std::uint32_t maxTotalVertices = 0;
    std::uint32_t minTruncateBits = 0;
};

/// The upper bounds a BuildBottoms is sized for: maxTotalClusters clusters
/// in all, maxClustersPerBottom in each bottom structure.
struct ClusterBottomLimits {
    std::uint32_t maxTotalClusters = 0;
    std::uint32_t maxClustersPerBottom = 0;
};

/// What a Move moves, and at most maxBytes of it; with noOverlap, no
/// destination overlaps a source, which stays valid, as after a copy.
struct ClusterMoveLimits {
    ClusterStructureType type = ClusterStructureType::Bottom;
    bool noOverlap = false;
    std::uint64_t maxBytes = 0;
};

/// A cluster operation's kind, mode and upper bounds: at most maxCount
/// structures, built with flags as other structures are; with
/// allowOpacityMicromaps, clusters may name opacity micromaps. The bounds
/// the operation reads: move for Move, bottoms for BuildBottoms, triangles
/// for the rest.
struct ClusterInputs {
    ClusterOperation operation = ClusterOperation::Move;
    ClusterMode mode = ClusterMode::ImplicitDestinations;
    AccelerationStructureFlags flags{};
    bool allowOpacityMicromaps = false;
    std::uint32_t maxCount = 0;
    ClusterMoveLimits move{};
    ClusterBottomLimits bottoms{};
    ClusterTriangleLimits triangles{};
};

/// Which of a ClusterBuild's regions hold, at their offset, the address of
/// the data rather than the data, which the GPU reads as it runs.
struct ClusterIndirection {
    bool destination = false;
    bool scratch = false;
    bool addresses = false;
    bool sizes = false;
    bool records = false;
    bool count = false;
};

/// A cluster operation, from inputs: the records of its kind from
/// recordOffset in records, recordStride bytes apart (0 packed), as many as
/// the uint32_t at countOffset, a multiple of 4, in count, or with no count
/// maxCount, never more than maxCount; records and count have
/// accelerationStructureInput usage. ImplicitDestinations packs the
/// structures from destinationOffset in destination, which has
/// accelerationStructureStorage usage and at least the size query's size
/// bytes, or for a Move the bytes moved, and writes their addresses to
/// addresses and their sizes to sizes, where given; ExplicitDestinations
/// reads each structure's address from addresses, each in a buffer with
/// accelerationStructureStorage usage, and its size from sizes, where given;
/// ComputeSizes writes each structure's size to sizes. Each destination is
/// a multiple of RayTracingLimits::clusterAlignment for clusters,
/// clusterTemplateAlignment for templates and clusterBottomAlignment for
/// bottom structures. addresses holds 8-byte addresses addressStride bytes
/// apart, at least 8, and has storage usage; sizes holds 4-byte sizes
/// sizeStride bytes apart, at least 4. Every buffer here has a usage
/// Device::bufferAddress gives an address for, and each of addresses, sizes
/// and records is read or written within its buffer, from its offset.
/// Scratch is scratchOffset bytes into scratch, which has
/// accelerationStructureScratch usage, at least the size query's
/// buildScratch bytes, a multiple of
/// RayTracingLimits::clusterScratchAlignment, and overlaps no destination.
struct ClusterBuild {
    ClusterInputs inputs{};
    BufferHandle destination{};
    std::uint64_t destinationOffset = 0;
    BufferHandle scratch{};
    std::uint64_t scratchOffset = 0;
    BufferHandle addresses{};
    std::uint64_t addressOffset = 0;
    std::uint64_t addressStride = 0;
    BufferHandle sizes{};
    std::uint64_t sizeOffset = 0;
    std::uint64_t sizeStride = 0;
    BufferHandle records{};
    std::uint64_t recordOffset = 0;
    std::uint64_t recordStride = 0;
    BufferHandle count{};
    std::uint64_t countOffset = 0;
    ClusterIndirection indirection{};
};

/// A Move record: the address of the structure moved. The layout the
/// backend reads.
struct ClusterMoveRecord {
    std::uint64_t source = 0;
};

/// A BuildBottoms record: a cluster bottom structure of clusterCount
/// clusters, each named once, whose addresses are clusterStride bytes apart,
/// at least 8, from clusters. The layout the backend reads.
struct ClusterBottomRecord {
    std::uint32_t clusterCount = 0;
    std::uint32_t clusterStride = 0;
    std::uint64_t clusters = 0;
};

/// The index types of a cluster record's indices and micromap indices. The
/// values are the records'.
enum class ClusterIndexFormat : std::uint32_t {
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 4,
};

/// The bits of a geometry word's flags. The values are the records'.
enum class ClusterGeometryFlag : std::uint32_t {
    CullDisable = 1,
    NoDuplicateAnyHit = 2,
    Opaque = 4,
};

/// The bits of a cluster record's clusterFlags. The values are the records'.
enum class ClusterFlag : std::uint32_t {
    /// Instances may disable its opacity micromaps.
    AllowDisableOpacityMicromaps = 1,
};

/// A geometry word: the geometry index in its low 24 bits, below
/// RayTracingLimits::maxClusterGeometryIndex, and ClusterGeometryFlag bits
/// from bit kClusterGeometryFlagsShift; the bits between are 0.
inline constexpr std::uint32_t kClusterGeometryFlagsShift = 29;

/// A cluster record's counts word: from bit 0 the triangles (9 bits), from
/// kClusterVertexCountShift the vertices (9 bits), from
/// kClusterTruncateShift the position bits truncated (6 bits), from
/// kClusterIndexFormatShift the indices' ClusterIndexFormat (4 bits), and
/// from kClusterMicromapIndexFormatShift the micromap indices' (4 bits).
inline constexpr std::uint32_t kClusterVertexCountShift = 9;
inline constexpr std::uint32_t kClusterTruncateShift = 18;
inline constexpr std::uint32_t kClusterIndexFormatShift = 24;
inline constexpr std::uint32_t kClusterMicromapIndexFormatShift = 28;

/// A BuildClusters record: a cluster of id clusterId, not 0xFFFFFFFF, with
/// clusterFlags (ClusterFlag bits) and counts (a counts word), at most
/// RayTracingLimits::maxClusterTriangles triangles and maxClusterVertices
/// vertices, its truncation at least the inputs' minTruncateBits.
/// baseGeometry is a geometry word for every triangle; geometries, where
/// not 0, holds a geometry word for each triangle, geometryStride apart,
/// which the backend combines with it.
/// indices and vertices are addresses, indexStride and vertexStride bytes
/// apart (0 packed); micromaps, where not 0, is an opacity micromap's
/// address, of operations with allowOpacityMicromaps, and micromapIndices,
/// where not 0, its indices, micromapIndexStride apart. The layout the
/// backend reads.
struct ClusterTriangleRecord {
    std::uint32_t clusterId = 0;
    std::uint32_t clusterFlags = 0;
    std::uint32_t counts = 0;
    std::uint32_t baseGeometry = 0;
    std::uint16_t indexStride = 0;
    std::uint16_t vertexStride = 0;
    std::uint16_t geometryStride = 0;
    std::uint16_t micromapIndexStride = 0;
    std::uint64_t indices = 0;
    std::uint64_t vertices = 0;
    std::uint64_t geometries = 0;
    std::uint64_t micromaps = 0;
    std::uint64_t micromapIndices = 0;
};

/// A BuildTemplates record: as a ClusterTriangleRecord, but vertices may be
/// 0, and boundsLimit, where not 0, is the address, a multiple of
/// RayTracingLimits::clusterTemplateBoundsAlignment, of a box of kBoxFloats
/// every instantiation lies within. The layout the backend reads.
struct ClusterTemplateRecord {
    ClusterTriangleRecord cluster{};
    std::uint64_t boundsLimit = 0;
};

/// An InstantiateTemplates record: the template at clusterTemplate, its
/// cluster id offset by clusterIdOffset and its geometry indices by
/// geometryIndexOffset (the low 24 bits, signed; the rest 0), with vertices
/// read from vertices, vertexStride bytes apart, or, where vertices is 0,
/// the template's own. The layout the backend reads.
struct ClusterInstantiateRecord {
    std::uint32_t clusterIdOffset = 0;
    std::uint32_t geometryIndexOffset = 0;
    std::uint64_t clusterTemplate = 0;
    std::uint64_t vertices = 0;
    std::uint64_t vertexStride = 0;
};

/// A TemplateIndices record: the address of the template whose indices are
/// written. The layout the backend reads.
struct ClusterTemplateIndicesRecord {
    std::uint64_t clusterTemplate = 0;
};

} // namespace bfr::rhi
