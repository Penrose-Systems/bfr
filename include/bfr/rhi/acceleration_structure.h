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
/// against. Only named, by an AccelerationStructureHandle.
struct AccelerationStructure;

using AccelerationStructureHandle = core::Handle<AccelerationStructure>;

enum class AccelerationStructureType : std::uint8_t {
    Bottom,
    Top,
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

/// What a structure is built from. A bottom one holds triangles or boxes,
/// not both; a geometry's index in its span is the geometry index a hit
/// group is chosen by. A top one holds instanceCount instance records from
/// instanceOffset, a multiple of 16, in instances, which has
/// accelerationStructureInput usage, laid out as writeInstances writes them
/// (bfr_write_instances in api/bfr.yaml). The size query reads the counts, formats, flags and
/// whether a transform is given.
struct AccelerationStructureInputs {
    AccelerationStructureType type = AccelerationStructureType::Bottom;
    AccelerationStructureFlags flags{};
    std::span<const TriangleGeometry> triangles{};
    std::span<const BoxGeometry> boxes{};
    BufferHandle instances{};
    std::uint64_t instanceOffset = 0;
    std::uint32_t instanceCount = 0;
};

/// The bytes a structure takes, and the scratch bytes its build and its
/// update take.
struct AccelerationStructureSizes {
    std::uint64_t size = 0;
    std::uint64_t buildScratch = 0;
    std::uint64_t updateScratch = 0;
};

/// A structure of size bytes, in memory the device allocates.
struct AccelerationStructureDesc {
    AccelerationStructureType type = AccelerationStructureType::Bottom;
    std::uint64_t size = 0;
};

/// A build of dst, of inputs.type, from inputs; or, with update, an update
/// of src into dst (they may be the same), src built with allowUpdate and
/// inputs as its last build gave them but for the data the buffers hold.
/// Scratch is scratchOffset bytes into scratch, which has
/// accelerationStructureScratch usage, a multiple of
/// RayTracingLimits::scratchAlignment, at least the build's or update's
/// scratch bytes, and overlaps neither structure.
struct AccelerationStructureBuild {
    AccelerationStructureHandle dst{};
    AccelerationStructureHandle src{};
    bool update = false;
    AccelerationStructureInputs inputs{};
    BufferHandle scratch{};
    std::uint64_t scratchOffset = 0;
};

/// The floats of a 3x4 row-major transform.
inline constexpr std::size_t kTransformFloats = 12;

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
    AccelerationStructureHandle bottomStructure{};
};

} // namespace bfr::rhi
