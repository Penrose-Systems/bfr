// What every backend does with acceleration structures, through the interface
// alone: builds of triangles, boxes and instances, rays traced inline against
// them from a compute shader, copies, updates, and the usage checks. Each case
// runs on every backend the build has; the shaders are the SPIR-V in
// tests/library/shaders, so a case that runs code skips on a backend with none
// checked in, and one that needs acceleration structures skips where the
// adapter has none.

#include "allocation/allocation_counter.h"
#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"
#include "shaders/spirv.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

using bfr::test::created;

// ray_query.comp's workgroup: one ray each, x = 5i apart.
constexpr std::uint32_t kRays = 8;
constexpr float kSpacing = 5;
constexpr std::uint32_t kMiss = 0xFFFFFFFF;
constexpr std::uint32_t kTableSlots = 4;
constexpr std::uint32_t kGeometries = 4;
constexpr std::uint32_t kPropertyPools = 4;
// A trace or a ray query writing its results, before a copy reads them.
constexpr BarrierScope kTraceWrites{.stages = {.all = true}, .accesses = {.storageWrite = true}};
// Property writes, before the copy that resolves them; the pool is BFR's, so
// they name no access.
constexpr BarrierScope kPropertiesWritten{.stages = {.accelerationStructureProperties = true}};
constexpr BarrierScope kPropertiesResolved{.stages = {.copy = true}};
// A compaction and property writes, before anything reads the compacted
// structure or resolves the properties.
constexpr BarrierScope kCompactedAndWritten{.stages = {.accelerationStructureCopy = true,
                                                       .accelerationStructureProperties = true},
                                            .accesses = {.accelerationStructureWrite = true}};
// What the scene's instances answer a hit with.
constexpr std::uint32_t kFirst = 7;
constexpr std::uint32_t kSecond = 9;
constexpr std::uint32_t kBox = 11;
constexpr std::uint32_t kInstances = 3;
// A vertex's floats and a triangle's vertices.
constexpr std::uint32_t kVertexFloats = 3;
constexpr std::uint32_t kTriangleVertices = 3;
// A size any backend takes for a structure no build fills.
constexpr std::uint64_t kSmall = 256;
constexpr std::uint8_t kAllRays = 0xFF;
// The queue every list here records on.
constexpr std::uint32_t kQueue = 0;

// An Instance and a device on its first adapter, presenting nowhere, with the
// ray tracing features and an acceleration structure table where the adapter
// has them.
struct RayDevice {
    explicit RayDevice(const InstanceDesc& desc, std::uint32_t tableSlots = kTableSlots,
                       bool validation = kValidation) {
        auto made = createInstance(desc, validation);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        instance = std::move(*made);
        caps = capsOf(instance, kFirstAdapter);
        const bool table = caps.bindless.maxAccelerationStructures != 0;
        auto opened = instance.createDevice(testDeviceWith(instance, kFirstAdapter, kNoWindow, [&](DeviceDesc& device) {
            device.features = rayTracingFeaturesOf(caps);
            device.maxBindlessAccelerationStructures = table ? tableSlots : 0;
            device.maxBindlessPartitionedTops = caps.features.partitionedTop ? tableSlots : 0;
            device.maxBuildGeometries = kGeometries;
            device.maxAccelerationStructurePropertyPools = kPropertyPools;
        }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        device = std::move(*opened);
    }

    Instance instance;
    Caps caps;
    Device device;
};

AccelerationStructure created(const Device& device, const AccelerationStructureDesc& desc) {
    auto structure = device.createAccelerationStructure(desc);
    INFO(failureOf(structure));
    REQUIRE(structure.has_value());
    return std::move(*structure);
}

// What bfr_get_acceleration_structure_reference says of structure, which may
// be stale.
Result referenceResult(const Device& device, bfr_acceleration_structure structure) {
    std::uint64_t reference = 0;
    return resultOf(bfr_get_acceleration_structure_reference(device.get(), structure, &reference));
}

// What bfr_reset_acceleration_structure_properties says of pool, which may
// be stale, resetting its first entry in a list recording on queue.
Result resetPropertiesResult(const Device& device, std::uint32_t queue, bfr_acceleration_structure_property_pool pool) {
    constexpr std::uint32_t kFirstEntry = 0;
    constexpr std::uint32_t kEntryCount = 1;
    return inList(device, queue, [pool](const Device& on, CommandList recording) {
        return bfr_reset_acceleration_structure_properties(on.get(), recording.get(), pool, kFirstEntry, kEntryCount);
    });
}

// What a ray tracing pipeline of shader alone fails with, or success: the
// shader's next use, which may be stale.
Result shaderResult(const Device& device, const RayTracingShader& shader) {
    return resultOf(device.createRayTracingPipeline({.shaders = std::span{&shader, 1}}));
}

// Whether letting owner go destroys it: its C handle's next use is then
// refused as stale. An owner of nothing has nothing to destroy, as the destroy
// of a null handle did nothing.
bool goneStale(const Device& device, Buffer& buffer) {
    return !buffer || slotsResult(device, destroyed(buffer)) == Result::ErrorInvalidHandle;
}

bool goneStale(const Device& device, AccelerationStructure& structure) {
    return !structure || captureResult(device, destroyed(structure)) == Result::ErrorInvalidHandle;
}

bool goneStale(const Device& device, Pipeline& pipeline) {
    return !pipeline || bindResult(device, kQueue, destroyed(pipeline)) == Result::ErrorInvalidHandle;
}

bool goneStale(const Device& device, CommandPool& pool) {
    return !pool || resetResult(device, destroyed(pool)) == Result::ErrorInvalidHandle;
}

bool goneStale(const Device& device, AccelerationStructurePropertyPool& pool) {
    return !pool || resetPropertiesResult(device, kQueue, destroyed(pool)) == Result::ErrorInvalidHandle;
}

// What a buffer that builds read is made for.
constexpr BufferUsage kBuildInput{.accelerationStructureInput = true};

// A 3x4 row-major transform moving x by x.
std::array<float, BFR_TRANSFORM_FLOATS> movedBy(float x) {
    return {1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0};
}

// A structure built from inputs, with the scratch its build and update take.
struct Built {
    AccelerationStructure structure;
    Buffer scratch;
};

Built allocated(const Device& device, const AccelerationStructureInputs& inputs) {
    const auto sizes = device.accelerationStructureSizes(inputs);
    INFO(failureOf(sizes));
    REQUIRE(sizes.has_value());
    CHECK(sizes->size != 0);
    return {.structure = created(device, AccelerationStructureDesc{.type = inputs.type, .size = sizes->size}),
            .scratch = created(device, BufferDesc{.size = std::max(sizes->buildScratch, sizes->updateScratch),
                                                  .usage = {.accelerationStructureScratch = true}})};
}

// A triangle about the origin in z = 0, a box about it, and a top structure
// of three instances: the triangle at x = 0 and at x = 10, and the box at
// x = 20, which rays 0, 2 and 4 hit; the others miss.
struct Scene {
    // topFlags build the top structure, bottomFlags the bottom ones. A
    // triangle that takes its opacity from micromap, counted by usages, is
    // not opaque by itself, so a ray query takes it as hit only where the
    // micromap says it is opaque.
    explicit Scene(const Device& device, const AccelerationStructureFlags& topFlags = {},
                   const AccelerationStructureFlags& bottomFlags = {}, const TriangleOpacityMicromap& micromap = {},
                   std::span<const OpacityMicromapUsage> usages = {})
        : device(&device),
          usages(usages),
          vertices(uploaded(device, std::array<float, kVertexFloats * kTriangleVertices>{-1, -1, 0, 1, -1, 0, 0, 1, 0},
                            kBuildInput)),
          boxes(uploaded(device, std::array<float, BFR_BOX_FLOATS>{-1, -1, -1, 1, 1, 1}, kBuildInput)) {
        triangleGeometry = {.vertices = vertices,
                            .vertexFormat = Format::Rgb32Float,
                            .vertexStride = kVertexFloats * sizeof(float),
                            .vertexCount = kTriangleVertices,
                            .triangleCount = 1,
                            .opaque = !micromap.micromap && !micromap.indices,
                            .opacityMicromap = micromap};
        boxGeometry = {.boxes = boxes, .stride = BFR_BOX_FLOATS * sizeof(float), .count = 1};
        flags = topFlags;
        this->bottomFlags = bottomFlags;
        triangles = allocated(device, triangleInputs());
        box = allocated(device, boxInputs());
    }

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(Scene&&) = delete;
    ~Scene() {
        for (AccelerationStructure* structure : {&triangles.structure, &box.structure, &top.structure}) {
            CHECK(goneStale(*device, *structure));
        }
        for (Buffer* buffer : {&vertices, &boxes, &instances, &triangles.scratch, &box.scratch, &top.scratch}) {
            CHECK(goneStale(*device, *buffer));
        }
    }

    [[nodiscard]] AccelerationStructureInputs triangleInputs() const {
        return {.type = AccelerationStructureType::Bottom,
                .flags = bottomFlags,
                .triangles = std::span{&triangleGeometry, 1},
                .micromapUsages = usages};
    }
    [[nodiscard]] AccelerationStructureInputs boxInputs() const {
        return {.type = AccelerationStructureType::Bottom, .flags = bottomFlags, .boxes = std::span{&boxGeometry, 1}};
    }
    [[nodiscard]] AccelerationStructureInputs topInputs() const {
        return {.type = AccelerationStructureType::Top,
                .flags = flags,
                .instances = instances,
                .instanceCount = kInstances};
    }

    // Writes the instances, the second triangle at secondX, into their
    // buffer, made the first time, and makes the top structure the first
    // time. The triangles' instances are of trianglesFrom, or of the scene's
    // own where it is null.
    void place(const Caps& caps, float secondX, AccelerationStructureRef trianglesFrom = {}) {
        const AccelerationStructureRef triangleStructure = trianglesFrom
                                                               ? trianglesFrom
                                                               : AccelerationStructureRef{triangles.structure};
        const std::uint32_t size = caps.rayTracing.instanceSize;
        REQUIRE(size != 0);
        const std::uint64_t bytes = std::uint64_t{size} * kInstances;
        if (!instances) {
            instances = created(*device, BufferDesc{.size = bytes,
                                                    .memory = BufferMemory::Upload,
                                                    .usage = {.accelerationStructureInput = true}});
        }
        const std::array<AccelerationStructureInstance, kInstances> placed{{
            {.transform = movedBy(0),
             .customIndex = kFirst,
             .mask = kAllRays,
             .disableOpacityMicromaps = disableOpacityMicromaps,
             .bottomStructure = triangleStructure},
            {.transform = movedBy(secondX),
             .customIndex = kSecond,
             .mask = kAllRays,
             .disableOpacityMicromaps = disableOpacityMicromaps,
             .bottomStructure = triangleStructure},
            // A ray tracing pipeline's second hit group is the box's.
            {.transform = movedBy(4 * kSpacing),
             .customIndex = kBox,
             .mask = kAllRays,
             .hitGroupOffset = 1,
             .bottomStructure = box.structure},
        }};
        const auto mapped = instances.map();
        REQUIRE(mapped.has_value());
        const auto written = device->writeInstances(placed, *mapped, bytes);
        INFO(failureOf(written));
        REQUIRE(written.has_value());
        if (!top.structure) {
            top = allocated(*device, topInputs());
        }
    }

    // Records builds of the bottom structures, then of the top one.
    void build(CommandList list) const {
        for (const auto& [built, inputs] : {std::pair{&triangles, triangleInputs()}, std::pair{&box, boxInputs()}}) {
            const auto recorded = device->buildAccelerationStructure(list, {.dst = built->structure,
                                                                            .inputs = inputs,
                                                                            .scratch = built->scratch});
            INFO(failureOf(recorded));
            CHECK(recorded.has_value());
        }
        CHECK(globalBarrier(*device, list, kStructureWrites, kStructureReads).has_value());
        buildTop(list, false);
    }

    void buildTop(CommandList list, bool update) const {
        const auto recorded = device->buildAccelerationStructure(
            list, {.dst = top.structure,
                   .src = update ? AccelerationStructureRef{top.structure} : AccelerationStructureRef{},
                   .update = update,
                   .inputs = topInputs(),
                   .scratch = top.scratch});
        INFO(failureOf(recorded));
        CHECK(recorded.has_value());
        CHECK(globalBarrier(*device, list, kStructureWrites, kStructureReads).has_value());
    }

    const Device* device;
    std::span<const OpacityMicromapUsage> usages;
    // The triangles' instances disable their opacity micromaps.
    bool disableOpacityMicromaps = false;
    Buffer vertices;
    Buffer boxes;
    Buffer instances;
    TriangleGeometry triangleGeometry;
    BoxGeometry boxGeometry;
    AccelerationStructureFlags flags;
    AccelerationStructureFlags bottomFlags;
    Built triangles;
    Built box;
    Built top;
};

// The push constants the test shaders read: a structure's slot and the
// results' slot, as bytes.
using Push = std::array<std::uint32_t, 2>;
using PushBytes = std::array<std::uint8_t, sizeof(Push)>;

// Records record, then trace, given the push constants the test shaders
// read, that sends rays against structure, and returns what each ray hit.
template <class Record, class Trace>
std::array<std::uint32_t, kRays> traced(const Device& device, AccelerationStructure& structure, const Record& record,
                                        const Trace& trace) {
    constexpr std::uint64_t kBytes = sizeof(std::uint32_t) * kRays;
    Buffer results = created(device, BufferDesc{.size = kBytes, .usage = {.copySrc = true, .storage = true}});
    Buffer readback = created(device,
                              BufferDesc{.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
    const auto slot = structure.accelerationStructureSlot();
    INFO(failureOf(slot));
    REQUIRE(slot.has_value());
    const auto resultSlot = storageSlot(results);
    REQUIRE(resultSlot.has_value());
    auto pool = device.createCommandPool({.queue = kQueue, .listCount = kPoolLists, .pushConstantUpdates = 1});
    REQUIRE(pool.has_value());
    const auto list = pool->beginCommands();
    REQUIRE(list.has_value());

    record(*list);
    const auto push = std::bit_cast<PushBytes>(Push{*slot, *resultSlot});
    trace(*list, std::span<const std::uint8_t>{push});
    CHECK(globalBarrier(device, *list, kTraceWrites, kCopyReads).has_value());
    const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
    CHECK(device.copyBuffer(*list, results, readback, std::span{&whole, 1}).has_value());
    const auto sent = device.submit({.queue = kQueue, .commandLists = std::span{&*list, 1}, .waits = {}});
    INFO(failureOf(sent));
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());

    std::array<std::uint32_t, kRays> hits{};
    const auto read = readback.map();
    REQUIRE(read.has_value());
    std::memcpy(hits.data(), *read, kBytes);
    CHECK(goneStale(device, results));
    CHECK(goneStale(device, readback));
    CHECK(goneStale(device, *pool));
    return hits;
}

// ray_query.comp, which traces from a compute shader.
// Or, given code, another shader that traces as it does.
struct RayQuery {
    explicit RayQuery(const Device& device, std::span<const std::uint8_t> code = bfr::test::spirv::kRayQueryComp)
        : device(&device) {
        auto made = device.createComputePipeline({.compute = shaderCode(code), .workgroupSize = {kRays, 1, 1}});
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        pipeline = std::move(*made);
    }
    RayQuery(const RayQuery&) = delete;
    RayQuery& operator=(const RayQuery&) = delete;
    RayQuery(RayQuery&&) = delete;
    RayQuery& operator=(RayQuery&&) = delete;
    ~RayQuery() { CHECK(goneStale(*device, pipeline)); }

    void operator()(CommandList list, std::span<const std::uint8_t> push) const {
        CHECK(device->bindPipeline(list, pipeline).has_value());
        CHECK(device->pushConstants(list, push).has_value());
        CHECK(device->dispatch(list, 1, 1, 1).has_value());
    }

    const Device* device;
    Pipeline pipeline;
};

// How a RayTracing's pipeline is made: with stackSize; with linked, its
// miss and hit groups in a library it links, which follow its ray generation
// group as they would in the pipeline itself; with captureReplay and the
// groups' replayIdentifiers.
struct TracingOptions {
    std::uint64_t stackSize = 0;
    bool linked = false;
    bool captureReplay = false;
    std::span<const std::uint8_t> replayIdentifiers{};
    bool allowOpacityMicromaps = false;
    // Made with allowMotion, its ray generation trace_motion.rgen, which
    // traces at the time a third push constant holds.
    bool motion = false;
    bool allowClusters = false;
};

// trace.rgen, trace.rmiss, and hit groups of trace.rchit for triangles and
// with box.rint for boxes, in a pipeline, and a binding table of their
// records: ray generation, then miss, then the two hit groups, each region at
// a table alignment.
struct RayTracing {
    RayTracing(const Device& device, const Caps& caps, const TracingOptions& options = {}) : device(&device) {
        const std::array codes{options.motion ? shaderCode(bfr::test::spirv::kTraceMotionRgen)
                                              : shaderCode(bfr::test::spirv::kTraceRgen),
                               shaderCode(bfr::test::spirv::kTraceRmiss), shaderCode(bfr::test::spirv::kTraceRchit),
                               shaderCode(bfr::test::spirv::kBoxRint)};
        std::array<Shader, kShaders> shaders{};
        for (std::size_t i = 0; i < kShaders; ++i) {
            auto shader = device.createShader(codes.at(i));
            INFO(failureOf(shader));
            REQUIRE(shader.has_value());
            shaders.at(i) = std::move(*shader);
        }
        const std::array<RayTracingShader, kShaders> stages{{
            {.stage = RayTracingStage::RayGeneration, .shader = shaders[0]},
            {.stage = RayTracingStage::Miss, .shader = shaders[1]},
            {.stage = RayTracingStage::ClosestHit, .shader = shaders[2]},
            {.stage = RayTracingStage::Intersection, .shader = shaders[3]},
        }};
        const std::array<RayTracingGroup, kGroups> groups{{
            {.type = RayTracingGroupType::General, .general = 0},
            {.type = RayTracingGroupType::General, .general = 1},
            {.type = RayTracingGroupType::TriangleHit, .closestHit = 2},
            {.type = RayTracingGroupType::BoxHit, .closestHit = 2, .intersection = 3},
        }};
        const RayTracingPipelineDesc whole{.shaders = stages,
                                           .groups = groups,
                                           .maxRecursionDepth = 1,
                                           .maxPayloadSize = sizeof(std::uint32_t),
                                           .stackSize = options.stackSize,
                                           .captureReplay = options.captureReplay,
                                           .replayIdentifiers = options.replayIdentifiers,
                                           .allowOpacityMicromaps = options.allowOpacityMicromaps,
                                           .allowMotion = options.motion,
                                           .allowClusters = options.allowClusters};
        if (options.linked) {
            // The library's shader indices are its own, from the miss shader.
            const std::array<RayTracingGroup, kGroups - 1> libraryGroups{{
                {.type = RayTracingGroupType::General, .general = 0},
                {.type = RayTracingGroupType::TriangleHit, .closestHit = 1},
                {.type = RayTracingGroupType::BoxHit, .closestHit = 1, .intersection = 2},
            }};
            RayTracingPipelineDesc part = whole;
            part.shaders = std::span{stages}.subspan(1);
            part.groups = libraryGroups;
            part.library = true;
            auto linked = device.createRayTracingPipeline(part);
            INFO(failureOf(linked));
            REQUIRE(linked.has_value());
            library = std::move(*linked);
        }
        const std::array<PipelineRef, 1> libraries{library};
        RayTracingPipelineDesc desc = whole;
        if (options.linked) {
            desc.shaders = std::span{stages}.first(1);
            desc.groups = std::span{groups}.first(1);
            desc.libraries = libraries;
        }
        auto made = device.createRayTracingPipeline(desc);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        pipeline = std::move(*made);
        // A pipeline keeps what it needs of its shaders.
        for (std::size_t i = 0; i < kShaders; ++i) {
            shaders.at(i) = {};
            CHECK(shaderResult(device, stages.at(i)) == Result::ErrorInvalidHandle);
        }

        const CapsRayTracing& limits = caps.rayTracing;
        stride = alignUp(limits.identifierSize, limits.identifierAlignment);
        spacing = alignUp(2 * stride, limits.tableAlignment);
        std::vector<std::byte> identifiers(std::size_t{limits.identifierSize} * kGroups);
        const auto written = pipeline.shaderIdentifiers(0, kGroups, identifiers.data(), identifiers.size());
        INFO(failureOf(written));
        REQUIRE(written.has_value());
        const std::uint64_t tableBytes = 3 * spacing;
        table = created(device, BufferDesc{.size = tableBytes,
                                           .memory = BufferMemory::Upload,
                                           .usage = {.shaderBindingTable = true}});
        const auto mapped = table.map();
        REQUIRE(mapped.has_value());
        const std::span<std::byte> bytes = mappedBytes(*mapped, tableBytes);
        // Group i at its region's start, the two hit groups a stride apart.
        const std::array<std::uint64_t, kGroups> at{0, spacing, 2 * spacing, 2 * spacing + stride};
        for (std::size_t i = 0; i < kGroups; ++i) {
            std::memcpy(bytes.subspan(at.at(i)).data(), identifiers.data() + i * limits.identifierSize,
                        limits.identifierSize);
        }
    }
    RayTracing(const RayTracing&) = delete;
    RayTracing& operator=(const RayTracing&) = delete;
    RayTracing(RayTracing&&) = delete;
    RayTracing& operator=(RayTracing&&) = delete;
    ~RayTracing() {
        CHECK(goneStale(*device, pipeline));
        CHECK(goneStale(*device, library));
        CHECK(goneStale(*device, table));
    }

    [[nodiscard]] TraceRaysDesc desc() const {
        return {.rayGeneration = {.buffer = table, .offset = 0, .size = stride, .stride = stride},
                .miss = {.buffer = table, .offset = spacing, .size = stride, .stride = stride},
                .hit = {.buffer = table, .offset = 2 * spacing, .size = 2 * stride, .stride = stride},
                .callable = {},
                .width = kRays,
                .height = 1,
                .depth = 1};
    }

    // desc()'s trace, its regions by address, in an Upload buffer of
    // indirect usage for traceRaysIndirect.
    [[nodiscard]] Buffer indirectArgs() {
        const auto address = table.bufferAddress();
        INFO(failureOf(address));
        REQUIRE(address.has_value());
        const TraceRaysDesc direct = desc();
        const auto region = [&](const ShaderTableRegion& from) {
            return bfr_shader_table_address_region{.address = *address + from.offset,
                                                   .size = from.size,
                                                   .stride = from.stride};
        };
        const bfr_trace_rays_indirect_args args{.ray_generation_address = *address,
                                                .ray_generation_size = direct.rayGeneration.size,
                                                .miss = region(direct.miss),
                                                .hit = region(direct.hit),
                                                .callable = {},
                                                .width = direct.width,
                                                .height = direct.height,
                                                .depth = direct.depth};
        Buffer buffer = created(*device, BufferDesc{.size = sizeof(args),
                                                    .memory = BufferMemory::Upload,
                                                    .usage = {.indirect = true}});
        const auto mapped = buffer.map();
        REQUIRE(mapped.has_value());
        std::memcpy(*mapped, &args, sizeof(args));
        return buffer;
    }

    static constexpr std::size_t kShaders = 4;
    static constexpr std::uint32_t kGroups = 4;
    const Device* device;
    Pipeline library;
    Pipeline pipeline;
    Buffer table;
    std::uint64_t stride = 0;
    std::uint64_t spacing = 0;
};

// Records record in a list on queue 0, submits it and waits for it.
template <class Record>
void run(const Device& device, const Record& record) {
    auto pool = device.createCommandPool({.queue = kQueue, .listCount = 1});
    REQUIRE(pool.has_value());
    const auto list = pool->beginCommands();
    REQUIRE(list.has_value());
    record(*list);
    const auto sent = device.submit({.queue = kQueue, .commandLists = std::span{&*list, 1}, .waits = {}});
    INFO(failureOf(sent));
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());
    CHECK(goneStale(device, *pool));
}

// Each structure's property, written after before, which builds them, and
// read back.
template <class Before>
std::vector<std::uint64_t> propertyOf(const Device& device, const Caps& caps, AccelerationStructureProperty property,
                                      std::span<const AccelerationStructureRef> structures, const Before& before) {
    const auto count = static_cast<std::uint32_t>(structures.size());
    auto pool = device.createAccelerationStructurePropertyPool({.property = property, .count = count});
    INFO(failureOf(pool));
    REQUIRE(pool.has_value());
    const std::uint32_t entry = caps.rayTracing.propertyEntryBytes;
    REQUIRE(entry == sizeof(std::uint64_t));
    Buffer readback = created(device, BufferDesc{.size = std::uint64_t{entry} * count,
                                                 .memory = BufferMemory::Readback,
                                                 .usage = {.copyDst = true}});
    run(device, [&](CommandList list) {
        CHECK(device.resetAccelerationStructureProperties(list, *pool, 0, count).has_value());
        before(list);
        const auto written = device.writeAccelerationStructureProperties(list, *pool, 0, structures);
        INFO(failureOf(written));
        CHECK(written.has_value());
        CHECK(globalBarrier(device, list, kPropertiesWritten, kPropertiesResolved).has_value());
        CHECK(device.resolveAccelerationStructureProperties(list, *pool, 0, count, readback, 0).has_value());
    });
    std::vector<std::uint64_t> values(count);
    const auto read = readback.map();
    REQUIRE(read.has_value());
    std::memcpy(values.data(), *read, values.size() * sizeof(std::uint64_t));
    CHECK(goneStale(device, readback));
    CHECK(goneStale(device, *pool));
    return values;
}

constexpr std::array<std::uint32_t, kRays> kPlaced{kFirst, kMiss, kSecond, kMiss, kBox, kMiss, kMiss, kMiss};
// The rays of a scene whose triangles are transparent: only the box is hit.
constexpr std::array<std::uint32_t, kRays> kBoxOnly{kMiss, kMiss, kMiss, kMiss, kBox, kMiss, kMiss, kMiss};

// One two-state micromap of subdivision level 0, whose one subtriangle is
// opaque where opaque, else transparent, and the usage that counts it.
constexpr OpacityMicromapUsage kOneMicromap{.count = 1,
                                            .subdivisionLevel = 0,
                                            .format = OpacityMicromapFormat::TwoState};
constexpr std::uint8_t kOpaqueState = 1;

struct Micromap {
    Micromap(const Device& device, bool opaque, const AccelerationStructureFlags& flags = {})
        : device(&device),
          flags(flags),
          data(uploaded(device, std::array<std::uint8_t, 1>{opaque ? kOpaqueState : std::uint8_t{0}}, kBuildInput)),
          records(uploaded(device,
                           std::array<bfr_opacity_micromap_record, 1>{
                               {{.data_offset = 0,
                                 .subdivision_level = 0,
                                 .format = static_cast<std::uint16_t>(OpacityMicromapFormat::TwoState)}}},
                           kBuildInput)),
          built(allocated(device, inputs())) {}
    Micromap(const Micromap&) = delete;
    Micromap& operator=(const Micromap&) = delete;
    Micromap(Micromap&&) = delete;
    Micromap& operator=(Micromap&&) = delete;
    ~Micromap() {
        CHECK(goneStale(*device, built.structure));
        for (Buffer* buffer : {&data, &records, &built.scratch}) {
            CHECK(goneStale(*device, *buffer));
        }
    }

    [[nodiscard]] AccelerationStructureInputs inputs() const {
        return {.type = AccelerationStructureType::OpacityMicromap,
                .flags = flags,
                .micromapUsages = std::span{&kOneMicromap, 1},
                .micromapData = data,
                .micromapRecords = records,
                .micromapRecordStride = sizeof(bfr_opacity_micromap_record)};
    }

    // Records its build and a barrier after it.
    void build(CommandList list) const {
        const auto recorded = device->buildAccelerationStructure(list, {.dst = built.structure,
                                                                        .inputs = inputs(),
                                                                        .scratch = built.scratch});
        INFO(failureOf(recorded));
        CHECK(recorded.has_value());
        CHECK(globalBarrier(*device, list, kStructureWrites, kStructureReads).has_value());
    }

    const Device* device;
    AccelerationStructureFlags flags;
    Buffer data;
    Buffer records;
    Built built;
};

} // namespace

TEST_CASE("Device builds triangles, boxes and instances, and rays traced inline hit them, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps.features.accelerationStructure || !made.caps.features.rayQuery) {
            SKIP("the adapter has no acceleration structures or ray queries");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        REQUIRE(made.caps.bindless.maxAccelerationStructures != 0);
        const RayQuery query{device};
        const auto built = [](Scene& scene) {
            return [&scene](CommandList list) {
                scene.build(list);
            };
        };

        SECTION("a build") {
            Scene scene{device};
            scene.place(made.caps, 2 * kSpacing);
            CHECK(traced(device, scene.top.structure, built(scene), query) == kPlaced);
        }
        SECTION("a copy of a top structure traces as it does") {
            Scene scene{device};
            scene.place(made.caps, 2 * kSpacing);
            const auto sizes = device.accelerationStructureSizes(scene.topInputs());
            REQUIRE(sizes.has_value());
            AccelerationStructure copy = created(device, {.type = AccelerationStructureType::Top, .size = sizes->size});
            const auto copied = [&](CommandList list) {
                scene.build(list);
                CHECK(device.copyAccelerationStructure(list, copy, scene.top.structure).has_value());
                CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
            };
            CHECK(traced(device, copy, copied, query) == kPlaced);
            CHECK(goneStale(device, copy));
        }
        SECTION("an update moves an instance") {
            Scene scene{device, {.allowUpdate = true}};
            scene.place(made.caps, 2 * kSpacing);
            CHECK(traced(device, scene.top.structure, built(scene), query) == kPlaced);
            // The second triangle moves from ray 2 to ray 1.
            scene.place(made.caps, kSpacing);
            constexpr std::array<std::uint32_t, kRays> kMoved{kFirst, kSecond, kMiss, kMiss, kBox, kMiss, kMiss, kMiss};
            const auto updated = [&](CommandList list) {
                scene.buildTop(list, true);
            };
            CHECK(traced(device, scene.top.structure, updated, query) == kMoved);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device traces rays through a ray tracing pipeline's binding table, directly and indirectly, where it has "
          "them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps.features.rayTracingPipeline) {
            SKIP("the adapter has no ray tracing pipelines");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        RayTracing tracing{device, made.caps};
        Scene scene{device};
        scene.place(made.caps, 2 * kSpacing);
        const auto built = [&](CommandList list) {
            scene.build(list);
        };
        const auto bind = [&](CommandList list, std::span<const std::uint8_t> push) {
            CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
            CHECK(device.pushConstants(list, push).has_value());
        };

        SECTION("a trace") {
            const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
                bind(list, push);
                const auto traced = device.traceRays(list, tracing.desc());
                INFO(failureOf(traced));
                CHECK(traced.has_value());
            };
            CHECK(traced(device, scene.top.structure, built, trace) == kPlaced);
        }
        SECTION("a trace with the stack size the caller gives") {
            // Every shader's stack, which is more than any trace takes.
            std::uint64_t stack = 0;
            for (const auto& [group, shader] :
                 {std::pair{0U, RayTracingGroupShader::General}, std::pair{1U, RayTracingGroupShader::General},
                  std::pair{2U, RayTracingGroupShader::ClosestHit}, std::pair{3U, RayTracingGroupShader::ClosestHit},
                  std::pair{3U, RayTracingGroupShader::Intersection}}) {
                const auto size = tracing.pipeline.rayTracingStackSize(group, shader);
                INFO(failureOf(size));
                REQUIRE(size.has_value());
                stack += *size;
            }
            const RayTracing sized{device, made.caps, {.stackSize = std::max<std::uint64_t>(stack, 1)}};
            const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
                CHECK(device.bindPipeline(list, sized.pipeline).has_value());
                CHECK(device.pushConstants(list, push).has_value());
                CHECK(device.traceRays(list, sized.desc()).has_value());
            };
            CHECK(traced(device, scene.top.structure, built, trace) == kPlaced);
        }
        SECTION("a trace through a pipeline that links a library of its miss and hit groups, where the device has "
                "them") {
            if (!made.caps.features.rayTracingPipelineLibrary) {
                SKIP("the adapter has no ray tracing pipeline libraries");
            }
            RayTracing linked{device, made.caps, {.linked = true}};
            const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
                CHECK(device.bindPipeline(list, linked.pipeline).has_value());
                CHECK(device.pushConstants(list, push).has_value());
                CHECK(device.traceRays(list, linked.desc()).has_value());
            };
            CHECK(traced(device, scene.top.structure, built, trace) == kPlaced);
            if (made.caps.features.pipelineLibraryIdentifiers) {
                // The library's groups have the identifiers the pipeline
                // gives them, after its ray generation group.
                const std::size_t size = made.caps.rayTracing.identifierSize;
                std::vector<std::byte> fromLibrary(size * (RayTracing::kGroups - 1));
                std::vector<std::byte> fromPipeline(size * (RayTracing::kGroups - 1));
                CHECK(
                    linked.library.shaderIdentifiers(0, RayTracing::kGroups - 1, fromLibrary.data(), fromLibrary.size())
                        .has_value());
                CHECK(linked.pipeline
                          .shaderIdentifiers(1, RayTracing::kGroups - 1, fromPipeline.data(), fromPipeline.size())
                          .has_value());
                CHECK(fromLibrary == fromPipeline);
            }
        }
        SECTION("an indirect trace, of regions by address, where the device has it") {
            if (!made.caps.features.traceRaysIndirect) {
                SKIP("the adapter has no indirect traces");
            }
            Buffer argsBuffer = tracing.indirectArgs();
            const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
                bind(list, push);
                const auto traced = device.traceRaysIndirect(list, argsBuffer, 0);
                INFO(failureOf(traced));
                CHECK(traced.has_value());
            };
            CHECK(traced(device, scene.top.structure, built, trace) == kPlaced);
            CHECK(goneStale(device, argsBuffer));
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

// A scene that moves: its instances written, its top structure rebuilt and
// rays traced against it each frame. Validation stays off: its layer
// allocates through the operator new this program counts.
TEST_CASE("Device traces rays through a pipeline replaying identifiers another device captured, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        Caps caps;
        std::vector<std::uint8_t> captured;
        std::vector<std::byte> first;
        {
            const RayDevice capturing{recordingInto(backend, messages)};
            caps = capturing.caps;
            if (!caps.features.rayTracingCaptureReplay) {
                SKIP("the adapter captures no ray tracing identifiers");
            }
            if (!takesSpirv(backend)) {
                SKIP("no " << backend.name << " bytecode is checked in");
            }
            captured.resize(std::size_t{caps.rayTracing.captureReplayIdentifierSize} * RayTracing::kGroups);
            first.resize(std::size_t{caps.rayTracing.identifierSize} * RayTracing::kGroups);
            RayTracing original{capturing.device, caps, {.captureReplay = true}};
            const auto written = original.pipeline.captureReplayIdentifiers(0, RayTracing::kGroups, captured.data(),
                                                                            captured.size());
            INFO(failureOf(written));
            REQUIRE(written.has_value());
            CHECK(original.pipeline.shaderIdentifiers(0, RayTracing::kGroups, first.data(), first.size()).has_value());
        }
        // A new device, whose first ray tracing pipeline replays, as a later
        // run's would: without Features::rayTracingCaptureReplayMixed it
        // makes none without replayed identifiers before one with them.
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        RayTracing replayed{device, made.caps, {.captureReplay = true, .replayIdentifiers = captured}};
        std::vector<std::byte> again(first.size());
        CHECK(replayed.pipeline.shaderIdentifiers(0, RayTracing::kGroups, again.data(), again.size()).has_value());
        CHECK(again == first);
        Scene scene{device};
        scene.place(made.caps, 2 * kSpacing);
        const auto built = [&](CommandList list) {
            scene.build(list);
        };
        const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
            CHECK(device.bindPipeline(list, replayed.pipeline).has_value());
            CHECK(device.pushConstants(list, push).has_value());
            CHECK(device.traceRays(list, replayed.desc()).has_value());
        };
        CHECK(traced(device, scene.top.structure, built, trace) == kPlaced);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device makes structures again at the references another device captured, and rays traced inline hit "
          "them, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        // A bottom and a top structure of a scene, and a partitioned top one
        // where the device has them, made where they may be captured, and
        // what replays them, as a capture tool keeps them.
        struct Captured {
            AccelerationStructureDesc desc;
            AccelerationStructureCapture capture;
            std::uint64_t reference = 0;
        };
        std::vector<Captured> captured;
        {
            const RayDevice capturing{recordingInto(backend, messages)};
            const Device& device = capturing.device;
            if (!capturing.caps.features.accelerationStructureCaptureReplay || !capturing.caps.features.rayQuery) {
                SKIP("the adapter captures no acceleration structure references, or has no ray queries");
            }
            if (!takesSpirv(backend)) {
                SKIP("no " << backend.name << " bytecode is checked in");
            }
            Scene scene{device};
            scene.place(capturing.caps, 2 * kSpacing);
            std::vector<AccelerationStructureDesc> descs;
            for (const AccelerationStructureInputs& inputs : {scene.triangleInputs(), scene.topInputs()}) {
                const auto sizes = device.accelerationStructureSizes(inputs);
                REQUIRE(sizes.has_value());
                descs.push_back({.type = inputs.type, .size = sizes->size, .captureReplay = true});
            }
            if (capturing.caps.features.partitionedTop) {
                descs.push_back(
                    {.type = AccelerationStructureType::PartitionedTop, .size = kSmall, .captureReplay = true});
            }
            // All held at once, as a captured run holds them.
            std::vector<AccelerationStructure> held;
            for (const AccelerationStructureDesc& desc : descs) {
                held.push_back(created(device, desc));
                const auto capture = held.back().capture();
                INFO(failureOf(capture));
                REQUIRE(capture.has_value());
                const auto reference = held.back().accelerationStructureReference();
                REQUIRE(reference.has_value());
                captured.push_back({.desc = desc, .capture = *capture, .reference = *reference});
            }
            for (AccelerationStructure& structure : held) {
                CHECK(goneStale(device, structure));
            }
        }
        // A new device, which makes the structures again before any other,
        // as a replay would, last first, so none is where it was only by
        // being made in the same order.
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        std::vector<AccelerationStructure> replayed(captured.size());
        for (std::size_t i = captured.size(); i-- > 0;) {
            AccelerationStructureDesc desc = captured.at(i).desc;
            desc.replay = captured.at(i).capture;
            replayed.at(i) = created(device, desc);
            const auto reference = replayed.at(i).accelerationStructureReference();
            REQUIRE(reference.has_value());
            CHECK(*reference == captured.at(i).reference);
        }
        const RayQuery query{device};
        Scene scene{device};
        scene.place(made.caps, 2 * kSpacing, replayed[0]);
        const auto built = [&](CommandList list) {
            for (const auto& [structure, inputs, scratch] :
                 {std::tuple{AccelerationStructureRef{replayed[0]}, scene.triangleInputs(),
                             BufferRef{scene.triangles.scratch}},
                  std::tuple{AccelerationStructureRef{scene.box.structure}, scene.boxInputs(),
                             BufferRef{scene.box.scratch}}}) {
                CHECK(device.buildAccelerationStructure(list, {.dst = structure, .inputs = inputs, .scratch = scratch})
                          .has_value());
            }
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
            CHECK(device
                      .buildAccelerationStructure(
                          list, {.dst = replayed[1], .inputs = scene.topInputs(), .scratch = scene.top.scratch})
                      .has_value());
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
        };
        CHECK(traced(device, replayed[1], built, query) == kPlaced);
        for (AccelerationStructure& structure : replayed) {
            CHECK(goneStale(device, structure));
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device allocates nothing tracing rays frame after frame", "[rhi][agnostic]") {
    if (!bfr::test::AllocationCounter::counts()) {
        SKIP("a sanitizer replaces operator new, so allocations are not counted");
    }
    constexpr int kFrames = 8;
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages), kTableSlots, kValidationOff};
        const Device& device = made.device;
        if (!made.caps.features.rayTracingPipeline) {
            SKIP("the adapter has no ray tracing pipelines");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const RayTracing tracing{device, made.caps};
        // A top structure compacted each frame, its compacted size written.
        Scene scene{device, {.allowCompaction = true}};
        scene.place(made.caps, 2 * kSpacing);
        run(device, [&](CommandList list) { scene.build(list); });
        const auto topSizes = device.accelerationStructureSizes(scene.topInputs());
        REQUIRE(topSizes.has_value());
        AccelerationStructure compact = created(device,
                                                {.type = AccelerationStructureType::Top, .size = topSizes->size});
        auto properties = device.createAccelerationStructurePropertyPool(
            {.property = AccelerationStructureProperty::CompactedSize, .count = 1});
        REQUIRE(properties.has_value());
        Buffer sizes = created(device, BufferDesc{.size = made.caps.rayTracing.propertyEntryBytes,
                                                  .memory = BufferMemory::Readback,
                                                  .usage = {.copyDst = true}});
        const std::array<AccelerationStructureRef, 1> compacted{scene.top.structure};
        Buffer results = created(device, BufferDesc{.size = sizeof(std::uint32_t) * kRays, .usage = {.storage = true}});
        const auto slot = scene.top.structure.accelerationStructureSlot();
        const auto resultSlot = storageSlot(results);
        REQUIRE(slot.has_value());
        REQUIRE(resultSlot.has_value());
        const auto push = std::bit_cast<PushBytes>(Push{*slot, *resultSlot});
        auto pool = device.createCommandPool({.queue = kQueue, .listCount = 1, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        const auto mapped = scene.instances.map();
        REQUIRE(mapped.has_value());
        const std::size_t instanceBytes = std::size_t{made.caps.rayTracing.instanceSize} * kInstances;
        const AccelerationStructureInputs topInputs = scene.topInputs();

        const char* failed = nullptr;
        std::size_t allocations = 0;
        std::size_t frees = 0;
        {
            const bfr::test::AllocationCounter counter;
            Token last;
            const auto check = [&failed](const char* call, const auto& result) {
                if (!result.has_value() && failed == nullptr) {
                    failed = call;
                }
                return result.has_value();
            };
            for (int frame = 0; frame < kFrames && failed == nullptr; ++frame) {
                check("wait", device.wait(last, kTestWait));
                check("resetCommandPool", pool->reset());
                const AccelerationStructureInstance placed{.transform = movedBy(static_cast<float>(frame)),
                                                           .mask = kAllRays,
                                                           .bottomStructure = scene.triangles.structure};
                check("writeInstances", device.writeInstances(std::span{&placed, 1}, *mapped, instanceBytes));
                const auto list = pool->beginCommands();
                if (!check("beginCommands", list)) {
                    break;
                }
                check("resetAccelerationStructureProperties",
                      device.resetAccelerationStructureProperties(*list, *properties, 0, 1));
                check("buildAccelerationStructure",
                      device.buildAccelerationStructure(
                          *list, {.dst = scene.top.structure, .inputs = topInputs, .scratch = scene.top.scratch}));
                check("barrier", globalBarrier(device, *list, kStructureWrites, kStructureReads));
                check("writeAccelerationStructureProperties",
                      device.writeAccelerationStructureProperties(*list, *properties, 0, compacted));
                check("compactAccelerationStructure",
                      device.compactAccelerationStructure(*list, compact, scene.top.structure));
                check("barrier", globalBarrier(device, *list, kCompactedAndWritten, kStructureReads));
                check("resolveAccelerationStructureProperties",
                      device.resolveAccelerationStructureProperties(*list, *properties, 0, 1, sizes, 0));
                check("bindPipeline", device.bindPipeline(*list, tracing.pipeline));
                check("pushConstants", device.pushConstants(*list, push));
                check("traceRays", device.traceRays(*list, tracing.desc()));
                const auto submitted = device.submit(
                    {.queue = kQueue, .commandLists = std::span{&*list, 1}, .waits = {}});
                if (check("submit", submitted)) {
                    last = *submitted;
                }
            }
            check("wait", device.wait(last, kTestWait));
            allocations = counter.allocations();
            frees = counter.frees();
        }
        INFO("first failure: " << (failed == nullptr ? "none" : failed));
        CHECK(failed == nullptr);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        CHECK(goneStale(device, *pool));
        CHECK(goneStale(device, results));
        CHECK(goneStale(device, sizes));
        CHECK(goneStale(device, *properties));
        CHECK(goneStale(device, compact));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses ray tracing pipeline work it can't pass on", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        auto pool = device.createCommandPool({.queue = kQueue, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        // d1: null destroy: no message (track B)
        bfr_destroy_shader(device.get(), {});
        Buffer readback = created(device, BufferDesc{.size = sizeof(std::uint32_t),
                                                     .memory = BufferMemory::Readback,
                                                     .usage = {.copyDst = true}});
        if (!made.caps.features.rayTracingPipeline) {
            CHECK(resultOf(device.createRayTracingPipeline({})) == Result::ErrorUnsupported);
            CHECK(resultOf(device.traceRays(*list, {})) == Result::ErrorUnsupported);
            // bfr.hpp names a pipeline only by a live Pipeline, so the null
            // one goes through bfr.h.
            std::vector<std::byte> none;
            CHECK(resultOf(bfr_get_shader_identifiers(device.get(), {}, 0, 0, none.data(), none.size())) ==
                  Result::ErrorUnsupported);
            CHECK(resultOf(bfr_get_capture_replay_identifiers(device.get(), {}, 0, 0, none.data(), none.size())) ==
                  Result::ErrorUnsupported);
        }
        if (!made.caps.features.traceRaysIndirect) {
            CHECK(resultOf(device.traceRaysIndirect(*list, Buffer{}, 0)) == Result::ErrorUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && !made.caps.features.rayTracingPipelineLibrary) {
            CHECK(resultOf(device.createRayTracingPipeline({.library = true})) == Result::ErrorUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && !made.caps.features.rayTracingMotion) {
            CHECK(resultOf(device.createRayTracingPipeline({.allowMotion = true})) == Result::ErrorUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && !made.caps.features.opacityMicromap) {
            CHECK(resultOf(device.createRayTracingPipeline({.allowOpacityMicromaps = true})) ==
                  Result::ErrorUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && !made.caps.features.clusterAccelerationStructure) {
            CHECK(resultOf(device.createRayTracingPipeline({.allowClusters = true})) == Result::ErrorUnsupported);
        }
        // Every buffer has an address where the device gives them, whatever its usage.
        const auto address = readback.bufferAddress();
        if (made.caps.features.bufferAddress) {
            INFO(failureOf(address));
            REQUIRE(address.has_value());
            CHECK(*address != 0);
        }
        else {
            CHECK(resultOf(address) == Result::ErrorUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && takesSpirv(backend)) {
            RayTracing tracing{device, made.caps};
            // Room for every identifier, of a ray tracing pipeline.
            std::vector<std::byte> out(made.caps.rayTracing.identifierSize);
            CHECK(tracing.pipeline.shaderIdentifiers(0, 1, out.data(), out.size()).has_value());
            CHECK(resultOf(tracing.pipeline.shaderIdentifiers(0, 2, out.data(), out.size())) ==
                  Result::ErrorInvalidArgument);
            if (made.caps.features.rayTracingCaptureReplay) {
                // Replay identifiers BFR would read past.
                const std::array<RayTracingGroup, 2> groups{};
                const std::vector<std::uint8_t> tooFew(made.caps.rayTracing.captureReplayIdentifierSize);
                CHECK(resultOf(device.createRayTracingPipeline(
                          {.groups = groups, .captureReplay = true, .replayIdentifiers = tooFew})) ==
                      Result::ErrorInvalidArgument);
            }
            // An enum value BFR doesn't have.
            constexpr std::int32_t kNoSuchStage = 0xFF;
            const RayTracingShader unknown{.stage = static_cast<RayTracingStage>(kNoSuchStage)};
            CHECK(resultOf(device.createRayTracingPipeline({.shaders = std::span{&unknown, 1}})) ==
                  Result::ErrorInvalidArgument);
        }
        CHECK(goneStale(device, readback));
        device.destroyCommandList(*list);
        CHECK(resultOf(device.barrier(*list, {})) == Result::ErrorInvalidHandle);
        CHECK(goneStale(device, *pool));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device compacts, serializes and deserializes structures, and writes their properties, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const Caps& caps = made.caps;
        if (!caps.features.accelerationStructure || !caps.features.rayQuery) {
            SKIP("the adapter has no acceleration structures or ray queries");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const RayQuery query{device};
        const auto rebuilt = [](Scene& scene) {
            return [&scene](CommandList list) {
                scene.buildTop(list, false);
            };
        };

        SECTION("a compact copy traces as its structure does") {
            Scene scene{device, {}, {.allowCompaction = true}};
            scene.place(caps, 2 * kSpacing);
            const std::array<AccelerationStructureRef, 1> structures{scene.triangles.structure};
            const auto compacted = propertyOf(device, caps, AccelerationStructureProperty::CompactedSize, structures,
                                              [&](CommandList list) { scene.build(list); });
            REQUIRE(compacted.front() != 0);
            AccelerationStructure compact = created(device, AccelerationStructureDesc{.size = compacted.front()});
            run(device, [&](CommandList list) {
                const auto copied = device.compactAccelerationStructure(list, compact, scene.triangles.structure);
                INFO(failureOf(copied));
                CHECK(copied.has_value());
            });
            scene.place(caps, 2 * kSpacing, compact);
            CHECK(traced(device, scene.top.structure, rebuilt(scene), query) == kPlaced);
            CHECK(goneStale(device, compact));
        }
        SECTION("a deserialized structure traces as its structure does, where the device serializes them") {
            if (!caps.features.accelerationStructureSerialization) {
                SKIP("the adapter serializes no acceleration structures");
            }
            Scene scene{device};
            scene.place(caps, 2 * kSpacing);
            const std::array<AccelerationStructureRef, 1> structures{scene.triangles.structure};
            const auto serialized = propertyOf(device, caps, AccelerationStructureProperty::SerializedSize, structures,
                                               [&](CommandList list) { scene.build(list); });
            REQUIRE(serialized.front() >= BFR_SERIALIZED_IDENTIFIER_SIZE);
            Buffer data = created(device, BufferDesc{.size = serialized.front(),
                                                     .memory = BufferMemory::Readback,
                                                     .usage = {.accelerationStructureData = true}});
            run(device, [&](CommandList list) {
                const auto written = device.serializeAccelerationStructure(list, scene.triangles.structure, data, 0);
                INFO(failureOf(written));
                CHECK(written.has_value());
            });
            const auto bytes = data.map();
            REQUIRE(bytes.has_value());
            const auto compatible = device.serializedCompatible(AccelerationStructureType::Bottom, *bytes);
            INFO(failureOf(compatible));
            REQUIRE(compatible.has_value());
            CHECK(*compatible);
            // As large as the structure it came from, which is at least the
            // size its data names.
            const auto sizes = device.accelerationStructureSizes(scene.triangleInputs());
            REQUIRE(sizes.has_value());
            AccelerationStructure copy = created(device, AccelerationStructureDesc{.size = sizes->size});
            run(device, [&](CommandList list) {
                const auto read = device.deserializeAccelerationStructure(list, data, 0, copy);
                INFO(failureOf(read));
                CHECK(read.has_value());
            });
            scene.place(caps, 2 * kSpacing, copy);
            CHECK(traced(device, scene.top.structure, rebuilt(scene), query) == kPlaced);
            CHECK(goneStale(device, copy));
            CHECK(goneStale(device, data));
        }
        SECTION("a structure's current size and a top structure's bottom count, where the device writes them") {
            Scene scene{device};
            scene.place(caps, 2 * kSpacing);
            const auto build = [&](CommandList list) {
                scene.build(list);
            };
            if (caps.features.accelerationStructureCurrentSize) {
                const std::array<AccelerationStructureRef, 1> structures{scene.triangles.structure};
                const auto current = propertyOf(device, caps, AccelerationStructureProperty::CurrentSize, structures,
                                                build);
                const auto sizes = device.accelerationStructureSizes(scene.triangleInputs());
                REQUIRE(sizes.has_value());
                CHECK(current.front() != 0);
                CHECK(current.front() <= sizes->size);
            }
            if (caps.features.serializedBottomCount) {
                const std::array<AccelerationStructureRef, 1> structures{scene.top.structure};
                const auto bottoms = propertyOf(device, caps, AccelerationStructureProperty::SerializedBottomCount,
                                                structures, build);
                CHECK(bottoms.front() != 0);
            }
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device lets shaders read a hit triangle's positions, of a structure built to allow it, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps.features.rayPositionFetch || !made.caps.features.rayQuery) {
            SKIP("the adapter fetches no hit positions or has no ray queries");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const RayQuery positions{device, bfr::test::spirv::kRayPositionsComp};
        Scene scene{device, {}, {.allowDataAccess = true}};
        scene.place(made.caps, 2 * kSpacing);
        // The triangles' first vertex is at x = -1 in object space, wherever
        // their instances are.
        const auto left = std::bit_cast<std::uint32_t>(-1.0F);
        constexpr std::uint32_t kAll = kRays;
        const std::array<std::uint32_t, kAll> expected{left, kMiss, left, kMiss, kMiss, kMiss, kMiss, kMiss};
        CHECK(traced(device, scene.top.structure, [&](CommandList list) { scene.build(list); }, positions) == expected);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device builds opacity micromaps, and rays take triangles' opacity from them, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps.features.opacityMicromap) {
            SKIP("the adapter has no opacity micromaps");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        CHECK(made.caps.rayTracing.maxOpacity2StateSubdivisionLevel != 0);
        // Every triangle a ray meets is hit unless a micromap makes it
        // transparent.
        const RayQuery query{device, bfr::test::spirv::kRayQueryAnyComp};
        const std::span usages{&kOneMicromap, 1};
        // The rays a scene whose triangles take their opacity from micromap,
        // or from indices, hit; with disabled, instances disable the
        // micromaps.
        const auto tracedWith = [&](const TriangleOpacityMicromap& linked, const Micromap* micromap,
                                    bool disabled = false) {
            Scene scene{device, {}, {.allowDisableOpacityMicromaps = disabled}, linked, usages};
            scene.disableOpacityMicromaps = disabled;
            scene.place(made.caps, 2 * kSpacing);
            const auto built = [&](CommandList list) {
                if (micromap != nullptr) {
                    micromap->build(list);
                }
                scene.build(list);
            };
            return traced(device, scene.top.structure, built, query);
        };

        SECTION("an opaque micromap's triangles are hit, a transparent one's are not") {
            const Micromap opaque{device, true};
            const Micromap clear{device, false};
            CHECK(tracedWith({.micromap = opaque.built.structure, .usageCount = 1}, &opaque) == kPlaced);
            CHECK(tracedWith({.micromap = clear.built.structure, .usageCount = 1}, &clear) == kBoxOnly);
            // Instances that disable it see its triangles as they are.
            CHECK(tracedWith({.micromap = clear.built.structure, .usageCount = 1}, &clear, true) == kPlaced);
        }
        SECTION("special indices make triangles opaque or transparent with no micromap") {
            const auto special = [&](OpacityMicromapSpecialIndex index) {
                Buffer indices = uploaded(device, std::array<std::int32_t, 1>{std::to_underlying(index)}, kBuildInput);
                const auto hits = tracedWith(
                    {.indices = indices, .indexType = IndexType::Uint32, .indexStride = sizeof(std::int32_t)}, nullptr);
                CHECK(goneStale(device, indices));
                return hits;
            };
            CHECK(special(OpacityMicromapSpecialIndex::FullyOpaque) == kPlaced);
            CHECK(special(OpacityMicromapSpecialIndex::FullyTransparent) == kBoxOnly);
        }
        SECTION("a ray tracing pipeline made to take micromaps takes them, where the device has one") {
            if (!made.caps.features.rayTracingPipeline) {
                SKIP("the adapter has no ray tracing pipelines");
            }
            const RayTracing tracing{device, made.caps, {.allowOpacityMicromaps = true}};
            const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
                CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
                CHECK(device.pushConstants(list, push).has_value());
                CHECK(device.traceRays(list, tracing.desc()).has_value());
            };
            const Micromap clear{device, false};
            Scene scene{device, {}, {}, {.micromap = clear.built.structure, .usageCount = 1}, usages};
            scene.place(made.caps, 2 * kSpacing);
            const auto built = [&](CommandList list) {
                clear.build(list);
                scene.build(list);
            };
            CHECK(traced(device, scene.top.structure, built, trace) == kBoxOnly);
        }
        SECTION("a micromap is not taken where another structure is, nor another where a micromap is") {
            const Micromap micromap{device, true};
            const Scene scene{device};
            auto pool = device.createCommandPool({.queue = kQueue, .listCount = 1});
            REQUIRE(pool.has_value());
            const auto list = pool->beginCommands();
            REQUIRE(list.has_value());
            CHECK(resultOf(device.copyAccelerationStructure(*list, scene.triangles.structure,
                                                            micromap.built.structure)) == Result::ErrorInvalidArgument);
            // Triangles take their opacity only from a micromap.
            TriangleGeometry named = scene.triangleGeometry;
            named.opacityMicromap = {.micromap = scene.box.structure, .usageCount = 1};
            CHECK(resultOf(device.accelerationStructureSizes({.type = AccelerationStructureType::Bottom,
                                                              .triangles = std::span{&named, 1},
                                                              .micromapUsages = usages})) ==
                  Result::ErrorInvalidArgument);
            // Usages past the inputs'.
            named.opacityMicromap = {.micromap = micromap.built.structure, .firstUsage = 1, .usageCount = 1};
            CHECK(resultOf(device.accelerationStructureSizes({.type = AccelerationStructureType::Bottom,
                                                              .triangles = std::span{&named, 1},
                                                              .micromapUsages = usages})) ==
                  Result::ErrorInvalidArgument);
            // Instances are of bottom structures.
            std::vector<std::byte> out(made.caps.rayTracing.instanceSize);
            const AccelerationStructureInstance instance{.bottomStructure = micromap.built.structure};
            CHECK(resultOf(device.writeInstances(std::span{&instance, 1}, out.data(), out.size())) ==
                  Result::ErrorInvalidArgument);
            device.destroyCommandList(*list);
            CHECK(resultOf(device.barrier(*list, {})) == Result::ErrorInvalidHandle);
            CHECK(goneStale(device, *pool));
        }
        SECTION("a copied micromap says what the built one does") {
            const Micromap original{device, true};
            const auto whole = device.accelerationStructureSizes(original.inputs());
            REQUIRE(whole.has_value());
            AccelerationStructure copy = created(device, {.type = AccelerationStructureType::OpacityMicromap,
                                                          .size = whole->size});
            run(device, [&](CommandList list) {
                original.build(list);
                const auto copied = device.copyAccelerationStructure(list, copy, original.built.structure);
                INFO(failureOf(copied));
                CHECK(copied.has_value());
            });
            CHECK(tracedWith({.micromap = copy, .usageCount = 1}, nullptr) == kPlaced);
            CHECK(goneStale(device, copy));
        }
        SECTION("a compacted micromap, and a deserialized one, say what the built one does") {
            const Micromap original{device, true, {.allowCompaction = true}};
            const std::array<AccelerationStructureRef, 1> structures{original.built.structure};
            // A driver may make no pool of micromaps' properties; it says so
            // as it fails (the RTX 3070's: VK_ERROR_UNKNOWN).
            {
                auto pool = device.createAccelerationStructurePropertyPool(
                    {.property = AccelerationStructureProperty::CompactedSize, .count = 1, .opacityMicromaps = true});
                // Only the driver's own refusal of the pool skips, after the
                // sections before have said whether validation spoke.
                if (!pool.has_value() && pool.error().code == Result::ErrorDeviceFailed &&
                    failureOf(pool).find("vkCreateQueryPool") != std::string::npos) {
                    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
                    SKIP("the driver makes no pool of micromaps' properties: " << failureOf(pool));
                }
                REQUIRE(pool.has_value());
                CHECK(goneStale(device, *pool));
            }
            const auto sizeOf = [&](AccelerationStructureProperty property) {
                auto pool = device.createAccelerationStructurePropertyPool(
                    {.property = property, .count = 1, .opacityMicromaps = true});
                INFO(failureOf(pool));
                REQUIRE(pool.has_value());
                Buffer readback = created(device, BufferDesc{.size = sizeof(std::uint64_t),
                                                             .memory = BufferMemory::Readback,
                                                             .usage = {.copyDst = true}});
                run(device, [&](CommandList list) {
                    CHECK(device.resetAccelerationStructureProperties(list, *pool, 0, 1).has_value());
                    original.build(list);
                    const auto written = device.writeAccelerationStructureProperties(list, *pool, 0, structures);
                    INFO(failureOf(written));
                    CHECK(written.has_value());
                    CHECK(globalBarrier(device, list, kPropertiesWritten, kPropertiesResolved).has_value());
                    CHECK(device.resolveAccelerationStructureProperties(list, *pool, 0, 1, readback, 0).has_value());
                });
                std::uint64_t value = 0;
                const auto read = readback.map();
                REQUIRE(read.has_value());
                std::memcpy(&value, *read, sizeof(value));
                CHECK(goneStale(device, readback));
                CHECK(goneStale(device, *pool));
                return value;
            };
            const std::uint64_t compacted = sizeOf(AccelerationStructureProperty::CompactedSize);
            REQUIRE(compacted != 0);
            AccelerationStructure small = created(device, {.type = AccelerationStructureType::OpacityMicromap,
                                                           .size = compacted});
            run(device, [&](CommandList list) {
                const auto copied = device.compactAccelerationStructure(list, small, original.built.structure);
                INFO(failureOf(copied));
                CHECK(copied.has_value());
            });
            CHECK(tracedWith({.micromap = small, .usageCount = 1}, nullptr) == kPlaced);

            const std::uint64_t serialized = sizeOf(AccelerationStructureProperty::SerializedSize);
            REQUIRE(serialized > BFR_SERIALIZED_IDENTIFIER_SIZE);
            Buffer bytes = created(device, BufferDesc{.size = serialized,
                                                      .memory = BufferMemory::Readback,
                                                      .usage = {.accelerationStructureData = true}});
            run(device, [&](CommandList list) {
                CHECK(device.serializeAccelerationStructure(list, original.built.structure, bytes, 0).has_value());
            });
            const auto mapped = bytes.map();
            REQUIRE(mapped.has_value());
            const auto compatible = device.serializedCompatible(AccelerationStructureType::OpacityMicromap, *mapped);
            INFO(failureOf(compatible));
            REQUIRE(compatible.has_value());
            CHECK(*compatible);
            // Deserialized at the size the original was made with.
            const auto whole = device.accelerationStructureSizes(original.inputs());
            REQUIRE(whole.has_value());
            AccelerationStructure restored = created(device, {.type = AccelerationStructureType::OpacityMicromap,
                                                              .size = whole->size});
            run(device, [&](CommandList list) {
                CHECK(device.deserializeAccelerationStructure(list, bytes, 0, restored).has_value());
            });
            CHECK(tracedWith({.micromap = restored, .usageCount = 1}, nullptr) == kPlaced);
            for (AccelerationStructure* structure : {&small, &restored}) {
                CHECK(goneStale(device, *structure));
            }
            CHECK(goneStale(device, bytes));
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device builds structures with motion, and rays traced at a time hit them where they are then, where it "
          "has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps.features.rayTracingMotion) {
            SKIP("the adapter has no structures with motion");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const CapsRayTracing& limits = made.caps.rayTracing;
        REQUIRE(limits.maxMotionKeyframes >= 2);
        constexpr std::uint32_t kKeyframes = 2;
        // A triangle about the origin at time 0 that has moved 10 along x by
        // time 1, and a box about the origin that stays.
        Buffer vertices = uploaded(device,
                                   std::array<float, kVertexFloats * kTriangleVertices>{-1, -1, 0, 1, -1, 0, 0, 1, 0},
                                   kBuildInput);
        constexpr float kMovedX = 2 * kSpacing;
        Buffer moved = uploaded(device,
                                std::array<float, kVertexFloats * kTriangleVertices>{kMovedX - 1, -1, 0, kMovedX + 1,
                                                                                     -1, 0, kMovedX, 1, 0},
                                kBuildInput);
        Buffer boxes = uploaded(device, std::array<float, BFR_BOX_FLOATS>{-1, -1, -1, 1, 1, 1}, kBuildInput);
        const TriangleGeometry triangle{.vertices = vertices,
                                        .vertexFormat = Format::Rgb32Float,
                                        .vertexStride = kVertexFloats * sizeof(float),
                                        .vertexCount = kTriangleVertices,
                                        .triangleCount = 1,
                                        .opaque = true,
                                        .motionVertices = moved};
        const BoxGeometry box{.boxes = boxes, .stride = BFR_BOX_FLOATS * sizeof(float), .count = 1};
        const AccelerationStructureInputs triangleInputs{.type = AccelerationStructureType::Bottom,
                                                         .triangles = std::span{&triangle, 1},
                                                         .motionKeyframes = kKeyframes};
        const AccelerationStructureInputs boxInputs{.type = AccelerationStructureType::Bottom,
                                                    .boxes = std::span{&box, 1}};
        // Instances: the moving triangle, still; a box moved by matrices
        // from x = 20 to 30; and one moved by SRT transforms to x = 25 from a
        // start that uses every part: scaled 3 in y, sheared 2 in y by z,
        // pivoted 4 in y, turned 90 degrees about z and moved 33 in x, so it
        // lies across x = 24 to 34 and only rays 5 and 6 hit it. A part read
        // into the wrong field moves it off those rays.
        const std::uint64_t recordBytes = std::uint64_t{limits.motionInstanceSize} * kInstances;
        Buffer records = created(device, BufferDesc{.size = recordBytes,
                                                    .memory = BufferMemory::Upload,
                                                    .usage = {.accelerationStructureInput = true}});
        const AccelerationStructureInputs topInputs{.type = AccelerationStructureType::Top,
                                                    .instances = records,
                                                    .instanceCount = kInstances,
                                                    .motionKeyframes = kKeyframes};
        const auto make = [&](const AccelerationStructureInputs& inputs, std::uint32_t motion) {
            const auto sizes = device.accelerationStructureSizes(inputs);
            INFO(failureOf(sizes));
            REQUIRE(sizes.has_value());
            return Built{.structure = created(device,
                                              AccelerationStructureDesc{.type = inputs.type,
                                                                        .size = sizes->size,
                                                                        .motionKeyframes = motion,
                                                                        .motionInstances = motion == 0 ? 0
                                                                                                       : kInstances}),
                         .scratch = created(device, BufferDesc{.size = sizes->buildScratch,
                                                               .usage = {.accelerationStructureScratch = true}})};
        };
        Built triangles = make(triangleInputs, kKeyframes);
        Built still = make(boxInputs, 0);
        Built top = make(topInputs, kKeyframes);

        const float halfTurn = std::sqrt(0.5F);
        const SrtTransform srtStart{.scale = {1, 3, 1},
                                    .shear = {0, 0, 2},
                                    .pivot = {0, 4, 0},
                                    .rotation = {0, 0, halfTurn, halfTurn},
                                    .translation = {33, 0, 0}};
        const SrtTransform srtEnd{.scale = {1, 1, 1}, .rotation = {0, 0, 0, 1}, .translation = {5 * kSpacing, 0, 0}};
        const std::array<MotionKeyframe, 4> keyframes{{
            {.matrix = movedBy(4 * kSpacing)},
            {.matrix = movedBy(6 * kSpacing)},
            {.srt = srtStart},
            {.srt = srtEnd},
        }};
        constexpr std::array<std::uint32_t, kRays> kAtStart{kFirst, kMiss, kMiss, kMiss, kBox, kSecond, kSecond, kMiss};
        constexpr std::array<std::uint32_t, kRays> kAtEnd{kMiss, kMiss, kFirst, kMiss, kMiss, kSecond, kBox, kMiss};
        const std::array<AccelerationStructureMotionInstance, kInstances> instances{{
            // Still, so its keyframes, past those given, are never read.
            {.instance = {.transform = movedBy(0),
                          .customIndex = kFirst,
                          .mask = kAllRays,
                          .bottomStructure = triangles.structure},
             .motion = InstanceMotion::None,
             .firstKeyframe = std::numeric_limits<std::uint32_t>::max(),
             .keyframeCount = kKeyframes},
            {.instance =
                 {.customIndex = kBox, .mask = kAllRays, .hitGroupOffset = 1, .bottomStructure = still.structure},
             .motion = InstanceMotion::Matrix,
             .firstKeyframe = 0,
             .keyframeCount = kKeyframes},
            {.instance =
                 {.customIndex = kSecond, .mask = kAllRays, .hitGroupOffset = 1, .bottomStructure = still.structure},
             .motion = InstanceMotion::Srt,
             .firstKeyframe = 2,
             .keyframeCount = kKeyframes},
        }};
        const auto mapped = records.map();
        REQUIRE(mapped.has_value());
        const auto written = device.writeMotionInstances(instances, keyframes, *mapped, recordBytes);
        INFO(failureOf(written));
        REQUIRE(written.has_value());

        RayTracing tracing{device, made.caps, {.motion = true}};
        const auto built = [&](CommandList list) {
            for (const auto& [structure, inputs] :
                 {std::pair{&triangles, triangleInputs}, std::pair{&still, boxInputs}}) {
                const auto recorded = device.buildAccelerationStructure(list, {.dst = structure->structure,
                                                                               .inputs = inputs,
                                                                               .scratch = structure->scratch});
                INFO(failureOf(recorded));
                CHECK(recorded.has_value());
            }
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
            const auto recorded = device.buildAccelerationStructure(list, {.dst = top.structure,
                                                                           .inputs = topInputs,
                                                                           .scratch = top.scratch});
            INFO(failureOf(recorded));
            CHECK(recorded.has_value());
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
        };
        // The rays traced at time, indirectly where args is a buffer.
        const auto at = [&](float time, const Buffer* args = nullptr) {
            const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
                std::array<std::uint8_t, 3 * sizeof(std::uint32_t)> timed{};
                std::memcpy(timed.data(), push.data(), push.size());
                std::memcpy(timed.data() + push.size(), &time, sizeof(time));
                CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
                CHECK(device.pushConstants(list, timed).has_value());
                const auto traced = args == nullptr ? device.traceRays(list, tracing.desc())
                                                    : device.traceRaysIndirect(list, *args, 0);
                INFO(failureOf(traced));
                CHECK(traced.has_value());
            };
            return traced(device, top.structure, built, trace);
        };
        CHECK(at(0) == kAtStart);
        CHECK(at(1) == kAtEnd);
        if (made.caps.features.traceRaysIndirect && made.caps.features.rayTracingMotionIndirect) {
            Buffer args = tracing.indirectArgs();
            CHECK(at(0, &args) == kAtStart);
            CHECK(at(1, &args) == kAtEnd);
            CHECK(goneStale(device, args));
        }
        else if (made.caps.features.traceRaysIndirect) {
            run(device, [&](CommandList list) {
                CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
                const auto traced = device.traceRaysIndirect(list, Buffer{}, 0);
                REQUIRE_FALSE(traced.has_value());
                CHECK(traced.error().code == Result::ErrorUnsupported);
            });
        }

        for (Built* done : {&triangles, &still, &top}) {
            CHECK(goneStale(device, done->structure));
            CHECK(goneStale(device, done->scratch));
        }
        for (Buffer* buffer : {&vertices, &moved, &boxes, &records}) {
            CHECK(goneStale(device, *buffer));
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

namespace {

// An Upload buffer with accelerationStructureInput usage holding records,
// and its address.
template <class T, std::size_t N>
std::pair<Buffer, std::uint64_t> addressed(const Device& device, const std::array<T, N>& records) {
    Buffer buffer = uploaded(device, records, kBuildInput);
    const auto address = buffer.bufferAddress();
    INFO(failureOf(address));
    REQUIRE(address.has_value());
    return {std::move(buffer), *address};
}

// Where partitionedInstance places an instance.
struct Placement {
    float x = 0;
    std::uint32_t customIndex = 0;
    std::uint32_t instanceIndex = 0;
    std::uint32_t partitionIndex = 0;
    std::uint64_t reference = 0;
};

// A write_instance record: instance instanceIndex of the bottom structure
// reference names, moved by x, in partition partitionIndex, answering a hit
// with customIndex. bfr.h's record holds its transform as a plain array.
bfr_partitioned_instance_record partitionedInstance(const Placement& placed) {
    bfr_partitioned_instance_record record{.transform = {},
                                           .bounds = {},
                                           .custom_index = placed.customIndex,
                                           .mask = kAllRays,
                                           .hit_group_offset = 0,
                                           .flags = 0,
                                           .instance_index = placed.instanceIndex,
                                           .partition_index = placed.partitionIndex,
                                           .bottom_reference = placed.reference};
    std::ranges::copy(movedBy(placed.x), std::begin(record.transform));
    return record;
}

} // namespace

TEST_CASE("Device builds partitioned top structures from records the GPU reads, updates them, and rays traced "
          "inline hit them, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps.features.partitionedTop || !made.caps.features.rayQuery) {
            SKIP("the adapter has no partitioned top structures or no ray queries");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        REQUIRE(made.caps.features.bufferAddress);
        const RayQuery query{device, bfr::test::spirv::kRayQueryPartitionedComp};
        Scene scene{device};
        // The triangle at x = 0 in partition 0, at x = 10 in partition 1, and
        // the box at x = 20 in the global partition.
        constexpr std::uint32_t kPartitions = 2;
        const auto reference = [&](AccelerationStructure& structure) {
            const auto named = structure.accelerationStructureReference();
            REQUIRE(named.has_value());
            return *named;
        };
        auto [instances, instancesAddress] =
            addressed(device, std::array<bfr_partitioned_instance_record, kInstances>{
                                  partitionedInstance({.x = 0,
                                                       .customIndex = kFirst,
                                                       .instanceIndex = 0,
                                                       .partitionIndex = 0,
                                                       .reference = reference(scene.triangles.structure)}),
                                  partitionedInstance({.x = 2 * kSpacing,
                                                       .customIndex = kSecond,
                                                       .instanceIndex = 1,
                                                       .partitionIndex = 1,
                                                       .reference = reference(scene.triangles.structure)}),
                                  partitionedInstance({.x = 4 * kSpacing,
                                                       .customIndex = kBox,
                                                       .instanceIndex = 2,
                                                       .partitionIndex = BFR_GLOBAL_PARTITION,
                                                       .reference = reference(scene.box.structure)}),
                              });
        // The update moves partition 1 on by one ray's spacing.
        auto [translations,
              translationsAddress] = addressed(device, std::array<bfr_partition_translation_record, 1>{
                                                           {{.partition_index = 1, .translation = {kSpacing, 0, 0}}}});
        auto [ops, opsAddress] = addressed(
            device, std::array<bfr_partitioned_top_op, 2>{{
                        {.type = static_cast<std::uint32_t>(std::to_underlying(PartitionedTopOpType::WriteInstance)),
                         .count = kInstances,
                         .address = instancesAddress,
                         .stride = sizeof(bfr_partitioned_instance_record)},
                        {.type = static_cast<std::uint32_t>(
                             std::to_underlying(PartitionedTopOpType::WritePartitionTranslation)),
                         .count = 1,
                         .address = translationsAddress,
                         .stride = sizeof(bfr_partition_translation_record)},
                    }});
        // The build runs the first op, the update the second.
        Buffer opCount = uploaded(device, std::array<std::uint32_t, 1>{1}, kBuildInput);
        AccelerationStructureInputs inputs{.type = AccelerationStructureType::PartitionedTop,
                                           .partitioned = {.instanceCount = kInstances,
                                                           .maxPartitionInstances = 1,
                                                           .partitionCount = kPartitions,
                                                           .maxGlobalInstances = 1,
                                                           .partitionTranslation = true,
                                                           .ops = ops,
                                                           .opCount = opCount}};
        Built top = allocated(device, inputs);
        const auto build = [&](CommandList list, bool update) {
            const auto recorded = device.buildAccelerationStructure(
                list, {.dst = top.structure,
                       .src = update ? AccelerationStructureRef{top.structure} : AccelerationStructureRef{},
                       .update = update,
                       .inputs = inputs,
                       .scratch = top.scratch});
            INFO(failureOf(recorded));
            CHECK(recorded.has_value());
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
        };

        const std::array<std::uint32_t, kRays> built{kFirst, kMiss, kSecond, kMiss, kBox, kMiss, kMiss, kMiss};
        CHECK(traced(
                  device, top.structure,
                  [&](CommandList list) {
                      for (const auto& [bottom, bottomInputs] : {std::pair{&scene.triangles, scene.triangleInputs()},
                                                                 std::pair{&scene.box, scene.boxInputs()}}) {
                          CHECK(device
                                    .buildAccelerationStructure(list, {.dst = bottom->structure,
                                                                       .inputs = bottomInputs,
                                                                       .scratch = bottom->scratch})
                                    .has_value());
                      }
                      CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
                      build(list, false);
                  },
                  query) == built);

        inputs.partitioned.opOffset = sizeof(bfr_partitioned_top_op);
        const std::array<std::uint32_t, kRays> moved{kFirst, kMiss, kMiss, kSecond, kBox, kMiss, kMiss, kMiss};
        CHECK(traced(device, top.structure, [&](CommandList list) { build(list, true); }, query) == moved);

        CHECK(goneStale(device, top.structure));
        for (Buffer* buffer : {&top.scratch, &instances, &translations, &ops, &opCount}) {
            CHECK(goneStale(device, *buffer));
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device builds clusters and cluster bottom structures from records the GPU reads, and rays traced inline "
          "hit them, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps.features.clusterAccelerationStructure || !made.caps.features.rayQuery) {
            SKIP("the adapter has no cluster acceleration structures or no ray queries");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        REQUIRE(made.caps.features.bufferAddress);
        const CapsRayTracing& limits = made.caps.rayTracing;
        const RayQuery query{device};
        // Two clusters of the one triangle about the origin, each in its own
        // cluster bottom structure, at addresses the CPU picks.
        constexpr std::uint32_t kClusters = 2;
        auto [vertices, verticesAddress] =
            addressed(device, std::array<float, kVertexFloats * kTriangleVertices>{-1, -1, 0, 1, -1, 0, 0, 1, 0});
        auto [indices, indicesAddress] = addressed(device, std::array<std::uint8_t, kTriangleVertices>{0, 1, 2});
        const ClusterInputs clusterInputs{.operation = ClusterOperation::BuildClusters,
                                          .mode = ClusterMode::ExplicitDestinations,
                                          .maxCount = kClusters,
                                          .triangles = {.vertexFormat = Format::Rgb32Float,
                                                        .maxGeometryIndex = 0,
                                                        .maxUniqueGeometries = 1,
                                                        .maxTriangles = 1,
                                                        .maxVertices = kTriangleVertices,
                                                        .maxTotalTriangles = kClusters,
                                                        .maxTotalVertices = kClusters * kTriangleVertices}};
        const ClusterInputs bottomInputs{.operation = ClusterOperation::BuildBottoms,
                                         .mode = ClusterMode::ExplicitDestinations,
                                         .maxCount = kClusters,
                                         .bottoms = {.maxTotalClusters = kClusters, .maxClustersPerBottom = 1}};
        const auto clusterSizes = device.clusterSizes(clusterInputs);
        INFO(failureOf(clusterSizes));
        REQUIRE(clusterSizes.has_value());
        const auto bottomSizes = device.clusterSizes(bottomInputs);
        INFO(failureOf(bottomSizes));
        REQUIRE(bottomSizes.has_value());
        REQUIRE(clusterSizes->size != 0);
        REQUIRE(bottomSizes->size != 0);

        // Each cluster, then each bottom structure, at its alignment.
        const std::uint64_t clusterStep = alignUp(clusterSizes->size, limits.clusterAlignment);
        const std::uint64_t bottomsAt = alignUp(clusterStep * kClusters, limits.clusterBottomAlignment);
        const std::uint64_t bottomStep = alignUp(bottomSizes->size, limits.clusterBottomAlignment);
        Buffer storage = created(device, BufferDesc{.size = bottomsAt + (bottomStep * kClusters),
                                                    .usage = {.accelerationStructureStorage = true}});
        const auto storageAddress = storage.bufferAddress();
        INFO(failureOf(storageAddress));
        REQUIRE(storageAddress.has_value());
        std::array<std::uint64_t, kClusters> clusterAddresses{};
        std::array<std::uint64_t, kClusters> bottomAddresses{};
        for (std::uint32_t i = 0; i < kClusters; ++i) {
            clusterAddresses.at(i) = *storageAddress + (clusterStep * i);
            bottomAddresses.at(i) = *storageAddress + bottomsAt + (bottomStep * i);
        }
        // The addresses each build reads its destinations from; the bottom
        // records name the clusters' by where they are in this buffer.
        const std::uint64_t destinationBytes = sizeof(std::uint64_t) * kClusters * 2;
        Buffer destinations = created(device,
                                      BufferDesc{.size = destinationBytes,
                                                 .memory = BufferMemory::Upload,
                                                 .usage = {.storage = true, .accelerationStructureInput = true}});
        const auto destinationsAddress = destinations.bufferAddress();
        REQUIRE(destinationsAddress.has_value());
        {
            const auto mapped = destinations.map();
            REQUIRE(mapped.has_value());
            const std::span<std::byte> bytes = mappedBytes(*mapped, destinationBytes);
            std::memcpy(bytes.data(), clusterAddresses.data(), sizeof(clusterAddresses));
            std::memcpy(bytes.subspan(sizeof(clusterAddresses)).data(), bottomAddresses.data(),
                        sizeof(bottomAddresses));
        }

        constexpr std::uint32_t kCounts = 1U | (kTriangleVertices << BFR_CLUSTER_VERTEX_COUNT_SHIFT) |
                                          (static_cast<std::uint32_t>(std::to_underlying(ClusterIndexFormat::Uint8))
                                           << BFR_CLUSTER_INDEX_FORMAT_SHIFT);
        constexpr auto kOpaque = static_cast<std::uint32_t>(std::to_underlying(ClusterGeometryFlag::Opaque))
                                 << BFR_CLUSTER_GEOMETRY_FLAGS_SHIFT;
        const bfr_cluster_triangle_record cluster{.cluster_id = 0,
                                                  .cluster_flags = 0,
                                                  .counts = kCounts,
                                                  .base_geometry = kOpaque,
                                                  .index_stride = sizeof(std::uint8_t),
                                                  .vertex_stride = kVertexFloats * sizeof(float),
                                                  .geometry_stride = 0,
                                                  .micromap_index_stride = 0,
                                                  .indices = indicesAddress,
                                                  .vertices = verticesAddress,
                                                  .geometries = 0,
                                                  .micromaps = 0,
                                                  .micromap_indices = 0};
        Buffer clusterRecords = uploaded(device, std::array<bfr_cluster_triangle_record, kClusters>{cluster, cluster},
                                         kBuildInput);
        std::array<bfr_cluster_bottom_record, kClusters> bottoms{};
        for (std::uint32_t i = 0; i < kClusters; ++i) {
            bottoms.at(i) = {.cluster_count = 1,
                             .cluster_stride = sizeof(std::uint64_t),
                             .clusters = *destinationsAddress + (sizeof(std::uint64_t) * i)};
        }
        Buffer bottomRecords = uploaded(device, bottoms, kBuildInput);
        // Vulkan may need no scratch, and a buffer has bytes.
        const std::uint64_t scratchBytes = std::max(clusterSizes->buildScratch, bottomSizes->buildScratch);
        Buffer scratch = scratchBytes == 0
                             ? Buffer{}
                             : created(device, BufferDesc{.size = scratchBytes,
                                                          .usage = {.accelerationStructureScratch = true}});

        // A top structure of the two bottom structures, the second at x = 10.
        const std::uint64_t instanceBytes = std::uint64_t{limits.instanceSize} * kClusters;
        Buffer instances = created(device, BufferDesc{.size = instanceBytes,
                                                      .memory = BufferMemory::Upload,
                                                      .usage = {.accelerationStructureInput = true}});
        {
            const std::array<AccelerationStructureInstance, kClusters> placed{{
                {.transform = movedBy(0),
                 .customIndex = kFirst,
                 .mask = kAllRays,
                 .bottomReference = bottomAddresses[0]},
                {.transform = movedBy(2 * kSpacing),
                 .customIndex = kSecond,
                 .mask = kAllRays,
                 .bottomReference = bottomAddresses[1]},
            }};
            const auto mapped = instances.map();
            REQUIRE(mapped.has_value());
            const auto written = device.writeInstances(placed, *mapped, instanceBytes);
            INFO(failureOf(written));
            REQUIRE(written.has_value());
        }
        const AccelerationStructureInputs topInputs{.type = AccelerationStructureType::Top,
                                                    .instances = instances,
                                                    .instanceCount = kClusters};
        Built top = allocated(device, topInputs);

        const auto built = [&](CommandList list) {
            const auto clustersBuilt = device.buildClusters(list, {.inputs = clusterInputs,
                                                                   .scratch = scratch,
                                                                   .addresses = destinations,
                                                                   .addressStride = sizeof(std::uint64_t),
                                                                   .records = clusterRecords});
            INFO(failureOf(clustersBuilt));
            CHECK(clustersBuilt.has_value());
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
            const auto bottomsBuilt = device.buildClusters(list, {.inputs = bottomInputs,
                                                                  .scratch = scratch,
                                                                  .addresses = destinations,
                                                                  .addressOffset = sizeof(clusterAddresses),
                                                                  .addressStride = sizeof(std::uint64_t),
                                                                  .records = bottomRecords});
            INFO(failureOf(bottomsBuilt));
            CHECK(bottomsBuilt.has_value());
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
            CHECK(device
                      .buildAccelerationStructure(list,
                                                  {.dst = top.structure, .inputs = topInputs, .scratch = top.scratch})
                      .has_value());
            CHECK(globalBarrier(device, list, kStructureWrites, kStructureReads).has_value());
        };
        constexpr std::array<std::uint32_t, kRays> kClustered{kFirst, kMiss, kSecond, kMiss,
                                                              kMiss,  kMiss, kMiss,   kMiss};
        CHECK(traced(device, top.structure, built, query) == kClustered);
        // A trace reaches cluster bottom structures through a pipeline made
        // to.
        if (made.caps.features.rayTracingPipeline) {
            const RayTracing tracing{device, made.caps, {.allowClusters = true}};
            const auto trace = [&](CommandList list, std::span<const std::uint8_t> push) {
                CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
                CHECK(device.pushConstants(list, push).has_value());
                const auto traced = device.traceRays(list, tracing.desc());
                INFO(failureOf(traced));
                CHECK(traced.has_value());
            };
            CHECK(traced(device, top.structure, built, trace) == kClustered);
        }

        CHECK(goneStale(device, top.structure));
        for (Buffer* buffer : {&top.scratch, &instances, &scratch, &bottomRecords, &clusterRecords, &destinations,
                               &storage, &indices, &vertices}) {
            CHECK(goneStale(device, *buffer));
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses acceleration structure work it can't pass on", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        auto pool = device.createCommandPool({.queue = kQueue, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        // d1: null destroy: no message (track B)
        bfr_destroy_acceleration_structure(device.get(), {});
        if (made.caps.features.accelerationStructure && !made.caps.features.rayTracingMotion) {
            std::array<std::byte, 1> out{};
            CHECK(resultOf(device.createAccelerationStructure({.size = kSmall, .motionKeyframes = 2})) ==
                  Result::ErrorUnsupported);
            CHECK(resultOf(device.writeMotionInstances({}, {}, out.data(), out.size())) == Result::ErrorUnsupported);
            const AccelerationStructureInputs moving{.motionKeyframes = 2};
            CHECK(resultOf(device.accelerationStructureSizes(moving)) == Result::ErrorUnsupported);
            CHECK(resultOf(device.buildAccelerationStructure(*list, {.inputs = moving})) == Result::ErrorUnsupported);
        }
        if (made.caps.features.accelerationStructure && !made.caps.features.opacityMicromap) {
            CHECK(resultOf(device.createAccelerationStructure({.type = AccelerationStructureType::OpacityMicromap,
                                                               .size = kSmall})) == Result::ErrorUnsupported);
            CHECK(resultOf(device.accelerationStructureSizes({.type = AccelerationStructureType::OpacityMicromap})) ==
                  Result::ErrorUnsupported);
            CHECK(resultOf(device.createAccelerationStructurePropertyPool({.count = 1, .opacityMicromaps = true})) ==
                  Result::ErrorUnsupported);
        }
        if (made.caps.features.accelerationStructure && !made.caps.features.partitionedTop) {
            CHECK(resultOf(device.createAccelerationStructure({.type = AccelerationStructureType::PartitionedTop,
                                                               .size = kSmall})) == Result::ErrorUnsupported);
            CHECK(resultOf(device.accelerationStructureSizes({.type = AccelerationStructureType::PartitionedTop})) ==
                  Result::ErrorUnsupported);
        }
        if (!made.caps.features.clusterAccelerationStructure) {
            CHECK(resultOf(device.clusterSizes({})) == Result::ErrorUnsupported);
            CHECK(resultOf(device.buildClusters(*list, {})) == Result::ErrorUnsupported);
        }
        else {
            // A value past every enum BFR has.
            constexpr std::uint8_t kNoSuchValue = 0xFF;
            CHECK(resultOf(device.clusterSizes({.operation = static_cast<ClusterOperation>(kNoSuchValue)})) ==
                  Result::ErrorInvalidArgument);
            CHECK(resultOf(device.buildClusters(*list, {.inputs = {.mode = static_cast<ClusterMode>(kNoSuchValue)}})) ==
                  Result::ErrorInvalidArgument);
        }
        if (!made.caps.features.partitionedTop) {
            const auto refused = made.instance.createDevice(
                testDeviceWith(made.instance, kFirstAdapter, kNoWindow,
                               [](DeviceDesc& desc) { desc.maxBindlessPartitionedTops = 1; }));
            REQUIRE_FALSE(refused.has_value());
            CHECK(refused.error().code == Result::ErrorLimitReached);
        }
        if (!made.caps.features.accelerationStructure) {
            std::array<std::byte, 1> out{};
            CHECK(resultOf(device.accelerationStructureSizes({})) == Result::ErrorUnsupported);
            CHECK(resultOf(device.createAccelerationStructure({.size = 1})) == Result::ErrorUnsupported);
            CHECK(resultOf(device.writeInstances({}, out.data(), out.size())) == Result::ErrorUnsupported);
            CHECK(resultOf(device.buildAccelerationStructure(*list, {})) == Result::ErrorUnsupported);
            CHECK(resultOf(device.copyAccelerationStructure(*list, AccelerationStructure{}, AccelerationStructure{})) ==
                  Result::ErrorUnsupported);
        }
        else {
            Buffer vertices = uploaded(device, std::array<float, kVertexFloats * kTriangleVertices>{}, kBuildInput);
            const TriangleGeometry triangle{.vertices = vertices,
                                            .vertexFormat = Format::Rgb32Float,
                                            .vertexStride = kVertexFloats * sizeof(float),
                                            .vertexCount = kTriangleVertices,
                                            .triangleCount = 1};
            std::array<TriangleGeometry, kGeometries + 1> tooMany{};
            tooMany.fill(triangle);
            AccelerationStructure bottom = created(device, {.type = AccelerationStructureType::Bottom, .size = kSmall});
            AccelerationStructure top = created(device, {.type = AccelerationStructureType::Top, .size = kSmall});

            // A build of at most maxBuildGeometries, which BFR holds room for.
            CHECK(resultOf(device.accelerationStructureSizes({.triangles = tooMany})) == Result::ErrorLimitReached);
            // Only top structures are in the table.
            CHECK(resultOf(bottom.accelerationStructureSlot()) == Result::ErrorInvalidArgument);
            CHECK(top.accelerationStructureSlot().has_value() == (made.caps.bindless.maxAccelerationStructures != 0));
            // Room for every record BFR writes.
            std::vector<std::byte> out(made.caps.rayTracing.instanceSize);
            const AccelerationStructureInstance one{.bottomStructure = bottom};
            CHECK(device.writeInstances(std::span{&one, 1}, out.data(), out.size()).has_value());
            const std::array<AccelerationStructureInstance, 2> two{one, one};
            CHECK(resultOf(device.writeInstances(two, out.data(), out.size())) == Result::ErrorInvalidArgument);

            // A structure made without captureReplay has no capture.
            const auto none = bottom.capture();
            REQUIRE(none.has_value());
            CHECK(none->words == AccelerationStructureCapture{}.words);
            // Kept to name them once they are gone: bfr.hpp names a structure
            // only by a live one.
            const AccelerationStructureRef staleBottom{bottom};
            const AccelerationStructureRef staleTop{top};
            for (AccelerationStructure* structure : {&bottom, &top}) {
                CHECK(goneStale(device, *structure));
            }
            CHECK(referenceResult(device, staleTop.get()) == Result::ErrorInvalidHandle);
            CHECK(captureResult(device, staleTop.get()) == Result::ErrorInvalidHandle);
            // A property the device writes, and no more structures than its
            // pool holds.
            auto properties = device.createAccelerationStructurePropertyPool(
                {.property = AccelerationStructureProperty::CompactedSize, .count = 1});
            REQUIRE(properties.has_value());
            const std::array<AccelerationStructureRef, 2> twoStructures{staleBottom, staleBottom};
            CHECK(resultOf(device.writeAccelerationStructureProperties(*list, *properties, 0, twoStructures)) ==
                  Result::ErrorInvalidArgument);
            CHECK(resultOf(
                      device.writeAccelerationStructureProperties(*list, *properties, 1, std::span{&staleBottom, 1})) ==
                  Result::ErrorInvalidArgument);
            CHECK(goneStale(device, *properties));
            if (!made.caps.features.accelerationStructureCurrentSize) {
                CHECK(resultOf(device.createAccelerationStructurePropertyPool(
                          {.property = AccelerationStructureProperty::CurrentSize, .count = 1})) ==
                      Result::ErrorUnsupported);
            }
            CHECK(goneStale(device, vertices));
        }
        device.destroyCommandList(*list);
        CHECK(resultOf(device.barrier(*list, {})) == Result::ErrorInvalidHandle);
        CHECK(goneStale(device, *pool));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a top structure past its table, and slots from an adapter with no table", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages), 1};
        const Device& device = made.device;
        if (made.caps.bindless.maxAccelerationStructures == 0) {
            const auto refused = made.instance.createDevice(
                testDeviceWith(made.instance, kFirstAdapter, kNoWindow,
                               [](DeviceDesc& desc) { desc.maxBindlessAccelerationStructures = 1; }));
            REQUIRE_FALSE(refused.has_value());
            CHECK(refused.error().code == Result::ErrorLimitReached);
            SKIP("the adapter has no acceleration structure table");
        }
        AccelerationStructure first = created(device, {.type = AccelerationStructureType::Top, .size = kSmall});
        const auto second = device.createAccelerationStructure(
            {.type = AccelerationStructureType::Top, .size = kSmall});
        REQUIRE_FALSE(second.has_value());
        CHECK(second.error().code == Result::ErrorLimitReached);
        // Bottom structures take no slot.
        AccelerationStructure bottom = created(device, {.type = AccelerationStructureType::Bottom, .size = kSmall});
        CHECK(goneStale(device, first));
        CHECK(goneStale(device, bottom));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
