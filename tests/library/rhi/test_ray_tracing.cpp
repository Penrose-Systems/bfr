// What every backend does with acceleration structures, through the interface
// alone: builds of triangles, boxes and instances, rays traced inline against
// them from a compute shader, copies, updates, and the usage checks. Each case
// runs on every backend the build has; the shaders are the SPIR-V in
// tests/library/shaders, so a case that runs code skips on a backend with none
// checked in, and one that needs acceleration structures skips where the
// adapter has none.

#include "allocation/allocation_counter.h"
#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/rhi_config.h"
#include "rhi_test_support.h"
#include "shaders/spirv.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;
using bfr::core::ErrorCode;

namespace {

// ray_query.comp's workgroup: one ray each, x = 5i apart.
constexpr std::uint32_t kRays = 8;
constexpr float kSpacing = 5;
constexpr std::uint32_t kMiss = 0xFFFFFFFF;
constexpr std::uint32_t kTableSlots = 4;
constexpr std::uint32_t kGeometries = 4;
constexpr std::uint32_t kPropertyPools = 4;
// What the scene's instances answer a hit with.
constexpr std::uint32_t kFirst = 7;
constexpr std::uint32_t kSecond = 9;
constexpr std::uint32_t kBox = 11;
constexpr std::uint32_t kInstances = 3;
// A vertex's floats, a triangle's vertices, and a box's floats.
constexpr std::uint32_t kVertexFloats = 3;
constexpr std::uint32_t kTriangleVertices = 3;
constexpr std::uint32_t kBoxFloats = 6;
// A size any backend takes for a structure no build fills.
constexpr std::uint64_t kSmall = 256;
constexpr std::uint8_t kAllRays = 0xFF;

// An Instance and a device on its first adapter, presenting nowhere, with an
// acceleration structure table where the adapter has one.
struct RayDevice {
    explicit RayDevice(const InstanceDesc& desc, std::uint32_t tableSlots = kTableSlots,
                       const RhiConfig& config = kValidation) {
        auto created = createInstance(desc, config);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        caps = instance->adapters()[0].caps;
        const bool table = caps.bindless.maxAccelerationStructures != 0;
        auto made = instance->createDevice(0, kNoWindow, testDeviceWith([&](DeviceDesc& device) {
                                               device.maxBindlessAccelerationStructures = table ? tableSlots : 0;
                                               device.maxBuildGeometries = kGeometries;
                                               device.maxAccelerationStructurePropertyPools = kPropertyPools;
                                           }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    std::unique_ptr<Instance> instance;
    Caps caps;
    std::unique_ptr<Device> device;
};

// Whether backend takes the SPIR-V checked in, as test_pipeline.cpp says.
bool takesSpirv(const TestBackend& backend) {
    return backend.backend == Backend::Vulkan;
}

BufferHandle created(Device& device, const BufferDesc& desc) {
    const auto buffer = device.createBuffer(desc);
    INFO(failureOf(buffer));
    REQUIRE(buffer.has_value());
    return *buffer;
}

AccelerationStructureHandle created(Device& device, const AccelerationStructureDesc& desc) {
    const auto structure = device.createAccelerationStructure(desc);
    INFO(failureOf(structure));
    REQUIRE(structure.has_value());
    return *structure;
}

// An Upload buffer for builds to read holding data.
template <class T, std::size_t N>
BufferHandle input(Device& device, const std::array<T, N>& data) {
    const BufferHandle buffer = created(device, {.size = sizeof(data),
                                                 .memory = BufferMemory::Upload,
                                                 .usage = {.accelerationStructureInput = true}});
    const auto written = device.mapBuffer(buffer);
    REQUIRE(written.has_value());
    std::memcpy(written->data(), data.data(), sizeof(data));
    return buffer;
}

// A 3x4 row-major transform moving x by x.
std::array<float, kTransformFloats> movedBy(float x) {
    return {1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0};
}

// A structure built from inputs, with the scratch its build and update take.
struct Built {
    AccelerationStructureHandle structure;
    BufferHandle scratch;
};

Built allocated(Device& device, const AccelerationStructureInputs& inputs) {
    const auto sizes = device.accelerationStructureSizes(inputs);
    INFO(failureOf(sizes));
    REQUIRE(sizes.has_value());
    CHECK(sizes->size != 0);
    return {.structure = created(device, {.type = inputs.type, .size = sizes->size}),
            .scratch = created(device, {.size = std::max(sizes->buildScratch, sizes->updateScratch),
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
    explicit Scene(Device& device, const AccelerationStructureFlags& topFlags = {},
                   const AccelerationStructureFlags& bottomFlags = {}, const TriangleOpacityMicromap& micromap = {},
                   std::span<const OpacityMicromapUsage> usages = {})
        : device(&device),
          usages(usages),
          vertices(input(device, std::array<float, kVertexFloats * kTriangleVertices>{-1, -1, 0, 1, -1, 0, 0, 1, 0})),
          boxes(input(device, std::array<float, kBoxFloats>{-1, -1, -1, 1, 1, 1})) {
        triangleGeometry = {.vertices = vertices,
                            .vertexFormat = Format::Rgb32Float,
                            .vertexStride = kVertexFloats * sizeof(float),
                            .vertexCount = kTriangleVertices,
                            .triangleCount = 1,
                            .opaque = micromap.micromap.isNull() && micromap.indices.isNull(),
                            .opacityMicromap = micromap};
        boxGeometry = {.boxes = boxes, .stride = kBoxFloats * sizeof(float), .count = 1};
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
        for (const AccelerationStructureHandle structure : {triangles.structure, box.structure, top.structure}) {
            CHECK(device->destroyAccelerationStructure(structure).has_value());
        }
        for (const BufferHandle buffer : {vertices, boxes, instances, triangles.scratch, box.scratch, top.scratch}) {
            CHECK(device->destroyBuffer(buffer).has_value());
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
    void place(const Caps& caps, float secondX, AccelerationStructureHandle trianglesFrom = {}) {
        const AccelerationStructureHandle triangleStructure = trianglesFrom.isNull() ? triangles.structure
                                                                                     : trianglesFrom;
        const std::uint32_t size = caps.rayTracing.instanceSize;
        REQUIRE(size != 0);
        if (instances.isNull()) {
            instances = created(*device, {.size = std::uint64_t{size} * kInstances,
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
        const auto mapped = device->mapBuffer(instances);
        REQUIRE(mapped.has_value());
        const auto written = device->writeInstances(placed, *mapped);
        INFO(failureOf(written));
        REQUIRE(written.has_value());
        if (top.structure.isNull()) {
            top = allocated(*device, topInputs());
        }
    }

    // Records builds of the bottom structures, then of the top one.
    void build(CommandListHandle list) const {
        for (const auto& [built, inputs] : {std::pair{triangles, triangleInputs()}, std::pair{box, boxInputs()}}) {
            const auto recorded = device->buildAccelerationStructure(list, {.dst = built.structure,
                                                                            .inputs = inputs,
                                                                            .scratch = built.scratch});
            INFO(failureOf(recorded));
            CHECK(recorded.has_value());
        }
        CHECK(device->barrier(list).has_value());
        buildTop(list, false);
    }

    void buildTop(CommandListHandle list, bool update) const {
        const auto recorded = device->buildAccelerationStructure(list, {.dst = top.structure,
                                                                        .src = update ? top.structure
                                                                                      : AccelerationStructureHandle{},
                                                                        .update = update,
                                                                        .inputs = topInputs(),
                                                                        .scratch = top.scratch});
        INFO(failureOf(recorded));
        CHECK(recorded.has_value());
        CHECK(device->barrier(list).has_value());
    }

    Device* device;
    std::span<const OpacityMicromapUsage> usages;
    // The triangles' instances disable their opacity micromaps.
    bool disableOpacityMicromaps = false;
    BufferHandle vertices;
    BufferHandle boxes;
    BufferHandle instances;
    TriangleGeometry triangleGeometry;
    BoxGeometry boxGeometry;
    AccelerationStructureFlags flags;
    AccelerationStructureFlags bottomFlags;
    Built triangles;
    Built box;
    Built top;
};

// Records record, then trace, given the push constants the test shaders
// read, that sends rays against structure, and returns what each ray hit.
template <class Record, class Trace>
std::array<std::uint32_t, kRays> traced(Device& device, AccelerationStructureHandle structure, const Record& record,
                                        const Trace& trace) {
    constexpr std::uint64_t kBytes = sizeof(std::uint32_t) * kRays;
    const BufferHandle results = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
    const BufferHandle readback =
        created(device, {.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
    const auto slot = device.accelerationStructureSlot(structure);
    INFO(failureOf(slot));
    REQUIRE(slot.has_value());
    const auto resultSlot = device.bufferSlot(results);
    REQUIRE(resultSlot.has_value());
    const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
    REQUIRE(pool.has_value());
    const auto list = device.beginCommands(*pool);
    REQUIRE(list.has_value());

    record(*list);
    const std::array<std::uint32_t, 2> push{*slot, *resultSlot};
    trace(*list, std::as_bytes(std::span{push}));
    CHECK(device.barrier(*list).has_value());
    const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
    CHECK(device.copyBuffer(*list, results, readback, std::span{&whole, 1}).has_value());
    const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
    INFO(failureOf(sent));
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());

    std::array<std::uint32_t, kRays> hits{};
    const auto read = device.mapBuffer(readback);
    REQUIRE(read.has_value());
    std::memcpy(hits.data(), read->data(), kBytes);
    CHECK(device.destroyBuffer(results).has_value());
    CHECK(device.destroyBuffer(readback).has_value());
    CHECK(device.destroyCommandPool(*pool).has_value());
    return hits;
}

// ray_query.comp, which traces from a compute shader.
// Or, given code, another shader that traces as it does.
struct RayQuery {
    explicit RayQuery(Device& device, std::span<const std::uint8_t> code = bfr::test::spirv::kRayQueryComp)
        : device(&device) {
        const auto made = device.createComputePipeline(
            {.compute = {.code = std::as_bytes(code), .entryPoint = "main"}, .workgroupSize = {kRays, 1, 1}});
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        pipeline = *made;
    }
    RayQuery(const RayQuery&) = delete;
    RayQuery& operator=(const RayQuery&) = delete;
    RayQuery(RayQuery&&) = delete;
    RayQuery& operator=(RayQuery&&) = delete;
    ~RayQuery() { CHECK(device->destroyPipeline(pipeline).has_value()); }

    void operator()(CommandListHandle list, std::span<const std::byte> push) const {
        CHECK(device->bindPipeline(list, pipeline).has_value());
        CHECK(device->pushConstants(list, push).has_value());
        CHECK(device->dispatch(list, 1, 1, 1).has_value());
    }

    Device* device;
    PipelineHandle pipeline;
};

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

ShaderCode spirv(std::span<const std::uint8_t> bytes) {
    return {.code = std::as_bytes(bytes), .entryPoint = "main"};
}

// How a RayTracing's pipeline is made: with stackSize; with linked, its
// miss and hit groups in a library it links, which follow its ray generation
// group as they would in the pipeline itself; with captureReplay and the
// groups' replayIdentifiers.
struct TracingOptions {
    std::uint64_t stackSize = 0;
    bool linked = false;
    bool captureReplay = false;
    std::span<const std::byte> replayIdentifiers{};
    bool allowOpacityMicromaps = false;
    // Made with allowMotion, its ray generation trace_motion.rgen, which
    // traces at the time a third push constant holds.
    bool motion = false;
};

// trace.rgen, trace.rmiss, and hit groups of trace.rchit for triangles and
// with box.rint for boxes, in a pipeline, and a binding table of their
// records: ray generation, then miss, then the two hit groups, each region at
// a table alignment.
struct RayTracing {
    RayTracing(Device& device, const Caps& caps, const TracingOptions& options = {}) : device(&device) {
        const std::array codes{options.motion ? spirv(bfr::test::spirv::kTraceMotionRgen)
                                              : spirv(bfr::test::spirv::kTraceRgen),
                               spirv(bfr::test::spirv::kTraceRmiss), spirv(bfr::test::spirv::kTraceRchit),
                               spirv(bfr::test::spirv::kBoxRint)};
        std::array<ShaderHandle, kShaders> shaders{};
        for (std::size_t i = 0; i < kShaders; ++i) {
            const auto shader = device.createShader(codes.at(i));
            INFO(failureOf(shader));
            REQUIRE(shader.has_value());
            shaders.at(i) = *shader;
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
                                           .allowMotion = options.motion};
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
            const auto linked = device.createRayTracingPipeline(part);
            INFO(failureOf(linked));
            REQUIRE(linked.has_value());
            library = *linked;
        }
        RayTracingPipelineDesc desc = whole;
        if (options.linked) {
            desc.shaders = std::span{stages}.first(1);
            desc.groups = std::span{groups}.first(1);
            desc.libraries = std::span{&library, 1};
        }
        const auto made = device.createRayTracingPipeline(desc);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        pipeline = *made;
        // A pipeline keeps what it needs of its shaders.
        for (const ShaderHandle shader : shaders) {
            CHECK(device.destroyShader(shader).has_value());
        }

        const RayTracingLimits& limits = caps.rayTracing;
        stride = alignUp(limits.identifierSize, limits.identifierAlignment);
        spacing = alignUp(2 * stride, limits.tableAlignment);
        std::vector<std::byte> identifiers(std::size_t{limits.identifierSize} * kGroups);
        const auto written = device.shaderIdentifiers(pipeline, 0, kGroups, identifiers);
        INFO(failureOf(written));
        REQUIRE(written.has_value());
        table = created(device,
                        {.size = 3 * spacing, .memory = BufferMemory::Upload, .usage = {.shaderBindingTable = true}});
        const auto mapped = device.mapBuffer(table);
        REQUIRE(mapped.has_value());
        // Group i at its region's start, the two hit groups a stride apart.
        const std::array<std::uint64_t, kGroups> at{0, spacing, 2 * spacing, 2 * spacing + stride};
        for (std::size_t i = 0; i < kGroups; ++i) {
            std::memcpy(mapped->data() + at.at(i), identifiers.data() + i * limits.identifierSize,
                        limits.identifierSize);
        }
    }
    RayTracing(const RayTracing&) = delete;
    RayTracing& operator=(const RayTracing&) = delete;
    RayTracing(RayTracing&&) = delete;
    RayTracing& operator=(RayTracing&&) = delete;
    ~RayTracing() {
        CHECK(device->destroyPipeline(pipeline).has_value());
        CHECK(device->destroyPipeline(library).has_value());
        CHECK(device->destroyBuffer(table).has_value());
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

    static constexpr std::size_t kShaders = 4;
    static constexpr std::uint32_t kGroups = 4;
    Device* device;
    PipelineHandle library;
    PipelineHandle pipeline;
    BufferHandle table;
    std::uint64_t stride = 0;
    std::uint64_t spacing = 0;
};

// Records record in a list on queue 0, submits it and waits for it.
template <class Record>
void run(Device& device, const Record& record) {
    const auto pool = device.createCommandPool({.queue = 0, .listCount = 1});
    REQUIRE(pool.has_value());
    const auto list = device.beginCommands(*pool);
    REQUIRE(list.has_value());
    record(*list);
    const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
    INFO(failureOf(sent));
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());
    CHECK(device.destroyCommandPool(*pool).has_value());
}

// Each structure's property, written after before, which builds them, and
// read back.
template <class Before>
std::vector<std::uint64_t> propertyOf(Device& device, const Caps& caps, AccelerationStructureProperty property,
                                      std::span<const AccelerationStructureHandle> structures, const Before& before) {
    const auto count = static_cast<std::uint32_t>(structures.size());
    const auto pool = device.createAccelerationStructurePropertyPool({.property = property, .count = count});
    INFO(failureOf(pool));
    REQUIRE(pool.has_value());
    const std::uint32_t entry = caps.rayTracing.propertyEntryBytes;
    REQUIRE(entry == sizeof(std::uint64_t));
    const BufferHandle readback = created(device, {.size = std::uint64_t{entry} * count,
                                                   .memory = BufferMemory::Readback,
                                                   .usage = {.copyDst = true}});
    run(device, [&](CommandListHandle list) {
        CHECK(device.resetAccelerationStructureProperties(list, *pool, 0, count).has_value());
        before(list);
        const auto written = device.writeAccelerationStructureProperties(list, *pool, 0, structures);
        INFO(failureOf(written));
        CHECK(written.has_value());
        CHECK(device.barrier(list).has_value());
        CHECK(device.resolveAccelerationStructureProperties(list, *pool, 0, count, readback, 0).has_value());
    });
    std::vector<std::uint64_t> values(count);
    const auto read = device.mapBuffer(readback);
    REQUIRE(read.has_value());
    std::memcpy(values.data(), read->data(), values.size() * sizeof(std::uint64_t));
    CHECK(device.destroyBuffer(readback).has_value());
    CHECK(device.destroyAccelerationStructurePropertyPool(*pool).has_value());
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
    Micromap(Device& device, bool opaque, const AccelerationStructureFlags& flags = {})
        : device(&device),
          flags(flags),
          data(input(device, std::array<std::uint8_t, 1>{opaque ? kOpaqueState : std::uint8_t{0}})),
          records(input(device, std::array<OpacityMicromapRecord, 1>{{{.dataOffset = 0,
                                                                       .subdivisionLevel = 0,
                                                                       .format = static_cast<std::uint16_t>(
                                                                           OpacityMicromapFormat::TwoState)}}})),
          built(allocated(device, inputs())) {}
    Micromap(const Micromap&) = delete;
    Micromap& operator=(const Micromap&) = delete;
    Micromap(Micromap&&) = delete;
    Micromap& operator=(Micromap&&) = delete;
    ~Micromap() {
        CHECK(device->destroyAccelerationStructure(built.structure).has_value());
        for (const BufferHandle buffer : {data, records, built.scratch}) {
            CHECK(device->destroyBuffer(buffer).has_value());
        }
    }

    [[nodiscard]] AccelerationStructureInputs inputs() const {
        return {.type = AccelerationStructureType::OpacityMicromap,
                .flags = flags,
                .micromapUsages = std::span{&kOneMicromap, 1},
                .micromapData = data,
                .micromapRecords = records,
                .micromapRecordStride = sizeof(OpacityMicromapRecord)};
    }

    // Records its build and a barrier after it.
    void build(CommandListHandle list) const {
        const auto recorded = device->buildAccelerationStructure(list, {.dst = built.structure,
                                                                        .inputs = inputs(),
                                                                        .scratch = built.scratch});
        INFO(failureOf(recorded));
        CHECK(recorded.has_value());
        CHECK(device->barrier(list).has_value());
    }

    Device* device;
    AccelerationStructureFlags flags;
    BufferHandle data;
    BufferHandle records;
    Built built;
};

} // namespace

TEST_CASE("Device builds triangles, boxes and instances, and rays traced inline hit them, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        if (!made.caps.features.accelerationStructure || !made.caps.features.rayQuery) {
            SKIP("the adapter has no acceleration structures or ray queries");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        REQUIRE(made.caps.bindless.maxAccelerationStructures != 0);
        const RayQuery query{device};
        const auto built = [](Scene& scene) {
            return [&scene](CommandListHandle list) {
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
            const AccelerationStructureHandle copy = created(device, {.type = AccelerationStructureType::Top,
                                                                      .size = sizes->size});
            const auto copied = [&](CommandListHandle list) {
                scene.build(list);
                CHECK(device.copyAccelerationStructure(list, copy, scene.top.structure).has_value());
                CHECK(device.barrier(list).has_value());
            };
            CHECK(traced(device, copy, copied, query) == kPlaced);
            CHECK(device.destroyAccelerationStructure(copy).has_value());
        }
        SECTION("an update moves an instance") {
            Scene scene{device, {.allowUpdate = true}};
            scene.place(made.caps, 2 * kSpacing);
            CHECK(traced(device, scene.top.structure, built(scene), query) == kPlaced);
            // The second triangle moves from ray 2 to ray 1.
            scene.place(made.caps, kSpacing);
            constexpr std::array<std::uint32_t, kRays> kMoved{kFirst, kSecond, kMiss, kMiss, kBox, kMiss, kMiss, kMiss};
            const auto updated = [&](CommandListHandle list) {
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
        Device& device = *made.device;
        if (!made.caps.features.rayTracingPipeline) {
            SKIP("the adapter has no ray tracing pipelines");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const RayTracing tracing{device, made.caps};
        Scene scene{device};
        scene.place(made.caps, 2 * kSpacing);
        const auto built = [&](CommandListHandle list) {
            scene.build(list);
        };
        const auto bind = [&](CommandListHandle list, std::span<const std::byte> push) {
            CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
            CHECK(device.pushConstants(list, push).has_value());
        };

        SECTION("a trace") {
            const auto trace = [&](CommandListHandle list, std::span<const std::byte> push) {
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
                const auto size = device.rayTracingStackSize(tracing.pipeline, group, shader);
                INFO(failureOf(size));
                REQUIRE(size.has_value());
                stack += *size;
            }
            const RayTracing sized{device, made.caps, {.stackSize = std::max<std::uint64_t>(stack, 1)}};
            const auto trace = [&](CommandListHandle list, std::span<const std::byte> push) {
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
            const RayTracing linked{device, made.caps, {.linked = true}};
            const auto trace = [&](CommandListHandle list, std::span<const std::byte> push) {
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
                CHECK(device.shaderIdentifiers(linked.library, 0, RayTracing::kGroups - 1, fromLibrary).has_value());
                CHECK(device.shaderIdentifiers(linked.pipeline, 1, RayTracing::kGroups - 1, fromPipeline).has_value());
                CHECK(fromLibrary == fromPipeline);
            }
        }
        SECTION("an indirect trace, of regions by address, where the device has it") {
            if (!made.caps.features.traceRaysIndirect) {
                SKIP("the adapter has no indirect traces");
            }
            const auto table = device.bufferAddress(tracing.table);
            INFO(failureOf(table));
            REQUIRE(table.has_value());
            const TraceRaysDesc direct = tracing.desc();
            const auto region = [&](const ShaderTableRegion& from) {
                return ShaderTableAddressRegion{.address = *table + from.offset,
                                                .size = from.size,
                                                .stride = from.stride};
            };
            const std::array<TraceRaysIndirectArgs, 1> args{{{.rayGenerationAddress = *table,
                                                              .rayGenerationSize = direct.rayGeneration.size,
                                                              .miss = region(direct.miss),
                                                              .hit = region(direct.hit),
                                                              .callable = {},
                                                              .width = kRays,
                                                              .height = 1,
                                                              .depth = 1}}};
            const BufferHandle argsBuffer =
                created(device, {.size = sizeof(args), .memory = BufferMemory::Upload, .usage = {.indirect = true}});
            const auto mapped = device.mapBuffer(argsBuffer);
            REQUIRE(mapped.has_value());
            std::memcpy(mapped->data(), args.data(), sizeof(args));
            const auto trace = [&](CommandListHandle list, std::span<const std::byte> push) {
                bind(list, push);
                const auto traced = device.traceRaysIndirect(list, argsBuffer, 0);
                INFO(failureOf(traced));
                CHECK(traced.has_value());
            };
            CHECK(traced(device, scene.top.structure, built, trace) == kPlaced);
            CHECK(device.destroyBuffer(argsBuffer).has_value());
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
        std::vector<std::byte> captured;
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
            const RayTracing original{*capturing.device, caps, {.captureReplay = true}};
            const auto written = capturing.device->captureReplayIdentifiers(original.pipeline, 0, RayTracing::kGroups,
                                                                            captured);
            INFO(failureOf(written));
            REQUIRE(written.has_value());
            CHECK(capturing.device->shaderIdentifiers(original.pipeline, 0, RayTracing::kGroups, first).has_value());
        }
        // A new device, whose first ray tracing pipeline replays, as a later
        // run's would: without Features::rayTracingCaptureReplayMixed it
        // makes none without replayed identifiers before one with them.
        const RayDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const RayTracing replayed{device, made.caps, {.captureReplay = true, .replayIdentifiers = captured}};
        std::vector<std::byte> again(first.size());
        CHECK(device.shaderIdentifiers(replayed.pipeline, 0, RayTracing::kGroups, again).has_value());
        CHECK(again == first);
        Scene scene{device};
        scene.place(made.caps, 2 * kSpacing);
        const auto built = [&](CommandListHandle list) {
            scene.build(list);
        };
        const auto trace = [&](CommandListHandle list, std::span<const std::byte> push) {
            CHECK(device.bindPipeline(list, replayed.pipeline).has_value());
            CHECK(device.pushConstants(list, push).has_value());
            CHECK(device.traceRays(list, replayed.desc()).has_value());
        };
        CHECK(traced(device, scene.top.structure, built, trace) == kPlaced);
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
        const RayDevice made{recordingInto(backend, messages), kTableSlots, RhiConfig{}};
        Device& device = *made.device;
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
        run(device, [&](CommandListHandle list) { scene.build(list); });
        const auto topSizes = device.accelerationStructureSizes(scene.topInputs());
        REQUIRE(topSizes.has_value());
        const AccelerationStructureHandle compact = created(device, {.type = AccelerationStructureType::Top,
                                                                     .size = topSizes->size});
        const auto properties = device.createAccelerationStructurePropertyPool(
            {.property = AccelerationStructureProperty::CompactedSize, .count = 1});
        REQUIRE(properties.has_value());
        const BufferHandle sizes = created(device, {.size = made.caps.rayTracing.propertyEntryBytes,
                                                    .memory = BufferMemory::Readback,
                                                    .usage = {.copyDst = true}});
        const std::array compacted{scene.top.structure};
        const BufferHandle results = created(device,
                                             {.size = sizeof(std::uint32_t) * kRays, .usage = {.storage = true}});
        const auto slot = device.accelerationStructureSlot(scene.top.structure);
        const auto resultSlot = device.bufferSlot(results);
        REQUIRE(slot.has_value());
        REQUIRE(resultSlot.has_value());
        const std::array<std::uint32_t, 2> push{*slot, *resultSlot};
        const auto pool = device.createCommandPool({.queue = 0, .listCount = 1, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        const auto mapped = device.mapBuffer(scene.instances);
        REQUIRE(mapped.has_value());
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
                check("resetCommandPool", device.resetCommandPool(*pool));
                const AccelerationStructureInstance placed{.transform = movedBy(static_cast<float>(frame)),
                                                           .mask = kAllRays,
                                                           .bottomStructure = scene.triangles.structure};
                check("writeInstances", device.writeInstances(std::span{&placed, 1}, *mapped));
                const auto list = device.beginCommands(*pool);
                if (!check("beginCommands", list)) {
                    break;
                }
                check("resetAccelerationStructureProperties",
                      device.resetAccelerationStructureProperties(*list, *properties, 0, 1));
                check("buildAccelerationStructure",
                      device.buildAccelerationStructure(
                          *list, {.dst = scene.top.structure, .inputs = topInputs, .scratch = scene.top.scratch}));
                check("barrier", device.barrier(*list));
                check("writeAccelerationStructureProperties",
                      device.writeAccelerationStructureProperties(*list, *properties, 0, compacted));
                check("compactAccelerationStructure",
                      device.compactAccelerationStructure(*list, compact, scene.top.structure));
                check("barrier", device.barrier(*list));
                check("resolveAccelerationStructureProperties",
                      device.resolveAccelerationStructureProperties(*list, *properties, 0, 1, sizes, 0));
                check("bindPipeline", device.bindPipeline(*list, tracing.pipeline));
                check("pushConstants", device.pushConstants(*list, std::as_bytes(std::span{push})));
                check("traceRays", device.traceRays(*list, tracing.desc()));
                const auto submitted = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
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
        CHECK(device.destroyCommandPool(*pool).has_value());
        CHECK(device.destroyBuffer(results).has_value());
        CHECK(device.destroyBuffer(sizes).has_value());
        CHECK(device.destroyAccelerationStructurePropertyPool(*properties).has_value());
        CHECK(device.destroyAccelerationStructure(compact).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses ray tracing pipeline work it can't pass on", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const auto code = [](const auto& result) {
            return result.has_value() ? ErrorCode{} : result.error().code;
        };
        CHECK(device.destroyShader(ShaderHandle{}).has_value());
        const BufferHandle readback = created(device, {.size = sizeof(std::uint32_t),
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.copyDst = true}});
        if (!made.caps.features.rayTracingPipeline) {
            CHECK(code(device.createRayTracingPipeline({})) == ErrorCode::RhiUnsupported);
            CHECK(code(device.traceRays(*list, {})) == ErrorCode::RhiUnsupported);
        }
        if (!made.caps.features.traceRaysIndirect) {
            CHECK(code(device.traceRaysIndirect(*list, {}, 0)) == ErrorCode::RhiUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && !made.caps.features.rayTracingPipelineLibrary) {
            CHECK(code(device.createRayTracingPipeline({.library = true})) == ErrorCode::RhiUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && !made.caps.features.rayTracingMotion) {
            CHECK(code(device.createRayTracingPipeline({.allowMotion = true})) == ErrorCode::RhiUnsupported);
        }
        if (made.caps.features.rayTracingPipeline && !made.caps.features.opacityMicromap) {
            CHECK(code(device.createRayTracingPipeline({.allowOpacityMicromaps = true})) == ErrorCode::RhiUnsupported);
        }
        if (!made.caps.features.rayTracingCaptureReplay) {
            std::vector<std::byte> out(1);
            CHECK(code(device.captureReplayIdentifiers({}, 0, 1, out)) == ErrorCode::RhiUnsupported);
            if (made.caps.features.rayTracingPipeline) {
                CHECK(code(device.createRayTracingPipeline({.captureReplay = true})) == ErrorCode::RhiUnsupported);
            }
        }
        // An address only of a buffer with a usage that has one.
        CHECK(code(device.bufferAddress(readback)) ==
              (made.caps.features.bufferAddress ? ErrorCode::RhiInvalidCall : ErrorCode::RhiUnsupported));
        if (made.caps.features.rayTracingPipeline && takesSpirv(backend)) {
            const RayTracing tracing{device, made.caps};
            // Room for every identifier, of a ray tracing pipeline.
            std::vector<std::byte> out(made.caps.rayTracing.identifierSize);
            CHECK(device.shaderIdentifiers(tracing.pipeline, 0, 1, out).has_value());
            CHECK(code(device.shaderIdentifiers(tracing.pipeline, 0, 2, out)) == ErrorCode::RhiInvalidCall);
            const RayQuery query{device};
            CHECK(code(device.shaderIdentifiers(query.pipeline, 0, 1, out)) == ErrorCode::RhiInvalidCall);
            CHECK(code(device.rayTracingStackSize(query.pipeline, 0, RayTracingGroupShader::General)) ==
                  ErrorCode::RhiInvalidCall);
            if (made.caps.features.rayTracingCaptureReplay) {
                // Replay identifiers BFR would read past.
                const std::array<RayTracingGroup, 2> groups{};
                const std::vector<std::byte> tooFew(made.caps.rayTracing.captureReplayIdentifierSize);
                CHECK(code(device.createRayTracingPipeline(
                          {.groups = groups, .captureReplay = true, .replayIdentifiers = tooFew})) ==
                      ErrorCode::RhiInvalidCall);
                CHECK(code(device.captureReplayIdentifiers(query.pipeline, 0, 1, out)) == ErrorCode::RhiInvalidCall);
            }
            // An enum value BFR doesn't have.
            const RayTracingShader unknown{.stage = static_cast<RayTracingStage>(0xFF)};
            CHECK(code(device.createRayTracingPipeline({.shaders = std::span{&unknown, 1}})) ==
                  ErrorCode::RhiInvalidCall);
        }
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device compacts, serializes and deserializes structures, and writes their properties, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.caps;
        if (!caps.features.accelerationStructure || !caps.features.rayQuery) {
            SKIP("the adapter has no acceleration structures or ray queries");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const RayQuery query{device};
        const auto rebuilt = [](Scene& scene) {
            return [&scene](CommandListHandle list) {
                scene.buildTop(list, false);
            };
        };

        SECTION("a compact copy traces as its structure does") {
            Scene scene{device, {}, {.allowCompaction = true}};
            scene.place(caps, 2 * kSpacing);
            const std::array structures{scene.triangles.structure};
            const auto compacted = propertyOf(device, caps, AccelerationStructureProperty::CompactedSize, structures,
                                              [&](CommandListHandle list) { scene.build(list); });
            REQUIRE(compacted.front() != 0);
            const AccelerationStructureHandle compact = created(device,
                                                                AccelerationStructureDesc{.size = compacted.front()});
            run(device, [&](CommandListHandle list) {
                const auto copied = device.compactAccelerationStructure(list, compact, scene.triangles.structure);
                INFO(failureOf(copied));
                CHECK(copied.has_value());
            });
            scene.place(caps, 2 * kSpacing, compact);
            CHECK(traced(device, scene.top.structure, rebuilt(scene), query) == kPlaced);
            CHECK(device.destroyAccelerationStructure(compact).has_value());
        }
        SECTION("a deserialized structure traces as its structure does, where the device serializes them") {
            if (!caps.features.accelerationStructureSerialization) {
                SKIP("the adapter serializes no acceleration structures");
            }
            Scene scene{device};
            scene.place(caps, 2 * kSpacing);
            const std::array structures{scene.triangles.structure};
            const auto serialized = propertyOf(device, caps, AccelerationStructureProperty::SerializedSize, structures,
                                               [&](CommandListHandle list) { scene.build(list); });
            REQUIRE(serialized.front() >= kSerializedIdentifierSize);
            const BufferHandle data = created(device, {.size = serialized.front(),
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.accelerationStructureData = true}});
            run(device, [&](CommandListHandle list) {
                const auto written = device.serializeAccelerationStructure(list, scene.triangles.structure, data, 0);
                INFO(failureOf(written));
                CHECK(written.has_value());
            });
            const auto bytes = device.mapBuffer(data);
            REQUIRE(bytes.has_value());
            const auto compatible = device.serializedCompatible(AccelerationStructureType::Bottom,
                                                                std::span{*bytes}.first<kSerializedIdentifierSize>());
            INFO(failureOf(compatible));
            REQUIRE(compatible.has_value());
            CHECK(*compatible);
            // As large as the structure it came from, which is at least the
            // size its data names.
            const auto sizes = device.accelerationStructureSizes(scene.triangleInputs());
            REQUIRE(sizes.has_value());
            const AccelerationStructureHandle copy = created(device, AccelerationStructureDesc{.size = sizes->size});
            run(device, [&](CommandListHandle list) {
                const auto read = device.deserializeAccelerationStructure(list, data, 0, copy);
                INFO(failureOf(read));
                CHECK(read.has_value());
            });
            scene.place(caps, 2 * kSpacing, copy);
            CHECK(traced(device, scene.top.structure, rebuilt(scene), query) == kPlaced);
            CHECK(device.destroyAccelerationStructure(copy).has_value());
            CHECK(device.destroyBuffer(data).has_value());
        }
        SECTION("a structure's current size and a top structure's bottom count, where the device writes them") {
            Scene scene{device};
            scene.place(caps, 2 * kSpacing);
            const auto build = [&](CommandListHandle list) {
                scene.build(list);
            };
            if (caps.features.accelerationStructureCurrentSize) {
                const std::array structures{scene.triangles.structure};
                const auto current = propertyOf(device, caps, AccelerationStructureProperty::CurrentSize, structures,
                                                build);
                const auto sizes = device.accelerationStructureSizes(scene.triangleInputs());
                REQUIRE(sizes.has_value());
                CHECK(current.front() != 0);
                CHECK(current.front() <= sizes->size);
            }
            if (caps.features.serializedBottomCount) {
                const std::array structures{scene.top.structure};
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
        Device& device = *made.device;
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
        CHECK(traced(
                  device, scene.top.structure, [&](CommandListHandle list) { scene.build(list); }, positions) ==
              expected);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device builds opacity micromaps, and rays take triangles' opacity from them, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
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
            const auto built = [&](CommandListHandle list) {
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
                const BufferHandle indices = input(device, std::array<std::int32_t, 1>{std::to_underlying(index)});
                const auto hits = tracedWith(
                    {.indices = indices, .indexType = IndexType::Uint32, .indexStride = sizeof(std::int32_t)}, nullptr);
                CHECK(device.destroyBuffer(indices).has_value());
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
            const auto trace = [&](CommandListHandle list, std::span<const std::byte> push) {
                CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
                CHECK(device.pushConstants(list, push).has_value());
                CHECK(device.traceRays(list, tracing.desc()).has_value());
            };
            const Micromap clear{device, false};
            Scene scene{device, {}, {}, {.micromap = clear.built.structure, .usageCount = 1}, usages};
            scene.place(made.caps, 2 * kSpacing);
            const auto built = [&](CommandListHandle list) {
                clear.build(list);
                scene.build(list);
            };
            CHECK(traced(device, scene.top.structure, built, trace) == kBoxOnly);
        }
        SECTION("a micromap is not taken where another structure is, nor another where a micromap is") {
            const auto code = [](const auto& result) {
                return result.has_value() ? ErrorCode{} : result.error().code;
            };
            const Micromap micromap{device, true};
            Scene scene{device};
            const auto pool = device.createCommandPool({.queue = 0, .listCount = 1});
            REQUIRE(pool.has_value());
            const auto list = device.beginCommands(*pool);
            REQUIRE(list.has_value());
            CHECK(code(device.copyAccelerationStructure(*list, scene.triangles.structure, micromap.built.structure)) ==
                  ErrorCode::RhiInvalidCall);
            // Triangles take their opacity only from a micromap.
            TriangleGeometry named = scene.triangleGeometry;
            named.opacityMicromap = {.micromap = scene.box.structure, .usageCount = 1};
            CHECK(code(device.accelerationStructureSizes({.type = AccelerationStructureType::Bottom,
                                                          .triangles = std::span{&named, 1},
                                                          .micromapUsages = usages})) == ErrorCode::RhiInvalidCall);
            // Usages past the inputs'.
            named.opacityMicromap = {.micromap = micromap.built.structure, .firstUsage = 1, .usageCount = 1};
            CHECK(code(device.accelerationStructureSizes({.type = AccelerationStructureType::Bottom,
                                                          .triangles = std::span{&named, 1},
                                                          .micromapUsages = usages})) == ErrorCode::RhiInvalidCall);
            // Instances are of bottom structures.
            std::vector<std::byte> out(made.caps.rayTracing.instanceSize);
            const AccelerationStructureInstance instance{.bottomStructure = micromap.built.structure};
            CHECK(code(device.writeInstances(std::span{&instance, 1}, out)) == ErrorCode::RhiInvalidCall);
            CHECK(device.destroyCommandList(*list).has_value());
            CHECK(device.destroyCommandPool(*pool).has_value());
        }
        SECTION("a copied micromap says what the built one does") {
            const Micromap original{device, true};
            const auto whole = device.accelerationStructureSizes(original.inputs());
            REQUIRE(whole.has_value());
            const AccelerationStructureHandle copy = created(device,
                                                             {.type = AccelerationStructureType::OpacityMicromap,
                                                              .size = whole->size});
            run(device, [&](CommandListHandle list) {
                original.build(list);
                const auto copied = device.copyAccelerationStructure(list, copy, original.built.structure);
                INFO(failureOf(copied));
                CHECK(copied.has_value());
            });
            CHECK(tracedWith({.micromap = copy, .usageCount = 1}, nullptr) == kPlaced);
            CHECK(device.destroyAccelerationStructure(copy).has_value());
        }
        SECTION("a compacted micromap, and a deserialized one, say what the built one does") {
            const Micromap original{device, true, {.allowCompaction = true}};
            const std::array structures{original.built.structure};
            // A driver may make no pool of micromaps' properties; it says so
            // as it fails (the RTX 3070's: VK_ERROR_UNKNOWN).
            {
                const auto pool = device.createAccelerationStructurePropertyPool(
                    {.property = AccelerationStructureProperty::CompactedSize, .count = 1, .opacityMicromaps = true});
                // Only the driver's own refusal of the pool skips, after the
                // sections before have said whether validation spoke.
                if (!pool.has_value() && pool.error().code == ErrorCode::RhiDeviceFailed &&
                    failureOf(pool).find("vkCreateQueryPool") != std::string::npos) {
                    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
                    SKIP("the driver makes no pool of micromaps' properties: " << failureOf(pool));
                }
                REQUIRE(pool.has_value());
                CHECK(device.destroyAccelerationStructurePropertyPool(*pool).has_value());
            }
            const auto sizeOf = [&](AccelerationStructureProperty property) {
                const auto pool = device.createAccelerationStructurePropertyPool(
                    {.property = property, .count = 1, .opacityMicromaps = true});
                INFO(failureOf(pool));
                REQUIRE(pool.has_value());
                const BufferHandle readback = created(device, {.size = sizeof(std::uint64_t),
                                                               .memory = BufferMemory::Readback,
                                                               .usage = {.copyDst = true}});
                run(device, [&](CommandListHandle list) {
                    CHECK(device.resetAccelerationStructureProperties(list, *pool, 0, 1).has_value());
                    original.build(list);
                    const auto written = device.writeAccelerationStructureProperties(list, *pool, 0, structures);
                    INFO(failureOf(written));
                    CHECK(written.has_value());
                    CHECK(device.barrier(list).has_value());
                    CHECK(device.resolveAccelerationStructureProperties(list, *pool, 0, 1, readback, 0).has_value());
                });
                std::uint64_t value = 0;
                const auto read = device.mapBuffer(readback);
                REQUIRE(read.has_value());
                std::memcpy(&value, read->data(), sizeof(value));
                CHECK(device.destroyBuffer(readback).has_value());
                CHECK(device.destroyAccelerationStructurePropertyPool(*pool).has_value());
                return value;
            };
            const std::uint64_t compacted = sizeOf(AccelerationStructureProperty::CompactedSize);
            REQUIRE(compacted != 0);
            const AccelerationStructureHandle small = created(device,
                                                              {.type = AccelerationStructureType::OpacityMicromap,
                                                               .size = compacted});
            run(device, [&](CommandListHandle list) {
                const auto copied = device.compactAccelerationStructure(list, small, original.built.structure);
                INFO(failureOf(copied));
                CHECK(copied.has_value());
            });
            CHECK(tracedWith({.micromap = small, .usageCount = 1}, nullptr) == kPlaced);

            const std::uint64_t serialized = sizeOf(AccelerationStructureProperty::SerializedSize);
            REQUIRE(serialized > kSerializedIdentifierSize);
            const BufferHandle bytes = created(device, {.size = serialized,
                                                        .memory = BufferMemory::Readback,
                                                        .usage = {.accelerationStructureData = true}});
            run(device, [&](CommandListHandle list) {
                CHECK(device.serializeAccelerationStructure(list, original.built.structure, bytes, 0).has_value());
            });
            const auto mapped = device.mapBuffer(bytes);
            REQUIRE(mapped.has_value());
            const auto compatible = device.serializedCompatible(AccelerationStructureType::OpacityMicromap,
                                                                mapped->first<kSerializedIdentifierSize>());
            INFO(failureOf(compatible));
            REQUIRE(compatible.has_value());
            CHECK(*compatible);
            // Deserialized at the size the original was made with.
            const auto whole = device.accelerationStructureSizes(original.inputs());
            REQUIRE(whole.has_value());
            const AccelerationStructureHandle restored = created(device,
                                                                 {.type = AccelerationStructureType::OpacityMicromap,
                                                                  .size = whole->size});
            run(device, [&](CommandListHandle list) {
                CHECK(device.deserializeAccelerationStructure(list, bytes, 0, restored).has_value());
            });
            CHECK(tracedWith({.micromap = restored, .usageCount = 1}, nullptr) == kPlaced);
            for (const AccelerationStructureHandle structure : {small, restored}) {
                CHECK(device.destroyAccelerationStructure(structure).has_value());
            }
            CHECK(device.destroyBuffer(bytes).has_value());
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
        Device& device = *made.device;
        if (!made.caps.features.rayTracingMotion) {
            SKIP("the adapter has no structures with motion");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const RayTracingLimits& limits = made.caps.rayTracing;
        REQUIRE(limits.maxMotionKeyframes >= 2);
        constexpr std::uint32_t kKeyframes = 2;
        // A triangle about the origin at time 0 that has moved 10 along x by
        // time 1, and a box about the origin that stays.
        const BufferHandle vertices = input(device,
                                            std::array<float, kVertexFloats * kTriangleVertices>{-1, -1, 0, 1, -1, 0, 0,
                                                                                                 1, 0});
        const BufferHandle moved = input(device, std::array<float, kVertexFloats * kTriangleVertices>{9, -1, 0, 11, -1,
                                                                                                      0, 10, 1, 0});
        const BufferHandle boxes = input(device, std::array<float, kBoxFloats>{-1, -1, -1, 1, 1, 1});
        const TriangleGeometry triangle{.vertices = vertices,
                                        .vertexFormat = Format::Rgb32Float,
                                        .vertexStride = kVertexFloats * sizeof(float),
                                        .vertexCount = kTriangleVertices,
                                        .triangleCount = 1,
                                        .opaque = true,
                                        .motionVertices = moved};
        const BoxGeometry box{.boxes = boxes, .stride = kBoxFloats * sizeof(float), .count = 1};
        const AccelerationStructureInputs triangleInputs{.type = AccelerationStructureType::Bottom,
                                                         .triangles = std::span{&triangle, 1},
                                                         .motionKeyframes = kKeyframes};
        const AccelerationStructureInputs boxInputs{.type = AccelerationStructureType::Bottom,
                                                    .boxes = std::span{&box, 1}};
        // Instances: the moving triangle, still; a box moved by matrices
        // from x = 20 to 30; and one moved by SRT transforms from x = 35 to 25.
        const BufferHandle records = created(device, {.size = std::uint64_t{limits.motionInstanceSize} * kInstances,
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
            return Built{.structure = created(device, {.type = inputs.type,
                                                       .size = sizes->size,
                                                       .motionKeyframes = motion,
                                                       .motionInstances = motion == 0 ? 0 : kInstances}),
                         .scratch = created(device, {.size = sizes->buildScratch,
                                                     .usage = {.accelerationStructureScratch = true}})};
        };
        const Built triangles = make(triangleInputs, kKeyframes);
        const Built still = make(boxInputs, 0);
        const Built top = make(topInputs, kKeyframes);

        const auto srtAt = [](float x) {
            return SrtTransform{.scale = {1, 1, 1}, .rotation = {0, 0, 0, 1}, .translation = {x, 0, 0}};
        };
        const std::array<MotionKeyframe, 4> keyframes{{
            {.matrix = movedBy(4 * kSpacing)},
            {.matrix = movedBy(6 * kSpacing)},
            {.srt = srtAt(7 * kSpacing)},
            {.srt = srtAt(5 * kSpacing)},
        }};
        const std::array<AccelerationStructureMotionInstance, kInstances> instances{{
            {.instance = {.transform = movedBy(0),
                          .customIndex = kFirst,
                          .mask = kAllRays,
                          .bottomStructure = triangles.structure}},
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
        const auto mapped = device.mapBuffer(records);
        REQUIRE(mapped.has_value());
        const auto written = device.writeMotionInstances(instances, keyframes, *mapped);
        INFO(failureOf(written));
        REQUIRE(written.has_value());

        const RayTracing tracing{device, made.caps, {.motion = true}};
        const auto built = [&](CommandListHandle list) {
            for (const auto& [structure, inputs] :
                 {std::pair{triangles, triangleInputs}, std::pair{still, boxInputs}}) {
                const auto recorded = device.buildAccelerationStructure(list, {.dst = structure.structure,
                                                                               .inputs = inputs,
                                                                               .scratch = structure.scratch});
                INFO(failureOf(recorded));
                CHECK(recorded.has_value());
            }
            CHECK(device.barrier(list).has_value());
            const auto recorded = device.buildAccelerationStructure(list, {.dst = top.structure,
                                                                           .inputs = topInputs,
                                                                           .scratch = top.scratch});
            INFO(failureOf(recorded));
            CHECK(recorded.has_value());
            CHECK(device.barrier(list).has_value());
        };
        // The rays traced at time.
        const auto at = [&](float time) {
            const auto trace = [&](CommandListHandle list, std::span<const std::byte> push) {
                std::array<std::byte, 3 * sizeof(std::uint32_t)> timed{};
                std::memcpy(timed.data(), push.data(), push.size());
                std::memcpy(timed.data() + push.size(), &time, sizeof(time));
                CHECK(device.bindPipeline(list, tracing.pipeline).has_value());
                CHECK(device.pushConstants(list, timed).has_value());
                const auto traced = device.traceRays(list, tracing.desc());
                INFO(failureOf(traced));
                CHECK(traced.has_value());
            };
            return traced(device, top.structure, built, trace);
        };
        CHECK(at(0) == std::array<std::uint32_t, kRays>{kFirst, kMiss, kMiss, kMiss, kBox, kMiss, kMiss, kSecond});
        CHECK(at(1) == std::array<std::uint32_t, kRays>{kMiss, kMiss, kFirst, kMiss, kMiss, kSecond, kBox, kMiss});

        for (const Built& done : {triangles, still, top}) {
            CHECK(device.destroyAccelerationStructure(done.structure).has_value());
            CHECK(device.destroyBuffer(done.scratch).has_value());
        }
        for (const BufferHandle buffer : {vertices, moved, boxes, records}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses acceleration structure work it can't pass on", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const auto code = [](const auto& result) {
            return result.has_value() ? ErrorCode{} : result.error().code;
        };
        CHECK(device.destroyAccelerationStructure(AccelerationStructureHandle{}).has_value());
        if (!made.caps.features.accelerationStructureIndirectBuild) {
            CHECK(code(device.buildAccelerationStructureIndirect(*list, {}, {}, 0, 0)) == ErrorCode::RhiUnsupported);
        }
        if (made.caps.features.accelerationStructure && !made.caps.features.rayTracingMotion) {
            std::array<std::byte, 1> out{};
            CHECK(code(device.createAccelerationStructure({.size = kSmall, .motionKeyframes = 2})) ==
                  ErrorCode::RhiUnsupported);
            CHECK(code(device.writeMotionInstances({}, {}, out)) == ErrorCode::RhiUnsupported);
        }
        if (made.caps.features.accelerationStructure && !made.caps.features.opacityMicromap) {
            CHECK(code(device.createAccelerationStructure({.type = AccelerationStructureType::OpacityMicromap,
                                                           .size = kSmall})) == ErrorCode::RhiUnsupported);
            CHECK(code(device.accelerationStructureSizes({.type = AccelerationStructureType::OpacityMicromap})) ==
                  ErrorCode::RhiUnsupported);
            CHECK(code(device.createAccelerationStructurePropertyPool({.count = 1, .opacityMicromaps = true})) ==
                  ErrorCode::RhiUnsupported);
        }
        if (!made.caps.features.accelerationStructure) {
            std::array<std::byte, 1> out{};
            CHECK(code(device.accelerationStructureSizes({})) == ErrorCode::RhiUnsupported);
            CHECK(code(device.createAccelerationStructure({.size = 1})) == ErrorCode::RhiUnsupported);
            CHECK(code(device.writeInstances({}, out)) == ErrorCode::RhiUnsupported);
            CHECK(code(device.buildAccelerationStructure(*list, {})) == ErrorCode::RhiUnsupported);
            CHECK(code(device.copyAccelerationStructure(*list, {}, {})) == ErrorCode::RhiUnsupported);
        }
        else {
            const BufferHandle vertices = input(device, std::array<float, kVertexFloats * kTriangleVertices>{});
            const TriangleGeometry triangle{.vertices = vertices,
                                            .vertexFormat = Format::Rgb32Float,
                                            .vertexStride = kVertexFloats * sizeof(float),
                                            .vertexCount = kTriangleVertices,
                                            .triangleCount = 1};
            const BoxGeometry box{.boxes = vertices, .stride = kBoxFloats * sizeof(float), .count = 1};
            std::array<TriangleGeometry, kGeometries + 1> tooMany{};
            tooMany.fill(triangle);
            const AccelerationStructureHandle bottom = created(device, {.type = AccelerationStructureType::Bottom,
                                                                        .size = kSmall});
            const AccelerationStructureHandle top = created(device,
                                                            {.type = AccelerationStructureType::Top, .size = kSmall});

            // A bottom structure holds triangles or boxes, not both, and a
            // build at most maxBuildGeometries.
            CHECK(code(device.accelerationStructureSizes({.triangles = std::span{&triangle, 1},
                                                          .boxes = std::span{&box, 1}})) == ErrorCode::RhiInvalidCall);
            CHECK(code(device.accelerationStructureSizes({.triangles = tooMany})) == ErrorCode::RhiLimitReached);
            CHECK(code(device.createAccelerationStructure({.size = 0})) == ErrorCode::RhiInvalidCall);
            // Only top structures are in the table.
            CHECK(code(device.accelerationStructureSlot(bottom)) == ErrorCode::RhiInvalidCall);
            CHECK(device.accelerationStructureSlot(top).has_value() ==
                  (made.caps.bindless.maxAccelerationStructures != 0));
            // A build's inputs are of its structure's type.
            CHECK(code(device.buildAccelerationStructure(*list, {.dst = top,
                                                                 .inputs = {.triangles = std::span{&triangle, 1}}})) ==
                  ErrorCode::RhiInvalidCall);

            // An index or offset of 24 bits at most, room for every record,
            // and instances of bottom structures only.
            std::vector<std::byte> out(made.caps.rayTracing.instanceSize);
            const AccelerationStructureInstance fits{.customIndex = kInstanceIndexLimit - 1,
                                                     .hitGroupOffset = kInstanceIndexLimit - 1,
                                                     .bottomStructure = bottom};
            CHECK(device.writeInstances(std::span{&fits, 1}, out).has_value());
            AccelerationStructureInstance wide = fits;
            wide.customIndex = kInstanceIndexLimit;
            CHECK(code(device.writeInstances(std::span{&wide, 1}, out)) == ErrorCode::RhiInvalidCall);
            wide = fits;
            wide.hitGroupOffset = kInstanceIndexLimit;
            CHECK(code(device.writeInstances(std::span{&wide, 1}, out)) == ErrorCode::RhiInvalidCall);
            const std::array<AccelerationStructureInstance, 2> two{fits, fits};
            CHECK(code(device.writeInstances(two, out)) == ErrorCode::RhiInvalidCall);
            AccelerationStructureInstance ofTop = fits;
            ofTop.bottomStructure = top;
            CHECK(code(device.writeInstances(std::span{&ofTop, 1}, out)) == ErrorCode::RhiInvalidCall);

            for (const AccelerationStructureHandle structure : {bottom, top}) {
                CHECK(device.destroyAccelerationStructure(structure).has_value());
            }
            CHECK(code(device.accelerationStructureReference(top)) == ErrorCode::RhiInvalidHandle);
            // A property the device writes, and no more structures than its
            // pool holds.
            const auto pool = device.createAccelerationStructurePropertyPool(
                {.property = AccelerationStructureProperty::CompactedSize, .count = 1});
            REQUIRE(pool.has_value());
            const std::array<AccelerationStructureHandle, 2> twoStructures{bottom, bottom};
            CHECK(code(device.writeAccelerationStructureProperties(*list, *pool, 0, twoStructures)) ==
                  ErrorCode::RhiInvalidCall);
            CHECK(code(device.writeAccelerationStructureProperties(*list, *pool, 1, std::span{&bottom, 1})) ==
                  ErrorCode::RhiInvalidCall);
            CHECK(device.destroyAccelerationStructurePropertyPool(*pool).has_value());
            if (!made.caps.features.accelerationStructureCurrentSize) {
                CHECK(code(device.createAccelerationStructurePropertyPool(
                          {.property = AccelerationStructureProperty::CurrentSize, .count = 1})) ==
                      ErrorCode::RhiUnsupported);
            }
            CHECK(device.destroyBuffer(vertices).has_value());
        }
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a top structure past its table, and slots from an adapter with no table", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const RayDevice made{recordingInto(backend, messages), 1};
        Device& device = *made.device;
        if (made.caps.bindless.maxAccelerationStructures == 0) {
            const auto refused = made.instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                                                 desc.maxBindlessAccelerationStructures = 1;
                                                             }));
            REQUIRE_FALSE(refused.has_value());
            CHECK(refused.error().code == ErrorCode::RhiLimitReached);
            SKIP("the adapter has no acceleration structure table");
        }
        const AccelerationStructureHandle first = created(device,
                                                          {.type = AccelerationStructureType::Top, .size = kSmall});
        const auto second = device.createAccelerationStructure(
            {.type = AccelerationStructureType::Top, .size = kSmall});
        REQUIRE_FALSE(second.has_value());
        CHECK(second.error().code == ErrorCode::RhiLimitReached);
        // Bottom structures take no slot.
        const AccelerationStructureHandle bottom = created(device,
                                                           {.type = AccelerationStructureType::Bottom, .size = kSmall});
        CHECK(device.destroyAccelerationStructure(first).has_value());
        CHECK(device.destroyAccelerationStructure(bottom).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses acceleration structure work on a copy queue", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        const Caps& caps = instance->adapters()[0].caps;
        if (!caps.features.accelerationStructure || caps.queues.copy == 0) {
            SKIP("the adapter has no acceleration structures or no copy queue");
        }
        auto made = instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                               desc.copy = std::span{kTestQueues}.first(1);
                                               desc.maxBuildGeometries = kGeometries;
                                           }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        Device& device = **made;
        const auto pool = device.createCommandPool({.queue = 1, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const auto structure = device.createAccelerationStructure({.size = kSmall});
        REQUIRE(structure.has_value());
        const auto built = device.buildAccelerationStructure(*list, {.dst = *structure});
        const auto copied = device.copyAccelerationStructure(*list, *structure, *structure);
        // A trace, where the device has them, as builds.
        const auto traced = caps.features.rayTracingPipeline ? device.traceRays(*list, {}) : copied;
        for (const auto* result : {&built, &copied, &traced}) {
            REQUIRE_FALSE(result->has_value());
            CHECK(result->error().code == ErrorCode::RhiUnsupported);
        }
        CHECK(device.destroyAccelerationStructure(*structure).has_value());
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
