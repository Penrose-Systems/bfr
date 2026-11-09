// What every backend does with acceleration structures, through the interface
// alone: builds of triangles, boxes and instances, rays traced inline against
// them from a compute shader, copies, updates, and the usage checks. Each case
// runs on every backend the build has; the shaders are the SPIR-V in
// tests/library/shaders, so a case that runs code skips on a backend with none
// checked in, and one that needs acceleration structures skips where the
// adapter has none.

#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/pipeline.h"
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
    explicit RayDevice(const InstanceDesc& desc, std::uint32_t tableSlots = kTableSlots) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        caps = instance->adapters()[0].caps;
        const bool table = caps.bindless.maxAccelerationStructures != 0;
        auto made = instance->createDevice(0, kNoWindow, testDeviceWith([&](DeviceDesc& device) {
                                               device.maxBindlessAccelerationStructures = table ? tableSlots : 0;
                                               device.maxBuildGeometries = kGeometries;
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
    explicit Scene(Device& device, bool allowUpdate = false)
        : device(&device),
          vertices(input(device, std::array<float, kVertexFloats * kTriangleVertices>{-1, -1, 0, 1, -1, 0, 0, 1, 0})),
          boxes(input(device, std::array<float, kBoxFloats>{-1, -1, -1, 1, 1, 1})) {
        triangleGeometry = {.vertices = vertices,
                            .vertexFormat = Format::Rgb32Float,
                            .vertexStride = kVertexFloats * sizeof(float),
                            .vertexCount = kTriangleVertices,
                            .triangleCount = 1,
                            .opaque = true};
        boxGeometry = {.boxes = boxes, .stride = kBoxFloats * sizeof(float), .count = 1};
        triangles = allocated(device, triangleInputs());
        box = allocated(device, boxInputs());
        flags.allowUpdate = allowUpdate;
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
        return {.type = AccelerationStructureType::Bottom, .triangles = std::span{&triangleGeometry, 1}};
    }
    [[nodiscard]] AccelerationStructureInputs boxInputs() const {
        return {.type = AccelerationStructureType::Bottom, .boxes = std::span{&boxGeometry, 1}};
    }
    [[nodiscard]] AccelerationStructureInputs topInputs() const {
        return {.type = AccelerationStructureType::Top,
                .flags = flags,
                .instances = instances,
                .instanceCount = kInstances};
    }

    // Writes the instances, the second triangle at secondX, into their
    // buffer, made the first time, and makes the top structure the first
    // time.
    void place(const Caps& caps, float secondX) {
        const std::uint32_t size = caps.rayTracing.instanceSize;
        REQUIRE(size != 0);
        if (instances.isNull()) {
            instances = created(*device, {.size = std::uint64_t{size} * kInstances,
                                          .memory = BufferMemory::Upload,
                                          .usage = {.accelerationStructureInput = true}});
        }
        const std::array<AccelerationStructureInstance, kInstances> placed{{
            {.transform = movedBy(0), .customIndex = kFirst, .mask = kAllRays, .bottomStructure = triangles.structure},
            {.transform = movedBy(secondX),
             .customIndex = kSecond,
             .mask = kAllRays,
             .bottomStructure = triangles.structure},
            {.transform = movedBy(4 * kSpacing),
             .customIndex = kBox,
             .mask = kAllRays,
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
    BufferHandle vertices;
    BufferHandle boxes;
    BufferHandle instances;
    TriangleGeometry triangleGeometry;
    BoxGeometry boxGeometry;
    AccelerationStructureFlags flags;
    Built triangles;
    Built box;
    Built top;
};

// Runs ray_query.comp against structure, after record, and returns what each
// ray hit.
template <class Record>
std::array<std::uint32_t, kRays> traced(Device& device, AccelerationStructureHandle structure, Record record) {
    constexpr std::uint64_t kBytes = sizeof(std::uint32_t) * kRays;
    const BufferHandle results = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
    const BufferHandle readback =
        created(device, {.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
    const auto pipeline = device.createComputePipeline(
        {.compute = {.code = std::as_bytes(std::span{bfr::test::spirv::kRayQueryComp}), .entryPoint = "main"},
         .workgroupSize = {kRays, 1, 1}});
    INFO(failureOf(pipeline));
    REQUIRE(pipeline.has_value());
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
    CHECK(device.bindPipeline(*list, *pipeline).has_value());
    const std::array<std::uint32_t, 2> push{*slot, *resultSlot};
    CHECK(device.pushConstants(*list, std::as_bytes(std::span{push})).has_value());
    CHECK(device.dispatch(*list, 1, 1, 1).has_value());
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
    CHECK(device.destroyPipeline(*pipeline).has_value());
    CHECK(device.destroyBuffer(results).has_value());
    CHECK(device.destroyBuffer(readback).has_value());
    CHECK(device.destroyCommandPool(*pool).has_value());
    return hits;
}

constexpr std::array<std::uint32_t, kRays> kPlaced{kFirst, kMiss, kSecond, kMiss, kBox, kMiss, kMiss, kMiss};

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

        SECTION("a build") {
            Scene scene{device};
            scene.place(made.caps, 2 * kSpacing);
            CHECK(traced(device, scene.top.structure, [&](CommandListHandle list) { scene.build(list); }) == kPlaced);
        }
        SECTION("a copy of a top structure traces as it does") {
            Scene scene{device};
            scene.place(made.caps, 2 * kSpacing);
            const auto sizes = device.accelerationStructureSizes(scene.topInputs());
            REQUIRE(sizes.has_value());
            const AccelerationStructureHandle copy = created(device, {.type = AccelerationStructureType::Top,
                                                                      .size = sizes->size});
            CHECK(traced(device, copy, [&](CommandListHandle list) {
                      scene.build(list);
                      CHECK(device.copyAccelerationStructure(list, copy, scene.top.structure).has_value());
                      CHECK(device.barrier(list).has_value());
                  }) == kPlaced);
            CHECK(device.destroyAccelerationStructure(copy).has_value());
        }
        SECTION("an update moves an instance") {
            Scene scene{device, true};
            scene.place(made.caps, 2 * kSpacing);
            CHECK(traced(device, scene.top.structure, [&](CommandListHandle list) { scene.build(list); }) == kPlaced);
            // The second triangle moves from ray 2 to ray 1.
            scene.place(made.caps, kSpacing);
            constexpr std::array<std::uint32_t, kRays> kMoved{kFirst, kSecond, kMiss, kMiss, kBox, kMiss, kMiss, kMiss};
            CHECK(traced(device, scene.top.structure, [&](CommandListHandle list) { scene.buildTop(list, true); }) ==
                  kMoved);
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
            const AccelerationStructureHandle bigger = created(device, {.type = AccelerationStructureType::Bottom,
                                                                        .size = 2 * kSmall});
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
            // A build's inputs are of its structure's type, and a copy is of
            // one type and size.
            CHECK(code(device.buildAccelerationStructure(*list, {.dst = top,
                                                                 .inputs = {.triangles = std::span{&triangle, 1}}})) ==
                  ErrorCode::RhiInvalidCall);
            CHECK(code(device.copyAccelerationStructure(*list, bigger, bottom)) == ErrorCode::RhiInvalidCall);
            CHECK(code(device.copyAccelerationStructure(*list, top, bottom)) == ErrorCode::RhiInvalidCall);

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

            for (const AccelerationStructureHandle structure : {bottom, bigger, top}) {
                CHECK(device.destroyAccelerationStructure(structure).has_value());
            }
            CHECK(code(device.accelerationStructureReference(top)) == ErrorCode::RhiInvalidHandle);
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
        for (const auto* result : {&built, &copied}) {
            REQUIRE_FALSE(result->has_value());
            CHECK(result->error().code == ErrorCode::RhiUnsupported);
        }
        CHECK(device.destroyAccelerationStructure(*structure).has_value());
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
