// What every backend does with pipelines, through the interface alone. Each
// case runs on every backend the build has. The shaders are SPIR-V, compiled
// from tests/library/shaders, which the Vulkan backend takes; a case
// that needs code skips on a backend with none checked in.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;
using bfr::core::ErrorCode;

namespace {

constexpr std::uint32_t kWorkgroup = 64;

// An Instance and a device on its first adapter, presenting nowhere.
struct PipelineDevice {
    explicit PipelineDevice(const InstanceDesc& desc) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

// Whether backend takes the SPIR-V checked in: Vulkan does. A case that
// needs code skips elsewhere.
bool takesSpirv(const TestBackend& backend) {
    return backend.backend == Backend::Vulkan;
}

ShaderCode spirv(std::span<const std::uint8_t> bytes) {
    return {.code = std::as_bytes(bytes), .entryPoint = "main"};
}

const ComputePipelineDesc kCompute{.compute = spirv(bfr::test::spirv::kBindlessComp),
                                   .workgroupSize = {kWorkgroup, 1, 1}};

// A pipeline that renders a color target with depth, blending the color.
GraphicsPipelineDesc graphicsDesc() {
    GraphicsPipelineDesc desc{.vertex = spirv(bfr::test::spirv::kBindlessVert),
                              .fragment = spirv(bfr::test::spirv::kBindlessFrag),
                              .topology = Topology::TriangleList,
                              .primitiveRestart = PrimitiveRestart::Off,
                              .raster = {.cull = CullMode::Back, .frontFace = FrontFace::CounterClockwise},
                              .depthStencil = {.depthTest = true,
                                               .depthWrite = true,
                                               .depthCompare = CompareOp::Less,
                                               .stencilTest = false,
                                               .front = {},
                                               .back = {}},
                              .colorTargetCount = 1,
                              .depthStencilFormat = Format::D32Float,
                              .samples = 1,
                              .sampleMask = BFR_MASK_ALL};
    desc.colorTargets[0] = {.format = Format::Rgba8Unorm,
                            .blend = {.enable = true,
                                      .srcColor = BlendFactor::SrcAlpha,
                                      .dstColor = BlendFactor::OneMinusSrcAlpha,
                                      .colorOp = BlendOp::Add,
                                      .srcAlpha = BlendFactor::One,
                                      .dstAlpha = BlendFactor::Zero,
                                      .alphaOp = BlendOp::Add,
                                      .writeMask = BFR_MASK_ALL}};
    return desc;
}

} // namespace

TEST_CASE("Device creates and destroys compute and graphics pipelines", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const PipelineDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;

        const auto compute = device.createComputePipeline(kCompute);
        INFO(failureOf(compute));
        REQUIRE(compute.has_value());
        const auto graphics = device.createGraphicsPipeline(graphicsDesc());
        INFO(failureOf(graphics));
        REQUIRE(graphics.has_value());
        CHECK(*compute != *graphics);

        // A vertex-only pipeline writes depth alone, to no color target.
        GraphicsPipelineDesc depthOnly = graphicsDesc();
        depthOnly.fragment = {};
        depthOnly.colorTargetCount = 0;
        const auto depth = device.createGraphicsPipeline(depthOnly);
        INFO(failureOf(depth));
        REQUIRE(depth.has_value());

        CHECK(device.destroyPipeline(*compute).has_value());
        CHECK(device.destroyPipeline(*graphics).has_value());
        CHECK(device.destroyPipeline(*depth).has_value());
        CHECK(device.destroyPipeline(*compute).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyPipeline({}).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device keeps pipelines in a cache, and a cache made from its bytes makes them again", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const PipelineDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const bool cached = made.instance->adapters()[0].caps.features.pipelineCache;

        const auto first = device.createPipelineCache({});
        // Features::pipelineCache says whether caches are made.
        CHECK(first.has_value() == cached);
        if (!first) {
            CHECK(first.error().code == ErrorCode::RhiUnsupported);
            SKIP(failureOf(first));
        }
        ComputePipelineDesc compute = kCompute;
        compute.cache = *first;
        GraphicsPipelineDesc graphics = graphicsDesc();
        graphics.cache = *first;
        const auto computeMade = device.createComputePipeline(compute);
        INFO(failureOf(computeMade));
        REQUIRE(computeMade.has_value());
        const auto graphicsMade = device.createGraphicsPipeline(graphics);
        INFO(failureOf(graphicsMade));
        REQUIRE(graphicsMade.has_value());

        const auto size = device.pipelineCacheSize(*first);
        INFO(failureOf(size));
        REQUIRE(size.has_value());
        CHECK(*size > 0);
        std::vector<std::byte> bytes(*size);
        const auto written = device.pipelineCacheData(*first, bytes);
        INFO(failureOf(written));
        REQUIRE(written.has_value());
        CHECK(written->written == *size);
        CHECK(written->complete);
        bytes.resize(written->written);

        // Pipelines made with a cache stay once it goes.
        CHECK(device.destroyPipelineCache(*first).has_value());
        CHECK(device.pipelineCacheSize(*first).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.createComputePipeline(compute).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyPipelineCache(*first).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyPipelineCache({}).has_value());

        const auto second = device.createPipelineCache({.data = bytes});
        INFO(failureOf(second));
        REQUIRE(second.has_value());
        compute.cache = *second;
        graphics.cache = *second;
        const auto computeAgain = device.createComputePipeline(compute);
        INFO(failureOf(computeAgain));
        REQUIRE(computeAgain.has_value());
        const auto graphicsAgain = device.createGraphicsPipeline(graphics);
        INFO(failureOf(graphicsAgain));
        REQUIRE(graphicsAgain.has_value());

        for (const PipelineHandle pipeline : {*computeMade, *graphicsMade, *computeAgain, *graphicsAgain}) {
            CHECK(device.destroyPipeline(pipeline).has_value());
        }
        CHECK(device.destroyPipelineCache(*second).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses more color targets than a pipeline holds with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const PipelineDevice made{recordingInto(backend, messages)};
        GraphicsPipelineDesc tooManyTargets = graphicsDesc();
        tooManyTargets.colorTargetCount = kMaxColorTargets + 1;
        CHECK(made.device->createGraphicsPipeline(tooManyTargets).error().code == ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the pipelines left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const PipelineDevice made{recordingInto(backend, messages)};
        const auto pipeline = made.device->createComputePipeline(kCompute);
        INFO(failureOf(pipeline));
        REQUIRE(pipeline.has_value());
    }

    const std::size_t warnings = bfr::core::kLoggingEnabled ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 pipelines were left when the device went"));
    }
}
