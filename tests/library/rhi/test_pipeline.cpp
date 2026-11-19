// What every backend does with pipelines, through the interface alone. Each
// case runs on every backend the build has. The shaders are SPIR-V, compiled
// from tests/library/shaders, which the Vulkan backend takes; a case
// that needs code skips on a backend with none checked in.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"
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
#include <span>
#include <string>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint32_t kWorkgroup = 64;
const ComputePipelineDesc kCompute{.compute = shaderCode(bfr::test::spirv::kBindlessComp),
                                   .workgroupSize = {kWorkgroup, 1, 1}};

// A pipeline that renders a color target with depth, blending the color.
GraphicsPipelineDesc graphicsDesc() {
    GraphicsPipelineDesc desc{.vertex = shaderCode(bfr::test::spirv::kBindlessVert),
                              .fragment = shaderCode(bfr::test::spirv::kBindlessFrag),
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;

        auto compute = device.createComputePipeline(kCompute);
        INFO(failureOf(compute));
        REQUIRE(compute.has_value());
        auto graphics = device.createGraphicsPipeline(graphicsDesc());
        INFO(failureOf(graphics));
        REQUIRE(graphics.has_value());
        CHECK(compute->get().bits != graphics->get().bits);

        // A vertex-only pipeline writes depth alone, to no color target.
        GraphicsPipelineDesc depthOnly = graphicsDesc();
        depthOnly.fragment = {};
        depthOnly.colorTargetCount = 0;
        auto depth = device.createGraphicsPipeline(depthOnly);
        INFO(failureOf(depth));
        REQUIRE(depth.has_value());

        const bfr_pipeline stale = compute->get();
        for (Pipeline* pipeline : std::array{&*compute, &*graphics, &*depth}) {
            const bfr_pipeline gone = destroyed(*pipeline);
            CHECK(bindResult(device, 0, gone) == Result::ErrorInvalidHandle);
        }
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_pipeline(device.get(), stale);
        // d1: null destroy: no message (track B)
        bfr_destroy_pipeline(device.get(), {});
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const bool cached = capsOf(made.instance, kFirstAdapter).features.pipelineCache;

        auto first = device.createPipelineCache({});
        // CapsFeatures::pipelineCache says whether caches are made.
        CHECK(first.has_value() == cached);
        if (!first) {
            CHECK(first.error().code == Result::ErrorUnsupported);
            SKIP(failureOf(first));
        }
        ComputePipelineDesc compute = kCompute;
        compute.cache = *first;
        GraphicsPipelineDesc graphics = graphicsDesc();
        graphics.cache = *first;
        auto computeMade = device.createComputePipeline(compute);
        INFO(failureOf(computeMade));
        REQUIRE(computeMade.has_value());
        auto graphicsMade = device.createGraphicsPipeline(graphics);
        INFO(failureOf(graphicsMade));
        REQUIRE(graphicsMade.has_value());

        // bfr.hpp gives a cache's bytes whole; bfr.h gives their size, and
        // whether a buffer of that size took them all.
        std::size_t size = 0;
        const bfr_result sized = bfr_get_pipeline_cache_data(device.get(), first->get(), &size, nullptr);
        INFO(failureOf(resultOf(sized)));
        REQUIRE(sized >= 0);
        CHECK(size > 0);
        std::vector<std::uint8_t> bytes(size);
        std::size_t written = bytes.size();
        const bfr_result wrote = bfr_get_pipeline_cache_data(device.get(), first->get(), &written, bytes.data());
        INFO(failureOf(resultOf(wrote)));
        REQUIRE(wrote >= 0);
        CHECK(written == size);
        CHECK(resultOf(wrote) != Result::Incomplete);
        bytes.resize(written);

        // Pipelines made with a cache stay once it goes.
        const bfr_pipeline_cache stale = first->get();
        *first = {};
        std::size_t staleSize = 0;
        CHECK(resultOf(bfr_get_pipeline_cache_data(device.get(), stale, &staleSize, nullptr)) ==
              Result::ErrorInvalidHandle);
        CHECK(device.createComputePipeline(compute).error().code == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_pipeline_cache(device.get(), stale);
        // d1: null destroy: no message (track B)
        bfr_destroy_pipeline_cache(device.get(), {});

        auto second = device.createPipelineCache({.data = bytes});
        INFO(failureOf(second));
        REQUIRE(second.has_value());
        compute.cache = *second;
        graphics.cache = *second;
        auto computeAgain = device.createComputePipeline(compute);
        INFO(failureOf(computeAgain));
        REQUIRE(computeAgain.has_value());
        auto graphicsAgain = device.createGraphicsPipeline(graphics);
        INFO(failureOf(graphicsAgain));
        REQUIRE(graphicsAgain.has_value());

        for (Pipeline* pipeline : std::array{&*computeMade, &*graphicsMade, &*computeAgain, &*graphicsAgain}) {
            const bfr_pipeline gone = destroyed(*pipeline);
            CHECK(bindResult(device, 0, gone) == Result::ErrorInvalidHandle);
        }
        const bfr_pipeline_cache secondStale = second->get();
        *second = {};
        std::size_t secondStaleSize = 0;
        CHECK(resultOf(bfr_get_pipeline_cache_data(device.get(), secondStale, &secondStaleSize, nullptr)) ==
              Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses more color targets than a pipeline holds with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        GraphicsPipelineDesc tooManyTargets = graphicsDesc();
        tooManyTargets.colorTargetCount = BFR_MAX_COLOR_TARGETS + 1;
        CHECK(made.device.createGraphicsPipeline(tooManyTargets).error().code == Result::ErrorInvalidArgument);
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
        // A bfr.hpp Pipeline keeps its Device until it goes, so none is left
        // through it; bfr.h makes the one left.
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const bfr_compute_pipeline_desc desc = detail::toC(kCompute, nullptr);
        bfr_pipeline pipeline{};
        const bfr_result created = bfr_create_compute_pipeline(made.device.get(), &desc, &pipeline);
        INFO(failureOf(resultOf(created)));
        REQUIRE(created >= 0);
    }

    // BFR_LOGGING_ENABLED: the bfr target's definition of whether BFR_LOG_* is compiled in.
    const std::size_t warnings = BFR_LOGGING_ENABLED ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 pipelines were left when the device went"));
    }
}

TEST_CASE("Device refuses push constants past their size, and on Vulkan and D3D12 ones not whole 4-byte values",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        constexpr std::uint32_t kGraphicsQueue = 0;
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const std::uint32_t size = capsOf(made.instance, kFirstAdapter).limits.maxPushConstantsSize;
        const std::vector<std::uint8_t> tooMany(size + sizeof(std::uint32_t));
        constexpr std::size_t kPartValue = 2;
        const std::vector<std::uint8_t> partValue(kPartValue);
        const auto push = [&made](std::span<const std::uint8_t> data) {
            return inList(made.device, kGraphicsQueue, [data](const Device& on, CommandList recording) {
                return bfr_push_constants(on.get(), recording.get(), data.data(),
                                          static_cast<std::uint32_t>(data.size()));
            });
        };

        CHECK(push(tooMany) == Result::ErrorInvalidArgument);
        const bool takesWholeValues = backend.backend == Backend::Vulkan || backend.backend == Backend::D3d12;
        CHECK(push(partValue) == (takesWholeValues ? Result::ErrorInvalidArgument : Result{}));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
