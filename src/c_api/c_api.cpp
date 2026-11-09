// The C API in bfr.h, over the C++ implementation. No exception crosses it:
// each entry point turns one into a result.

#include "bfr/bfr.h"
#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/rhi_config.h"
#include "bfr/rhi/sampler.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// Shares the Instance with its devices, which keep it alive: garbage-collected
// wrappers destroy them in no fixed order.
struct bfr_instance {
    std::shared_ptr<bfr::rhi::Instance> instance;
    bfr_backend backend;
};

// The Instance comes first so that it outlives the device it made.
struct bfr_device {
    std::shared_ptr<bfr::rhi::Instance> instance;
    std::unique_ptr<bfr::rhi::Device> device;
    // What a build and the size query turn the caller's geometry arrays
    // into: made with the device, max_build_geometries of each, so no call
    // allocates; one call at a time uses them.
    std::mutex conversions;
    std::vector<bfr::rhi::TriangleGeometry> triangles;
    std::vector<bfr::rhi::BoxGeometry> boxes;
    // The adapter's bfr_caps_ray_tracing.instance_size: the bytes each record
    // bfr_write_instances writes takes.
    std::uint32_t instanceSize = 0;
};

namespace {

using bfr::core::ErrorCode;
using bfr::core::Severity;
using bfr::core::window::NativeWindow;

// The C and C++ severities are the same numbers.
static_assert(static_cast<int>(Severity::Trace) == BFR_MESSAGE_SEVERITY_TRACE);
static_assert(static_cast<int>(Severity::Error) == BFR_MESSAGE_SEVERITY_ERROR);

// The least severity a desc asks for. Nullopt for a value that is not a
// severity.
std::optional<Severity> severityFrom(bfr_message_severity severity) {
    if (severity < BFR_MESSAGE_SEVERITY_TRACE || severity > BFR_MESSAGE_SEVERITY_ERROR) {
        return std::nullopt;
    }
    return static_cast<Severity>(severity);
}

// value as a Format, Undefined included; nothing when it is not one BFR has.
std::optional<bfr::rhi::Format> formatIn(bfr_format value, const char* what, bfr::core::Message& error) {
    using namespace bfr::rhi;
    const auto number = static_cast<std::int64_t>(value);
    if (number == static_cast<std::int64_t>(Format::Undefined)) {
        return Format::Undefined;
    }
    const auto* found = std::ranges::find_if(kFormats, [number](Format known) {
        return static_cast<std::int64_t>(known) == number;
    });
    if (found == kFormats.end()) {
        error.format("{} {} is not one BFR has", what, number);
        return std::nullopt;
    }
    return *found;
}

// The swapchain desc asks for. Nothing when a value is one BFR doesn't have; error says which.
std::optional<bfr::rhi::SwapchainDesc> swapchainIn(const bfr_swapchain_desc& desc, bfr::core::Message& error) {
    using namespace bfr::rhi;
    SwapchainDesc swapchain;
    swapchain.queue = desc.queue;
    swapchain.imageCount = desc.image_count;
    switch (desc.present_mode) {
        case BFR_PRESENT_MODE_FIFO:
            swapchain.presentMode = PresentMode::Fifo;
            break;
        case BFR_PRESENT_MODE_FIFO_RELAXED:
            swapchain.presentMode = PresentMode::FifoRelaxed;
            break;
        case BFR_PRESENT_MODE_MAILBOX:
            swapchain.presentMode = PresentMode::Mailbox;
            break;
        case BFR_PRESENT_MODE_IMMEDIATE:
            swapchain.presentMode = PresentMode::Immediate;
            break;
        case BFR_PRESENT_MODE_FORCE32:
        default:
            error.format("present mode {} is not one BFR has", static_cast<std::int64_t>(desc.present_mode));
            return std::nullopt;
    }
    const auto format = formatIn(desc.format, "format", error);
    if (!format) {
        return std::nullopt;
    }
    swapchain.format = *format;
    const auto colorSpace = static_cast<std::int64_t>(desc.color_space);
    if (colorSpace < static_cast<std::int64_t>(ColorSpace::Srgb) ||
        colorSpace > static_cast<std::int64_t>(ColorSpace::DisplayNative)) {
        error.format("color space {} is not one BFR has", colorSpace);
        return std::nullopt;
    }
    swapchain.colorSpace = static_cast<ColorSpace>(colorSpace);
    switch (desc.alpha_mode) {
        case BFR_ALPHA_MODE_OPAQUE:
            swapchain.alphaMode = AlphaMode::Opaque;
            break;
        case BFR_ALPHA_MODE_PREMULTIPLIED:
            swapchain.alphaMode = AlphaMode::Premultiplied;
            break;
        case BFR_ALPHA_MODE_POSTMULTIPLIED:
            swapchain.alphaMode = AlphaMode::Postmultiplied;
            break;
        case BFR_ALPHA_MODE_INHERIT:
            swapchain.alphaMode = AlphaMode::Inherit;
            break;
        case BFR_ALPHA_MODE_FORCE32:
        default:
            error.format("alpha mode {} is not one BFR has", static_cast<std::int64_t>(desc.alpha_mode));
            return std::nullopt;
    }
    return swapchain;
}

// The calling thread's last error, which bfr_get_last_error returns. Fixed
// text, so recording a failure allocates nothing, on the frame path or off it.
bfr::core::Message& lastError() {
    thread_local bfr::core::Message error;
    return error;
}

// Records message as the calling thread's last error and returns result.
bfr_result fail(bfr_result result, std::string_view message) noexcept {
    lastError().assign(message);
    return result;
}

// Records format filled with args as the calling thread's last error, written
// straight into it, and returns result.
template <class... Args>
    requires(sizeof...(Args) > 0)
bfr_result fail(bfr_result result, std::format_string<Args...> format, Args&&... args) noexcept {
    lastError().format(format, std::forward<Args>(args)...);
    return result;
}

bfr_result resultFor(ErrorCode code) {
    switch (code) {
        case ErrorCode::RhiInstanceFailed:
            return BFR_RESULT_ERROR_INSTANCE_FAILED;
        case ErrorCode::RhiLoaderFailed:
            return BFR_RESULT_ERROR_LOADER_FAILED;
        case ErrorCode::RhiValidationUnavailable:
            return BFR_RESULT_ERROR_VALIDATION_UNAVAILABLE;
        case ErrorCode::RhiAdapterOutOfRange:
            return BFR_RESULT_ERROR_INVALID_ARGUMENT;
        case ErrorCode::RhiDeviceFailed:
            return BFR_RESULT_ERROR_DEVICE_FAILED;
        case ErrorCode::RhiNoAdapter:
            return BFR_RESULT_ERROR_NO_ADAPTER;
        case ErrorCode::RhiWindowFailed:
            return BFR_RESULT_ERROR_WINDOW_FAILED;
        case ErrorCode::RhiInvalidHandle:
            return BFR_RESULT_ERROR_INVALID_HANDLE;
        case ErrorCode::RhiPresentUnsupported:
            return BFR_RESULT_ERROR_PRESENT_UNSUPPORTED;
        case ErrorCode::RhiSwapchainFailed:
            return BFR_RESULT_ERROR_SWAPCHAIN_FAILED;
        case ErrorCode::RhiLimitReached:
            return BFR_RESULT_ERROR_LIMIT_REACHED;
        case ErrorCode::RhiOutOfMemory:
            return BFR_RESULT_ERROR_OUT_OF_MEMORY;
        case ErrorCode::RhiDeviceLost:
            return BFR_RESULT_ERROR_DEVICE_LOST;
        case ErrorCode::RhiWindowLost:
            return BFR_RESULT_ERROR_WINDOW_LOST;
        case ErrorCode::RhiUnsupported:
            return BFR_RESULT_ERROR_UNSUPPORTED;
        case ErrorCode::RhiBackendUnavailable:
            return BFR_RESULT_ERROR_BACKEND_UNAVAILABLE;
        case ErrorCode::RhiQueryFailed:
            return BFR_RESULT_ERROR_QUERY_FAILED;
        case ErrorCode::RhiOutOfDate:
            return BFR_RESULT_ERROR_OUT_OF_DATE;
        case ErrorCode::RhiInvalidCall:
            return BFR_RESULT_ERROR_INVALID_ARGUMENT;
        case ErrorCode::RhiTimeout:
            return BFR_RESULT_ERROR_TIMEOUT;
    }
    return BFR_RESULT_ERROR_INTERNAL;
}

bfr_result fail(const bfr::core::Error& error) noexcept {
    return fail(resultFor(error.code), error.message);
}

// Runs body, which returns a bfr_result, turning an exception into one.
template <class Body>
bfr_result guarded(Body&& body) noexcept {
    try {
        return std::forward<Body>(body)();
    } catch (const std::bad_alloc&) {
        return fail(BFR_RESULT_ERROR_OUT_OF_MEMORY, "out of memory");
    } catch (const std::exception& exception) {
        return fail(BFR_RESULT_ERROR_INTERNAL, exception.what());
    } catch (...) {
        return fail(BFR_RESULT_ERROR_INTERNAL, "an unknown exception");
    }
}

// Runs body, which returns nothing. The destroy calls have no result to
// report an exception through, so it is recorded as the last error only.
template <class Body>
void quietly(Body&& body) noexcept {
    try {
        std::forward<Body>(body)();
    } catch (const std::bad_alloc&) {
        fail(BFR_RESULT_ERROR_OUT_OF_MEMORY, "out of memory");
    } catch (const std::exception& exception) {
        fail(BFR_RESULT_ERROR_INTERNAL, exception.what());
    } catch (...) {
        fail(BFR_RESULT_ERROR_INTERNAL, "an unknown exception");
    }
}

// A C handle is a Handle's bits, so the two convert by copy.
bfr_swapchain toC(bfr::rhi::SwapchainHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::SwapchainHandle fromC(bfr_swapchain handle) {
    return bfr::rhi::SwapchainHandle::fromBits(handle.bits);
}

bfr_command_pool toC(bfr::rhi::CommandPoolHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::CommandPoolHandle fromC(bfr_command_pool handle) {
    return bfr::rhi::CommandPoolHandle::fromBits(handle.bits);
}

bfr_buffer toC(bfr::rhi::BufferHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::BufferHandle fromC(bfr_buffer handle) {
    return bfr::rhi::BufferHandle::fromBits(handle.bits);
}

// The buffer desc asks for. Nothing when its memory is not one BFR has; error
// says which.
std::optional<bfr::rhi::BufferDesc> bufferIn(const bfr_buffer_desc& desc, bfr::core::Message& error) {
    using namespace bfr::rhi;
    BufferDesc buffer{.size = desc.size,
                      .memory = BufferMemory::Device,
                      .usage = {.copySrc = desc.usage.copy_src != BFR_FALSE,
                                .copyDst = desc.usage.copy_dst != BFR_FALSE,
                                .uniform = desc.usage.uniform != BFR_FALSE,
                                .storage = desc.usage.storage != BFR_FALSE,
                                .index = desc.usage.index != BFR_FALSE,
                                .indirect = desc.usage.indirect != BFR_FALSE,
                                .accelerationStructureInput = desc.usage.acceleration_structure_input != BFR_FALSE,
                                .accelerationStructureScratch = desc.usage.acceleration_structure_scratch != BFR_FALSE,
                                .shaderBindingTable = desc.usage.shader_binding_table != BFR_FALSE},
                      .sparse = desc.sparse != BFR_FALSE,
                      .sparseAliased = desc.sparse_aliased != BFR_FALSE,
                      .tileBytes = desc.tile_bytes};
    switch (desc.memory) {
        case BFR_BUFFER_MEMORY_DEVICE:
            buffer.memory = BufferMemory::Device;
            break;
        case BFR_BUFFER_MEMORY_UPLOAD:
            buffer.memory = BufferMemory::Upload;
            break;
        case BFR_BUFFER_MEMORY_READBACK:
            buffer.memory = BufferMemory::Readback;
            break;
        case BFR_BUFFER_MEMORY_FORCE32:
        default:
            error.format("buffer memory {} is not one BFR has", static_cast<std::int64_t>(desc.memory));
            return std::nullopt;
    }
    return buffer;
}

bfr_texture toC(bfr::rhi::TextureHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::TextureHandle fromC(bfr_texture handle) {
    return bfr::rhi::TextureHandle::fromBits(handle.bits);
}

// The C and C++ texture enums are the same numbers, so they cross as they are.
static_assert(static_cast<int>(bfr::rhi::TextureDimension::Cube) == BFR_TEXTURE_DIMENSION_CUBE);
static_assert(static_cast<int>(bfr::rhi::TextureLayout::ResolveDst) == BFR_TEXTURE_LAYOUT_RESOLVE_DST);
static_assert(static_cast<int>(bfr::rhi::TextureAspect::Stencil) == BFR_TEXTURE_ASPECT_STENCIL);

// The texture desc asks for. Nothing when its dimension or format is not one
// BFR has; error says which.
std::optional<bfr::rhi::TextureDesc> textureIn(const bfr_texture_desc& desc, bfr::core::Message& error) {
    using namespace bfr::rhi;
    if (desc.dimension < BFR_TEXTURE_DIMENSION_D2 || desc.dimension > BFR_TEXTURE_DIMENSION_CUBE) {
        error.format("texture dimension {} is not one BFR has", static_cast<std::int64_t>(desc.dimension));
        return std::nullopt;
    }
    const auto format = static_cast<std::int64_t>(desc.format);
    const auto* found = std::ranges::find_if(kFormats, [format](Format known) {
        return static_cast<std::int64_t>(known) == format;
    });
    if (found == kFormats.end()) {
        error.format("format {} is not one BFR has", format);
        return std::nullopt;
    }
    return TextureDesc{.dimension = static_cast<TextureDimension>(desc.dimension),
                       .format = *found,
                       .width = desc.width,
                       .height = desc.height,
                       .depth = desc.depth,
                       .layers = desc.layers,
                       .mipLevels = desc.mip_levels,
                       .samples = desc.samples,
                       .usage = {.copySrc = desc.usage.copy_src != BFR_FALSE,
                                 .copyDst = desc.usage.copy_dst != BFR_FALSE,
                                 .sampled = desc.usage.sampled != BFR_FALSE,
                                 .storage = desc.usage.storage != BFR_FALSE,
                                 .colorAttachment = desc.usage.color_attachment != BFR_FALSE,
                                 .depthStencil = desc.usage.depth_stencil != BFR_FALSE},
                       .sparse = desc.sparse != BFR_FALSE,
                       .sparseAliased = desc.sparse_aliased != BFR_FALSE,
                       .tileBytes = desc.tile_bytes};
}

bfr_pipeline toC(bfr::rhi::PipelineHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::PipelineHandle fromC(bfr_pipeline handle) {
    return bfr::rhi::PipelineHandle::fromBits(handle.bits);
}

bfr_pipeline_cache toC(bfr::rhi::PipelineCacheHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::ShaderHandle fromC(bfr_shader handle) {
    return bfr::rhi::ShaderHandle::fromBits(handle.bits);
}

bfr_shader toC(bfr::rhi::ShaderHandle handle) {
    return {.bits = handle.bits};
}

static_assert(static_cast<int>(bfr::rhi::RayTracingStage::Callable) == BFR_RAY_TRACING_STAGE_CALLABLE);
static_assert(static_cast<int>(bfr::rhi::RayTracingGroupType::BoxHit) == BFR_RAY_TRACING_GROUP_TYPE_BOX_HIT);
// bfr_trace_rays_indirect's arguments are read in place on the GPU, so the C
// and C++ layouts are one.
static_assert(sizeof(bfr_trace_rays_indirect_args) == sizeof(bfr::rhi::TraceRaysIndirectArgs) &&
              offsetof(bfr_trace_rays_indirect_args, miss) == offsetof(bfr::rhi::TraceRaysIndirectArgs, miss) &&
              offsetof(bfr_trace_rays_indirect_args, hit) == offsetof(bfr::rhi::TraceRaysIndirectArgs, hit) &&
              offsetof(bfr_trace_rays_indirect_args, callable) == offsetof(bfr::rhi::TraceRaysIndirectArgs, callable) &&
              offsetof(bfr_trace_rays_indirect_args, width) == offsetof(bfr::rhi::TraceRaysIndirectArgs, width) &&
              offsetof(bfr_trace_rays_indirect_args, depth) == offsetof(bfr::rhi::TraceRaysIndirectArgs, depth));

bfr::rhi::PipelineCacheHandle fromC(bfr_pipeline_cache handle) {
    return bfr::rhi::PipelineCacheHandle::fromBits(handle.bits);
}

// The C and C++ pipeline enums are the same numbers, so they cross as they
// are, once checked to be a value BFR has.
static_assert(static_cast<int>(bfr::rhi::Topology::PointList) == BFR_TOPOLOGY_POINT_LIST);
static_assert(static_cast<int>(bfr::rhi::PrimitiveRestart::Index32) == BFR_PRIMITIVE_RESTART_INDEX32);
static_assert(static_cast<int>(bfr::rhi::ResolveMode::Max) == BFR_RESOLVE_MODE_MAX);
static_assert(static_cast<int>(bfr::rhi::CullMode::Back) == BFR_CULL_MODE_BACK);
static_assert(static_cast<int>(bfr::rhi::FrontFace::Clockwise) == BFR_FRONT_FACE_CLOCKWISE);
static_assert(static_cast<int>(bfr::rhi::FillMode::Wireframe) == BFR_FILL_MODE_WIREFRAME);
static_assert(static_cast<int>(bfr::rhi::CompareOp::GreaterEqual) == BFR_COMPARE_OP_GREATER_EQUAL);
static_assert(static_cast<int>(bfr::rhi::StencilOp::DecrementWrap) == BFR_STENCIL_OP_DECREMENT_WRAP);
static_assert(static_cast<int>(bfr::rhi::BlendFactor::SrcAlphaSaturate) == BFR_BLEND_FACTOR_SRC_ALPHA_SATURATE);
static_assert(static_cast<int>(bfr::rhi::BlendOp::Max) == BFR_BLEND_OP_MAX);
static_assert(static_cast<int>(bfr::rhi::IndexType::Uint16) == BFR_INDEX_TYPE_UINT16);

// value as the C++ enum E, whose values run from 0 to last; nothing when it is
// not one of them, and error says which.
template <class E, class C>
std::optional<E> enumIn(C value, E last, const char* what, bfr::core::Message& error) {
    const auto number = static_cast<std::int64_t>(value);
    if (number < 0 || number > static_cast<std::int64_t>(last)) {
        error.format("{} {} is not one BFR has", what, number);
        return std::nullopt;
    }
    return static_cast<E>(number);
}

static_assert(static_cast<int>(bfr::rhi::QueueLevel::Realtime) == BFR_QUEUE_LEVEL_REALTIME);
static_assert(static_cast<int>(bfr::rhi::SwapchainStatus::Suboptimal) == BFR_SWAPCHAIN_STATUS_SUBOPTIMAL);

// count queues from queues, which may be null only when count is 0. Nothing
// when it is null otherwise, or a level is one BFR doesn't have; error says which.
std::optional<std::vector<bfr::rhi::QueueDesc>> queuesIn(const bfr_queue_desc* queues, std::uint32_t count,
                                                         const char* what, bfr::core::Message& error) {
    if (queues == nullptr && count != 0) {
        error.format("{} is null with a count of {}", what, count);
        return std::nullopt;
    }
    std::vector<bfr::rhi::QueueDesc> out;
    out.reserve(count);
    for (const bfr_queue_desc& queue : std::span{queues, count}) {
        const auto level = enumIn(queue.level, bfr::rhi::QueueLevel::Realtime, "queue level", error);
        if (!level) {
            return std::nullopt;
        }
        out.push_back({.rank = queue.rank, .level = *level});
    }
    return out;
}

bfr_timestamp_pool toC(bfr::rhi::TimestampPoolHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::TimestampPoolHandle fromC(bfr_timestamp_pool handle) {
    return bfr::rhi::TimestampPoolHandle::fromBits(handle.bits);
}

bfr_sparse_memory toC(bfr::rhi::SparseMemoryHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::SparseMemoryHandle fromC(bfr_sparse_memory handle) {
    return bfr::rhi::SparseMemoryHandle::fromBits(handle.bits);
}

bfr_tile_shape toC(const bfr::rhi::TileShape& shape) {
    return {.width = shape.width, .height = shape.height, .depth = shape.depth};
}

static_assert(static_cast<int>(bfr::rhi::TileMapping::Unmap) == BFR_TILE_MAPPING_UNMAP);

// bfr_update_tile_mappings reads the caller's regions as the rhi types, in
// place, so a mapping copies and allocates nothing; each backend checks their
// enums.
static_assert(sizeof(bfr_texture_tile_region) == sizeof(bfr::rhi::TextureTileRegion) &&
              alignof(bfr_texture_tile_region) == alignof(bfr::rhi::TextureTileRegion) &&
              std::is_standard_layout_v<bfr::rhi::TextureTileRegion> &&
              std::is_trivially_copyable_v<bfr::rhi::TextureTileRegion> &&
              offsetof(bfr_texture_tile_region, mapping) == offsetof(bfr::rhi::TextureTileRegion, mapping) &&
              offsetof(bfr_texture_tile_region, aspect) == offsetof(bfr::rhi::TextureTileRegion, aspect) &&
              offsetof(bfr_texture_tile_region, mip_level) == offsetof(bfr::rhi::TextureTileRegion, mipLevel) &&
              offsetof(bfr_texture_tile_region, layer) == offsetof(bfr::rhi::TextureTileRegion, layer) &&
              offsetof(bfr_texture_tile_region, x) == offsetof(bfr::rhi::TextureTileRegion, x) &&
              offsetof(bfr_texture_tile_region, y) == offsetof(bfr::rhi::TextureTileRegion, y) &&
              offsetof(bfr_texture_tile_region, z) == offsetof(bfr::rhi::TextureTileRegion, z) &&
              offsetof(bfr_texture_tile_region, width) == offsetof(bfr::rhi::TextureTileRegion, width) &&
              offsetof(bfr_texture_tile_region, height) == offsetof(bfr::rhi::TextureTileRegion, height) &&
              offsetof(bfr_texture_tile_region, depth) == offsetof(bfr::rhi::TextureTileRegion, depth) &&
              offsetof(bfr_texture_tile_region, tail_tile) == offsetof(bfr::rhi::TextureTileRegion, tailTile) &&
              offsetof(bfr_texture_tile_region, tail_tiles) == offsetof(bfr::rhi::TextureTileRegion, tailTiles) &&
              offsetof(bfr_texture_tile_region, memory_tile) == offsetof(bfr::rhi::TextureTileRegion, memoryTile));
static_assert(sizeof(bfr_buffer_tile_region) == sizeof(bfr::rhi::BufferTileRegion) &&
              alignof(bfr_buffer_tile_region) == alignof(bfr::rhi::BufferTileRegion) &&
              std::is_standard_layout_v<bfr::rhi::BufferTileRegion> &&
              std::is_trivially_copyable_v<bfr::rhi::BufferTileRegion> &&
              offsetof(bfr_buffer_tile_region, mapping) == offsetof(bfr::rhi::BufferTileRegion, mapping) &&
              offsetof(bfr_buffer_tile_region, tile) == offsetof(bfr::rhi::BufferTileRegion, tile) &&
              offsetof(bfr_buffer_tile_region, tiles) == offsetof(bfr::rhi::BufferTileRegion, tiles) &&
              offsetof(bfr_buffer_tile_region, memory_tile) == offsetof(bfr::rhi::BufferTileRegion, memoryTile));

static_assert(static_cast<int>(bfr::rhi::TimestampStage::Copy) == BFR_TIMESTAMP_STAGE_COPY);
static_assert(static_cast<int>(bfr::rhi::TimestampStage::AccelerationStructureBuild) ==
              BFR_TIMESTAMP_STAGE_ACCELERATION_STRUCTURE_BUILD);
static_assert(static_cast<int>(bfr::rhi::TimestampGranularity::Precise) == BFR_TIMESTAMP_GRANULARITY_PRECISE);
static_assert(static_cast<int>(bfr::rhi::HostClock::PerformanceCounter) == BFR_HOST_CLOCK_PERFORMANCE_COUNTER);
static_assert(static_cast<int>(bfr::rhi::Call::InstanceSwapchainFormats) == BFR_CALL_INSTANCE_SWAPCHAIN_FORMATS);
static_assert(static_cast<int>(bfr::rhi::Call::SetTopology) == BFR_CALL_SET_TOPOLOGY);
static_assert(static_cast<int>(bfr::rhi::Call::ResolveTexture) == BFR_CALL_RESOLVE_TEXTURE);
static_assert(static_cast<int>(bfr::rhi::Call::DrawMeshIndirectCount) == BFR_CALL_DRAW_MESH_INDIRECT_COUNT);
static_assert(static_cast<int>(bfr::rhi::Call::BuildAccelerationStructure) == BFR_CALL_BUILD_ACCELERATION_STRUCTURE);
static_assert(static_cast<int>(bfr::rhi::Call::CopyAccelerationStructure) == BFR_CALL_COPY_ACCELERATION_STRUCTURE);
static_assert(static_cast<int>(bfr::rhi::Call::BufferAddress) == BFR_CALL_BUFFER_ADDRESS);
static_assert(bfr::rhi::kCallCount == BFR_CALL_BUFFER_ADDRESS + 1);

bfr_sampler toC(bfr::rhi::SamplerHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::SamplerHandle fromC(bfr_sampler handle) {
    return bfr::rhi::SamplerHandle::fromBits(handle.bits);
}

bfr_acceleration_structure toC(bfr::rhi::AccelerationStructureHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::AccelerationStructureHandle fromC(bfr_acceleration_structure handle) {
    return bfr::rhi::AccelerationStructureHandle::fromBits(handle.bits);
}

static_assert(static_cast<int>(bfr::rhi::AccelerationStructureType::Top) == BFR_ACCELERATION_STRUCTURE_TYPE_TOP);
static_assert(static_cast<int>(bfr::rhi::BuildPreference::FastBuild) == BFR_BUILD_PREFERENCE_FAST_BUILD);
static_assert(static_cast<int>(bfr::rhi::InstanceOpacity::ForceNonOpaque) == BFR_INSTANCE_OPACITY_FORCE_NON_OPAQUE);
static_assert(bfr::rhi::kInstanceIndexLimit == BFR_INSTANCE_INDEX_LIMIT);
static_assert(bfr::rhi::kTransformFloats == BFR_TRANSFORM_FLOATS);

// The rhi form of a structure's inputs, its triangles and boxes written to
// device's arrays from triangles and boxes on, which those advance past; a
// top structure's are none, as the rhi reads none. Nothing when an array is
// null with a count, there is no room left, or an enum is not one BFR has;
// error says which, and limit whether it was room.
std::optional<bfr::rhi::AccelerationStructureInputs> inputsIn(const bfr_acceleration_structure_inputs& inputs,
                                                              bfr_device& device, std::size_t& triangles,
                                                              std::size_t& boxes, bool& limit,
                                                              bfr::core::Message& error) {
    using namespace bfr::rhi;
    limit = false;
    const auto type = enumIn(inputs.type, AccelerationStructureType::Top, "acceleration structure type", error);
    const auto preference = enumIn(inputs.flags.preference, BuildPreference::FastBuild, "build preference", error);
    if (!type || !preference) {
        return std::nullopt;
    }
    AccelerationStructureInputs made{.type = *type,
                                     .flags = {.preference = *preference,
                                               .allowUpdate = inputs.flags.allow_update != BFR_FALSE,
                                               .minimizeMemory = inputs.flags.minimize_memory != BFR_FALSE},
                                     .triangles = {},
                                     .boxes = {},
                                     .instances = fromC(inputs.instances),
                                     .instanceOffset = inputs.instance_offset,
                                     .instanceCount = inputs.instance_count};
    if (made.type == AccelerationStructureType::Top) {
        return made;
    }
    if ((inputs.triangles == nullptr && inputs.triangle_count != 0) ||
        (inputs.boxes == nullptr && inputs.box_count != 0)) {
        error.format("triangles or boxes is null with a count");
        return std::nullopt;
    }
    if (inputs.triangle_count > device.triangles.size() - triangles || inputs.box_count > device.boxes.size() - boxes) {
        limit = true;
        error.format("the geometries are more than the device's max_build_geometries, {}", device.triangles.size());
        return std::nullopt;
    }
    const std::span triangleOut = std::span{device.triangles}.subspan(triangles, inputs.triangle_count);
    for (std::size_t i = 0; i < triangleOut.size(); ++i) {
        const bfr_triangle_geometry& from = std::span{inputs.triangles, inputs.triangle_count}[i];
        const auto format = formatIn(from.vertex_format, "vertex format", error);
        const auto indexType = enumIn(from.index_type, IndexType::Uint16, "index type", error);
        if (!format || !indexType) {
            return std::nullopt;
        }
        triangleOut[i] = {.vertices = fromC(from.vertices),
                          .vertexOffset = from.vertex_offset,
                          .vertexFormat = *format,
                          .vertexStride = from.vertex_stride,
                          .vertexCount = from.vertex_count,
                          .indices = fromC(from.indices),
                          .indexOffset = from.index_offset,
                          .indexType = *indexType,
                          .triangleCount = from.triangle_count,
                          .transform = fromC(from.transform),
                          .transformOffset = from.transform_offset,
                          .opaque = from.opaque != BFR_FALSE,
                          .noDuplicateAnyHit = from.no_duplicate_any_hit != BFR_FALSE};
    }
    const std::span boxOut = std::span{device.boxes}.subspan(boxes, inputs.box_count);
    for (std::size_t i = 0; i < boxOut.size(); ++i) {
        const bfr_box_geometry& from = std::span{inputs.boxes, inputs.box_count}[i];
        boxOut[i] = {.boxes = fromC(from.boxes),
                     .offset = from.offset,
                     .stride = from.stride,
                     .count = from.count,
                     .opaque = from.opaque != BFR_FALSE,
                     .noDuplicateAnyHit = from.no_duplicate_any_hit != BFR_FALSE};
    }
    triangles += triangleOut.size();
    boxes += boxOut.size();
    made.triangles = triangleOut;
    made.boxes = boxOut;
    return made;
}

static_assert(static_cast<int>(bfr::rhi::Filter::Linear) == BFR_FILTER_LINEAR);
static_assert(static_cast<int>(bfr::rhi::AddressMode::MirrorClampToEdge) == BFR_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE);
static_assert(static_cast<int>(bfr::rhi::SamplerReduction::Maximum) == BFR_SAMPLER_REDUCTION_MAXIMUM);

// The sampler desc asks for. Nothing when one of its enums holds a value BFR
// does not have; error says which.
std::optional<bfr::rhi::SamplerDesc> samplerIn(const bfr_sampler_desc& desc, bfr::core::Message& error) {
    using namespace bfr::rhi;
    const auto magFilter = enumIn(desc.mag_filter, Filter::Linear, "mag filter", error);
    const auto minFilter = enumIn(desc.min_filter, Filter::Linear, "min filter", error);
    const auto mipFilter = enumIn(desc.mip_filter, Filter::Linear, "mip filter", error);
    const auto addressU = enumIn(desc.address_u, AddressMode::MirrorClampToEdge, "address mode u", error);
    const auto addressV = enumIn(desc.address_v, AddressMode::MirrorClampToEdge, "address mode v", error);
    const auto addressW = enumIn(desc.address_w, AddressMode::MirrorClampToEdge, "address mode w", error);
    const auto reduction = enumIn(desc.reduction, SamplerReduction::Maximum, "sampler reduction", error);
    const auto compare = enumIn(desc.compare_op, CompareOp::GreaterEqual, "compare op", error);
    if (!magFilter || !minFilter || !mipFilter || !addressU || !addressV || !addressW || !reduction || !compare) {
        return std::nullopt;
    }
    SamplerDesc sampler{.magFilter = *magFilter,
                        .minFilter = *minFilter,
                        .mipFilter = *mipFilter,
                        .addressU = *addressU,
                        .addressV = *addressV,
                        .addressW = *addressW,
                        .lodBias = desc.lod_bias,
                        .minLod = desc.min_lod,
                        .maxLod = desc.max_lod,
                        .maxAnisotropy = desc.max_anisotropy,
                        .reduction = *reduction,
                        .compareOp = *compare,
                        .borderColor = {},
                        .borderInteger = desc.border_integer != BFR_FALSE,
                        .borderColorInteger = {},
                        .unnormalizedCoordinates = desc.unnormalized_coordinates != BFR_FALSE,
                        .lodAverage = desc.lod_average != BFR_FALSE};
    std::ranges::copy(desc.border_color, sampler.borderColor.begin());
    std::ranges::copy(desc.border_color_integer, sampler.borderColorInteger.begin());
    return sampler;
}

bfr::rhi::ShaderCode shaderIn(const bfr_shader_code& code) {
    return {.code = {reinterpret_cast<const std::byte*>(code.code), code.code_size},
            .entryPoint = {code.entry_point.data, code.entry_point.length}};
}

// The compute pipeline desc asks for.
bfr::rhi::ComputePipelineDesc computePipelineIn(const bfr_compute_pipeline_desc& desc) {
    return {.compute = shaderIn(desc.compute),
            .workgroupSize = std::to_array(desc.workgroup_size),
            .cache = fromC(desc.cache)};
}

std::optional<bfr::rhi::StencilFace> stencilFaceIn(const bfr_stencil_face& face, bfr::core::Message& error) {
    using namespace bfr::rhi;
    const auto fail = enumIn(face.fail, StencilOp::DecrementWrap, "stencil op", error);
    const auto pass = enumIn(face.pass, StencilOp::DecrementWrap, "stencil op", error);
    const auto depthFail = enumIn(face.depth_fail, StencilOp::DecrementWrap, "stencil op", error);
    const auto compare = enumIn(face.compare, CompareOp::GreaterEqual, "compare op", error);
    if (!fail || !pass || !depthFail || !compare) {
        return std::nullopt;
    }
    return StencilFace{.fail = *fail,
                       .pass = *pass,
                       .depthFail = *depthFail,
                       .compare = *compare,
                       .readMask = face.read_mask,
                       .writeMask = face.write_mask};
}

std::optional<bfr::rhi::BlendState> blendIn(const bfr_blend_state& blend, bfr::core::Message& error) {
    using namespace bfr::rhi;
    const auto srcColor = enumIn(blend.src_color, BlendFactor::SrcAlphaSaturate, "blend factor", error);
    const auto dstColor = enumIn(blend.dst_color, BlendFactor::SrcAlphaSaturate, "blend factor", error);
    const auto colorOp = enumIn(blend.color_op, BlendOp::Max, "blend op", error);
    const auto srcAlpha = enumIn(blend.src_alpha, BlendFactor::SrcAlphaSaturate, "blend factor", error);
    const auto dstAlpha = enumIn(blend.dst_alpha, BlendFactor::SrcAlphaSaturate, "blend factor", error);
    const auto alphaOp = enumIn(blend.alpha_op, BlendOp::Max, "blend op", error);
    if (!srcColor || !dstColor || !colorOp || !srcAlpha || !dstAlpha || !alphaOp) {
        return std::nullopt;
    }
    return BlendState{.enable = blend.enable != BFR_FALSE,
                      .srcColor = *srcColor,
                      .dstColor = *dstColor,
                      .colorOp = *colorOp,
                      .srcAlpha = *srcAlpha,
                      .dstAlpha = *dstAlpha,
                      .alphaOp = *alphaOp,
                      .writeMask = blend.write_mask};
}

// The graphics pipeline desc asks for. Nothing when a value is not one BFR
// has; error says which. The targets past color_target_count are not read.
std::optional<bfr::rhi::GraphicsPipelineDesc> graphicsPipelineIn(const bfr_graphics_pipeline_desc& desc,
                                                                 bfr::core::Message& error) {
    using namespace bfr::rhi;
    const auto topology = enumIn(desc.topology, Topology::PointList, "topology", error);
    const auto restart = enumIn(desc.primitive_restart, PrimitiveRestart::Index32, "primitive restart", error);
    const auto cull = enumIn(desc.raster.cull, CullMode::Back, "cull mode", error);
    const auto frontFace = enumIn(desc.raster.front_face, FrontFace::Clockwise, "front face", error);
    const auto fill = enumIn(desc.raster.fill, FillMode::Wireframe, "fill mode", error);
    const auto depthCompare = enumIn(desc.depth_stencil.depth_compare, CompareOp::GreaterEqual, "compare op", error);
    const auto front = stencilFaceIn(desc.depth_stencil.front, error);
    const auto back = stencilFaceIn(desc.depth_stencil.back, error);
    const auto depthStencilFormat = formatIn(desc.depth_stencil_format, "depth-stencil format", error);
    if (!topology || !restart || !cull || !frontFace || !fill || !depthCompare || !front || !back ||
        !depthStencilFormat) {
        return std::nullopt;
    }
    GraphicsPipelineDesc pipeline{
        .vertex = shaderIn(desc.vertex),
        .fragment = shaderIn(desc.fragment),
        .task = shaderIn(desc.task),
        .taskWorkgroupSize = std::to_array(desc.task_workgroup_size),
        .mesh = shaderIn(desc.mesh),
        .meshWorkgroupSize = std::to_array(desc.mesh_workgroup_size),
        .topology = *topology,
        .primitiveRestart = *restart,
        .raster = {.cull = *cull,
                   .frontFace = *frontFace,
                   .fill = *fill,
                   .depthClamp = desc.raster.depth_clamp != BFR_FALSE,
                   .depthBias = desc.raster.depth_bias,
                   .depthBiasSlope = desc.raster.depth_bias_slope,
                   .depthBiasClamp = desc.raster.depth_bias_clamp},
        .depthStencil = {.depthTest = desc.depth_stencil.depth_test != BFR_FALSE,
                         .depthWrite = desc.depth_stencil.depth_write != BFR_FALSE,
                         .depthCompare = *depthCompare,
                         .stencilTest = desc.depth_stencil.stencil_test != BFR_FALSE,
                         .front = *front,
                         .back = *back},
        .colorTargetCount = desc.color_target_count,
        .colorTargets = {},
        .depthStencilFormat = *depthStencilFormat,
        .samples = desc.samples,
        .alphaToCoverage = desc.alpha_to_coverage != BFR_FALSE,
        .sampleMask = desc.sample_mask,
        .alphaToOne = desc.alpha_to_one != BFR_FALSE,
        .minSampleShading = desc.min_sample_shading,
        .cache = fromC(desc.cache),
    };
    const std::span targets{desc.color_targets};
    for (std::uint32_t i = 0; i < std::min<std::uint32_t>(desc.color_target_count, kMaxColorTargets); ++i) {
        const auto format = formatIn(targets[i].format, "color target format", error);
        const auto blend = blendIn(targets[i].blend, error);
        if (!format || !blend) {
            return std::nullopt;
        }
        pipeline.colorTargets.at(i) = {.format = *format, .blend = *blend};
    }
    return pipeline;
}

bfr_command_list toC(bfr::rhi::CommandListHandle handle) {
    return {.bits = handle.bits};
}

bfr_token toC(bfr::rhi::Token token) {
    return {.queue = token.queue, .value = token.value};
}

// bfr_submit reads the caller's arrays as the rhi types, which are laid out
// exactly as the C ones, so a submit copies and allocates nothing.
static_assert(sizeof(bfr_command_list) == sizeof(bfr::rhi::CommandListHandle) &&
              alignof(bfr_command_list) == alignof(bfr::rhi::CommandListHandle) &&
              std::is_standard_layout_v<bfr::rhi::CommandListHandle> &&
              std::is_trivially_copyable_v<bfr::rhi::CommandListHandle> &&
              offsetof(bfr_command_list, bits) == offsetof(bfr::rhi::CommandListHandle, bits));
static_assert(sizeof(bfr_token) == sizeof(bfr::rhi::Token) && alignof(bfr_token) == alignof(bfr::rhi::Token) &&
              std::is_standard_layout_v<bfr::rhi::Token> && std::is_trivially_copyable_v<bfr::rhi::Token> &&
              offsetof(bfr_token, queue) == offsetof(bfr::rhi::Token, queue) &&
              offsetof(bfr_token, value) == offsetof(bfr::rhi::Token, value));

// bfr_copy_buffer reads the caller's regions as the rhi type, in place.
static_assert(sizeof(bfr_buffer_copy) == sizeof(bfr::rhi::BufferCopy) &&
              alignof(bfr_buffer_copy) == alignof(bfr::rhi::BufferCopy) &&
              std::is_standard_layout_v<bfr::rhi::BufferCopy> && std::is_trivially_copyable_v<bfr::rhi::BufferCopy> &&
              offsetof(bfr_buffer_copy, src_offset) == offsetof(bfr::rhi::BufferCopy, srcOffset) &&
              offsetof(bfr_buffer_copy, dst_offset) == offsetof(bfr::rhi::BufferCopy, dstOffset) &&
              offsetof(bfr_buffer_copy, size) == offsetof(bfr::rhi::BufferCopy, size));

// The texture copies read the caller's regions as the rhi types, in place.
static_assert(sizeof(bfr_buffer_texture_copy) == sizeof(bfr::rhi::BufferTextureCopy) &&
              alignof(bfr_buffer_texture_copy) == alignof(bfr::rhi::BufferTextureCopy) &&
              std::is_standard_layout_v<bfr::rhi::BufferTextureCopy> &&
              std::is_trivially_copyable_v<bfr::rhi::BufferTextureCopy> &&
              offsetof(bfr_buffer_texture_copy, buffer_offset) == offsetof(bfr::rhi::BufferTextureCopy, bufferOffset) &&
              offsetof(bfr_buffer_texture_copy, bytes_per_row) == offsetof(bfr::rhi::BufferTextureCopy, bytesPerRow) &&
              offsetof(bfr_buffer_texture_copy, rows_per_image) ==
                  offsetof(bfr::rhi::BufferTextureCopy, rowsPerImage) &&
              offsetof(bfr_buffer_texture_copy, mip_level) == offsetof(bfr::rhi::BufferTextureCopy, mipLevel) &&
              offsetof(bfr_buffer_texture_copy, layer) == offsetof(bfr::rhi::BufferTextureCopy, layer) &&
              offsetof(bfr_buffer_texture_copy, aspect) == offsetof(bfr::rhi::BufferTextureCopy, aspect) &&
              offsetof(bfr_buffer_texture_copy, x) == offsetof(bfr::rhi::BufferTextureCopy, x) &&
              offsetof(bfr_buffer_texture_copy, y) == offsetof(bfr::rhi::BufferTextureCopy, y) &&
              offsetof(bfr_buffer_texture_copy, z) == offsetof(bfr::rhi::BufferTextureCopy, z) &&
              offsetof(bfr_buffer_texture_copy, width) == offsetof(bfr::rhi::BufferTextureCopy, width) &&
              offsetof(bfr_buffer_texture_copy, height) == offsetof(bfr::rhi::BufferTextureCopy, height) &&
              offsetof(bfr_buffer_texture_copy, depth) == offsetof(bfr::rhi::BufferTextureCopy, depth));
static_assert(sizeof(bfr_texture_copy) == sizeof(bfr::rhi::TextureCopy) &&
              alignof(bfr_texture_copy) == alignof(bfr::rhi::TextureCopy) &&
              std::is_standard_layout_v<bfr::rhi::TextureCopy> && std::is_trivially_copyable_v<bfr::rhi::TextureCopy> &&
              offsetof(bfr_texture_copy, src_mip_level) == offsetof(bfr::rhi::TextureCopy, srcMipLevel) &&
              offsetof(bfr_texture_copy, src_layer) == offsetof(bfr::rhi::TextureCopy, srcLayer) &&
              offsetof(bfr_texture_copy, src_x) == offsetof(bfr::rhi::TextureCopy, srcX) &&
              offsetof(bfr_texture_copy, src_y) == offsetof(bfr::rhi::TextureCopy, srcY) &&
              offsetof(bfr_texture_copy, src_z) == offsetof(bfr::rhi::TextureCopy, srcZ) &&
              offsetof(bfr_texture_copy, dst_mip_level) == offsetof(bfr::rhi::TextureCopy, dstMipLevel) &&
              offsetof(bfr_texture_copy, dst_layer) == offsetof(bfr::rhi::TextureCopy, dstLayer) &&
              offsetof(bfr_texture_copy, dst_x) == offsetof(bfr::rhi::TextureCopy, dstX) &&
              offsetof(bfr_texture_copy, dst_y) == offsetof(bfr::rhi::TextureCopy, dstY) &&
              offsetof(bfr_texture_copy, dst_z) == offsetof(bfr::rhi::TextureCopy, dstZ) &&
              offsetof(bfr_texture_copy, width) == offsetof(bfr::rhi::TextureCopy, width) &&
              offsetof(bfr_texture_copy, height) == offsetof(bfr::rhi::TextureCopy, height) &&
              offsetof(bfr_texture_copy, depth) == offsetof(bfr::rhi::TextureCopy, depth));

// Fails, naming the value, when a region's aspect is not one BFR has: the
// regions are read in place, so nothing else stops an unknown aspect.
bfr_result checkAspects(std::span<const bfr_buffer_texture_copy> regions) {
    for (const bfr_buffer_texture_copy& region : regions) {
        if (region.aspect < BFR_TEXTURE_ASPECT_COLOR || region.aspect > BFR_TEXTURE_ASPECT_STENCIL) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "texture aspect {} is not one BFR has",
                        static_cast<std::int64_t>(region.aspect));
        }
    }
    return BFR_RESULT_SUCCESS;
}

bfr::rhi::Token fromC(const bfr_token& token) {
    return {.queue = token.queue, .value = token.value};
}

bfr::rhi::CommandListHandle fromC(bfr_command_list handle) {
    return bfr::rhi::CommandListHandle::fromBits(handle.bits);
}

// The rendering desc asks for, each 0 as its documented default. Nothing when
// a value is one BFR doesn't have, or more attachments are counted than
// there is room for; error says which.
std::optional<bfr::rhi::RenderingDesc> renderingIn(const bfr_rendering_desc& desc, bfr::core::Message& error) {
    using namespace bfr::rhi;
    if (desc.color_attachment_count > kMaxColorTargets) {
        error.format("color_attachment_count {} is more than the {} attachments there is room for",
                     desc.color_attachment_count, kMaxColorTargets);
        return std::nullopt;
    }
    RenderingDesc rendering{.colorAttachmentCount = desc.color_attachment_count,
                            .colorAttachments = {},
                            .depthStencil = {}};
    for (std::uint32_t i = 0; i < desc.color_attachment_count; ++i) {
        const bfr_color_attachment& attachment = std::span{desc.color_attachments}[i];
        const auto loadOp = enumIn(attachment.load_op, LoadOp::DontCare, "load op", error);
        const auto storeOp = enumIn(attachment.store_op, StoreOp::DontCare, "store op", error);
        if (!loadOp || !storeOp) {
            return std::nullopt;
        }
        rendering.colorAttachments.at(i) = {.texture = fromC(attachment.texture),
                                            .loadOp = *loadOp,
                                            .storeOp = *storeOp,
                                            .clearColor = std::to_array(attachment.clear_color),
                                            .clearInteger = attachment.clear_integer != BFR_FALSE,
                                            .clearColorInteger = std::to_array(attachment.clear_color_integer),
                                            .resolveTexture = fromC(attachment.resolve_texture)};
    }
    const bfr_depth_stencil_attachment& depth = desc.depth_stencil;
    const auto depthLoad = enumIn(depth.depth_load_op, LoadOp::DontCare, "load op", error);
    const auto depthStore = enumIn(depth.depth_store_op, StoreOp::DontCare, "store op", error);
    const auto stencilLoad = enumIn(depth.stencil_load_op, LoadOp::DontCare, "load op", error);
    const auto stencilStore = enumIn(depth.stencil_store_op, StoreOp::DontCare, "store op", error);
    const auto depthResolve = enumIn(depth.depth_resolve, ResolveMode::Max, "resolve mode", error);
    const auto stencilResolve = enumIn(depth.stencil_resolve, ResolveMode::Max, "resolve mode", error);
    if (!depthLoad || !depthStore || !stencilLoad || !stencilStore || !depthResolve || !stencilResolve) {
        return std::nullopt;
    }
    rendering.depthStencil = {.texture = fromC(depth.texture),
                              .depthLoadOp = *depthLoad,
                              .depthStoreOp = *depthStore,
                              .stencilLoadOp = *stencilLoad,
                              .stencilStoreOp = *stencilStore,
                              .clearDepth = depth.clear_depth,
                              .clearStencil = depth.clear_stencil,
                              .readOnly = depth.read_only != BFR_FALSE,
                              .resolveTexture = fromC(depth.resolve_texture),
                              .depthResolve = *depthResolve,
                              .stencilResolve = *stencilResolve};
    return rendering;
}

// The window in window, as the library keeps it: monostate for all zero,
// which means no window. Fails for a window system this platform lacks, or a
// window system with no window.
std::optional<NativeWindow> windowIn(const bfr_window& window, bfr::core::Message& error) {
    switch (window.system) {
        case BFR_WINDOW_SYSTEM_HEADLESS:
            return bfr::core::window::Headless{};
        case BFR_WINDOW_SYSTEM_DEFAULT:
            if (window.window == nullptr && window.display == nullptr) {
                return NativeWindow{};
            }
            break;
        case BFR_WINDOW_SYSTEM_WAYLAND:
#if defined(__linux__)
            if (window.window != nullptr) {
                return bfr::core::window::Wayland{.display = window.display, .surface = window.window};
            }
#else
            error = "this platform has no Wayland";
            return std::nullopt;
#endif
            break;
        case BFR_WINDOW_SYSTEM_FORCE32:
        default:
            error.format("window system {} is not one BFR has", static_cast<std::int64_t>(window.system));
            return std::nullopt;
    }
    if (window.window == nullptr) {
        error = "a window needs its window value set";
        return std::nullopt;
    }
#if defined(_WIN32)
    return bfr::core::window::Win32{.instance = window.display, .window = window.window};
#elif defined(__APPLE__)
    return bfr::core::window::Metal{.layer = window.window};
#else
    // An X11 Window is an XID, carried in the pointer.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto xid = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(window.window));
    return bfr::core::window::Xlib{.display = window.display, .window = xid};
#endif
}

bfr_backend toC(bfr::rhi::Backend backend) {
    switch (backend) {
        case bfr::rhi::Backend::Vulkan:
            return BFR_BACKEND_VULKAN;
        case bfr::rhi::Backend::D3D12:
            return BFR_BACKEND_D3D12;
        case bfr::rhi::Backend::Metal:
            return BFR_BACKEND_METAL;
    }
    return BFR_BACKEND_FORCE32;
}

std::optional<bfr::rhi::Backend> backendIn(bfr_backend backend) {
    switch (backend) {
        case BFR_BACKEND_VULKAN:
            return bfr::rhi::Backend::Vulkan;
        case BFR_BACKEND_D3D12:
            return bfr::rhi::Backend::D3D12;
        case BFR_BACKEND_METAL:
            return bfr::rhi::Backend::Metal;
        case BFR_BACKEND_FORCE32:
        default:
            return std::nullopt;
    }
}

bfr_adapter_type toC(bfr::rhi::AdapterType type) {
    switch (type) {
        case bfr::rhi::AdapterType::Discrete:
            return BFR_ADAPTER_TYPE_DISCRETE;
        case bfr::rhi::AdapterType::Integrated:
            return BFR_ADAPTER_TYPE_INTEGRATED;
        case bfr::rhi::AdapterType::Virtual:
            return BFR_ADAPTER_TYPE_VIRTUAL;
        case bfr::rhi::AdapterType::Cpu:
            return BFR_ADAPTER_TYPE_CPU;
        case bfr::rhi::AdapterType::Other:
            break;
    }
    return BFR_ADAPTER_TYPE_OTHER;
}

bfr_string toC(const std::string& text) {
    return {.data = text.data(), .length = text.size()};
}

bfr_bool toC(bool value) {
    return static_cast<bfr_bool>(value);
}

// The C and C++ enums below are the same numbers.
static_assert(static_cast<int>(bfr::rhi::Format::R8Unorm) == BFR_FORMAT_R8_UNORM);
static_assert(static_cast<int>(bfr::rhi::Format::Astc12x12Float) == BFR_FORMAT_ASTC_12X12_FLOAT);
static_assert(static_cast<int>(bfr::rhi::ColorSpace::DisplayNative) == BFR_COLOR_SPACE_DISPLAY_NATIVE);
static_assert(static_cast<int>(bfr::rhi::DisplayRotation::Other) == BFR_DISPLAY_ROTATION_OTHER);

bfr_format toC(bfr::rhi::Format format) {
    return static_cast<bfr_format>(format);
}

bfr_color_space toC(bfr::rhi::ColorSpace colorSpace) {
    return static_cast<bfr_color_space>(colorSpace);
}

bfr_present_mode toC(bfr::rhi::PresentMode mode) {
    return static_cast<bfr_present_mode>(mode);
}

bfr_alpha_mode toC(bfr::rhi::AlphaMode mode) {
    return static_cast<bfr_alpha_mode>(mode);
}

static_assert(static_cast<int>(bfr::rhi::PresentMode::Immediate) == BFR_PRESENT_MODE_IMMEDIATE);
static_assert(static_cast<int>(bfr::rhi::AlphaMode::Inherit) == BFR_ALPHA_MODE_INHERIT);

bfr_swapchain_images toC(const bfr::rhi::SwapchainImages& images) {
    return {.width = images.width,
            .height = images.height,
            .count = images.count,
            .format = toC(images.format),
            .color_space = toC(images.colorSpace),
            .present_mode = toC(images.presentMode),
            .alpha_mode = toC(images.alphaMode)};
}

bfr_caps_info toC(const bfr::rhi::Adapter& adapter, bfr_backend backend) {
    const bfr::rhi::AdapterInfo& info = adapter.caps.info;
    bfr_caps_info out{
        .name = toC(adapter.name),
        .type = toC(adapter.type),
        .backend = backend,
        .vendor_id = info.vendorId,
        .device_id = info.deviceId,
        .driver_name = toC(info.driverName),
        .driver_info = toC(info.driverInfo),
        .driver_version = info.driverVersion,
        .api_major = info.apiMajor,
        .api_minor = info.apiMinor,
        .api_patch = info.apiPatch,
        .device_uuid = {},
        .driver_uuid = {},
        .luid = {},
        .luid_valid = toC(info.luidValid),
        .cache_uuid = {},
    };
    std::ranges::copy(info.deviceUuid, std::begin(out.device_uuid));
    std::ranges::copy(info.driverUuid, std::begin(out.driver_uuid));
    std::ranges::copy(info.luid, std::begin(out.luid));
    std::ranges::copy(info.cacheUuid, std::begin(out.cache_uuid));
    return out;
}

bfr_caps_limits toC(const bfr::rhi::Limits& limits) {
    bfr_caps_limits out{
        .max_texture_1d = limits.maxTexture1d,
        .max_texture_2d = limits.maxTexture2d,
        .max_texture_3d = limits.maxTexture3d,
        .max_texture_cube = limits.maxTextureCube,
        .max_texture_layers = limits.maxTextureLayers,
        .max_buffer_size = limits.maxBufferSize,
        .max_allocation_size = limits.maxAllocationSize,
        .max_allocations = limits.maxAllocations,
        .max_sampler_objects = limits.maxSamplerObjects,
        .max_uniform_buffer_range = limits.maxUniformBufferRange,
        .max_storage_buffer_range = limits.maxStorageBufferRange,
        .max_texel_buffer_elements = limits.maxTexelBufferElements,
        .max_push_constants_size = limits.maxPushConstantsSize,
        .max_color_attachments = limits.maxColorAttachments,
        .max_render_width = limits.maxRenderWidth,
        .max_render_height = limits.maxRenderHeight,
        .max_render_layers = limits.maxRenderLayers,
        .color_sample_counts = limits.colorSampleCounts,
        .integer_sample_counts = limits.integerSampleCounts,
        .depth_sample_counts = limits.depthSampleCounts,
        .stencil_sample_counts = limits.stencilSampleCounts,
        .no_attachment_sample_counts = limits.noAttachmentSampleCounts,
        .storage_texture_sample_counts = limits.storageTextureSampleCounts,
        .max_viewports = limits.maxViewports,
        .max_viewport_width = limits.maxViewportWidth,
        .max_viewport_height = limits.maxViewportHeight,
        .max_draw_indirect_count = limits.maxDrawIndirectCount,
        .max_clip_distances = limits.maxClipDistances,
        .max_cull_distances = limits.maxCullDistances,
        .max_sampler_anisotropy = limits.maxSamplerAnisotropy,
        .max_sampler_lod_bias = limits.maxSamplerLodBias,
        .max_custom_border_color_samplers = limits.maxCustomBorderColorSamplers,
        .min_shading_rate_tile = {},
        .max_shading_rate_tile = {},
        .min_uniform_buffer_alignment = limits.minUniformBufferAlignment,
        .min_storage_buffer_alignment = limits.minStorageBufferAlignment,
        .min_texel_buffer_alignment = limits.minTexelBufferAlignment,
        .copy_offset_alignment = limits.copyOffsetAlignment,
        .copy_row_pitch_alignment = limits.copyRowPitchAlignment,
        .non_coherent_atom_size = limits.nonCoherentAtomSize,
        .buffer_texture_granularity = limits.bufferTextureGranularity,
    };
    std::ranges::copy(limits.minShadingRateTile, std::begin(out.min_shading_rate_tile));
    std::ranges::copy(limits.maxShadingRateTile, std::begin(out.max_shading_rate_tile));
    return out;
}

bfr_caps_bindless toC(const bfr::rhi::BindlessLimits& bindless) {
    return {
        .max_textures = bindless.maxTextures,
        .max_storage_textures = bindless.maxStorageTextures,
        .max_storage_buffers = bindless.maxStorageBuffers,
        .max_uniform_buffers = bindless.maxUniformBuffers,
        .max_samplers = bindless.maxSamplers,
        .max_per_stage = bindless.maxPerStage,
        .max_acceleration_structures = bindless.maxAccelerationStructures,
    };
}

bfr_caps_compute toC(const bfr::rhi::ComputeLimits& compute) {
    bfr_caps_compute out{
        .max_workgroup_count = {},
        .max_workgroup_size = {},
        .max_workgroup_invocations = compute.maxWorkgroupInvocations,
        .max_shared_memory = compute.maxSharedMemory,
        .subgroup_size = compute.subgroupSize,
        .min_subgroup_size = compute.minSubgroupSize,
        .max_subgroup_size = compute.maxSubgroupSize,
    };
    std::ranges::copy(compute.maxWorkgroupCount, std::begin(out.max_workgroup_count));
    std::ranges::copy(compute.maxWorkgroupSize, std::begin(out.max_workgroup_size));
    return out;
}

bfr_caps_mesh toC(const bfr::rhi::MeshLimits& mesh) {
    bfr_caps_mesh out{
        .max_mesh_workgroup_size = {},
        .max_mesh_invocations = mesh.maxMeshInvocations,
        .max_mesh_output_vertices = mesh.maxMeshOutputVertices,
        .max_mesh_output_primitives = mesh.maxMeshOutputPrimitives,
        .max_mesh_shared_memory = mesh.maxMeshSharedMemory,
        .max_task_workgroup_size = {},
        .max_task_invocations = mesh.maxTaskInvocations,
        .max_task_payload_size = mesh.maxTaskPayloadSize,
        .max_task_workgroup_count = {},
        .max_task_workgroup_total_count = mesh.maxTaskWorkgroupTotalCount,
        .max_task_shared_memory = mesh.maxTaskSharedMemory,
        .max_task_payload_and_shared_memory = mesh.maxTaskPayloadAndSharedMemory,
        .max_mesh_workgroup_count = {},
        .max_mesh_workgroup_total_count = mesh.maxMeshWorkgroupTotalCount,
        .max_mesh_output_memory = mesh.maxMeshOutputMemory,
        .mesh_output_per_vertex_granularity = mesh.meshOutputPerVertexGranularity,
        .mesh_output_per_primitive_granularity = mesh.meshOutputPerPrimitiveGranularity,
        .max_mesh_payload_and_shared_memory = mesh.maxMeshPayloadAndSharedMemory,
        .max_mesh_payload_and_output_memory = mesh.maxMeshPayloadAndOutputMemory,
        .max_mesh_output_components = mesh.maxMeshOutputComponents,
        .max_mesh_output_layers = mesh.maxMeshOutputLayers,
        .max_mesh_multiview_views = mesh.maxMeshMultiviewViews,
    };
    std::ranges::copy(mesh.maxMeshWorkgroupSize, std::begin(out.max_mesh_workgroup_size));
    std::ranges::copy(mesh.maxTaskWorkgroupSize, std::begin(out.max_task_workgroup_size));
    std::ranges::copy(mesh.maxTaskWorkgroupCount, std::begin(out.max_task_workgroup_count));
    std::ranges::copy(mesh.maxMeshWorkgroupCount, std::begin(out.max_mesh_workgroup_count));
    return out;
}

bfr_caps_ray_tracing toC(const bfr::rhi::RayTracingLimits& rayTracing) {
    return {
        .max_geometries = rayTracing.maxGeometries,
        .max_instances = rayTracing.maxInstances,
        .max_primitives = rayTracing.maxPrimitives,
        .scratch_alignment = rayTracing.scratchAlignment,
        .max_recursion_depth = rayTracing.maxRecursionDepth,
        .identifier_size = rayTracing.identifierSize,
        .identifier_alignment = rayTracing.identifierAlignment,
        .table_alignment = rayTracing.tableAlignment,
        .max_shader_group_stride = rayTracing.maxShaderGroupStride,
        .max_trace_rays = rayTracing.maxTraceRays,
        .max_hit_attribute_size = rayTracing.maxHitAttributeSize,
        .instance_size = rayTracing.instanceSize,
    };
}

bfr_caps_features toC(const bfr::rhi::Features& features) {
    return {
        .ray_query = toC(features.rayQuery),
        .ray_query_graphics = toC(features.rayQueryGraphics),
        .ray_tracing_pipeline = toC(features.rayTracingPipeline),
        .trace_rays_indirect = toC(features.traceRaysIndirect),
        .ray_primitive_culling = toC(features.rayPrimitiveCulling),
        .acceleration_structure = toC(features.accelerationStructure),
        .buffer_address = toC(features.bufferAddress),
        .mesh_shader = toC(features.meshShader),
        .task_shader = toC(features.taskShader),
        .shading_rate = toC(features.shadingRate),
        .shading_rate_texture = toC(features.shadingRateTexture),
        .shader_float16 = toC(features.shaderFloat16),
        .shader_float64 = toC(features.shaderFloat64),
        .shader_int8 = toC(features.shaderInt8),
        .shader_int16 = toC(features.shaderInt16),
        .shader_int64 = toC(features.shaderInt64),
        .atomic_int64 = toC(features.atomicInt64),
        .atomic_int64_texture = toC(features.atomicInt64Texture),
        .atomic_float32_add = toC(features.atomicFloat32Add),
        .barycentrics = toC(features.barycentrics),
        .conservative_raster = toC(features.conservativeRaster),
        .geometry_shader = toC(features.geometryShader),
        .tessellation = toC(features.tessellation),
        .multiview = toC(features.multiview),
        .sample_rate_shading = toC(features.sampleRateShading),
        .dual_source_blend = toC(features.dualSourceBlend),
        .independent_blend = toC(features.independentBlend),
        .logic_op = toC(features.logicOp),
        .depth_clamp = toC(features.depthClamp),
        .depth_bounds = toC(features.depthBounds),
        .wireframe = toC(features.wireframe),
        .wide_lines = toC(features.wideLines),
        .sampler_anisotropy = toC(features.samplerAnisotropy),
        .sampler_minmax = toC(features.samplerMinmax),
        .sampler_mirror_clamp_to_edge = toC(features.samplerMirrorClampToEdge),
        .sampler_anisotropy_nearest_mip = toC(features.samplerAnisotropyNearestMip),
        .sampler_anisotropy_nearest_filter = toC(features.samplerAnisotropyNearestFilter),
        .sampler_custom_border_color = toC(features.samplerCustomBorderColor),
        .sampler_integer_border_color = toC(features.samplerIntegerBorderColor),
        .sampler_unnormalized_coordinates = toC(features.samplerUnnormalizedCoordinates),
        .sampler_lod_bias = toC(features.samplerLodBias),
        .sampler_lod_average = toC(features.samplerLodAverage),
        .compression_bc = toC(features.compressionBc),
        .compression_etc2 = toC(features.compressionEtc2),
        .compression_astc = toC(features.compressionAstc),
        .compression_astc_hdr = toC(features.compressionAstcHdr),
        .multi_draw_indirect = toC(features.multiDrawIndirect),
        .draw_indirect_count = toC(features.drawIndirectCount),
        .draw_indirect_first_instance = toC(features.drawIndirectFirstInstance),
        .triangle_fans = toC(features.triangleFans),
        .list_restart = toC(features.listRestart),
        .dynamic_topology = toC(features.dynamicTopology),
        .mesh_multiview = toC(features.meshMultiview),
        .mesh_statistics = toC(features.meshStatistics),
        .alpha_to_one = toC(features.alphaToOne),
        .sample_mask = toC(features.sampleMask),
        .pipeline_cache = toC(features.pipelineCache),
        .storage_read_without_format = toC(features.storageReadWithoutFormat),
        .storage_write_without_format = toC(features.storageWriteWithoutFormat),
        .storage_texture_non_uniform_indexing = toC(features.storageTextureNonUniformIndexing),
        .timestamps_on_all_queues = toC(features.timestampsOnAllQueues),
        .pipeline_statistics = toC(features.pipelineStatistics),
        .occlusion_query_precise = toC(features.occlusionQueryPrecise),
    };
}

// Up to the C arrays' lengths, which are every backend's most.
bfr_caps_memory toC(const bfr::rhi::Memory& memory) {
    bfr_caps_memory out{.heap_count = 0, .heaps = {}, .type_count = 0, .types = {}, .unified = toC(memory.unified)};
    const std::span heaps{out.heaps};
    for (const bfr::rhi::MemoryHeap& heap : memory.heaps | std::views::take(heaps.size())) {
        heaps[out.heap_count++] = {.size = heap.size,
                                   .device_local = toC(heap.deviceLocal),
                                   .host_visible = toC(heap.hostVisible)};
    }
    const std::span types{out.types};
    for (const bfr::rhi::MemoryType& type : memory.types | std::views::take(types.size())) {
        types[out.type_count++] = {
            .heap = type.heap,
            .device_local = toC(type.deviceLocal),
            .host_visible = toC(type.hostVisible),
            .host_coherent = toC(type.hostCoherent),
            .host_cached = toC(type.hostCached),
            .lazily_allocated = toC(type.lazilyAllocated),
        };
    }
    return out;
}

bfr_queue_levels toC(const bfr::rhi::QueueLevels& levels) {
    return {
        .normal = toC(levels.normal),
        .low = toC(levels.low),
        .high = toC(levels.high),
        .realtime = toC(levels.realtime),
    };
}

bfr_timestamp_stages toC(const bfr::rhi::TimestampStages& stages) {
    return {.end = toC(stages.end),
            .start = toC(stages.start),
            .vertex = toC(stages.vertex),
            .fragment = toC(stages.fragment),
            .compute = toC(stages.compute),
            .copy = toC(stages.copy),
            .task = toC(stages.task),
            .mesh = toC(stages.mesh),
            .ray_tracing = toC(stages.rayTracing),
            .acceleration_structure_build = toC(stages.accelerationStructureBuild)};
}

bfr_caps_timestamps toC(const bfr::rhi::Timestamps& timestamps) {
    const bfr::rhi::HostClocks& clocks = timestamps.hostClocks;
    return {.graphics = toC(timestamps.graphics),
            .compute = toC(timestamps.compute),
            .copy = toC(timestamps.copy),
            .outside_rendering = toC(timestamps.outsideRendering),
            .inside_rendering = toC(timestamps.insideRendering),
            .compute_stages = toC(timestamps.computeStages),
            .copy_stages = toC(timestamps.copyStages),
            .graphics_resets = toC(timestamps.graphicsResets),
            .compute_resets = toC(timestamps.computeResets),
            .copy_resets = toC(timestamps.copyResets),
            .granularity = toC(timestamps.granularity),
            .entry_bytes = timestamps.entryBytes,
            .host_clocks = {.monotonic = toC(clocks.monotonic),
                            .monotonic_raw = toC(clocks.monotonicRaw),
                            .performance_counter = toC(clocks.performanceCounter)}};
}

// Writes timings as a bfr_get_*_call_timings listing does.
bfr_result listTimings(std::span<const bfr::rhi::CallTiming> timings, std::uint32_t* count, bfr_call_timing* out) {
    const auto total = static_cast<std::uint32_t>(timings.size());
    if (out == nullptr) {
        *count = total;
        return BFR_RESULT_SUCCESS;
    }
    const std::uint32_t written = std::min(*count, total);
    for (std::uint32_t i = 0; i < written; ++i) {
        out[i] = {.count = timings[i].count, .total_ns = timings[i].totalNs, .max_ns = timings[i].maxNs};
    }
    *count = written;
    return written < total ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS;
}

bfr_copy_granularity toC(const bfr::rhi::CopyGranularity& granularity) {
    return {.width = granularity.width,
            .height = granularity.height,
            .depth = granularity.depth,
            .whole_levels = toC(granularity.wholeLevels)};
}

bfr_caps_queues toC(const bfr::rhi::Queues& queues) {
    return {
        .graphics = queues.graphics,
        .compute = queues.compute,
        .copy = queues.copy,
        .graphics_timestamp_bits = queues.graphicsTimestampBits,
        .compute_timestamp_bits = queues.computeTimestampBits,
        .copy_timestamp_bits = queues.copyTimestampBits,
        .ranks = queues.ranks,
        .graphics_levels = toC(queues.graphicsLevels),
        .compute_levels = toC(queues.computeLevels),
        .copy_levels = toC(queues.copyLevels),
        .graphics_sparse = toC(queues.graphicsSparse),
        .compute_sparse = toC(queues.computeSparse),
        .copy_sparse = toC(queues.copySparse),
        .graphics_granularity = toC(queues.graphicsGranularity),
        .compute_granularity = toC(queues.computeGranularity),
        .copy_granularity = toC(queues.copyGranularity),
    };
}

bfr_caps_sparse toC(const bfr::rhi::Sparse& sparse) {
    bfr_caps_sparse out{
        .binding = toC(sparse.binding),
        .buffers = toC(sparse.buffers),
        .textures_2d = toC(sparse.textures2D),
        .textures_3d = toC(sparse.textures3D),
        .samples_2 = toC(sparse.samples2),
        .samples_4 = toC(sparse.samples4),
        .samples_8 = toC(sparse.samples8),
        .samples_16 = toC(sparse.samples16),
        .aliased = toC(sparse.aliased),
        .standard_2d_shape = toC(sparse.standard2DShape),
        .standard_2d_multisample_shape = toC(sparse.standard2DMultisampleShape),
        .standard_3d_shape = toC(sparse.standard3DShape),
        .aligned_mip_size = toC(sparse.alignedMipSize),
        .non_resident_strict = toC(sparse.nonResidentStrict),
        .shader_residency = toC(sparse.shaderResidency),
        .mixed_resources = toC(sparse.mixedResources),
        .layer_tails = toC(sparse.layerTails),
        .address_space_bytes = sparse.addressSpaceBytes,
        .address_bits_per_resource = sparse.addressBitsPerResource,
        .address_bits_per_process = sparse.addressBitsPerProcess,
        .tile_byte_count = 0,
        .tile_bytes = {},
    };
    // Each backend lists at most kMaxSparseTileSizes.
    const std::size_t count = std::min(sparse.tileBytes.size(), bfr::rhi::kMaxSparseTileSizes);
    std::ranges::copy_n(sparse.tileBytes.begin(), static_cast<std::ptrdiff_t>(count), std::begin(out.tile_bytes));
    out.tile_byte_count = static_cast<std::uint32_t>(count);
    return out;
}

bfr_caps_resolve toC(const bfr::rhi::ResolveCaps& resolve) {
    return {.depth_modes = resolve.depthModes,
            .stencil_modes = resolve.stencilModes,
            .independent = toC(resolve.independent),
            .independent_none = toC(resolve.independentNone)};
}

bfr_format_caps toC(const bfr::rhi::FormatCaps& caps) {
    return {
        .format = toC(caps.format),
        .sampled = toC(caps.sampled),
        .filter_linear = toC(caps.filterLinear),
        .filter_minmax = toC(caps.filterMinmax),
        .storage = toC(caps.storage),
        .storage_atomic = toC(caps.storageAtomic),
        .color_attachment = toC(caps.colorAttachment),
        .blend = toC(caps.blend),
        .depth_stencil = toC(caps.depthStencil),
        .copy_src = toC(caps.copySrc),
        .copy_dst = toC(caps.copyDst),
        .texel_buffer = toC(caps.texelBuffer),
        .storage_texel_buffer = toC(caps.storageTexelBuffer),
        .storage_texel_buffer_atomic = toC(caps.storageTexelBufferAtomic),
        .sample_counts = caps.sampleCounts,
        .resolve = toC(caps.resolve),
        .acceleration_structure_vertices = toC(caps.accelerationStructureVertices),
    };
}

bfr_swapchain_caps toC(const bfr::rhi::SwapchainCaps& caps) {
    return {
        .min_image_count = caps.minImageCount,
        .max_image_count = caps.maxImageCount,
        .width = caps.width,
        .height = caps.height,
        .min_width = caps.minWidth,
        .min_height = caps.minHeight,
        .max_width = caps.maxWidth,
        .max_height = caps.maxHeight,
        .rotation = static_cast<bfr_display_rotation>(caps.rotation),
        .fifo = toC(caps.fifo),
        .fifo_relaxed = toC(caps.fifoRelaxed),
        .mailbox = toC(caps.mailbox),
        .immediate = toC(caps.immediate),
        .alpha_opaque = toC(caps.alphaOpaque),
        .alpha_premultiplied = toC(caps.alphaPremultiplied),
        .alpha_postmultiplied = toC(caps.alphaPostmultiplied),
        .alpha_inherit = toC(caps.alphaInherit),
    };
}

bfr_swapchain_format toC(const bfr::rhi::SwapchainFormat& format) {
    return {.format = toC(format.format), .color_space = static_cast<bfr_color_space>(format.colorSpace)};
}

// Fails unless adapter is an index into instance's adapters.
std::optional<bfr_result> checkAdapter(const bfr_instance& instance, std::uint32_t adapter) {
    const std::size_t count = instance.instance->adapters().size();
    if (adapter >= count) {
        return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "adapter {} of {}", adapter, count);
    }
    return std::nullopt;
}

// unusable one.
std::optional<NativeWindow> surfaceWindow(const bfr_swapchain_query& desc, bfr_result& failed) {
    bfr::core::Message error;
    auto window = windowIn(desc.window, error);
    if (window && std::holds_alternative<std::monostate>(*window)) {
        window.reset();
        error = "bfr_swapchain_query needs a window";
    }
    if (!window) {
        failed = fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
    }
    return window;
}

// Writes up to *count of items into out as C values, sets *count to how many
// it wrote, and returns incomplete if there were more.
template <class Item, class Out>
bfr_result writeList(std::span<const Item> items, std::uint32_t* count, Out* out) {
    const auto total = static_cast<std::uint32_t>(items.size());
    const std::uint32_t written = std::min(*count, total);
    std::ranges::transform(items.first(written), out, [](const Item& item) { return toC(item); });
    *count = written;
    return written < total ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS;
}

} // namespace

extern "C" {

bfr_string bfr_get_last_error(void) {
    const std::string_view error = lastError().view();
    return {.data = error.data(), .length = error.size()};
}

bfr_result bfr_get_backends(uint32_t* count, bfr_backend* backends) {
    return guarded([&] {
        if (count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "count must not be null");
        }
        const auto listed = bfr::rhi::backends();
        const auto total = static_cast<std::uint32_t>(listed.size());
        if (backends == nullptr) {
            *count = total;
            return BFR_RESULT_SUCCESS;
        }
        const std::uint32_t written = std::min(*count, total);
        for (std::uint32_t i = 0; i < written; ++i) {
            backends[i] = toC(listed[i]);
        }
        *count = written;
        return written < total ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_create_instance(const bfr_instance_desc* desc, bfr_instance** instance) {
    return guarded([&] {
        if (desc == nullptr || instance == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "desc and instance must not be null");
        }
        *instance = nullptr;
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_instance_desc takes no chained structs");
        }
        const auto backend = backendIn(desc->backend);
        if (!backend) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "backend {} is not one BFR has",
                        static_cast<std::int64_t>(desc->backend));
        }
        if ((desc->application_name.data == nullptr && desc->application_name.length != 0) ||
            (desc->engine_name.data == nullptr && desc->engine_name.length != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "application_name or engine_name is null but its length is not 0");
        }
        bfr::rhi::InstanceDesc rhiDesc{.backend = *backend,
                                       .apiMajor = desc->api_major,
                                       .apiMinor = desc->api_minor,
                                       .applicationName = {desc->application_name.data, desc->application_name.length},
                                       .applicationVersion = desc->application_version,
                                       .engineName = {desc->engine_name.data, desc->engine_name.length},
                                       .engineVersion = desc->engine_version,
                                       .callTimings = desc->call_timings != BFR_FALSE};
        // The severity is read only with a callback, which is all it filters.
        if (desc->message_callback != nullptr) {
            const auto least = severityFrom(desc->min_severity);
            if (!least) {
                return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "min_severity {} is not a severity",
                            static_cast<std::int64_t>(desc->min_severity));
            }
            rhiDesc.logger = {[callback = desc->message_callback,
                               userData = desc->message_user_data](Severity severity, std::string_view message) {
                                  callback(static_cast<bfr_message_severity>(severity),
                                           {.data = message.data(), .length = message.size()}, userData);
                              },
                              *least};
        }
        auto rhi = bfr::rhi::Instance::create(rhiDesc,
                                              bfr::rhi::RhiConfig{.validation = desc->validation != BFR_FALSE});
        if (!rhi) {
            return fail(rhi.error());
        }
        // The caller owns it until bfr_destroy_instance.
        *instance = std::make_unique<bfr_instance>(bfr_instance{.instance = std::move(*rhi), .backend = desc->backend})
                        .release();
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_instance(bfr_instance* instance) {
    quietly([&] { std::unique_ptr<bfr_instance>{instance}.reset(); });
}

bfr_result bfr_get_adapters(bfr_instance* instance, uint32_t* count, bfr_adapter_info* adapters) {
    return guarded([&] {
        if (instance == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and count must not be null");
        }
        const std::span<const bfr::rhi::Adapter> listed = instance->instance->adapters();
        const auto total = static_cast<std::uint32_t>(listed.size());
        if (adapters == nullptr) {
            *count = total;
            return BFR_RESULT_SUCCESS;
        }
        const std::uint32_t written = std::min(*count, total);
        for (std::uint32_t i = 0; i < written; ++i) {
            const bfr::rhi::Adapter& adapter = listed[i];
            adapters[i] = {.name = {.data = adapter.name.data(), .length = adapter.name.size()},
                           .type = toC(adapter.type)};
        }
        *count = written;
        return written < total ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_create_device(bfr_instance* instance, const bfr_device_desc* desc, bfr_device** device) {
    return guarded([&] {
        if (instance == nullptr || desc == nullptr || device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance, desc and device must not be null");
        }
        *device = nullptr;
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_device_desc takes no chained structs");
        }
        bfr::core::Message error;
        const auto window = windowIn(desc->window, error);
        if (!window) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto graphics = queuesIn(desc->graphics_queues, desc->graphics_queue_count, "graphics_queues", error);
        if (!graphics) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto compute = queuesIn(desc->compute_queues, desc->compute_queue_count, "compute_queues", error);
        if (!compute) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto copy = queuesIn(desc->copy_queues, desc->copy_queue_count, "copy_queues", error);
        if (!copy) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        if (desc->indirect_strides == nullptr && desc->indirect_stride_count != 0) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "indirect_strides is null but indirect_stride_count is not 0");
        }
        auto created = instance->instance->createDevice(
            desc->adapter, *window,
            {.graphics = *graphics,
             .compute = *compute,
             .copy = *copy,
             .maxCommandLists = desc->max_command_lists,
             .maxCommandPools = desc->max_command_pools,
             .maxSwapchains = desc->max_swapchains,
             .maxBindlessTextures = desc->max_bindless_textures,
             .maxBindlessStorageTextures = desc->max_bindless_storage_textures,
             .maxBindlessStorageBuffers = desc->max_bindless_storage_buffers,
             .maxBindlessSamplers = desc->max_bindless_samplers,
             .maxTimestampPools = desc->max_timestamp_pools,
             .maxSparseMemories = desc->max_sparse_memories,
             .maxTileMappings = desc->max_tile_mappings,
             .maxBindlessAccelerationStructures = desc->max_bindless_acceleration_structures,
             .maxBuildGeometries = desc->max_build_geometries,
             .indirectStrides = {desc->indirect_strides, desc->indirect_stride_count}});
        if (!created) {
            return fail(created.error());
        }
        auto made = std::make_unique<bfr_device>();
        made->instance = instance->instance;
        made->device = std::move(*created);
        made->triangles.resize(desc->max_build_geometries);
        made->boxes.resize(desc->max_build_geometries);
        made->instanceSize = instance->instance->adapters()[desc->adapter].caps.rayTracing.instanceSize;
        // The caller owns it until bfr_destroy_device.
        *device = made.release();
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_device(bfr_device* device) {
    quietly([&] { std::unique_ptr<bfr_device>{device}.reset(); });
}

bfr_result bfr_create_swapchain(bfr_device* device, const bfr_swapchain_desc* desc, bfr_swapchain* swapchain) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || swapchain == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and swapchain must not be null");
        }
        *swapchain = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_swapchain_desc takes no chained structs");
        }
        bfr::core::Message error;
        const auto window = windowIn(desc->window, error);
        if (!window) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        if (std::holds_alternative<std::monostate>(*window)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_swapchain_desc needs a window");
        }
        const auto made = swapchainIn(*desc, error);
        if (!made) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto created = device->device->createSwapchain(*window, *made);
        if (!created) {
            return fail(created.error());
        }
        *swapchain = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_resize_swapchain(bfr_device* device, bfr_swapchain swapchain, uint32_t width, uint32_t height,
                                bfr_swapchain_images* images) {
    return guarded([&] {
        if (device == nullptr || images == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and images must not be null");
        }
        const auto resized = device->device->resizeSwapchain(fromC(swapchain), width, height);
        if (!resized) {
            return fail(resized.error());
        }
        *images = toC(*resized);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_swapchain(bfr_device* device, bfr_swapchain swapchain) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        // A stale handle is already gone, which is all this call promises; the
        // reason stays readable for anyone debugging.
        if (const auto destroyed = device->device->destroySwapchain(fromC(swapchain)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_acquire(bfr_device* device, bfr_swapchain swapchain, uint64_t timeout_ns, bfr_swapchain_status* status) {
    return guarded([&] {
        if (device == nullptr || status == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and status must not be null");
        }
        const auto acquired = device->device->acquire(fromC(swapchain), timeout_ns);
        if (!acquired) {
            return fail(acquired.error());
        }
        *status = static_cast<bfr_swapchain_status>(*acquired);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_present(bfr_device* device, bfr_swapchain swapchain, bfr_swapchain_status* status) {
    return guarded([&] {
        if (device == nullptr || status == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and status must not be null");
        }
        const auto presented = device->device->present(fromC(swapchain));
        if (!presented) {
            return fail(presented.error());
        }
        *status = static_cast<bfr_swapchain_status>(*presented);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_create_command_pool(bfr_device* device, const bfr_command_pool_desc* desc,
                                   bfr_command_pool* command_pool) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || command_pool == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and command_pool must not be null");
        }
        *command_pool = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_command_pool_desc takes no chained structs");
        }
        const auto created = device->device->createCommandPool(
            {.queue = desc->queue, .listCount = desc->list_count, .pushConstantUpdates = desc->push_constant_updates});
        if (!created) {
            return fail(created.error());
        }
        *command_pool = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_reset_command_pool(bfr_device* device, bfr_command_pool command_pool) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto reset = device->device->resetCommandPool(fromC(command_pool));
        return reset ? BFR_RESULT_SUCCESS : fail(reset.error());
    });
}

void bfr_destroy_command_pool(bfr_device* device, bfr_command_pool command_pool) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyCommandPool(fromC(command_pool)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_begin_commands(bfr_device* device, bfr_command_pool command_pool, bfr_command_list* command_list) {
    return guarded([&] {
        if (device == nullptr || command_list == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and command_list must not be null");
        }
        *command_list = {};
        const auto begun = device->device->beginCommands(fromC(command_pool));
        if (!begun) {
            return fail(begun.error());
        }
        *command_list = toC(*begun);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_begin_rendering(bfr_device* device, bfr_command_list command_list, bfr_swapchain swapchain,
                               const bfr_rendering_desc* desc) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and desc must not be null");
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_rendering_desc takes no chained structs");
        }
        bfr::core::Message error;
        const auto rendering = renderingIn(*desc, error);
        if (!rendering) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto begun = device->device->beginRendering(fromC(command_list), fromC(swapchain), *rendering);
        return begun ? BFR_RESULT_SUCCESS : fail(begun.error());
    });
}

bfr_result bfr_end_rendering(bfr_device* device, bfr_command_list command_list) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto ended = device->device->endRendering(fromC(command_list));
        return ended ? BFR_RESULT_SUCCESS : fail(ended.error());
    });
}

bfr_result bfr_set_viewport(bfr_device* device, bfr_command_list command_list, const bfr_viewport* viewport) {
    return guarded([&] {
        if (device == nullptr || viewport == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and viewport must not be null");
        }
        const auto set = device->device->setViewport(fromC(command_list), {.x = viewport->x,
                                                                           .y = viewport->y,
                                                                           .width = viewport->width,
                                                                           .height = viewport->height,
                                                                           .minDepth = viewport->min_depth,
                                                                           .maxDepth = viewport->max_depth});
        return set ? BFR_RESULT_SUCCESS : fail(set.error());
    });
}

bfr_result bfr_set_scissor(bfr_device* device, bfr_command_list command_list, const bfr_rect* scissor) {
    return guarded([&] {
        if (device == nullptr || scissor == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and scissor must not be null");
        }
        const auto set = device->device->setScissor(fromC(command_list), {.x = scissor->x,
                                                                          .y = scissor->y,
                                                                          .width = scissor->width,
                                                                          .height = scissor->height});
        return set ? BFR_RESULT_SUCCESS : fail(set.error());
    });
}

bfr_result bfr_set_blend_constants(bfr_device* device, bfr_command_list command_list, float red, float green,
                                   float blue, float alpha) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto set = device->device->setBlendConstants(fromC(command_list), {red, green, blue, alpha});
        return set ? BFR_RESULT_SUCCESS : fail(set.error());
    });
}

bfr_result bfr_set_stencil_reference(bfr_device* device, bfr_command_list command_list, uint32_t reference) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto set = device->device->setStencilReference(fromC(command_list), reference);
        return set ? BFR_RESULT_SUCCESS : fail(set.error());
    });
}

bfr_result bfr_bind_pipeline(bfr_device* device, bfr_command_list command_list, bfr_pipeline pipeline) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto bound = device->device->bindPipeline(fromC(command_list), fromC(pipeline));
        return bound ? BFR_RESULT_SUCCESS : fail(bound.error());
    });
}

bfr_result bfr_push_constants(bfr_device* device, bfr_command_list command_list, const uint8_t* data, uint32_t size) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        if (size != 0 && data == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "data must not be null when size is not 0");
        }
        const auto pushed = device->device->pushConstants(fromC(command_list),
                                                          std::as_bytes(std::span<const std::uint8_t>{data, size}));
        return pushed ? BFR_RESULT_SUCCESS : fail(pushed.error());
    });
}

bfr_result bfr_bind_index_buffer(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer, uint64_t offset,
                                 bfr_index_type type) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        bfr::core::Message error;
        const auto indexType = enumIn(type, bfr::rhi::IndexType::Uint16, "index type", error);
        if (!indexType) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto bound = device->device->bindIndexBuffer(fromC(command_list), fromC(buffer), offset, *indexType);
        return bound ? BFR_RESULT_SUCCESS : fail(bound.error());
    });
}

bfr_result bfr_draw(bfr_device* device, bfr_command_list command_list, uint32_t vertex_count, uint32_t instance_count,
                    uint32_t first_vertex, uint32_t first_instance) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->draw(fromC(command_list), vertex_count, instance_count, first_vertex,
                                                first_instance);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_draw_indexed(bfr_device* device, bfr_command_list command_list, uint32_t index_count,
                            uint32_t instance_count, uint32_t first_index, int32_t vertex_offset,
                            uint32_t first_instance) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawIndexed(fromC(command_list), index_count, instance_count, first_index,
                                                       vertex_offset, first_instance);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_dispatch(bfr_device* device, bfr_command_list command_list, uint32_t x, uint32_t y, uint32_t z) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto dispatched = device->device->dispatch(fromC(command_list), x, y, z);
        return dispatched ? BFR_RESULT_SUCCESS : fail(dispatched.error());
    });
}

// The indirect arguments cross as they are: the C and C++ layouts are one.
static_assert(sizeof(bfr_draw_indirect_args) == sizeof(bfr::rhi::DrawIndirectArgs));
static_assert(sizeof(bfr_draw_indexed_indirect_args) == sizeof(bfr::rhi::DrawIndexedIndirectArgs));
static_assert(sizeof(bfr_dispatch_indirect_args) == sizeof(bfr::rhi::DispatchIndirectArgs));

bfr_result bfr_set_topology(bfr_device* device, bfr_command_list command_list, bfr_topology topology) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        bfr::core::Message error;
        const auto known = enumIn(topology, bfr::rhi::Topology::PointList, "topology", error);
        if (!known) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto set = device->device->setTopology(fromC(command_list), *known);
        return set ? BFR_RESULT_SUCCESS : fail(set.error());
    });
}

bfr_result bfr_draw_indirect(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer, uint64_t offset,
                             uint32_t draw_count, uint32_t stride) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawIndirect(fromC(command_list), fromC(buffer), offset, draw_count, stride);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_draw_indexed_indirect(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                     uint64_t offset, uint32_t draw_count, uint32_t stride) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawIndexedIndirect(fromC(command_list), fromC(buffer), offset, draw_count,
                                                               stride);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_draw_indirect_count(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                   uint64_t offset, bfr_buffer count_buffer, uint64_t count_offset,
                                   uint32_t max_draw_count, uint32_t stride) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawIndirectCount(fromC(command_list), fromC(buffer), offset,
                                                             fromC(count_buffer), count_offset, max_draw_count, stride);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_draw_indexed_indirect_count(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                           uint64_t offset, bfr_buffer count_buffer, uint64_t count_offset,
                                           uint32_t max_draw_count, uint32_t stride) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawIndexedIndirectCount(fromC(command_list), fromC(buffer), offset,
                                                                    fromC(count_buffer), count_offset, max_draw_count,
                                                                    stride);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_dispatch_indirect(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                 uint64_t offset) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto dispatched = device->device->dispatchIndirect(fromC(command_list), fromC(buffer), offset);
        return dispatched ? BFR_RESULT_SUCCESS : fail(dispatched.error());
    });
}

bfr_result bfr_draw_mesh(bfr_device* device, bfr_command_list command_list, uint32_t x, uint32_t y, uint32_t z) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawMesh(fromC(command_list), x, y, z);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_draw_mesh_indirect(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer, uint64_t offset,
                                  uint32_t draw_count, uint32_t stride) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawMeshIndirect(fromC(command_list), fromC(buffer), offset, draw_count,
                                                            stride);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_draw_mesh_indirect_count(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                        uint64_t offset, bfr_buffer count_buffer, uint64_t count_offset,
                                        uint32_t max_draw_count, uint32_t stride) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto drawn = device->device->drawMeshIndirectCount(fromC(command_list), fromC(buffer), offset,
                                                                 fromC(count_buffer), count_offset, max_draw_count,
                                                                 stride);
        return drawn ? BFR_RESULT_SUCCESS : fail(drawn.error());
    });
}

bfr_result bfr_copy_buffer(bfr_device* device, bfr_command_list command_list, bfr_buffer src, bfr_buffer dst,
                           const bfr_buffer_copy* regions, uint32_t region_count) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        if (region_count != 0 && regions == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "regions must not be null when region_count is not 0");
        }
        const std::span<const bfr::rhi::BufferCopy> copies{reinterpret_cast<const bfr::rhi::BufferCopy*>(regions),
                                                           region_count};
        const auto copied = device->device->copyBuffer(fromC(command_list), fromC(src), fromC(dst), copies);
        return copied ? BFR_RESULT_SUCCESS : fail(copied.error());
    });
}

bfr_result bfr_barrier(bfr_device* device, bfr_command_list command_list) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto recorded = device->device->barrier(fromC(command_list));
        return recorded ? BFR_RESULT_SUCCESS : fail(recorded.error());
    });
}

bfr_result bfr_texture_barrier(bfr_device* device, bfr_command_list command_list, bfr_texture texture,
                               bfr_texture_layout before, bfr_texture_layout after) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        for (const bfr_texture_layout layout : {before, after}) {
            if (layout < BFR_TEXTURE_LAYOUT_UNDEFINED || layout > BFR_TEXTURE_LAYOUT_RESOLVE_DST) {
                return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "texture layout {} is not one BFR has",
                            static_cast<std::int64_t>(layout));
            }
        }
        const auto recorded = device->device->textureBarrier(fromC(command_list), fromC(texture),
                                                             static_cast<bfr::rhi::TextureLayout>(before),
                                                             static_cast<bfr::rhi::TextureLayout>(after));
        return recorded ? BFR_RESULT_SUCCESS : fail(recorded.error());
    });
}

bfr_result bfr_copy_buffer_to_texture(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                      bfr_texture texture, const bfr_buffer_texture_copy* regions,
                                      uint32_t region_count) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        if (region_count != 0 && regions == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "regions must not be null when region_count is not 0");
        }
        if (const bfr_result checked = checkAspects({regions, region_count}); checked < 0) {
            return checked;
        }
        const std::span copies{reinterpret_cast<const bfr::rhi::BufferTextureCopy*>(regions), region_count};
        const auto copied = device->device->copyBufferToTexture(fromC(command_list), fromC(buffer), fromC(texture),
                                                                copies);
        return copied ? BFR_RESULT_SUCCESS : fail(copied.error());
    });
}

bfr_result bfr_copy_texture_to_buffer(bfr_device* device, bfr_command_list command_list, bfr_texture texture,
                                      bfr_buffer buffer, const bfr_buffer_texture_copy* regions,
                                      uint32_t region_count) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        if (region_count != 0 && regions == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "regions must not be null when region_count is not 0");
        }
        if (const bfr_result checked = checkAspects({regions, region_count}); checked < 0) {
            return checked;
        }
        const std::span copies{reinterpret_cast<const bfr::rhi::BufferTextureCopy*>(regions), region_count};
        const auto copied = device->device->copyTextureToBuffer(fromC(command_list), fromC(texture), fromC(buffer),
                                                                copies);
        return copied ? BFR_RESULT_SUCCESS : fail(copied.error());
    });
}

bfr_result bfr_copy_texture(bfr_device* device, bfr_command_list command_list, bfr_texture src, bfr_texture dst,
                            const bfr_texture_copy* regions, uint32_t region_count) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        if (region_count != 0 && regions == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "regions must not be null when region_count is not 0");
        }
        const std::span copies{reinterpret_cast<const bfr::rhi::TextureCopy*>(regions), region_count};
        const auto copied = device->device->copyTexture(fromC(command_list), fromC(src), fromC(dst), copies);
        return copied ? BFR_RESULT_SUCCESS : fail(copied.error());
    });
}

bfr_result bfr_resolve_texture(bfr_device* device, bfr_command_list command_list, bfr_texture src,
                               uint32_t src_mip_level, uint32_t src_layer, bfr_texture dst, uint32_t dst_mip_level,
                               uint32_t dst_layer) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto resolved = device->device->resolveTexture(fromC(command_list), fromC(src), src_mip_level, src_layer,
                                                             fromC(dst), dst_mip_level, dst_layer);
        return resolved ? BFR_RESULT_SUCCESS : fail(resolved.error());
    });
}

bfr_result bfr_submit(bfr_device* device, const bfr_submit_desc* desc, bfr_token* token) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and desc must not be null");
        }
        if (token != nullptr) {
            *token = {};
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_submit_desc takes no chained structs");
        }
        if (desc->command_list_count != 0 && desc->command_lists == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "command_lists must not be null when command_list_count is not 0");
        }
        if (desc->wait_count != 0 && desc->waits == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "waits must not be null when wait_count is not 0");
        }
        // The caller's arrays, read in place as the rhi types; see the
        // layout asserts above.
        const bfr::rhi::SubmitDesc submitDesc{
            .queue = desc->queue,
            .commandLists = {reinterpret_cast<const bfr::rhi::CommandListHandle*>(desc->command_lists),
                             desc->command_list_count},
            .waits = {reinterpret_cast<const bfr::rhi::Token*>(desc->waits), desc->wait_count},
        };
        const auto submitted = device->device->submit(submitDesc);
        if (!submitted) {
            return fail(submitted.error());
        }
        if (token != nullptr) {
            *token = toC(*submitted);
        }
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_wait(bfr_device* device, const bfr_token* token, uint64_t timeout_ns) {
    return guarded([&] {
        if (device == nullptr || token == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and token must not be null");
        }
        const auto waited = device->device->wait(fromC(*token), timeout_ns);
        return waited ? BFR_RESULT_SUCCESS : fail(waited.error());
    });
}

bfr_result bfr_poll(bfr_device* device, const bfr_token* token, bfr_bool* complete) {
    return guarded([&] {
        if (device == nullptr || token == nullptr || complete == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, token and complete must not be null");
        }
        *complete = BFR_FALSE;
        const auto polled = device->device->poll(fromC(*token));
        if (!polled) {
            return fail(polled.error());
        }
        *complete = toC(*polled);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_command_list(bfr_device* device, bfr_command_list command_list) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        // A stale handle is already gone, which is all this call promises.
        if (const auto destroyed = device->device->destroyCommandList(fromC(command_list)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_get_caps(bfr_instance* instance, uint32_t adapter, bfr_caps* caps) {
    return guarded([&] {
        if (instance == nullptr || caps == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and caps must not be null");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        const bfr::rhi::Adapter& listed = instance->instance->adapters()[adapter];
        const bfr::rhi::Caps& found = listed.caps;
        *caps = {
            .info = toC(listed, instance->backend),
            .limits = toC(found.limits),
            .bindless = toC(found.bindless),
            .compute = toC(found.compute),
            .mesh = toC(found.mesh),
            .ray_tracing = toC(found.rayTracing),
            .features = toC(found.features),
            .memory = toC(found.memory),
            .queues = toC(found.queues),
            .timestamps = toC(found.timestamps),
            .sparse = toC(found.sparse),
            .resolve = toC(found.resolve),
        };
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_formats(bfr_instance* instance, uint32_t adapter, uint32_t* count, bfr_format_caps* formats) {
    return guarded([&] {
        if (instance == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and count must not be null");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        const auto total = static_cast<std::uint32_t>(bfr::rhi::kFormats.size());
        if (formats == nullptr) {
            *count = total;
            return BFR_RESULT_SUCCESS;
        }
        // Asked one at a time, and only for those written.
        const std::uint32_t written = std::min(*count, total);
        const std::span out{formats, written};
        for (std::uint32_t i = 0; i < written; ++i) {
            const auto found = instance->instance->formatCaps(adapter, bfr::rhi::kFormats.at(i));
            if (!found) {
                return fail(found.error());
            }
            out[i] = toC(*found);
        }
        *count = written;
        return written < total ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_swapchain_caps(bfr_instance* instance, uint32_t adapter, const bfr_swapchain_query* desc,
                                  bfr_swapchain_caps* caps) {
    return guarded([&] {
        if (instance == nullptr || desc == nullptr || caps == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance, desc and caps must not be null");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        bfr_result failed = BFR_RESULT_SUCCESS;
        const auto window = surfaceWindow(*desc, failed);
        if (!window) {
            return failed;
        }
        const auto found = instance->instance->swapchainCaps(adapter, *window);
        if (!found) {
            return fail(found.error());
        }
        *caps = toC(*found);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_swapchain_formats(bfr_instance* instance, uint32_t adapter, const bfr_swapchain_query* desc,
                                     uint32_t* count, bfr_swapchain_format* formats) {
    return guarded([&] {
        if (instance == nullptr || desc == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance, desc and count must not be null");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        bfr_result failed = BFR_RESULT_SUCCESS;
        const auto window = surfaceWindow(*desc, failed);
        if (!window) {
            return failed;
        }
        const auto found = instance->instance->swapchainFormats(adapter, *window);
        if (!found) {
            return fail(found.error());
        }
        if (formats == nullptr) {
            *count = static_cast<std::uint32_t>(found->size());
            return BFR_RESULT_SUCCESS;
        }
        return writeList(std::span<const bfr::rhi::SwapchainFormat>{*found}, count, formats);
    });
}

bfr_result bfr_get_budget(bfr_device* device, bfr_memory_budget* budget) {
    return guarded([&] {
        if (device == nullptr || budget == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and budget must not be null");
        }
        const auto found = device->device->budget();
        if (!found) {
            return fail(found.error());
        }
        *budget = {};
        const std::span heaps{budget->heaps};
        for (const bfr::rhi::HeapBudget& heap : found->heaps | std::views::take(heaps.size())) {
            heaps[budget->heap_count++] = {.budget = heap.budget, .usage = heap.usage};
        }
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_create_buffer(bfr_device* device, const bfr_buffer_desc* desc, bfr_buffer* buffer) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || buffer == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and buffer must not be null");
        }
        *buffer = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_buffer_desc takes no chained structs");
        }
        bfr::core::Message error;
        const auto wanted = bufferIn(*desc, error);
        if (!wanted) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto created = device->device->createBuffer(*wanted);
        if (!created) {
            return fail(created.error());
        }
        *buffer = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_map_buffer(bfr_device* device, bfr_buffer buffer, void** data) {
    return guarded([&] {
        if (device == nullptr || data == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and data must not be null");
        }
        *data = nullptr;
        const auto mapped = device->device->mapBuffer(fromC(buffer));
        if (!mapped) {
            return fail(mapped.error());
        }
        *data = mapped->data();
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_buffer_slot(bfr_device* device, bfr_buffer buffer, uint32_t* slot) {
    return guarded([&] {
        if (device == nullptr || slot == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and slot must not be null");
        }
        *slot = 0;
        const auto found = device->device->bufferSlot(fromC(buffer));
        if (!found) {
            return fail(found.error());
        }
        *slot = *found;
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_buffer(bfr_device* device, bfr_buffer buffer) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyBuffer(fromC(buffer)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_create_texture(bfr_device* device, const bfr_texture_desc* desc, bfr_texture* texture) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || texture == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and texture must not be null");
        }
        *texture = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_texture_desc takes no chained structs");
        }
        bfr::core::Message error;
        const auto wanted = textureIn(*desc, error);
        if (!wanted) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto created = device->device->createTexture(*wanted);
        if (!created) {
            return fail(created.error());
        }
        *texture = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_texture_slots(bfr_device* device, bfr_texture texture, bfr_texture_slots* slots) {
    return guarded([&] {
        if (device == nullptr || slots == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and slots must not be null");
        }
        *slots = {};
        const auto found = device->device->textureSlots(fromC(texture));
        if (!found) {
            return fail(found.error());
        }
        *slots = {.sampled = found->sampled, .storage = found->storage};
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_texture(bfr_device* device, bfr_texture texture) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyTexture(fromC(texture)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_create_sampler(bfr_device* device, const bfr_sampler_desc* desc, bfr_sampler* sampler) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || sampler == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and sampler must not be null");
        }
        *sampler = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_sampler_desc takes no chained structs");
        }
        bfr::core::Message error;
        const auto wanted = samplerIn(*desc, error);
        if (!wanted) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto created = device->device->createSampler(*wanted);
        if (!created) {
            return fail(created.error());
        }
        *sampler = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_sampler_slot(bfr_device* device, bfr_sampler sampler, uint32_t* slot) {
    return guarded([&] {
        if (device == nullptr || slot == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and slot must not be null");
        }
        *slot = 0;
        const auto found = device->device->samplerSlot(fromC(sampler));
        if (!found) {
            return fail(found.error());
        }
        *slot = *found;
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_sampler(bfr_device* device, bfr_sampler sampler) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroySampler(fromC(sampler)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_create_compute_pipeline(bfr_device* device, const bfr_compute_pipeline_desc* desc,
                                       bfr_pipeline* pipeline) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || pipeline == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and pipeline must not be null");
        }
        *pipeline = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_compute_pipeline_desc takes no chained structs");
        }
        if (desc->compute.code == nullptr && desc->compute.code_size != 0) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "compute.code is null but compute.code_size is not 0");
        }
        if (desc->compute.entry_point.data == nullptr && desc->compute.entry_point.length != 0) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "compute.entry_point is null but its length is not 0");
        }
        const auto created = device->device->createComputePipeline(computePipelineIn(*desc));
        if (!created) {
            return fail(created.error());
        }
        *pipeline = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_create_graphics_pipeline(bfr_device* device, const bfr_graphics_pipeline_desc* desc,
                                        bfr_pipeline* pipeline) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || pipeline == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and pipeline must not be null");
        }
        *pipeline = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_graphics_pipeline_desc takes no chained structs");
        }
        for (const auto& [name, code] : {std::pair{"vertex", &desc->vertex}, std::pair{"fragment", &desc->fragment}}) {
            if (code->code == nullptr && code->code_size != 0) {
                return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "{}.code is null but {}.code_size is not 0", name, name);
            }
            if (code->entry_point.data == nullptr && code->entry_point.length != 0) {
                return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "{}.entry_point is null but its length is not 0", name);
            }
        }
        bfr::core::Message error;
        const auto wanted = graphicsPipelineIn(*desc, error);
        if (!wanted) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto created = device->device->createGraphicsPipeline(*wanted);
        if (!created) {
            return fail(created.error());
        }
        *pipeline = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_pipeline(bfr_device* device, bfr_pipeline pipeline) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyPipeline(fromC(pipeline)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_create_pipeline_cache(bfr_device* device, const bfr_pipeline_cache_desc* desc,
                                     bfr_pipeline_cache* cache) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || cache == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and cache must not be null");
        }
        *cache = {};
        if (desc->data == nullptr && desc->data_size != 0) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "data is null but data_size is not 0");
        }
        const auto created = device->device->createPipelineCache(
            {.data = std::as_bytes(std::span{desc->data, desc->data_size})});
        if (!created) {
            return fail(created.error());
        }
        *cache = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_pipeline_cache_data(bfr_device* device, bfr_pipeline_cache cache, size_t* size, uint8_t* data) {
    return guarded([&] {
        if (device == nullptr || size == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and size must not be null");
        }
        if (data == nullptr) {
            const auto total = device->device->pipelineCacheSize(fromC(cache));
            if (!total) {
                return fail(total.error());
            }
            *size = *total;
            return BFR_RESULT_SUCCESS;
        }
        const auto written = device->device->pipelineCacheData(fromC(cache),
                                                               std::as_writable_bytes(std::span{data, *size}));
        if (!written) {
            return fail(written.error());
        }
        *size = written->written;
        return written->complete ? BFR_RESULT_SUCCESS : BFR_RESULT_INCOMPLETE;
    });
}

void bfr_destroy_pipeline_cache(bfr_device* device, bfr_pipeline_cache cache) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyPipelineCache(fromC(cache)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_create_timestamp_pool(bfr_device* device, const bfr_timestamp_pool_desc* desc,
                                     bfr_timestamp_pool* pool) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || pool == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and pool must not be null");
        }
        *pool = {};
        const auto created = device->device->createTimestampPool({.queue = desc->queue, .count = desc->count});
        if (!created) {
            return fail(created.error());
        }
        *pool = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_timestamp_pool(bfr_device* device, bfr_timestamp_pool pool) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyTimestampPool(fromC(pool)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_reset_timestamps(bfr_device* device, bfr_command_list command_list, bfr_timestamp_pool pool,
                                uint32_t first, uint32_t count) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto reset = device->device->resetTimestamps(fromC(command_list), fromC(pool), first, count);
        return reset ? BFR_RESULT_SUCCESS : fail(reset.error());
    });
}

bfr_result bfr_write_timestamp(bfr_device* device, bfr_command_list command_list, bfr_timestamp_pool pool,
                               uint32_t index, bfr_timestamp_stage stage, bfr_timestamp_granularity granularity) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        bfr::core::Message error;
        const auto at = enumIn(stage, bfr::rhi::TimestampStage::AccelerationStructureBuild, "timestamp stage", error);
        if (!at) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto how = enumIn(granularity, bfr::rhi::TimestampGranularity::Precise, "timestamp granularity", error);
        if (!how) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto written = device->device->writeTimestamp(fromC(command_list), fromC(pool), index, *at, *how);
        return written ? BFR_RESULT_SUCCESS : fail(written.error());
    });
}

bfr_result bfr_resolve_timestamps(bfr_device* device, bfr_command_list command_list, bfr_timestamp_pool pool,
                                  uint32_t first, uint32_t count, bfr_buffer buffer, uint64_t offset) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto resolved = device->device->resolveTimestamps(fromC(command_list), fromC(pool), first, count,
                                                                fromC(buffer), offset);
        return resolved ? BFR_RESULT_SUCCESS : fail(resolved.error());
    });
}

bfr_result bfr_get_timestamp_rate(bfr_device* device, uint32_t queue, bfr_timestamp_rate* rate) {
    return guarded([&] {
        if (device == nullptr || rate == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and rate must not be null");
        }
        *rate = {};
        const auto found = device->device->timestampRate(queue);
        if (!found) {
            return fail(found.error());
        }
        *rate = {.period_ns = found->periodNs, .frequency = found->frequency};
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_calibrate_timestamps(bfr_device* device, uint32_t queue, bfr_host_clock clock,
                                    bfr_calibrated_timestamps* timestamps) {
    return guarded([&] {
        if (device == nullptr || timestamps == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and timestamps must not be null");
        }
        *timestamps = {};
        bfr::core::Message error;
        const auto read = enumIn(clock, bfr::rhi::HostClock::PerformanceCounter, "host clock", error);
        if (!read) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto taken = device->device->calibrateTimestamps(queue, *read);
        if (!taken) {
            return fail(taken.error());
        }
        *timestamps = {.gpu = taken->gpu,
                       .host = taken->host,
                       .max_deviation_ns = taken->maxDeviationNs,
                       .deviation_given = toC(taken->deviationGiven)};
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_create_sparse_memory(bfr_device* device, const bfr_sparse_memory_desc* desc, bfr_sparse_memory* memory) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || memory == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and memory must not be null");
        }
        *memory = {};
        const auto created = device->device->createSparseMemory({.size = desc->size,
                                                                 .tileBytes = desc->tile_bytes,
                                                                 .texture = fromC(desc->texture),
                                                                 .buffer = fromC(desc->buffer)});
        if (!created) {
            return fail(created.error());
        }
        *memory = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_sparse_memory(bfr_device* device, bfr_sparse_memory memory) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroySparseMemory(fromC(memory)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_get_texture_tiling(bfr_device* device, bfr_texture texture, bfr_texture_aspect aspect,
                                  bfr_texture_tiling* tiling) {
    return guarded([&] {
        if (device == nullptr || tiling == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and tiling must not be null");
        }
        *tiling = {};
        bfr::core::Message error;
        const auto read = enumIn(aspect, bfr::rhi::TextureAspect::Stencil, "texture aspect", error);
        if (!read) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto found = device->device->textureTiling(fromC(texture), *read);
        if (!found) {
            return fail(found.error());
        }
        *tiling = {.tile_bytes = found->tileBytes,
                   .shape = toC(found->shape),
                   .first_tail_mip = found->firstTailMip,
                   .tail_bytes = found->tailBytes,
                   .tail_stride = found->tailStride,
                   .single_tail = toC(found->singleTail)};
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_buffer_tiling(bfr_device* device, bfr_buffer buffer, bfr_buffer_tiling* tiling) {
    return guarded([&] {
        if (device == nullptr || tiling == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and tiling must not be null");
        }
        *tiling = {};
        const auto found = device->device->bufferTiling(fromC(buffer));
        if (!found) {
            return fail(found.error());
        }
        *tiling = {.tile_bytes = found->tileBytes};
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_sparse_format(bfr_device* device, const bfr_sparse_format_desc* desc, bfr_sparse_format* format) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || format == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and format must not be null");
        }
        *format = {};
        // The texture the desc names, read as bfr_texture_desc reads its
        // members, so its enums are checked the same way.
        const bfr_texture_desc texture{.next = nullptr,
                                       .dimension = desc->dimension,
                                       .format = desc->format,
                                       .width = 0,
                                       .height = 0,
                                       .depth = 0,
                                       .layers = 0,
                                       .mip_levels = 0,
                                       .samples = desc->samples,
                                       .usage = desc->usage,
                                       .sparse = BFR_TRUE,
                                       .sparse_aliased = BFR_FALSE,
                                       .tile_bytes = desc->tile_bytes};
        bfr::core::Message error;
        const auto wanted = textureIn(texture, error);
        if (!wanted) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto aspect = enumIn(desc->aspect, bfr::rhi::TextureAspect::Stencil, "texture aspect", error);
        if (!aspect) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto found = device->device->sparseFormat({.format = wanted->format,
                                                         .dimension = wanted->dimension,
                                                         .samples = wanted->samples,
                                                         .usage = wanted->usage,
                                                         .aspect = *aspect,
                                                         .tileBytes = wanted->tileBytes});
        if (!found) {
            return fail(found.error());
        }
        *format = {.supported = toC(found->supported),
                   .shape = toC(found->shape),
                   .single_tail = toC(found->singleTail),
                   .aligned_mip_size = toC(found->alignedMipSize)};
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_update_tile_mappings(bfr_device* device, const bfr_tile_mapping_desc* desc, bfr_token* token) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and desc must not be null");
        }
        if (token != nullptr) {
            *token = {};
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_tile_mapping_desc takes no chained structs");
        }
        if ((desc->wait_count != 0 && desc->waits == nullptr) ||
            (desc->texture_region_count != 0 && desc->texture_regions == nullptr) ||
            (desc->buffer_region_count != 0 && desc->buffer_regions == nullptr)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "an array must not be null when its count is not 0");
        }
        // The caller's arrays, read in place as the rhi types; see the
        // layout asserts above.
        const bfr::rhi::TileMappingDesc mappingDesc{
            .queue = desc->queue,
            .waits = {reinterpret_cast<const bfr::rhi::Token*>(desc->waits), desc->wait_count},
            .memory = fromC(desc->memory),
            .texture = fromC(desc->texture),
            .textureRegions = {reinterpret_cast<const bfr::rhi::TextureTileRegion*>(desc->texture_regions),
                               desc->texture_region_count},
            .buffer = fromC(desc->buffer),
            .bufferRegions = {reinterpret_cast<const bfr::rhi::BufferTileRegion*>(desc->buffer_regions),
                              desc->buffer_region_count},
        };
        const auto mapped = device->device->updateTileMappings(mappingDesc);
        if (!mapped) {
            return fail(mapped.error());
        }
        if (token != nullptr) {
            *token = toC(*mapped);
        }
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_acceleration_structure_sizes(bfr_device* device, const bfr_acceleration_structure_inputs* inputs,
                                                bfr_acceleration_structure_sizes* sizes) {
    return guarded([&] {
        if (device == nullptr || inputs == nullptr || sizes == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, inputs and sizes must not be null");
        }
        *sizes = {};
        const std::scoped_lock lock{device->conversions};
        bfr::core::Message error;
        std::size_t triangles = 0;
        std::size_t boxes = 0;
        bool limit = false;
        const auto made = inputsIn(*inputs, *device, triangles, boxes, limit, error);
        if (!made) {
            return fail(limit ? BFR_RESULT_ERROR_LIMIT_REACHED : BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto found = device->device->accelerationStructureSizes(*made);
        if (!found) {
            return fail(found.error());
        }
        *sizes = {.size = found->size, .build_scratch = found->buildScratch, .update_scratch = found->updateScratch};
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_create_acceleration_structure(bfr_device* device, const bfr_acceleration_structure_desc* desc,
                                             bfr_acceleration_structure* structure) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || structure == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and structure must not be null");
        }
        *structure = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_acceleration_structure_desc takes no chained structs");
        }
        bfr::core::Message error;
        const auto type = enumIn(desc->type, bfr::rhi::AccelerationStructureType::Top, "acceleration structure type",
                                 error);
        if (!type) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto made = device->device->createAccelerationStructure({.type = *type, .size = desc->size});
        if (!made) {
            return fail(made.error());
        }
        *structure = toC(*made);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_acceleration_structure_slot(bfr_device* device, bfr_acceleration_structure structure,
                                               uint32_t* slot) {
    return guarded([&] {
        if (device == nullptr || slot == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and slot must not be null");
        }
        *slot = 0;
        const auto found = device->device->accelerationStructureSlot(fromC(structure));
        if (!found) {
            return fail(found.error());
        }
        *slot = *found;
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_acceleration_structure_reference(bfr_device* device, bfr_acceleration_structure structure,
                                                    uint64_t* reference) {
    return guarded([&] {
        if (device == nullptr || reference == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and reference must not be null");
        }
        *reference = 0;
        const auto found = device->device->accelerationStructureReference(fromC(structure));
        if (!found) {
            return fail(found.error());
        }
        *reference = *found;
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_acceleration_structure(bfr_device* device, bfr_acceleration_structure structure) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyAccelerationStructure(fromC(structure)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_write_instances(bfr_device* device, const bfr_acceleration_structure_instance* instances,
                               uint32_t instance_count, void* data, size_t size) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        if ((instances == nullptr && instance_count != 0) || (data == nullptr && size != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instances and data must not be null with a count");
        }
        using namespace bfr::rhi;
        const std::span out{static_cast<std::byte*>(data), size};
        const std::span from{instances, instance_count};
        const std::size_t recordSize = device->instanceSize;
        // Converted a chunk at a time on the stack, so no call allocates; a
        // device without acceleration structures refuses the first.
        constexpr std::size_t kChunk = 64;
        std::array<AccelerationStructureInstance, kChunk> chunk{};
        if (recordSize != 0 && size / recordSize < from.size()) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "{} bytes hold fewer than {} instance records of {}", size,
                        from.size(), recordSize);
        }
        std::size_t done = 0;
        do {
            const std::size_t count = std::min(kChunk, from.size() - done);
            for (std::size_t i = 0; i < count; ++i) {
                const bfr_acceleration_structure_instance& instance = from[done + i];
                bfr::core::Message error;
                const auto opacity = enumIn(instance.opacity, InstanceOpacity::ForceNonOpaque, "instance opacity",
                                            error);
                if (!opacity) {
                    return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
                }
                chunk.at(i) = {.transform = std::to_array(instance.transform),
                               .customIndex = instance.custom_index,
                               .mask = instance.mask,
                               .hitGroupOffset = instance.hit_group_offset,
                               .cullDisable = instance.cull_disable != BFR_FALSE,
                               .frontCounterClockwise = instance.front_counter_clockwise != BFR_FALSE,
                               .opacity = *opacity,
                               .bottomStructure = fromC(instance.bottom_structure)};
            }
            const auto written = device->device->writeInstances(std::span{chunk}.first(count),
                                                                out.subspan(std::min(done * recordSize, out.size())));
            if (!written) {
                return fail(written.error());
            }
            done += count;
        } while (done < from.size());
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_build_acceleration_structure(bfr_device* device, bfr_command_list command_list,
                                            const bfr_acceleration_structure_build* build) {
    return guarded([&] {
        if (device == nullptr || build == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and build must not be null");
        }
        const std::scoped_lock lock{device->conversions};
        bfr::core::Message error;
        std::size_t triangles = 0;
        std::size_t boxes = 0;
        bool limit = false;
        const auto inputs = inputsIn(build->inputs, *device, triangles, boxes, limit, error);
        if (!inputs) {
            return fail(limit ? BFR_RESULT_ERROR_LIMIT_REACHED : BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        const auto built = device->device->buildAccelerationStructure(fromC(command_list),
                                                                      {.dst = fromC(build->dst),
                                                                       .src = fromC(build->src),
                                                                       .update = build->update != BFR_FALSE,
                                                                       .inputs = *inputs,
                                                                       .scratch = fromC(build->scratch),
                                                                       .scratchOffset = build->scratch_offset});
        return built ? BFR_RESULT_SUCCESS : fail(built.error());
    });
}

bfr_result bfr_copy_acceleration_structure(bfr_device* device, bfr_command_list command_list,
                                           bfr_acceleration_structure dst, bfr_acceleration_structure src) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto copied = device->device->copyAccelerationStructure(fromC(command_list), fromC(dst), fromC(src));
        return copied ? BFR_RESULT_SUCCESS : fail(copied.error());
    });
}

bfr_result bfr_create_shader(bfr_device* device, const bfr_shader_code* code, bfr_shader* shader) {
    return guarded([&] {
        if (device == nullptr || code == nullptr || shader == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, code and shader must not be null");
        }
        *shader = {};
        if ((code->code == nullptr && code->code_size != 0) ||
            (code->entry_point.data == nullptr && code->entry_point.length != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "code or entry_point is null with a size");
        }
        const auto made = device->device->createShader(shaderIn(*code));
        if (!made) {
            return fail(made.error());
        }
        *shader = toC(*made);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_shader(bfr_device* device, bfr_shader shader) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyShader(fromC(shader)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_create_ray_tracing_pipeline(bfr_device* device, const bfr_ray_tracing_pipeline_desc* desc,
                                           bfr_pipeline* pipeline) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || pipeline == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and pipeline must not be null");
        }
        *pipeline = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_ray_tracing_pipeline_desc takes no chained structs");
        }
        if ((desc->shaders == nullptr && desc->shader_count != 0) ||
            (desc->groups == nullptr && desc->group_count != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "shaders or groups is null with a count");
        }
        using namespace bfr::rhi;
        bfr::core::Message error;
        // Cold, so the converted arrays are vectors.
        std::vector<RayTracingShader> shaders;
        shaders.reserve(desc->shader_count);
        for (const bfr_ray_tracing_shader& shader : std::span{desc->shaders, desc->shader_count}) {
            const auto stage = enumIn(shader.stage, RayTracingStage::Callable, "ray tracing stage", error);
            if (!stage) {
                return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
            }
            shaders.push_back({.stage = *stage, .shader = fromC(shader.shader)});
        }
        std::vector<RayTracingGroup> groups;
        groups.reserve(desc->group_count);
        for (const bfr_ray_tracing_group& group : std::span{desc->groups, desc->group_count}) {
            const auto type = enumIn(group.type, RayTracingGroupType::BoxHit, "ray tracing group type", error);
            if (!type) {
                return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
            }
            groups.push_back({.type = *type,
                              .general = group.general,
                              .closestHit = group.closest_hit,
                              .anyHit = group.any_hit,
                              .intersection = group.intersection});
        }
        const auto created = device->device->createRayTracingPipeline({.shaders = shaders,
                                                                       .groups = groups,
                                                                       .maxRecursionDepth = desc->max_recursion_depth,
                                                                       .maxPayloadSize = desc->max_payload_size,
                                                                       .maxAttributeSize = desc->max_attribute_size,
                                                                       .cache = fromC(desc->cache)});
        if (!created) {
            return fail(created.error());
        }
        *pipeline = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_shader_identifiers(bfr_device* device, bfr_pipeline pipeline, uint32_t first, uint32_t count,
                                      void* data, size_t size) {
    return guarded([&] {
        if (device == nullptr || (data == nullptr && size != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null, nor data with a size");
        }
        const auto written = device->device->shaderIdentifiers(fromC(pipeline), first, count,
                                                               {static_cast<std::byte*>(data), size});
        return written ? BFR_RESULT_SUCCESS : fail(written.error());
    });
}

namespace {

bfr::rhi::ShaderTableRegion regionIn(const bfr_shader_table_region& region) {
    return {.buffer = fromC(region.buffer), .offset = region.offset, .size = region.size, .stride = region.stride};
}

} // namespace

bfr_result bfr_trace_rays(bfr_device* device, bfr_command_list command_list, const bfr_trace_rays_desc* desc) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and desc must not be null");
        }
        const auto traced = device->device->traceRays(fromC(command_list),
                                                      {.rayGeneration = regionIn(desc->ray_generation),
                                                       .miss = regionIn(desc->miss),
                                                       .hit = regionIn(desc->hit),
                                                       .callable = regionIn(desc->callable),
                                                       .width = desc->width,
                                                       .height = desc->height,
                                                       .depth = desc->depth});
        return traced ? BFR_RESULT_SUCCESS : fail(traced.error());
    });
}

bfr_result bfr_trace_rays_indirect(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                   uint64_t offset) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto traced = device->device->traceRaysIndirect(fromC(command_list), fromC(buffer), offset);
        return traced ? BFR_RESULT_SUCCESS : fail(traced.error());
    });
}

bfr_result bfr_get_buffer_address(bfr_device* device, bfr_buffer buffer, uint64_t* address) {
    return guarded([&] {
        if (device == nullptr || address == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and address must not be null");
        }
        *address = 0;
        const auto found = device->device->bufferAddress(fromC(buffer));
        if (!found) {
            return fail(found.error());
        }
        *address = *found;
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_call_timings(bfr_device* device, uint32_t* count, bfr_call_timing* timings) {
    return guarded([&] {
        if (device == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and count must not be null");
        }
        std::array<bfr::rhi::CallTiming, bfr::rhi::kCallCount> read{};
        if (const auto written = device->device->callTimings(read); !written) {
            return fail(written.error());
        }
        return listTimings(read, count, timings);
    });
}

bfr_result bfr_get_instance_call_timings(bfr_instance* instance, uint32_t* count, bfr_call_timing* timings) {
    return guarded([&] {
        if (instance == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and count must not be null");
        }
        std::array<bfr::rhi::CallTiming, bfr::rhi::kCallCount> read{};
        if (const auto written = instance->instance->callTimings(read); !written) {
            return fail(written.error());
        }
        return listTimings(read, count, timings);
    });
}

} // extern "C"
