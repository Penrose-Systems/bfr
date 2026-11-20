// The C API in bfr.h, over the C++ implementation. No exception crosses it:
// each entry point turns one into a result.

#include "bfr/bfr.h"
#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/text.h"
#include "bfr/core/window.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/call_timing.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/query.h"
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

// The caller's message callback and its user_data, which the Instance's
// Logger points at, so they live as long as it does.
struct CMessages {
    bfr_message_callback callback = nullptr;
    void* userData = nullptr;
};

// An Instance and the CMessages its Logger reads, which outlive it.
struct InstanceHolder {
    CMessages messages;
    std::unique_ptr<bfr::rhi::Instance> instance;
};

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
    // The adapter's bfr_caps_ray_tracing.instance_size and
    // motion_instance_size: the bytes each record bfr_write_instances, and
    // bfr_write_motion_instances, writes takes.
    std::uint32_t instanceSize = 0;
    std::uint32_t motionInstanceSize = 0;
    // The adapter the device is on, and its instance's backend, which
    // bfr_get_device_caps' info names.
    std::uint32_t adapter = 0;
    bfr_backend backend{};
};

namespace {

// A string that is null with a length is no string.
bool isString(bfr_string text) {
    return text.data != nullptr || text.length == 0;
}

std::string_view fromC(bfr_string text) {
    return {text.data, text.length};
}

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

// The swapchain desc asks for.
bfr::rhi::SwapchainDesc swapchainIn(const bfr_swapchain_desc& desc) {
    using namespace bfr;
    using namespace bfr::rhi;
    SwapchainDesc swapchain;
    swapchain.queue = desc.queue;
    swapchain.imageCount = desc.image_count;
    swapchain.presentMode = static_cast<PresentMode>(desc.present_mode);
    swapchain.format = static_cast<Format>(desc.format);
    swapchain.colorSpace = static_cast<ColorSpace>(desc.color_space);
    swapchain.alphaMode = static_cast<AlphaMode>(desc.alpha_mode);
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

// The buffer desc asks for.
bfr::rhi::BufferDesc bufferIn(const bfr_buffer_desc& desc) {
    using namespace bfr;
    using namespace bfr::rhi;
    BufferDesc buffer{.size = desc.size,
                      .memory = static_cast<BufferMemory>(desc.memory),
                      .usage = {.copySrc = desc.usage.copy_src != BFR_FALSE,
                                .copyDst = desc.usage.copy_dst != BFR_FALSE,
                                .uniform = desc.usage.uniform != BFR_FALSE,
                                .storage = desc.usage.storage != BFR_FALSE,
                                .index = desc.usage.index != BFR_FALSE,
                                .indirect = desc.usage.indirect != BFR_FALSE,
                                .accelerationStructureInput = desc.usage.acceleration_structure_input != BFR_FALSE,
                                .accelerationStructureScratch = desc.usage.acceleration_structure_scratch != BFR_FALSE,
                                .shaderBindingTable = desc.usage.shader_binding_table != BFR_FALSE,
                                .accelerationStructureData = desc.usage.acceleration_structure_data != BFR_FALSE,
                                .accelerationStructureStorage = desc.usage.acceleration_structure_storage != BFR_FALSE},
                      .sparse = desc.sparse != BFR_FALSE,
                      .sparseAliased = desc.sparse_aliased != BFR_FALSE,
                      .tileBytes = desc.tile_bytes,
                      .name = fromC(desc.name)};
    return buffer;
}

bfr_texture toC(bfr::rhi::TextureHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::TextureHandle fromC(bfr_texture handle) {
    return bfr::rhi::TextureHandle::fromBits(handle.bits);
}

// The texture desc asks for.
bfr::rhi::TextureDesc textureIn(const bfr_texture_desc& desc) {
    using namespace bfr;
    using namespace bfr::rhi;
    return TextureDesc{.dimension = static_cast<TextureDimension>(desc.dimension),
                       .format = static_cast<Format>(desc.format),
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
                                 .depthStencil = desc.usage.depth_stencil != BFR_FALSE,
                                 .storageAtomic = desc.usage.storage_atomic != BFR_FALSE},
                       .sparse = desc.sparse != BFR_FALSE,
                       .sparseAliased = desc.sparse_aliased != BFR_FALSE,
                       .tileBytes = desc.tile_bytes,
                       .name = fromC(desc.name)};
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

bfr::rhi::AccelerationStructurePropertyPoolHandle fromC(bfr_acceleration_structure_property_pool handle) {
    return bfr::rhi::AccelerationStructurePropertyPoolHandle::fromBits(handle.bits);
}

bfr_acceleration_structure_property_pool toC(bfr::rhi::AccelerationStructurePropertyPoolHandle handle) {
    return {.bits = handle.bits};
}

static_assert(bfr::rhi::kSerializedIdentifierSize == BFR_SERIALIZED_IDENTIFIER_SIZE);
// bfr_write_acceleration_structure_properties reads the caller's handles in
// place as the rhi's.
static_assert(sizeof(bfr_acceleration_structure) == sizeof(bfr::rhi::AccelerationStructureHandle) &&
              alignof(bfr_acceleration_structure) == alignof(bfr::rhi::AccelerationStructureHandle) &&
              std::is_standard_layout_v<bfr::rhi::AccelerationStructureHandle> &&
              std::is_trivially_copyable_v<bfr::rhi::AccelerationStructureHandle>);

bfr::rhi::ShaderHandle fromC(bfr_shader handle) {
    return bfr::rhi::ShaderHandle::fromBits(handle.bits);
}

bfr_shader toC(bfr::rhi::ShaderHandle handle) {
    return {.bits = handle.bits};
}

// bfr_build_acceleration_structure_indirect's ranges are read in place on the
// GPU, so the C layout is the one the backends read.
static_assert(sizeof(bfr_acceleration_structure_build_range) == sizeof(bfr::rhi::AccelerationStructureBuildRange) &&
              offsetof(bfr_acceleration_structure_build_range, primitive_offset) ==
                  offsetof(bfr::rhi::AccelerationStructureBuildRange, primitiveOffset) &&
              offsetof(bfr_acceleration_structure_build_range, first_vertex) ==
                  offsetof(bfr::rhi::AccelerationStructureBuildRange, firstVertex) &&
              offsetof(bfr_acceleration_structure_build_range, transform_offset) ==
                  offsetof(bfr::rhi::AccelerationStructureBuildRange, transformOffset));

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

// count queues from queues, which may be null only when count is 0. Nothing
// when it is null otherwise; error says so.
std::optional<std::vector<bfr::rhi::QueueDesc>> queuesIn(const bfr_queue_desc* queues, std::uint32_t count,
                                                         bfr::core::Message& error) {
    if (queues == nullptr && count != 0) {
        error.format("queues is null with a count of {}", count);
        return std::nullopt;
    }
    std::vector<bfr::rhi::QueueDesc> out;
    out.reserve(count);
    for (const bfr_queue_desc& queue : std::span{queues, count}) {
        out.push_back({.family = queue.family, .rank = queue.rank, .level = static_cast<bfr::QueueLevel>(queue.level)});
    }
    return out;
}

bfr_timestamp_pool toC(bfr::rhi::TimestampPoolHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::TimestampPoolHandle fromC(bfr_timestamp_pool handle) {
    return bfr::rhi::TimestampPoolHandle::fromBits(handle.bits);
}

bfr_query_pool toC(bfr::rhi::QueryPoolHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::QueryPoolHandle fromC(bfr_query_pool handle) {
    return bfr::rhi::QueryPoolHandle::fromBits(handle.bits);
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

static_assert(bfr::rhi::kCallCount == BFR_CALL_CAPS + 1);
static_assert(bfr::rhi::kPipelineStatistics == BFR_PIPELINE_STATISTIC_MESH_INVOCATIONS + 1);

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

// Records are read in place on the GPU, so the C layout is the one the
// backends read.
static_assert(sizeof(bfr_opacity_micromap_record) == sizeof(bfr::rhi::OpacityMicromapRecord) &&
              offsetof(bfr_opacity_micromap_record, subdivision_level) ==
                  offsetof(bfr::rhi::OpacityMicromapRecord, subdivisionLevel) &&
              offsetof(bfr_opacity_micromap_record, format) == offsetof(bfr::rhi::OpacityMicromapRecord, format));
static_assert(bfr::rhi::kInstanceIndexLimit == BFR_INSTANCE_INDEX_LIMIT);
static_assert(bfr::rhi::kTransformFloats == BFR_TRANSFORM_FLOATS);
static_assert(bfr::rhi::kAccelerationStructureCaptureWords == BFR_ACCELERATION_STRUCTURE_CAPTURE_WORDS);
static_assert(bfr::rhi::kBoxFloats == BFR_BOX_FLOATS);
static_assert(bfr::rhi::kGlobalPartition == BFR_GLOBAL_PARTITION);
// Partitioned top records are read in place on the GPU too.
static_assert(sizeof(bfr_partitioned_top_op) == sizeof(bfr::rhi::PartitionedTopOp) &&
              offsetof(bfr_partitioned_top_op, count) == offsetof(bfr::rhi::PartitionedTopOp, count) &&
              offsetof(bfr_partitioned_top_op, address) == offsetof(bfr::rhi::PartitionedTopOp, address) &&
              offsetof(bfr_partitioned_top_op, stride) == offsetof(bfr::rhi::PartitionedTopOp, stride));
static_assert(sizeof(bfr_partitioned_instance_record) == sizeof(bfr::rhi::PartitionedInstanceRecord) &&
              offsetof(bfr_partitioned_instance_record, bounds) ==
                  offsetof(bfr::rhi::PartitionedInstanceRecord, bounds) &&
              offsetof(bfr_partitioned_instance_record, custom_index) ==
                  offsetof(bfr::rhi::PartitionedInstanceRecord, customIndex) &&
              offsetof(bfr_partitioned_instance_record, mask) == offsetof(bfr::rhi::PartitionedInstanceRecord, mask) &&
              offsetof(bfr_partitioned_instance_record, hit_group_offset) ==
                  offsetof(bfr::rhi::PartitionedInstanceRecord, hitGroupOffset) &&
              offsetof(bfr_partitioned_instance_record, flags) ==
                  offsetof(bfr::rhi::PartitionedInstanceRecord, flags) &&
              offsetof(bfr_partitioned_instance_record, instance_index) ==
                  offsetof(bfr::rhi::PartitionedInstanceRecord, instanceIndex) &&
              offsetof(bfr_partitioned_instance_record, partition_index) ==
                  offsetof(bfr::rhi::PartitionedInstanceRecord, partitionIndex) &&
              offsetof(bfr_partitioned_instance_record, bottom_reference) ==
                  offsetof(bfr::rhi::PartitionedInstanceRecord, bottomReference));
static_assert(sizeof(bfr_partitioned_instance_update_record) == sizeof(bfr::rhi::PartitionedInstanceUpdateRecord) &&
              offsetof(bfr_partitioned_instance_update_record, hit_group_offset) ==
                  offsetof(bfr::rhi::PartitionedInstanceUpdateRecord, hitGroupOffset) &&
              offsetof(bfr_partitioned_instance_update_record, bottom_reference) ==
                  offsetof(bfr::rhi::PartitionedInstanceUpdateRecord, bottomReference));
static_assert(sizeof(bfr_partition_translation_record) == sizeof(bfr::rhi::PartitionTranslationRecord) &&
              offsetof(bfr_partition_translation_record, translation) ==
                  offsetof(bfr::rhi::PartitionTranslationRecord, translation));

// The rhi form of a structure's build flags.
bfr::rhi::AccelerationStructureFlags flagsIn(const bfr_acceleration_structure_flags& flags) {
    using namespace bfr;
    using namespace bfr::rhi;
    return AccelerationStructureFlags{.preference = static_cast<BuildPreference>(flags.preference),
                                      .allowUpdate = flags.allow_update != BFR_FALSE,
                                      .minimizeMemory = flags.minimize_memory != BFR_FALSE,
                                      .allowCompaction = flags.allow_compaction != BFR_FALSE,
                                      .allowDataAccess = flags.allow_data_access != BFR_FALSE,
                                      .allowOpacityMicromapUpdate = flags.allow_opacity_micromap_update != BFR_FALSE,
                                      .allowDisableOpacityMicromaps = flags.allow_disable_opacity_micromaps !=
                                                                      BFR_FALSE,
                                      .allowOpacityMicromapDataUpdate = flags.allow_opacity_micromap_data_update !=
                                                                        BFR_FALSE};
}

// The rhi form of a cluster operation's inputs.
bfr::rhi::ClusterInputs clusterInputsIn(const bfr_cluster_inputs& inputs) {
    using namespace bfr;
    using namespace bfr::rhi;
    const auto operation = static_cast<ClusterOperation>(inputs.operation);
    const auto mode = static_cast<ClusterMode>(inputs.mode);
    const auto moved = static_cast<ClusterStructureType>(inputs.move.type);
    const auto format = static_cast<bfr::Format>(inputs.triangles.vertex_format);
    const auto flags = flagsIn(inputs.flags);
    const bfr_cluster_triangle_limits& triangles = inputs.triangles;
    return ClusterInputs{.operation = operation,
                         .mode = mode,
                         .flags = flags,
                         .allowOpacityMicromaps = inputs.allow_opacity_micromaps != BFR_FALSE,
                         .maxCount = inputs.max_count,
                         .move = {.type = moved,
                                  .noOverlap = inputs.move.no_overlap != BFR_FALSE,
                                  .maxBytes = inputs.move.max_bytes},
                         .bottoms = {.maxTotalClusters = inputs.bottoms.max_total_clusters,
                                     .maxClustersPerBottom = inputs.bottoms.max_clusters_per_bottom},
                         .triangles = {.vertexFormat = format,
                                       .maxGeometryIndex = triangles.max_geometry_index,
                                       .maxUniqueGeometries = triangles.max_unique_geometries,
                                       .maxTriangles = triangles.max_triangles,
                                       .maxVertices = triangles.max_vertices,
                                       .maxTotalTriangles = triangles.max_total_triangles,
                                       .maxTotalVertices = triangles.max_total_vertices,
                                       .minTruncateBits = triangles.min_truncate_bits}};
}

static_assert(bfr::rhi::kClusterGeometryFlagsShift == BFR_CLUSTER_GEOMETRY_FLAGS_SHIFT &&
              bfr::rhi::kClusterVertexCountShift == BFR_CLUSTER_VERTEX_COUNT_SHIFT &&
              bfr::rhi::kClusterTruncateShift == BFR_CLUSTER_TRUNCATE_SHIFT &&
              bfr::rhi::kClusterIndexFormatShift == BFR_CLUSTER_INDEX_FORMAT_SHIFT &&
              bfr::rhi::kClusterMicromapIndexFormatShift == BFR_CLUSTER_MICROMAP_INDEX_FORMAT_SHIFT);
// Cluster records are read in place on the GPU too.
static_assert(sizeof(bfr_cluster_move_record) == sizeof(bfr::rhi::ClusterMoveRecord));
static_assert(sizeof(bfr_cluster_bottom_record) == sizeof(bfr::rhi::ClusterBottomRecord) &&
              offsetof(bfr_cluster_bottom_record, cluster_stride) ==
                  offsetof(bfr::rhi::ClusterBottomRecord, clusterStride) &&
              offsetof(bfr_cluster_bottom_record, clusters) == offsetof(bfr::rhi::ClusterBottomRecord, clusters));
static_assert(
    sizeof(bfr_cluster_triangle_record) == sizeof(bfr::rhi::ClusterTriangleRecord) &&
    offsetof(bfr_cluster_triangle_record, cluster_flags) == offsetof(bfr::rhi::ClusterTriangleRecord, clusterFlags) &&
    offsetof(bfr_cluster_triangle_record, counts) == offsetof(bfr::rhi::ClusterTriangleRecord, counts) &&
    offsetof(bfr_cluster_triangle_record, base_geometry) == offsetof(bfr::rhi::ClusterTriangleRecord, baseGeometry) &&
    offsetof(bfr_cluster_triangle_record, index_stride) == offsetof(bfr::rhi::ClusterTriangleRecord, indexStride) &&
    offsetof(bfr_cluster_triangle_record, vertex_stride) == offsetof(bfr::rhi::ClusterTriangleRecord, vertexStride) &&
    offsetof(bfr_cluster_triangle_record, geometry_stride) ==
        offsetof(bfr::rhi::ClusterTriangleRecord, geometryStride) &&
    offsetof(bfr_cluster_triangle_record, micromap_index_stride) ==
        offsetof(bfr::rhi::ClusterTriangleRecord, micromapIndexStride) &&
    offsetof(bfr_cluster_triangle_record, indices) == offsetof(bfr::rhi::ClusterTriangleRecord, indices) &&
    offsetof(bfr_cluster_triangle_record, vertices) == offsetof(bfr::rhi::ClusterTriangleRecord, vertices) &&
    offsetof(bfr_cluster_triangle_record, geometries) == offsetof(bfr::rhi::ClusterTriangleRecord, geometries) &&
    offsetof(bfr_cluster_triangle_record, micromaps) == offsetof(bfr::rhi::ClusterTriangleRecord, micromaps) &&
    offsetof(bfr_cluster_triangle_record, micromap_indices) ==
        offsetof(bfr::rhi::ClusterTriangleRecord, micromapIndices));
static_assert(sizeof(bfr_cluster_template_record) == sizeof(bfr::rhi::ClusterTemplateRecord) &&
              offsetof(bfr_cluster_template_record, bounds_limit) ==
                  offsetof(bfr::rhi::ClusterTemplateRecord, boundsLimit));
static_assert(sizeof(bfr_cluster_instantiate_record) == sizeof(bfr::rhi::ClusterInstantiateRecord) &&
              offsetof(bfr_cluster_instantiate_record, geometry_index_offset) ==
                  offsetof(bfr::rhi::ClusterInstantiateRecord, geometryIndexOffset) &&
              offsetof(bfr_cluster_instantiate_record, cluster_template) ==
                  offsetof(bfr::rhi::ClusterInstantiateRecord, clusterTemplate) &&
              offsetof(bfr_cluster_instantiate_record, vertices) ==
                  offsetof(bfr::rhi::ClusterInstantiateRecord, vertices) &&
              offsetof(bfr_cluster_instantiate_record, vertex_stride) ==
                  offsetof(bfr::rhi::ClusterInstantiateRecord, vertexStride));
static_assert(sizeof(bfr_cluster_template_indices_record) == sizeof(bfr::rhi::ClusterTemplateIndicesRecord));

// The rhi form of a structure's inputs, its triangles and boxes written to
// device's arrays from triangles and boxes on, which those advance past; a
// top structure's are none, as the rhi reads none. Nothing when an array is
// null with a count or there is no room left; error says which, and limit
// whether it was room.
std::optional<bfr::rhi::AccelerationStructureInputs> inputsIn(const bfr_acceleration_structure_inputs& inputs,
                                                              bfr_device& device, std::size_t& triangles,
                                                              std::size_t& boxes, bool& limit,
                                                              bfr::core::Message& error) {
    using namespace bfr;
    using namespace bfr::rhi;
    limit = false;
    const auto type = static_cast<AccelerationStructureType>(inputs.type);
    const auto flags = flagsIn(inputs.flags);
    AccelerationStructureInputs made{
        .type = type,
        .flags = flags,
        .triangles = {},
        .boxes = {},
        .instances = fromC(inputs.instances),
        .instanceOffset = inputs.instance_offset,
        .instanceCount = inputs.instance_count,
        .micromapUsages = {},
        .micromapData = fromC(inputs.micromap_data),
        .micromapDataOffset = inputs.micromap_data_offset,
        .micromapRecords = fromC(inputs.micromap_records),
        .micromapRecordOffset = inputs.micromap_record_offset,
        .micromapRecordStride = inputs.micromap_record_stride,
        .motionKeyframes = inputs.motion_keyframes,
        .partitioned = {.instanceCount = inputs.partitioned.instance_count,
                        .maxPartitionInstances = inputs.partitioned.max_partition_instances,
                        .partitionCount = inputs.partitioned.partition_count,
                        .maxGlobalInstances = inputs.partitioned.max_global_instances,
                        .partitionTranslation = inputs.partitioned.partition_translation != BFR_FALSE,
                        .ops = fromC(inputs.partitioned.ops),
                        .opOffset = inputs.partitioned.op_offset,
                        .opCount = fromC(inputs.partitioned.op_count),
                        .opCountOffset = inputs.partitioned.op_count_offset}};
    if (inputs.micromap_usages == nullptr && inputs.micromap_usage_count != 0) {
        error.format("micromap_usages is null with a count");
        return std::nullopt;
    }
    // The usages are read in place: the C and C++ layouts are one, and the
    // backend checks their formats.
    static_assert(sizeof(bfr_opacity_micromap_usage) == sizeof(OpacityMicromapUsage) &&
                  offsetof(bfr_opacity_micromap_usage, subdivision_level) ==
                      offsetof(OpacityMicromapUsage, subdivisionLevel) &&
                  offsetof(bfr_opacity_micromap_usage, format) == offsetof(OpacityMicromapUsage, format));
    made.micromapUsages = {reinterpret_cast<const OpacityMicromapUsage*>(inputs.micromap_usages),
                           inputs.micromap_usage_count};
    if (made.type != AccelerationStructureType::Bottom) {
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
        const auto format = static_cast<bfr::Format>(from.vertex_format);
        const auto indexType = static_cast<IndexType>(from.index_type);
        const auto micromapIndexType = static_cast<IndexType>(from.opacity_micromap.index_type);
        const bfr_triangle_opacity_micromap& micromap = from.opacity_micromap;
        triangleOut[i] = {.vertices = fromC(from.vertices),
                          .vertexOffset = from.vertex_offset,
                          .vertexFormat = format,
                          .vertexStride = from.vertex_stride,
                          .vertexCount = from.vertex_count,
                          .indices = fromC(from.indices),
                          .indexOffset = from.index_offset,
                          .indexType = indexType,
                          .triangleCount = from.triangle_count,
                          .transform = fromC(from.transform),
                          .transformOffset = from.transform_offset,
                          .opaque = from.opaque != BFR_FALSE,
                          .noDuplicateAnyHit = from.no_duplicate_any_hit != BFR_FALSE,
                          .opacityMicromap = {.micromap = fromC(micromap.micromap),
                                              .indices = fromC(micromap.indices),
                                              .indexOffset = micromap.index_offset,
                                              .indexType = micromapIndexType,
                                              .indexStride = micromap.index_stride,
                                              .baseTriangle = micromap.base_triangle,
                                              .firstUsage = micromap.first_usage,
                                              .usageCount = micromap.usage_count},
                          .motionVertices = fromC(from.motion_vertices),
                          .motionVertexOffset = from.motion_vertex_offset,
                          .motionKeyframeStride = from.motion_keyframe_stride};
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

// The sampler desc asks for.
bfr::rhi::SamplerDesc samplerIn(const bfr_sampler_desc& desc) {
    using namespace bfr;
    using namespace bfr::rhi;
    const auto magFilter = static_cast<Filter>(desc.mag_filter);
    const auto minFilter = static_cast<Filter>(desc.min_filter);
    const auto mipFilter = static_cast<Filter>(desc.mip_filter);
    const auto addressU = static_cast<AddressMode>(desc.address_u);
    const auto addressV = static_cast<AddressMode>(desc.address_v);
    const auto addressW = static_cast<AddressMode>(desc.address_w);
    const auto reduction = static_cast<SamplerReduction>(desc.reduction);
    const auto compare = static_cast<CompareOp>(desc.compare_op);
    SamplerDesc sampler{.magFilter = magFilter,
                        .minFilter = minFilter,
                        .mipFilter = mipFilter,
                        .addressU = addressU,
                        .addressV = addressV,
                        .addressW = addressW,
                        .lodBias = desc.lod_bias,
                        .minLod = desc.min_lod,
                        .maxLod = desc.max_lod,
                        .maxAnisotropy = desc.max_anisotropy,
                        .reduction = reduction,
                        .compareOp = compare,
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
            .cache = fromC(desc.cache),
            .name = fromC(desc.name)};
}

bfr::rhi::StencilFace stencilFaceIn(const bfr_stencil_face& face) {
    using namespace bfr;
    using namespace bfr::rhi;
    const auto fail = static_cast<StencilOp>(face.fail);
    const auto pass = static_cast<StencilOp>(face.pass);
    const auto depthFail = static_cast<StencilOp>(face.depth_fail);
    const auto compare = static_cast<CompareOp>(face.compare);
    return StencilFace{.fail = fail,
                       .pass = pass,
                       .depthFail = depthFail,
                       .compare = compare,
                       .readMask = face.read_mask,
                       .writeMask = face.write_mask};
}

bfr::rhi::BlendState blendIn(const bfr_blend_state& blend) {
    using namespace bfr;
    using namespace bfr::rhi;
    const auto srcColor = static_cast<BlendFactor>(blend.src_color);
    const auto dstColor = static_cast<BlendFactor>(blend.dst_color);
    const auto colorOp = static_cast<BlendOp>(blend.color_op);
    const auto srcAlpha = static_cast<BlendFactor>(blend.src_alpha);
    const auto dstAlpha = static_cast<BlendFactor>(blend.dst_alpha);
    const auto alphaOp = static_cast<BlendOp>(blend.alpha_op);
    return BlendState{.enable = blend.enable != BFR_FALSE,
                      .srcColor = srcColor,
                      .dstColor = dstColor,
                      .colorOp = colorOp,
                      .srcAlpha = srcAlpha,
                      .dstAlpha = dstAlpha,
                      .alphaOp = alphaOp,
                      .writeMask = blend.write_mask};
}

// The graphics pipeline desc asks for. The targets past color_target_count
// are not read.
bfr::rhi::GraphicsPipelineDesc graphicsPipelineIn(const bfr_graphics_pipeline_desc& desc) {
    using namespace bfr;
    using namespace bfr::rhi;
    const auto topology = static_cast<Topology>(desc.topology);
    const auto restart = static_cast<PrimitiveRestart>(desc.primitive_restart);
    const auto cull = static_cast<CullMode>(desc.raster.cull);
    const auto frontFace = static_cast<FrontFace>(desc.raster.front_face);
    const auto fill = static_cast<FillMode>(desc.raster.fill);
    const auto depthCompare = static_cast<CompareOp>(desc.depth_stencil.depth_compare);
    const auto front = stencilFaceIn(desc.depth_stencil.front);
    const auto back = stencilFaceIn(desc.depth_stencil.back);
    const auto depthStencilFormat = static_cast<bfr::Format>(desc.depth_stencil_format);
    GraphicsPipelineDesc pipeline{
        .vertex = shaderIn(desc.vertex),
        .fragment = shaderIn(desc.fragment),
        .task = shaderIn(desc.task),
        .taskWorkgroupSize = std::to_array(desc.task_workgroup_size),
        .mesh = shaderIn(desc.mesh),
        .meshWorkgroupSize = std::to_array(desc.mesh_workgroup_size),
        .topology = topology,
        .primitiveRestart = restart,
        .raster = {.cull = cull,
                   .frontFace = frontFace,
                   .fill = fill,
                   .depthClamp = desc.raster.depth_clamp != BFR_FALSE,
                   .depthBias = desc.raster.depth_bias,
                   .depthBiasSlope = desc.raster.depth_bias_slope,
                   .depthBiasClamp = desc.raster.depth_bias_clamp},
        .depthStencil = {.depthTest = desc.depth_stencil.depth_test != BFR_FALSE,
                         .depthWrite = desc.depth_stencil.depth_write != BFR_FALSE,
                         .depthCompare = depthCompare,
                         .stencilTest = desc.depth_stencil.stencil_test != BFR_FALSE,
                         .front = front,
                         .back = back},
        .colorTargetCount = desc.color_target_count,
        .colorTargets = {},
        .depthStencilFormat = depthStencilFormat,
        .samples = desc.samples,
        .alphaToCoverage = desc.alpha_to_coverage != BFR_FALSE,
        .sampleMask = desc.sample_mask,
        .alphaToOne = desc.alpha_to_one != BFR_FALSE,
        .minSampleShading = desc.min_sample_shading,
        .cache = fromC(desc.cache),
        .name = fromC(desc.name),
    };
    const std::span targets{desc.color_targets};
    for (std::uint32_t i = 0; i < std::min<std::uint32_t>(desc.color_target_count, kMaxColorTargets); ++i) {
        const auto format = static_cast<bfr::Format>(targets[i].format);
        const auto blend = blendIn(targets[i].blend);
        pipeline.colorTargets.at(i) = {.format = format, .blend = blend};
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

bfr::rhi::Token fromC(const bfr_token& token) {
    return {.queue = token.queue, .value = token.value};
}

bfr::rhi::CommandListHandle fromC(bfr_command_list handle) {
    return bfr::rhi::CommandListHandle::fromBits(handle.bits);
}

// The rendering desc asks for, each 0 as its documented default. Nothing when
// more attachments are counted than there is room for; error says so.
std::optional<bfr::rhi::RenderingDesc> renderingIn(const bfr_rendering_desc& desc, bfr::core::Message& error) {
    using namespace bfr;
    using namespace bfr::rhi;
    if (desc.color_attachment_count > kMaxColorTargets) {
        error.format("color_attachment_count {} is more than the {} attachments there is room for",
                     desc.color_attachment_count, kMaxColorTargets);
        return std::nullopt;
    }
    RenderingDesc rendering{.colorAttachmentCount = desc.color_attachment_count,
                            .colorAttachments = {},
                            .depthStencil = {},
                            .occlusionQueries = fromC(desc.occlusion_queries)};
    for (std::uint32_t i = 0; i < desc.color_attachment_count; ++i) {
        const bfr_color_attachment& attachment = std::span{desc.color_attachments}[i];
        const auto loadOp = static_cast<LoadOp>(attachment.load_op);
        const auto storeOp = static_cast<StoreOp>(attachment.store_op);
        rendering.colorAttachments.at(i) = {.texture = fromC(attachment.texture),
                                            .mipLevel = attachment.mip_level,
                                            .layer = attachment.layer,
                                            .layerCount = attachment.layer_count,
                                            .loadOp = loadOp,
                                            .storeOp = storeOp,
                                            .clearColor = std::to_array(attachment.clear_color),
                                            .clearInteger = attachment.clear_integer != BFR_FALSE,
                                            .clearColorInteger = std::to_array(attachment.clear_color_integer),
                                            .resolveTexture = fromC(attachment.resolve_texture)};
    }
    const bfr_depth_stencil_attachment& depth = desc.depth_stencil;
    const auto depthLoad = static_cast<LoadOp>(depth.depth_load_op);
    const auto depthStore = static_cast<StoreOp>(depth.depth_store_op);
    const auto stencilLoad = static_cast<LoadOp>(depth.stencil_load_op);
    const auto stencilStore = static_cast<StoreOp>(depth.stencil_store_op);
    const auto depthResolve = static_cast<ResolveMode>(depth.depth_resolve);
    const auto stencilResolve = static_cast<ResolveMode>(depth.stencil_resolve);
    rendering.depthStencil = {.texture = fromC(depth.texture),
                              .mipLevel = depth.mip_level,
                              .layer = depth.layer,
                              .layerCount = depth.layer_count,
                              .depthLoadOp = depthLoad,
                              .depthStoreOp = depthStore,
                              .stencilLoadOp = stencilLoad,
                              .stencilStoreOp = stencilStore,
                              .clearDepth = depth.clear_depth,
                              .clearStencil = depth.clear_stencil,
                              .readOnly = depth.read_only != BFR_FALSE,
                              .resolveTexture = fromC(depth.resolve_texture),
                              .depthResolve = depthResolve,
                              .stencilResolve = stencilResolve};
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
    const auto xid = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(window.window));
    return bfr::core::window::Xlib{.display = window.display, .window = xid};
#endif
}

bfr_backend toC(bfr::Backend backend) {
    return static_cast<bfr_backend>(backend);
}

bfr_adapter_type toC(bfr::AdapterType type) {
    return static_cast<bfr_adapter_type>(type);
}

// Fails for an object's name that is null with a length.
bfr_result checkName(bfr_string name) {
    return isString(name) ? BFR_RESULT_SUCCESS
                          : fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "name is null but its length is not 0");
}

// A null label, or one whose name is no string, is no label.
bool isLabel(const bfr_debug_label* label) {
    return label != nullptr && isString(label->name);
}

bfr::rhi::DebugLabel fromC(const bfr_debug_label& label) {
    return {.name = fromC(label.name), .color = std::to_array(label.color)};
}

bfr_string toC(const std::string& text) {
    return {.data = text.data(), .length = text.size()};
}

bfr_bool toC(bool value) {
    return static_cast<bfr_bool>(value);
}

bfr_format toC(bfr::Format format) {
    return static_cast<bfr_format>(format);
}

bfr_color_space toC(bfr::ColorSpace colorSpace) {
    return static_cast<bfr_color_space>(colorSpace);
}

bfr_present_mode toC(bfr::PresentMode mode) {
    return static_cast<bfr_present_mode>(mode);
}

bfr_alpha_mode toC(bfr::AlphaMode mode) {
    return static_cast<bfr_alpha_mode>(mode);
}

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
        .property_entry_bytes = rayTracing.propertyEntryBytes,
        .reorders_invocations = toC(rayTracing.reordersInvocations),
        .max_binding_table_record_index = rayTracing.maxBindingTableRecordIndex,
        .instance_size = rayTracing.instanceSize,
        .capture_replay_identifier_size = rayTracing.captureReplayIdentifierSize,
        .max_opacity_2_state_subdivision_level = rayTracing.maxOpacity2StateSubdivisionLevel,
        .max_opacity_4_state_subdivision_level = rayTracing.maxOpacity4StateSubdivisionLevel,
        .max_motion_keyframes = rayTracing.maxMotionKeyframes,
        .motion_instance_size = rayTracing.motionInstanceSize,
        .max_partitions = rayTracing.maxPartitions,
        .max_cluster_vertices = rayTracing.maxClusterVertices,
        .max_cluster_triangles = rayTracing.maxClusterTriangles,
        .cluster_scratch_alignment = rayTracing.clusterScratchAlignment,
        .cluster_alignment = rayTracing.clusterAlignment,
        .cluster_template_alignment = rayTracing.clusterTemplateAlignment,
        .cluster_bottom_alignment = rayTracing.clusterBottomAlignment,
        .cluster_template_bounds_alignment = rayTracing.clusterTemplateBoundsAlignment,
        .max_cluster_geometry_index = rayTracing.maxClusterGeometryIndex,
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
        .ray_position_fetch = toC(features.rayPositionFetch),
        .acceleration_structure_serialization = toC(features.accelerationStructureSerialization),
        .acceleration_structure_current_size = toC(features.accelerationStructureCurrentSize),
        .serialized_bottom_count = toC(features.serializedBottomCount),
        .ray_invocation_reorder = toC(features.rayInvocationReorder),
        .ray_tracing_pipeline_library = toC(features.rayTracingPipelineLibrary),
        .pipeline_library_identifiers = toC(features.pipelineLibraryIdentifiers),
        .ray_tracing_capture_replay = toC(features.rayTracingCaptureReplay),
        .ray_tracing_capture_replay_mixed = toC(features.rayTracingCaptureReplayMixed),
        .acceleration_structure_indirect_build = toC(features.accelerationStructureIndirectBuild),
        .opacity_micromap = toC(features.opacityMicromap),
        .ray_tracing_motion = toC(features.rayTracingMotion),
        .ray_tracing_motion_indirect = toC(features.rayTracingMotionIndirect),
        .partitioned_top = toC(features.partitionedTop),
        .cluster_acceleration_structure = toC(features.clusterAccelerationStructure),
        .acceleration_structure_capture_replay = toC(features.accelerationStructureCaptureReplay),
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
        .volume_color_targets = toC(features.volumeColorTargets),
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
        .uniform_buffer_non_uniform_indexing = toC(features.uniformBufferNonUniformIndexing),
        .timestamps_on_all_queues = toC(features.timestampsOnAllQueues),
        .queries = toC(features.queries),
        .pipeline_statistics = toC(features.pipelineStatistics),
        .occlusion_query_precise = toC(features.occlusionQueryPrecise),
        .swapchain = toC(features.swapchain),
        .robust_buffer_access = toC(features.robustBufferAccess),
        .robust_image_access = toC(features.robustImageAccess),
        .mesh_shading_rate = toC(features.meshShadingRate),
        .atomic_int64_shared = toC(features.atomicInt64Shared),
        .sparse_atomic_int64_texture = toC(features.sparseAtomicInt64Texture),
        .atomic_float32 = toC(features.atomicFloat32),
        .atomic_float64 = toC(features.atomicFloat64),
        .atomic_float64_add = toC(features.atomicFloat64Add),
        .atomic_float32_shared = toC(features.atomicFloat32Shared),
        .atomic_float32_add_shared = toC(features.atomicFloat32AddShared),
        .atomic_float64_shared = toC(features.atomicFloat64Shared),
        .atomic_float64_add_shared = toC(features.atomicFloat64AddShared),
        .atomic_float32_texture = toC(features.atomicFloat32Texture),
        .atomic_float32_add_texture = toC(features.atomicFloat32AddTexture),
        .sparse_atomic_float32_texture = toC(features.sparseAtomicFloat32Texture),
        .sparse_atomic_float32_add_texture = toC(features.sparseAtomicFloat32AddTexture),
        .multiview_geometry_shader = toC(features.multiviewGeometryShader),
        .multiview_tessellation_shader = toC(features.multiviewTessellationShader),
        .depth_bias_clamp = toC(features.depthBiasClamp),
        .image_cube_array = toC(features.imageCubeArray),
        .format_bgra4_unorm = toC(features.formatBgra4Unorm),
        .full_draw_index_uint32 = toC(features.fullDrawIndexUint32),
        .storage_texture_multisample = toC(features.storageTextureMultisample),
        .texture_dynamic_indexing = toC(features.textureDynamicIndexing),
        .storage_texture_dynamic_indexing = toC(features.storageTextureDynamicIndexing),
        .storage_buffer_dynamic_indexing = toC(features.storageBufferDynamicIndexing),
        .uniform_buffer_dynamic_indexing = toC(features.uniformBufferDynamicIndexing),
        .vertex_stores_and_atomics = toC(features.vertexStoresAndAtomics),
        .fragment_stores_and_atomics = toC(features.fragmentStoresAndAtomics),
        .tessellation_geometry_point_size = toC(features.tessellationGeometryPointSize),
        .texture_gather_extended = toC(features.textureGatherExtended),
        .clip_distance = toC(features.clipDistance),
        .cull_distance = toC(features.cullDistance),
        .shader_min_lod = toC(features.shaderMinLod),
        .storage_buffer_16bit_access = toC(features.storageBuffer16bitAccess),
        .uniform_and_storage_buffer_16bit_access = toC(features.uniformAndStorageBuffer16bitAccess),
        .push_constant_16bit = toC(features.pushConstant16bit),
        .input_output_16bit = toC(features.inputOutput16bit),
        .storage_buffer_8bit_access = toC(features.storageBuffer8bitAccess),
        .uniform_and_storage_buffer_8bit_access = toC(features.uniformAndStorageBuffer8bitAccess),
        .push_constant_8bit = toC(features.pushConstant8bit),
        .variable_pointers_storage_buffer = toC(features.variablePointersStorageBuffer),
        .variable_pointers = toC(features.variablePointers),
        .draw_parameters = toC(features.drawParameters),
        .scalar_block_layout = toC(features.scalarBlockLayout),
        .uniform_buffer_standard_layout = toC(features.uniformBufferStandardLayout),
        .subgroup_extended_types = toC(features.subgroupExtendedTypes),
        .memory_model = toC(features.memoryModel),
        .memory_model_device_scope = toC(features.memoryModelDeviceScope),
        .memory_model_visibility_chains = toC(features.memoryModelVisibilityChains),
        .shader_output_layer = toC(features.shaderOutputLayer),
        .subgroup_broadcast_dynamic_id = toC(features.subgroupBroadcastDynamicId),
        .demote_to_helper_invocation = toC(features.demoteToHelperInvocation),
        .terminate_invocation = toC(features.terminateInvocation),
        .zero_initialize_workgroup_memory = toC(features.zeroInitializeWorkgroupMemory),
        .integer_dot_product = toC(features.integerDotProduct),
        .local_size_id = toC(features.localSizeId),
        .subgroup_rotate = toC(features.subgroupRotate),
        .subgroup_rotate_clustered = toC(features.subgroupRotateClustered),
        .float_controls2 = toC(features.floatControls2),
        .expect_assume = toC(features.expectAssume),
        .non_semantic_info = toC(features.nonSemanticInfo),
        .debug_labels = toC(features.debugLabels),
        .queue_debug_labels = toC(features.queueDebugLabels),
        .object_names = toC(features.objectNames),
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
    return {.inside_rendering = toC(timestamps.insideRendering),
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

bfr_queue_family toC(const bfr::rhi::QueueFamily& family) {
    return {
        .graphics = toC(family.graphics),
        .compute = toC(family.compute),
        .copy = toC(family.copy),
        .sparse = toC(family.sparse),
        .count = family.count,
        .timestamp_bits = family.timestampBits,
        .timestamps = toC(family.timestamps),
        .timestamp_stages = toC(family.timestampStages),
        .timestamp_resets = toC(family.timestampResets),
        .levels = toC(family.levels),
        .granularity = toC(family.granularity),
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
    return {.texture = toC(resolve.texture),
            .depth_modes = resolve.depthModes,
            .stencil_modes = resolve.stencilModes,
            .independent = toC(resolve.independent),
            .independent_none = toC(resolve.independentNone)};
}

// caps, those of adapter or of a device on it, named as the adapter is.
bfr_caps toC(const bfr::rhi::Adapter& adapter, const bfr::rhi::Caps& caps, bfr_backend backend) {
    return {
        .info = toC(adapter, backend),
        .limits = toC(caps.limits),
        .bindless = toC(caps.bindless),
        .compute = toC(caps.compute),
        .mesh = toC(caps.mesh),
        .ray_tracing = toC(caps.rayTracing),
        .features = toC(caps.features),
        .memory = toC(caps.memory),
        .queue_ranks = caps.queueRanks,
        .timestamps = toC(caps.timestamps),
        .sparse = toC(caps.sparse),
        .resolve = toC(caps.resolve),
    };
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

// The window a swapchain is on, or the result for none or an unusable one; the
// backends take it as Device::createSwapchain requires.
std::optional<NativeWindow> surfaceWindow(const bfr_window& from, bfr_result& failed) {
    bfr::core::Message error;
    auto window = windowIn(from, error);
    if (window && std::holds_alternative<std::monostate>(*window)) {
        window.reset();
        error = "a swapchain needs a window";
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
        if ((desc->application_name.data == nullptr && desc->application_name.length != 0) ||
            (desc->engine_name.data == nullptr && desc->engine_name.length != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "application_name or engine_name is null but its length is not 0");
        }
        auto holder = std::make_shared<InstanceHolder>();
        bfr::rhi::InstanceDesc rhiDesc{.backend = static_cast<bfr::Backend>(desc->backend),
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
            holder->messages = {.callback = desc->message_callback, .userData = desc->message_user_data};
            rhiDesc.logger = {[](Severity severity, std::string_view message, void* userData) {
                                  const auto& messages = *static_cast<const CMessages*>(userData);
                                  messages.callback(static_cast<bfr_message_severity>(severity),
                                                    {.data = message.data(), .length = message.size()},
                                                    messages.userData);
                              },
                              &holder->messages, *least};
        }
        auto rhi = bfr::rhi::Instance::create(rhiDesc,
                                              bfr::rhi::RhiConfig{.validation = desc->validation != BFR_FALSE});
        if (!rhi) {
            return fail(rhi.error());
        }
        holder->instance = std::move(*rhi);
        // The caller owns it until bfr_destroy_instance; the shared pointer
        // keeps the whole holder, CMessages too, while it or a device lives.
        std::shared_ptr<bfr::rhi::Instance> shared{holder, holder->instance.get()};
        *instance = std::make_unique<bfr_instance>(
                        bfr_instance{.instance = std::move(shared), .backend = desc->backend})
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

namespace {

bfr::rhi::FeatureRequest fromC(const bfr_features& features) {
    return {
        .swapchain = features.swapchain != 0,
        .timestampCalibration = features.timestamp_calibration != 0,
        .robustBufferAccess = features.robust_buffer_access != 0,
        .robustImageAccess = features.robust_image_access != 0,
        .rayQuery = features.ray_query != 0,
        .rayQueryGraphics = features.ray_query_graphics != 0,
        .rayTracingPipeline = features.ray_tracing_pipeline != 0,
        .traceRaysIndirect = features.trace_rays_indirect != 0,
        .rayPrimitiveCulling = features.ray_primitive_culling != 0,
        .accelerationStructure = features.acceleration_structure != 0,
        .bufferAddress = features.buffer_address != 0,
        .rayPositionFetch = features.ray_position_fetch != 0,
        .accelerationStructureCurrentSize = features.acceleration_structure_current_size != 0,
        .serializedBottomCount = features.serialized_bottom_count != 0,
        .rayInvocationReorder = features.ray_invocation_reorder != 0,
        .rayTracingPipelineLibrary = features.ray_tracing_pipeline_library != 0,
        .pipelineLibraryIdentifiers = features.pipeline_library_identifiers != 0,
        .rayTracingCaptureReplay = features.ray_tracing_capture_replay != 0,
        .rayTracingCaptureReplayMixed = features.ray_tracing_capture_replay_mixed != 0,
        .accelerationStructureIndirectBuild = features.acceleration_structure_indirect_build != 0,
        .opacityMicromap = features.opacity_micromap != 0,
        .rayTracingMotion = features.ray_tracing_motion != 0,
        .rayTracingMotionIndirect = features.ray_tracing_motion_indirect != 0,
        .partitionedTop = features.partitioned_top != 0,
        .clusterAccelerationStructure = features.cluster_acceleration_structure != 0,
        .accelerationStructureCaptureReplay = features.acceleration_structure_capture_replay != 0,
        .meshShader = features.mesh_shader != 0,
        .taskShader = features.task_shader != 0,
        .meshMultiview = features.mesh_multiview != 0,
        .meshStatistics = features.mesh_statistics != 0,
        .meshShadingRate = features.mesh_shading_rate != 0,
        .shaderFloat16 = features.shader_float16 != 0,
        .shaderFloat64 = features.shader_float64 != 0,
        .shaderInt8 = features.shader_int8 != 0,
        .shaderInt16 = features.shader_int16 != 0,
        .shaderInt64 = features.shader_int64 != 0,
        .atomicInt64 = features.atomic_int64 != 0,
        .atomicInt64Shared = features.atomic_int64_shared != 0,
        .atomicInt64Texture = features.atomic_int64_texture != 0,
        .sparseAtomicInt64Texture = features.sparse_atomic_int64_texture != 0,
        .atomicFloat32 = features.atomic_float32 != 0,
        .atomicFloat32Add = features.atomic_float32_add != 0,
        .atomicFloat64 = features.atomic_float64 != 0,
        .atomicFloat64Add = features.atomic_float64_add != 0,
        .atomicFloat32Shared = features.atomic_float32_shared != 0,
        .atomicFloat32AddShared = features.atomic_float32_add_shared != 0,
        .atomicFloat64Shared = features.atomic_float64_shared != 0,
        .atomicFloat64AddShared = features.atomic_float64_add_shared != 0,
        .atomicFloat32Texture = features.atomic_float32_texture != 0,
        .atomicFloat32AddTexture = features.atomic_float32_add_texture != 0,
        .sparseAtomicFloat32Texture = features.sparse_atomic_float32_texture != 0,
        .sparseAtomicFloat32AddTexture = features.sparse_atomic_float32_add_texture != 0,
        .barycentrics = features.barycentrics != 0,
        .geometryShader = features.geometry_shader != 0,
        .tessellation = features.tessellation != 0,
        .multiview = features.multiview != 0,
        .multiviewGeometryShader = features.multiview_geometry_shader != 0,
        .multiviewTessellationShader = features.multiview_tessellation_shader != 0,
        .sampleRateShading = features.sample_rate_shading != 0,
        .dualSourceBlend = features.dual_source_blend != 0,
        .independentBlend = features.independent_blend != 0,
        .logicOp = features.logic_op != 0,
        .depthClamp = features.depth_clamp != 0,
        .depthBiasClamp = features.depth_bias_clamp != 0,
        .depthBounds = features.depth_bounds != 0,
        .wireframe = features.wireframe != 0,
        .wideLines = features.wide_lines != 0,
        .samplerAnisotropy = features.sampler_anisotropy != 0,
        .samplerMinmax = features.sampler_minmax != 0,
        .samplerMirrorClampToEdge = features.sampler_mirror_clamp_to_edge != 0,
        .samplerCustomBorderColor = features.sampler_custom_border_color != 0,
        .volumeColorTargets = features.volume_color_targets != 0,
        .samplerLodBias = features.sampler_lod_bias != 0,
        .imageCubeArray = features.image_cube_array != 0,
        .compressionAstcHdr = features.compression_astc_hdr != 0,
        .formatBgra4Unorm = features.format_bgra4_unorm != 0,
        .multiDrawIndirect = features.multi_draw_indirect != 0,
        .drawIndirectCount = features.draw_indirect_count != 0,
        .drawIndirectFirstInstance = features.draw_indirect_first_instance != 0,
        .fullDrawIndexUint32 = features.full_draw_index_uint32 != 0,
        .triangleFans = features.triangle_fans != 0,
        .listRestart = features.list_restart != 0,
        .dynamicTopology = features.dynamic_topology != 0,
        .alphaToOne = features.alpha_to_one != 0,
        .storageReadWithoutFormat = features.storage_read_without_format != 0,
        .storageWriteWithoutFormat = features.storage_write_without_format != 0,
        .storageTextureMultisample = features.storage_texture_multisample != 0,
        .textureDynamicIndexing = features.texture_dynamic_indexing != 0,
        .storageTextureDynamicIndexing = features.storage_texture_dynamic_indexing != 0,
        .storageBufferDynamicIndexing = features.storage_buffer_dynamic_indexing != 0,
        .uniformBufferDynamicIndexing = features.uniform_buffer_dynamic_indexing != 0,
        .storageTextureNonUniformIndexing = features.storage_texture_non_uniform_indexing != 0,
        .uniformBufferNonUniformIndexing = features.uniform_buffer_non_uniform_indexing != 0,
        .vertexStoresAndAtomics = features.vertex_stores_and_atomics != 0,
        .fragmentStoresAndAtomics = features.fragment_stores_and_atomics != 0,
        .tessellationGeometryPointSize = features.tessellation_geometry_point_size != 0,
        .textureGatherExtended = features.texture_gather_extended != 0,
        .clipDistance = features.clip_distance != 0,
        .cullDistance = features.cull_distance != 0,
        .shaderMinLod = features.shader_min_lod != 0,
        .storageBuffer16bitAccess = features.storage_buffer_16bit_access != 0,
        .uniformAndStorageBuffer16bitAccess = features.uniform_and_storage_buffer_16bit_access != 0,
        .pushConstant16bit = features.push_constant_16bit != 0,
        .inputOutput16bit = features.input_output_16bit != 0,
        .storageBuffer8bitAccess = features.storage_buffer_8bit_access != 0,
        .uniformAndStorageBuffer8bitAccess = features.uniform_and_storage_buffer_8bit_access != 0,
        .pushConstant8bit = features.push_constant_8bit != 0,
        .variablePointersStorageBuffer = features.variable_pointers_storage_buffer != 0,
        .variablePointers = features.variable_pointers != 0,
        .drawParameters = features.draw_parameters != 0,
        .scalarBlockLayout = features.scalar_block_layout != 0,
        .uniformBufferStandardLayout = features.uniform_buffer_standard_layout != 0,
        .subgroupExtendedTypes = features.subgroup_extended_types != 0,
        .memoryModel = features.memory_model != 0,
        .memoryModelDeviceScope = features.memory_model_device_scope != 0,
        .memoryModelVisibilityChains = features.memory_model_visibility_chains != 0,
        .shaderOutputLayer = features.shader_output_layer != 0,
        .subgroupBroadcastDynamicId = features.subgroup_broadcast_dynamic_id != 0,
        .demoteToHelperInvocation = features.demote_to_helper_invocation != 0,
        .terminateInvocation = features.terminate_invocation != 0,
        .zeroInitializeWorkgroupMemory = features.zero_initialize_workgroup_memory != 0,
        .integerDotProduct = features.integer_dot_product != 0,
        .localSizeId = features.local_size_id != 0,
        .subgroupRotate = features.subgroup_rotate != 0,
        .subgroupRotateClustered = features.subgroup_rotate_clustered != 0,
        .floatControls2 = features.float_controls2 != 0,
        .expectAssume = features.expect_assume != 0,
        .nonSemanticInfo = features.non_semantic_info != 0,
        .pipelineStatistics = features.pipeline_statistics != 0,
        .occlusionQueryPrecise = features.occlusion_query_precise != 0,
        .sparseBinding = features.sparse_binding != 0,
        .sparseBuffers = features.sparse_buffers != 0,
        .sparseTextures2d = features.sparse_textures_2d != 0,
        .sparseTextures3d = features.sparse_textures_3d != 0,
        .sparseSamples2 = features.sparse_samples_2 != 0,
        .sparseSamples4 = features.sparse_samples_4 != 0,
        .sparseSamples8 = features.sparse_samples_8 != 0,
        .sparseSamples16 = features.sparse_samples_16 != 0,
        .sparseAliased = features.sparse_aliased != 0,
        .sparseShaderResidency = features.sparse_shader_residency != 0,
    };
}

} // namespace

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
        const auto queues = queuesIn(desc->queues, desc->queue_count, error);
        if (!queues) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        if (desc->indirect_strides == nullptr && desc->indirect_stride_count != 0) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "indirect_strides is null but indirect_stride_count is not 0");
        }
        auto created = instance->instance->createDevice(
            desc->adapter, {.queues = *queues,
                            .features = fromC(desc->features),
                            .maxCommandLists = desc->max_command_lists,
                            .maxCommandPools = desc->max_command_pools,
                            .maxSwapchains = desc->max_swapchains,
                            .maxBindlessTextures = desc->max_bindless_textures,
                            .maxBindlessStorageTextures = desc->max_bindless_storage_textures,
                            .maxBindlessStorageBuffers = desc->max_bindless_storage_buffers,
                            .maxBindlessSamplers = desc->max_bindless_samplers,
                            .maxBindlessUniformBuffers = desc->max_bindless_uniform_buffers,
                            .maxTimestampPools = desc->max_timestamp_pools,
                            .maxQueryPools = desc->max_query_pools,
                            .maxDebugLabelLength = desc->max_debug_label_length,
                            .maxSparseMemories = desc->max_sparse_memories,
                            .maxTileMappings = desc->max_tile_mappings,
                            .maxBindlessAccelerationStructures = desc->max_bindless_acceleration_structures,
                            .maxBindlessPartitionedTops = desc->max_bindless_partitioned_tops,
                            .maxBuildGeometries = desc->max_build_geometries,
                            .maxAccelerationStructurePropertyPools = desc->max_acceleration_structure_property_pools,
                            .maxBuffers = desc->max_buffers,
                            .maxTextures = desc->max_textures,
                            .maxShaders = desc->max_shaders,
                            .maxPipelines = desc->max_pipelines,
                            .maxPipelineCaches = desc->max_pipeline_caches,
                            .maxAccelerationStructures = desc->max_acceleration_structures,
                            .indirectStrides = {desc->indirect_strides, desc->indirect_stride_count}});
        if (!created) {
            return fail(created.error());
        }
        auto made = std::make_unique<bfr_device>();
        made->instance = instance->instance;
        made->device = std::move(*created);
        made->triangles.resize(desc->max_build_geometries);
        made->boxes.resize(desc->max_build_geometries);
        made->instanceSize = made->device->caps().rayTracing.instanceSize;
        made->motionInstanceSize = made->device->caps().rayTracing.motionInstanceSize;
        made->adapter = desc->adapter;
        made->backend = instance->backend;
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
        bfr_result failed = BFR_RESULT_SUCCESS;
        const auto window = surfaceWindow(desc->window, failed);
        if (!window) {
            return failed;
        }
        const auto made = swapchainIn(*desc);
        const auto created = device->device->createSwapchain(*window, made);
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
        const auto indexType = static_cast<bfr::IndexType>(type);
        const auto bound = device->device->bindIndexBuffer(fromC(command_list), fromC(buffer), offset, indexType);
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
        const auto known = static_cast<bfr::Topology>(topology);
        const auto set = device->device->setTopology(fromC(command_list), known);
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
                               bfr_texture_layout before, bfr_texture_layout after, const bfr_texture_range* range) {
    return guarded([&] {
        if (device == nullptr || range == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and range must not be null");
        }
        const auto recorded = device->device->textureBarrier(fromC(command_list), fromC(texture),
                                                             static_cast<bfr::TextureLayout>(before),
                                                             static_cast<bfr::TextureLayout>(after),
                                                             {.mipLevel = range->mip_level,
                                                              .mipCount = range->mip_count,
                                                              .layer = range->layer,
                                                              .layerCount = range->layer_count});
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

bfr_result bfr_run_garbage_collection(bfr_device* device) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto collected = device->device->runGarbageCollection();
        return collected ? BFR_RESULT_SUCCESS : fail(collected.error());
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
        *caps = toC(listed, listed.caps, instance->backend);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_device_caps(bfr_device* device, bfr_caps* caps) {
    return guarded([&] {
        if (device == nullptr || caps == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and caps must not be null");
        }
        *caps = toC(device->instance->adapters()[device->adapter], device->device->caps(), device->backend);
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
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_swapchain_query takes no chained structs");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        bfr_result failed = BFR_RESULT_SUCCESS;
        const auto window = surfaceWindow(desc->window, failed);
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
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_swapchain_query takes no chained structs");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        bfr_result failed = BFR_RESULT_SUCCESS;
        const auto window = surfaceWindow(desc->window, failed);
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

bfr_result bfr_get_queue_families(bfr_instance* instance, uint32_t adapter, uint32_t* count,
                                  bfr_queue_family* families) {
    return guarded([&] {
        if (instance == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and count must not be null");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        const std::span<const bfr::rhi::QueueFamily> found = instance->instance->adapters()[adapter].caps.queueFamilies;
        if (families == nullptr) {
            *count = static_cast<std::uint32_t>(found.size());
            return BFR_RESULT_SUCCESS;
        }
        return writeList(found, count, families);
    });
}

bfr_result bfr_get_queue_family_presents(bfr_instance* instance, uint32_t adapter, uint32_t family,
                                         const bfr_swapchain_query* desc, bfr_bool* presents) {
    return guarded([&] {
        if (instance == nullptr || desc == nullptr || presents == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance, desc and presents must not be null");
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_swapchain_query takes no chained structs");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        bfr_result failed = BFR_RESULT_SUCCESS;
        const auto window = surfaceWindow(desc->window, failed);
        if (!window) {
            return failed;
        }
        const auto answer = instance->instance->queueFamilyPresents(adapter, family, *window);
        if (!answer) {
            return fail(answer.error());
        }
        *presents = toC(*answer);
        return BFR_RESULT_SUCCESS;
    });
}

// Built only on bfr_get_queue_families and bfr_get_queue_family_presents, as
// a caller could build it: the families with query's flags, in the backend's
// order, then the first of them that presents where query names a window.
bfr_result bfr_find_queue_family(bfr_instance* instance, uint32_t adapter, const bfr_queue_family_query* query,
                                 uint32_t* family) {
    return guarded([&] {
        if (query == nullptr || family == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "query and family must not be null");
        }
        std::uint32_t count = 0;
        if (const bfr_result listed = bfr_get_queue_families(instance, adapter, &count, nullptr); listed < 0) {
            return listed;
        }
        std::vector<bfr_queue_family> families(count);
        if (const bfr_result listed = bfr_get_queue_families(instance, adapter, &count, families.data()); listed < 0) {
            return listed;
        }
        const auto runs = [query](const bfr_queue_family& candidate) {
            return (query->graphics == BFR_FALSE || candidate.graphics != BFR_FALSE) &&
                   (query->compute == BFR_FALSE || candidate.compute != BFR_FALSE) &&
                   (query->copy == BFR_FALSE || candidate.copy != BFR_FALSE);
        };
        const bfr_swapchain_query presentsTo{.next = nullptr,
                                             .window = query->presents != nullptr ? *query->presents : bfr_window{}};
        for (std::uint32_t index = 0; index < count; ++index) {
            if (!runs(families[index])) {
                continue;
            }
            bfr_bool presents = BFR_TRUE;
            if (query->presents != nullptr) {
                if (const bfr_result asked = bfr_get_queue_family_presents(instance, adapter, index, &presentsTo,
                                                                           &presents);
                    asked < 0) {
                    return asked;
                }
            }
            if (presents != BFR_FALSE) {
                *family = index;
                return BFR_RESULT_SUCCESS;
            }
        }
        std::vector<std::string_view> asked;
        for (const auto& [wanted, what] :
             {std::pair{query->graphics, "runs graphics"}, std::pair{query->compute, "runs compute"},
              std::pair{query->copy, "runs copies"},
              std::pair{toC(query->presents != nullptr), "presents to the window"}}) {
            if (wanted != BFR_FALSE) {
                asked.emplace_back(what);
            }
        }
        if (asked.empty()) {
            return fail(BFR_RESULT_ERROR_UNSUPPORTED, "adapter {} has no queue family", adapter);
        }
        return fail(BFR_RESULT_ERROR_UNSUPPORTED, "adapter {} has no queue family that {}", adapter,
                    bfr::core::text::join(asked, " and "));
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_buffer_desc takes no chained structs");
        }
        const auto wanted = bufferIn(*desc);
        const auto created = device->device->createBuffer(wanted);
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

bfr_result bfr_get_buffer_slots(bfr_device* device, bfr_buffer buffer, bfr_buffer_slots* slots) {
    return guarded([&] {
        if (device == nullptr || slots == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and slots must not be null");
        }
        *slots = {.storage = BFR_NO_SLOT, .uniform = BFR_NO_SLOT};
        const auto found = device->device->bufferSlots(fromC(buffer));
        if (!found) {
            return fail(found.error());
        }
        *slots = {.storage = found->storage, .uniform = found->uniform};
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_texture_desc takes no chained structs");
        }
        const auto wanted = textureIn(*desc);
        const auto created = device->device->createTexture(wanted);
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
        const auto wanted = samplerIn(*desc);
        const auto created = device->device->createSampler(wanted);
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
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
        const auto wanted = graphicsPipelineIn(*desc);
        const auto created = device->device->createGraphicsPipeline(wanted);
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
        if (desc->data == nullptr && desc->data_size != 0) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "data is null but data_size is not 0");
        }
        const auto created = device->device->createPipelineCache(
            {.data = std::as_bytes(std::span{desc->data, desc->data_size}), .name = fromC(desc->name)});
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
        const auto created = device->device->createTimestampPool(
            {.queue = desc->queue, .count = desc->count, .name = fromC(desc->name)});
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
        const auto at = static_cast<bfr::TimestampStage>(stage);
        const auto how = static_cast<bfr::TimestampGranularity>(granularity);
        const auto written = device->device->writeTimestamp(fromC(command_list), fromC(pool), index, at, how);
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

bfr_result bfr_create_query_pool(bfr_device* device, const bfr_query_pool_desc* desc, bfr_query_pool* pool) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || pool == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and pool must not be null");
        }
        *pool = {};
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
        const auto type = static_cast<bfr::QueryType>(desc->type);
        const bfr_pipeline_statistics_mask& mask = desc->statistics;
        const auto created = device->device->createQueryPool(
            {.type = type,
             .count = desc->count,
             .statistics = {.inputVertices = mask.input_vertices != BFR_FALSE,
                            .inputPrimitives = mask.input_primitives != BFR_FALSE,
                            .vertexInvocations = mask.vertex_invocations != BFR_FALSE,
                            .geometryInvocations = mask.geometry_invocations != BFR_FALSE,
                            .geometryPrimitives = mask.geometry_primitives != BFR_FALSE,
                            .clipInvocations = mask.clip_invocations != BFR_FALSE,
                            .clipPrimitives = mask.clip_primitives != BFR_FALSE,
                            .fragmentInvocations = mask.fragment_invocations != BFR_FALSE,
                            .tessellationControl = mask.tessellation_control != BFR_FALSE,
                            .tessellationEvaluationInvocations = mask.tessellation_evaluation_invocations != BFR_FALSE,
                            .computeInvocations = mask.compute_invocations != BFR_FALSE,
                            .taskInvocations = mask.task_invocations != BFR_FALSE,
                            .meshInvocations = mask.mesh_invocations != BFR_FALSE},
             .name = fromC(desc->name)});
        if (!created) {
            return fail(created.error());
        }
        *pool = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_query_pool(bfr_device* device, bfr_query_pool pool) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyQueryPool(fromC(pool)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_get_query_layout(bfr_device* device, bfr_query_pool pool, bfr_query_layout* layout) {
    return guarded([&] {
        if (device == nullptr || layout == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and layout must not be null");
        }
        *layout = {};
        const auto found = device->device->queryLayout(fromC(pool));
        if (!found) {
            return fail(found.error());
        }
        layout->stride = found->stride;
        layout->value = found->value;
        std::ranges::copy(found->statistics, std::span{layout->statistics}.begin());
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_reset_queries(bfr_device* device, bfr_command_list command_list, bfr_query_pool pool, uint32_t first,
                             uint32_t count) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto reset = device->device->resetQueries(fromC(command_list), fromC(pool), first, count);
        return reset ? BFR_RESULT_SUCCESS : fail(reset.error());
    });
}

bfr_result bfr_begin_query(bfr_device* device, bfr_command_list command_list, bfr_query_pool pool, uint32_t index,
                           bfr_bool precise) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto begun = device->device->beginQuery(fromC(command_list), fromC(pool), index, precise != BFR_FALSE);
        return begun ? BFR_RESULT_SUCCESS : fail(begun.error());
    });
}

bfr_result bfr_end_query(bfr_device* device, bfr_command_list command_list, bfr_query_pool pool, uint32_t index) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto ended = device->device->endQuery(fromC(command_list), fromC(pool), index);
        return ended ? BFR_RESULT_SUCCESS : fail(ended.error());
    });
}

bfr_result bfr_resolve_queries(bfr_device* device, bfr_command_list command_list, bfr_query_pool pool, uint32_t first,
                               uint32_t count, bfr_buffer buffer, uint64_t offset) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto resolved = device->device->resolveQueries(fromC(command_list), fromC(pool), first, count,
                                                             fromC(buffer), offset);
        return resolved ? BFR_RESULT_SUCCESS : fail(resolved.error());
    });
}

bfr_result bfr_begin_debug_label(bfr_device* device, bfr_command_list command_list, const bfr_debug_label* label) {
    return guarded([&] {
        if (device == nullptr || !isLabel(label)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "device and label must not be null, nor the label's name where its length is not 0");
        }
        const auto begun = device->device->beginDebugLabel(fromC(command_list), fromC(*label));
        return begun ? BFR_RESULT_SUCCESS : fail(begun.error());
    });
}

bfr_result bfr_end_debug_label(bfr_device* device, bfr_command_list command_list) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto ended = device->device->endDebugLabel(fromC(command_list));
        return ended ? BFR_RESULT_SUCCESS : fail(ended.error());
    });
}

bfr_result bfr_insert_debug_label(bfr_device* device, bfr_command_list command_list, const bfr_debug_label* label) {
    return guarded([&] {
        if (device == nullptr || !isLabel(label)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "device and label must not be null, nor the label's name where its length is not 0");
        }
        const auto inserted = device->device->insertDebugLabel(fromC(command_list), fromC(*label));
        return inserted ? BFR_RESULT_SUCCESS : fail(inserted.error());
    });
}

bfr_result bfr_begin_queue_debug_label(bfr_device* device, uint32_t queue, const bfr_debug_label* label) {
    return guarded([&] {
        if (device == nullptr || !isLabel(label)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "device and label must not be null, nor the label's name where its length is not 0");
        }
        const auto begun = device->device->beginQueueDebugLabel(queue, fromC(*label));
        return begun ? BFR_RESULT_SUCCESS : fail(begun.error());
    });
}

bfr_result bfr_end_queue_debug_label(bfr_device* device, uint32_t queue) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto ended = device->device->endQueueDebugLabel(queue);
        return ended ? BFR_RESULT_SUCCESS : fail(ended.error());
    });
}

bfr_result bfr_insert_queue_debug_label(bfr_device* device, uint32_t queue, const bfr_debug_label* label) {
    return guarded([&] {
        if (device == nullptr || !isLabel(label)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "device and label must not be null, nor the label's name where its length is not 0");
        }
        const auto inserted = device->device->insertQueueDebugLabel(queue, fromC(*label));
        return inserted ? BFR_RESULT_SUCCESS : fail(inserted.error());
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
        const auto read = static_cast<bfr::HostClock>(clock);
        const auto taken = device->device->calibrateTimestamps(queue, read);
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
        const auto read = static_cast<bfr::TextureAspect>(aspect);
        const auto found = device->device->textureTiling(fromC(texture), read);
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
        // members.
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
                                       .tile_bytes = desc->tile_bytes,
                                       .name = {}};
        const auto wanted = textureIn(texture);
        const auto aspect = static_cast<bfr::TextureAspect>(desc->aspect);
        const auto found = device->device->sparseFormat({.format = wanted.format,
                                                         .dimension = wanted.dimension,
                                                         .samples = wanted.samples,
                                                         .usage = wanted.usage,
                                                         .aspect = aspect,
                                                         .tileBytes = wanted.tileBytes});
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
        *sizes = {.size = found->size,
                  .build_scratch = found->buildScratch,
                  .update_scratch = found->updateScratch,
                  .discardable = toC(found->discardable)};
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_acceleration_structure_desc takes no chained structs");
        }
        const auto type = static_cast<bfr::AccelerationStructureType>(desc->type);
        const auto made = device->device->createAccelerationStructure(
            {.type = type,
             .size = desc->size,
             .motionKeyframes = desc->motion_keyframes,
             .motionInstances = desc->motion_instances,
             .captureReplay = desc->capture_replay != BFR_FALSE,
             .replay = {.words = std::to_array(desc->replay.words)},
             .name = fromC(desc->name)});
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

bfr_result bfr_capture_acceleration_structure(bfr_device* device, bfr_acceleration_structure structure,
                                              bfr_acceleration_structure_capture* capture) {
    return guarded([&] {
        if (device == nullptr || capture == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and capture must not be null");
        }
        *capture = {};
        const auto found = device->device->captureAccelerationStructure(fromC(structure));
        if (!found) {
            return fail(found.error());
        }
        std::ranges::copy(found->words, capture->words);
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

namespace {

// instance as C++ has it.
bfr::rhi::AccelerationStructureInstance instanceIn(const bfr_acceleration_structure_instance& instance) {
    using namespace bfr;
    using namespace bfr::rhi;
    const auto opacity = static_cast<InstanceOpacity>(instance.opacity);
    return AccelerationStructureInstance{.transform = std::to_array(instance.transform),
                                         .customIndex = instance.custom_index,
                                         .mask = instance.mask,
                                         .hitGroupOffset = instance.hit_group_offset,
                                         .cullDisable = instance.cull_disable != BFR_FALSE,
                                         .frontCounterClockwise = instance.front_counter_clockwise != BFR_FALSE,
                                         .opacity = opacity,
                                         .forceOpacityMicromap2State = instance.force_opacity_micromap_2_state !=
                                                                       BFR_FALSE,
                                         .disableOpacityMicromaps = instance.disable_opacity_micromaps != BFR_FALSE,
                                         .bottomStructure = fromC(instance.bottom_structure),
                                         .bottomReference = instance.bottom_reference};
}

} // namespace

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
                chunk.at(i) = instanceIn(from[done + i]);
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

// Keyframes are read in place: the C and C++ layouts are one, all floats.
static_assert(sizeof(bfr_motion_keyframe) == sizeof(bfr::rhi::MotionKeyframe) &&
              offsetof(bfr_motion_keyframe, srt) == offsetof(bfr::rhi::MotionKeyframe, srt) &&
              sizeof(bfr_srt_transform) == sizeof(bfr::rhi::SrtTransform) &&
              offsetof(bfr_srt_transform, rotation) == offsetof(bfr::rhi::SrtTransform, rotation) &&
              offsetof(bfr_srt_transform, translation) == offsetof(bfr::rhi::SrtTransform, translation));

bfr_result bfr_write_motion_instances(bfr_device* device, const bfr_acceleration_structure_motion_instance* instances,
                                      uint32_t instance_count, const bfr_motion_keyframe* keyframes,
                                      uint32_t keyframe_count, void* data, size_t size) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        if ((instances == nullptr && instance_count != 0) || (keyframes == nullptr && keyframe_count != 0) ||
            (data == nullptr && size != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "instances, keyframes and data must not be null with a count");
        }
        using namespace bfr;
        using namespace bfr::rhi;
        const std::span out{static_cast<std::byte*>(data), size};
        const std::span from{instances, instance_count};
        const std::span<const MotionKeyframe> frames{reinterpret_cast<const MotionKeyframe*>(keyframes),
                                                     keyframe_count};
        const std::size_t recordSize = device->motionInstanceSize;
        // Converted a chunk at a time on the stack, as bfr_write_instances
        // does. Each chunk's write checks the bytes it takes: the last
        // record needs only the backend's own bytes, which caps don't give.
        constexpr std::size_t kChunk = 64;
        std::array<AccelerationStructureMotionInstance, kChunk> chunk{};
        std::size_t done = 0;
        do {
            const std::size_t count = std::min(kChunk, from.size() - done);
            for (std::size_t i = 0; i < count; ++i) {
                const bfr_acceleration_structure_motion_instance& moving = from[done + i];
                const auto instance = instanceIn(moving.instance);
                const auto motion = static_cast<InstanceMotion>(moving.motion);
                chunk.at(i) = {.instance = instance,
                               .motion = motion,
                               .firstKeyframe = moving.first_keyframe,
                               .keyframeCount = moving.keyframe_count};
            }
            const auto written = device->device->writeMotionInstances(std::span{chunk}.first(count), frames,
                                                                      out.subspan(
                                                                          std::min(done * recordSize, out.size())));
            if (!written) {
                return fail(written.error());
            }
            done += count;
        } while (done < from.size());
        return BFR_RESULT_SUCCESS;
    });
}

namespace {

// build with inputs, converted from its C inputs.
bfr::rhi::AccelerationStructureBuild buildIn(const bfr_acceleration_structure_build& build,
                                             const bfr::rhi::AccelerationStructureInputs& inputs) {
    return {.dst = fromC(build.dst),
            .src = fromC(build.src),
            .update = build.update != BFR_FALSE,
            .inputs = inputs,
            .scratch = fromC(build.scratch),
            .scratchOffset = build.scratch_offset};
}

} // namespace

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
        const auto built = device->device->buildAccelerationStructure(fromC(command_list), buildIn(*build, *inputs));
        return built ? BFR_RESULT_SUCCESS : fail(built.error());
    });
}

bfr_result bfr_build_acceleration_structure_indirect(bfr_device* device, bfr_command_list command_list,
                                                     const bfr_acceleration_structure_build* build, bfr_buffer ranges,
                                                     uint64_t offset, uint32_t stride) {
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
        const auto built = device->device->buildAccelerationStructureIndirect(fromC(command_list),
                                                                              buildIn(*build, *inputs), fromC(ranges),
                                                                              offset, stride);
        return built ? BFR_RESULT_SUCCESS : fail(built.error());
    });
}

bfr_result bfr_get_cluster_sizes(bfr_device* device, const bfr_cluster_inputs* inputs,
                                 bfr_acceleration_structure_sizes* sizes) {
    return guarded([&] {
        if (device == nullptr || inputs == nullptr || sizes == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, inputs and sizes must not be null");
        }
        *sizes = {};
        const auto made = clusterInputsIn(*inputs);
        const auto found = device->device->clusterSizes(made);
        if (!found) {
            return fail(found.error());
        }
        *sizes = {.size = found->size,
                  .build_scratch = found->buildScratch,
                  .update_scratch = found->updateScratch,
                  .discardable = toC(found->discardable)};
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_build_clusters(bfr_device* device, bfr_command_list command_list, const bfr_cluster_build* build) {
    return guarded([&] {
        if (device == nullptr || build == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and build must not be null");
        }
        const auto inputs = clusterInputsIn(build->inputs);
        const bfr_cluster_indirection& indirection = build->indirection;
        const auto built = device->device->buildClusters(
            fromC(command_list), {.inputs = inputs,
                                  .destination = fromC(build->destination),
                                  .destinationOffset = build->destination_offset,
                                  .scratch = fromC(build->scratch),
                                  .scratchOffset = build->scratch_offset,
                                  .addresses = fromC(build->addresses),
                                  .addressOffset = build->address_offset,
                                  .addressStride = build->address_stride,
                                  .sizes = fromC(build->sizes),
                                  .sizeOffset = build->size_offset,
                                  .sizeStride = build->size_stride,
                                  .records = fromC(build->records),
                                  .recordOffset = build->record_offset,
                                  .recordStride = build->record_stride,
                                  .count = fromC(build->count),
                                  .countOffset = build->count_offset,
                                  .indirection = {.destination = indirection.destination != BFR_FALSE,
                                                  .scratch = indirection.scratch != BFR_FALSE,
                                                  .addresses = indirection.addresses != BFR_FALSE,
                                                  .sizes = indirection.sizes != BFR_FALSE,
                                                  .records = indirection.records != BFR_FALSE,
                                                  .count = indirection.count != BFR_FALSE}});
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
        if (const bfr_result named = checkName(desc->name); named != BFR_RESULT_SUCCESS) {
            return named;
        }
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_ray_tracing_pipeline_desc takes no chained structs");
        }
        if ((desc->shaders == nullptr && desc->shader_count != 0) ||
            (desc->groups == nullptr && desc->group_count != 0) ||
            (desc->libraries == nullptr && desc->library_count != 0) ||
            (desc->replay_identifiers == nullptr && desc->replay_identifiers_size != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "shaders, groups, libraries or replay_identifiers is null with a count");
        }
        using namespace bfr;
        using namespace bfr::rhi;
        // Cold, so the converted arrays are vectors.
        std::vector<RayTracingShader> shaders;
        shaders.reserve(desc->shader_count);
        for (const bfr_ray_tracing_shader& shader : std::span{desc->shaders, desc->shader_count}) {
            const auto stage = static_cast<RayTracingStage>(shader.stage);
            shaders.push_back({.stage = stage, .shader = fromC(shader.shader)});
        }
        std::vector<RayTracingGroup> groups;
        groups.reserve(desc->group_count);
        for (const bfr_ray_tracing_group& group : std::span{desc->groups, desc->group_count}) {
            const auto type = static_cast<RayTracingGroupType>(group.type);
            groups.push_back({.type = type,
                              .general = group.general,
                              .closestHit = group.closest_hit,
                              .anyHit = group.any_hit,
                              .intersection = group.intersection});
        }
        std::vector<PipelineHandle> libraries;
        libraries.reserve(desc->library_count);
        for (const bfr_pipeline library : std::span{desc->libraries, desc->library_count}) {
            libraries.push_back(fromC(library));
        }
        const auto created = device->device->createRayTracingPipeline(
            {.shaders = shaders,
             .groups = groups,
             .maxRecursionDepth = desc->max_recursion_depth,
             .maxPayloadSize = desc->max_payload_size,
             .maxAttributeSize = desc->max_attribute_size,
             .cache = fromC(desc->cache),
             .stackSize = desc->stack_size,
             .library = desc->library != BFR_FALSE,
             .libraries = libraries,
             .captureReplay = desc->capture_replay != BFR_FALSE,
             .replayIdentifiers = std::as_bytes(std::span{desc->replay_identifiers, desc->replay_identifiers_size}),
             .allowOpacityMicromaps = desc->allow_opacity_micromaps != BFR_FALSE,
             .allowMotion = desc->allow_motion != BFR_FALSE,
             .allowClusters = desc->allow_clusters != BFR_FALSE,
             .name = fromC(desc->name)});
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

bfr_result bfr_get_capture_replay_identifiers(bfr_device* device, bfr_pipeline pipeline, uint32_t first, uint32_t count,
                                              void* data, size_t size) {
    return guarded([&] {
        if (device == nullptr || (data == nullptr && size != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null, nor data with a size");
        }
        const auto written = device->device->captureReplayIdentifiers(fromC(pipeline), first, count,
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

bfr_result bfr_create_acceleration_structure_property_pool(bfr_device* device,
                                                           const bfr_acceleration_structure_property_pool_desc* desc,
                                                           bfr_acceleration_structure_property_pool* pool) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || pool == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and pool must not be null");
        }
        *pool = {};
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        "bfr_acceleration_structure_property_pool_desc takes no chained structs");
        }
        const auto property = static_cast<bfr::AccelerationStructureProperty>(desc->property);
        const auto made = device->device->createAccelerationStructurePropertyPool(
            {.property = property, .count = desc->count, .opacityMicromaps = desc->opacity_micromaps != BFR_FALSE});
        if (!made) {
            return fail(made.error());
        }
        *pool = toC(*made);
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_acceleration_structure_property_pool(bfr_device* device,
                                                      bfr_acceleration_structure_property_pool pool) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        if (const auto destroyed = device->device->destroyAccelerationStructurePropertyPool(fromC(pool)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

bfr_result bfr_reset_acceleration_structure_properties(bfr_device* device, bfr_command_list command_list,
                                                       bfr_acceleration_structure_property_pool pool, uint32_t first,
                                                       uint32_t count) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto reset = device->device->resetAccelerationStructureProperties(fromC(command_list), fromC(pool), first,
                                                                                count);
        return reset ? BFR_RESULT_SUCCESS : fail(reset.error());
    });
}

bfr_result bfr_write_acceleration_structure_properties(bfr_device* device, bfr_command_list command_list,
                                                       bfr_acceleration_structure_property_pool pool, uint32_t first,
                                                       const bfr_acceleration_structure* structures,
                                                       uint32_t structure_count) {
    return guarded([&] {
        if (device == nullptr || (structures == nullptr && structure_count != 0)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null, nor structures with a count");
        }
        // The caller's handles, read in place; see the layout assert above.
        const auto written = device->device->writeAccelerationStructureProperties(
            fromC(command_list), fromC(pool), first,
            {reinterpret_cast<const bfr::rhi::AccelerationStructureHandle*>(structures), structure_count});
        return written ? BFR_RESULT_SUCCESS : fail(written.error());
    });
}

bfr_result bfr_resolve_acceleration_structure_properties(bfr_device* device, bfr_command_list command_list,
                                                         bfr_acceleration_structure_property_pool pool, uint32_t first,
                                                         uint32_t count, bfr_buffer buffer, uint64_t offset) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto resolved = device->device->resolveAccelerationStructureProperties(fromC(command_list), fromC(pool),
                                                                                     first, count, fromC(buffer),
                                                                                     offset);
        return resolved ? BFR_RESULT_SUCCESS : fail(resolved.error());
    });
}

bfr_result bfr_compact_acceleration_structure(bfr_device* device, bfr_command_list command_list,
                                              bfr_acceleration_structure dst, bfr_acceleration_structure src) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto copied = device->device->compactAccelerationStructure(fromC(command_list), fromC(dst), fromC(src));
        return copied ? BFR_RESULT_SUCCESS : fail(copied.error());
    });
}

bfr_result bfr_serialize_acceleration_structure(bfr_device* device, bfr_command_list command_list,
                                                bfr_acceleration_structure src, bfr_buffer buffer, uint64_t offset) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto serialized = device->device->serializeAccelerationStructure(fromC(command_list), fromC(src),
                                                                               fromC(buffer), offset);
        return serialized ? BFR_RESULT_SUCCESS : fail(serialized.error());
    });
}

bfr_result bfr_deserialize_acceleration_structure(bfr_device* device, bfr_command_list command_list, bfr_buffer buffer,
                                                  uint64_t offset, bfr_acceleration_structure dst) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto deserialized = device->device->deserializeAccelerationStructure(fromC(command_list), fromC(buffer),
                                                                                   offset, fromC(dst));
        return deserialized ? BFR_RESULT_SUCCESS : fail(deserialized.error());
    });
}

bfr_result bfr_serialized_compatible(bfr_device* device, bfr_acceleration_structure_type type, void* identifier,
                                     bfr_bool* compatible) {
    return guarded([&] {
        if (device == nullptr || identifier == nullptr || compatible == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, identifier and compatible must not be null");
        }
        *compatible = BFR_FALSE;
        const auto which = static_cast<bfr::AccelerationStructureType>(type);
        const auto found = device->device->serializedCompatible(
            which,
            std::span<const std::byte, bfr::rhi::kSerializedIdentifierSize>{static_cast<const std::byte*>(identifier),
                                                                            bfr::rhi::kSerializedIdentifierSize});
        if (!found) {
            return fail(found.error());
        }
        *compatible = toC(*found);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_ray_tracing_stack_size(bfr_device* device, bfr_pipeline pipeline, uint32_t group,
                                          bfr_ray_tracing_group_shader shader, uint64_t* stack_size) {
    return guarded([&] {
        if (device == nullptr || stack_size == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and stack_size must not be null");
        }
        *stack_size = 0;
        const auto which = static_cast<bfr::RayTracingGroupShader>(shader);
        const auto found = device->device->rayTracingStackSize(fromC(pipeline), group, which);
        if (!found) {
            return fail(found.error());
        }
        *stack_size = *found;
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
