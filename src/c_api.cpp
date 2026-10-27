// The C API in bfr.h, over the C++ implementation. No exception crosses it:
// each entry point turns one into a result.

#include "bfr.h"
#include "caps.h"
#include "device.h"
#include "error.h"
#include "format.h"
#include "log.h"
#include "rhi.h"
#include "rhi_config.h"
#include "swapchain.h"
#include "window.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

// Shares the Rhi with its devices, which keep it alive: garbage-collected
// wrappers destroy them in no fixed order.
struct bfr_instance {
    std::shared_ptr<bfr::rhi::Rhi> rhi;
    bfr_backend backend;
};

// The Rhi comes first so that it outlives the device it made.
struct bfr_device {
    std::shared_ptr<bfr::rhi::Rhi> rhi;
    std::unique_ptr<bfr::rhi::Device> device;
};

namespace {

using bfr::core::ErrorCode;
using bfr::core::Severity;
using bfr::core::window::NativeWindow;

// The C and C++ severities are the same numbers.
static_assert(static_cast<int>(Severity::Trace) == BFR_MESSAGE_SEVERITY_TRACE);
static_assert(static_cast<int>(Severity::Error) == BFR_MESSAGE_SEVERITY_ERROR);

// The least severity a desc asks for: 0 means warning. Nullopt for a value
// that is not a severity.
std::optional<Severity> severityFrom(bfr_message_severity severity) {
    if (severity == 0) {
        return Severity::Warning;
    }
    if (severity < BFR_MESSAGE_SEVERITY_TRACE || severity > BFR_MESSAGE_SEVERITY_ERROR) {
        return std::nullopt;
    }
    return static_cast<Severity>(severity);
}

// What bfr_swapchain_desc.image_count = 0 asks for.
constexpr std::uint32_t kDefaultSwapchainImages = 3;

// The calling thread's last error, which bfr_get_last_error returns.
std::string& lastError() {
    thread_local std::string error;
    return error;
}

// Records message as the calling thread's last error and returns result.
bfr_result fail(bfr_result result, std::string_view message) noexcept {
    try {
        lastError() = message;
    } catch (...) {
        lastError().clear();
    }
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
        case ErrorCode::RhiSurfaceFailed:
            return BFR_RESULT_ERROR_SURFACE_FAILED;
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
        case ErrorCode::RhiSurfaceLost:
            return BFR_RESULT_ERROR_SURFACE_LOST;
        case ErrorCode::RhiUnsupported:
            return BFR_RESULT_ERROR_UNSUPPORTED;
        case ErrorCode::RhiQueryFailed:
            return BFR_RESULT_ERROR_QUERY_FAILED;
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

// The extension struct that chain heads. Every one starts with its bfr_chain,
// so the two share an address; chain->type has said which struct it is.
template <class Extension>
const Extension* extension(const bfr_chain* chain) {
    static_assert(offsetof(Extension, chain) == 0);
    return reinterpret_cast<const Extension*>(chain); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

// The window chained to next: none, one, or an error for a second window or a
// struct this desc does not take.
std::optional<NativeWindow> windowIn(const bfr_chain* next, std::string& error) {
    NativeWindow window;
    for (; next != nullptr; next = next->next) {
        const bool first = std::holds_alternative<std::monostate>(window);
        switch (next->type) {
            case BFR_STRUCT_TYPE_WINDOW_XLIB: {
                const auto* xlib = extension<bfr_window_xlib>(next);
                window = bfr::core::window::Xlib{.display = xlib->display, .window = xlib->window};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_WAYLAND: {
                const auto* wayland = extension<bfr_window_wayland>(next);
                window = bfr::core::window::Wayland{.display = wayland->display, .surface = wayland->surface};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_WIN32: {
                const auto* win32 = extension<bfr_window_win32>(next);
                window = bfr::core::window::Win32{.instance = win32->instance, .window = win32->window};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_METAL: {
                const auto* metal = extension<bfr_window_metal>(next);
                window = bfr::core::window::Metal{.layer = metal->layer};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_HEADLESS:
                window = bfr::core::window::Headless{};
                break;
            case BFR_STRUCT_TYPE_FORCE32:
            default:
                error = std::format("struct type {} cannot be chained here", static_cast<std::int64_t>(next->type));
                return std::nullopt;
        }
        if (!first) {
            error = "more than one window is chained";
            return std::nullopt;
        }
    }
    return window;
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
    return value ? BFR_TRUE : BFR_FALSE;
}

// The C and C++ enums below are the same numbers.
static_assert(static_cast<int>(bfr::rhi::Format::R8Unorm) == BFR_FORMAT_R8_UNORM);
static_assert(static_cast<int>(bfr::rhi::Format::Astc12x12Float) == BFR_FORMAT_ASTC_12X12_FLOAT);
static_assert(static_cast<int>(bfr::rhi::ColorSpace::DisplayNative) == BFR_COLOR_SPACE_DISPLAY_NATIVE);
static_assert(static_cast<int>(bfr::rhi::SurfaceRotation::Other) == BFR_SURFACE_ROTATION_OTHER);

bfr_format toC(bfr::rhi::Format format) {
    return static_cast<bfr_format>(format);
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
    return {
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
        .min_shading_rate_tile = limits.minShadingRateTile,
        .max_shading_rate_tile = limits.maxShadingRateTile,
        .min_uniform_buffer_alignment = limits.minUniformBufferAlignment,
        .min_storage_buffer_alignment = limits.minStorageBufferAlignment,
        .min_texel_buffer_alignment = limits.minTexelBufferAlignment,
        .copy_offset_alignment = limits.copyOffsetAlignment,
        .copy_row_pitch_alignment = limits.copyRowPitchAlignment,
        .non_coherent_atom_size = limits.nonCoherentAtomSize,
        .buffer_texture_granularity = limits.bufferTextureGranularity,
        .timestamp_period = limits.timestampPeriod,
    };
}

bfr_caps_bindless toC(const bfr::rhi::BindlessLimits& bindless) {
    return {
        .max_textures = bindless.maxTextures,
        .max_storage_textures = bindless.maxStorageTextures,
        .max_storage_buffers = bindless.maxStorageBuffers,
        .max_uniform_buffers = bindless.maxUniformBuffers,
        .max_samplers = bindless.maxSamplers,
        .max_per_stage = bindless.maxPerStage,
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
    };
    std::ranges::copy(mesh.maxMeshWorkgroupSize, std::begin(out.max_mesh_workgroup_size));
    std::ranges::copy(mesh.maxTaskWorkgroupSize, std::begin(out.max_task_workgroup_size));
    return out;
}

bfr_caps_ray_tracing toC(const bfr::rhi::RayTracingLimits& rayTracing) {
    return {
        .max_geometries = rayTracing.maxGeometries,
        .max_instances = rayTracing.maxInstances,
        .max_primitives = rayTracing.maxPrimitives,
        .scratch_alignment = rayTracing.scratchAlignment,
        .max_recursion_depth = rayTracing.maxRecursionDepth,
        .handle_size = rayTracing.handleSize,
        .handle_alignment = rayTracing.handleAlignment,
        .table_alignment = rayTracing.tableAlignment,
    };
}

bfr_caps_features toC(const bfr::rhi::Features& features) {
    return {
        .ray_query = toC(features.rayQuery),
        .ray_tracing_pipeline = toC(features.rayTracingPipeline),
        .acceleration_structure = toC(features.accelerationStructure),
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
        .compression_bc = toC(features.compressionBc),
        .compression_etc2 = toC(features.compressionEtc2),
        .compression_astc = toC(features.compressionAstc),
        .compression_astc_hdr = toC(features.compressionAstcHdr),
        .multi_draw_indirect = toC(features.multiDrawIndirect),
        .draw_indirect_count = toC(features.drawIndirectCount),
        .draw_indirect_first_instance = toC(features.drawIndirectFirstInstance),
        .storage_read_without_format = toC(features.storageReadWithoutFormat),
        .storage_write_without_format = toC(features.storageWriteWithoutFormat),
        .timestamps_on_all_queues = toC(features.timestampsOnAllQueues),
        .pipeline_statistics = toC(features.pipelineStatistics),
        .occlusion_query_precise = toC(features.occlusionQueryPrecise),
        .sparse_binding = toC(features.sparseBinding),
        .sparse_textures = toC(features.sparseTextures),
        .sparse_buffers = toC(features.sparseBuffers),
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

bfr_caps_queues toC(const bfr::rhi::Queues& queues) {
    return {
        .graphics = queues.graphics,
        .compute = queues.compute,
        .copy = queues.copy,
        .graphics_timestamp_bits = queues.graphicsTimestampBits,
        .compute_timestamp_bits = queues.computeTimestampBits,
        .copy_timestamp_bits = queues.copyTimestampBits,
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
    };
}

bfr_surface_caps toC(const bfr::rhi::SurfaceCaps& caps) {
    return {
        .min_image_count = caps.minImageCount,
        .max_image_count = caps.maxImageCount,
        .width = caps.width,
        .height = caps.height,
        .min_width = caps.minWidth,
        .min_height = caps.minHeight,
        .max_width = caps.maxWidth,
        .max_height = caps.maxHeight,
        .rotation = static_cast<bfr_surface_rotation>(caps.rotation),
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

bfr_surface_format toC(const bfr::rhi::SurfaceFormat& format) {
    return {.format = toC(format.format), .color_space = static_cast<bfr_color_space>(format.colorSpace)};
}

// Fails unless adapter is an index into instance's adapters.
std::optional<bfr_result> checkAdapter(const bfr_instance& instance, std::uint32_t adapter) {
    const std::size_t count = instance.rhi->adapters().size();
    if (adapter >= count) {
        return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, std::format("adapter {} of {}", adapter, count));
    }
    return std::nullopt;
}

// The window chained to a bfr_surface_desc, or the result for one that
// chains none or something else.
std::optional<NativeWindow> surfaceWindow(const bfr_surface_desc& desc, bfr_result& failed) {
    std::string error;
    auto window = windowIn(desc.next, error);
    if (window && std::holds_alternative<std::monostate>(*window)) {
        window.reset();
        error = "bfr_surface_desc needs a window chained to it";
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
    const std::string& error = lastError();
    return {.data = error.data(), .length = error.size()};
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
        if (desc->backend != BFR_BACKEND_VULKAN) {
            return fail(BFR_RESULT_ERROR_BACKEND_UNAVAILABLE,
                        std::format("backend {} is not in this build", static_cast<std::int64_t>(desc->backend)));
        }
        const auto least = severityFrom(desc->min_severity);
        if (!least) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, std::format("min_severity {} is not a severity",
                                                                       static_cast<std::int64_t>(desc->min_severity)));
        }
        bfr::rhi::RhiDesc rhiDesc;
        if (desc->message_callback != nullptr) {
            rhiDesc.logger = {[callback = desc->message_callback,
                               userData = desc->message_user_data](Severity severity, std::string_view message) {
                                  callback(static_cast<bfr_message_severity>(severity),
                                           {.data = message.data(), .length = message.size()}, userData);
                              },
                              *least};
        }
        auto rhi = bfr::rhi::Rhi::create(rhiDesc, bfr::rhi::RhiConfig{.validation = desc->validation != BFR_FALSE});
        if (!rhi) {
            return fail(rhi.error());
        }
        // The caller owns it until bfr_destroy_instance.
        *instance =
            std::make_unique<bfr_instance>(bfr_instance{.rhi = std::move(*rhi), .backend = desc->backend}).release();
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_instance(bfr_instance* instance) {
    quietly([&] { std::unique_ptr<bfr_instance>{instance}.reset(); });
}

bfr_result bfr_instance_get_adapters(bfr_instance* instance, uint32_t* count, bfr_adapter_info* adapters) {
    return guarded([&] {
        if (instance == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and count must not be null");
        }
        const std::span<const bfr::rhi::Adapter> listed = instance->rhi->adapters();
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

bfr_result bfr_instance_create_device(bfr_instance* instance, const bfr_device_desc* desc, bfr_device** device) {
    return guarded([&] {
        if (instance == nullptr || desc == nullptr || device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance, desc and device must not be null");
        }
        *device = nullptr;
        std::string error;
        const auto window = windowIn(desc->next, error);
        if (!window) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        auto created = instance->rhi->createDevice(desc->adapter, *window);
        if (!created) {
            return fail(created.error());
        }
        // The caller owns it until bfr_destroy_device.
        *device =
            std::make_unique<bfr_device>(bfr_device{.rhi = instance->rhi, .device = std::move(*created)}).release();
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_device(bfr_device* device) {
    quietly([&] { std::unique_ptr<bfr_device>{device}.reset(); });
}

bfr_result bfr_device_create_swapchain(bfr_device* device, const bfr_swapchain_desc* desc, bfr_swapchain* swapchain) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || swapchain == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and swapchain must not be null");
        }
        *swapchain = {};
        std::string error;
        const auto window = windowIn(desc->next, error);
        if (!window) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        if (std::holds_alternative<std::monostate>(*window)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_swapchain_desc needs a window chained to it");
        }
        const std::uint32_t images = desc->image_count == 0 ? kDefaultSwapchainImages : desc->image_count;
        const auto created = device->device->createSwapchain(*window, images);
        if (!created) {
            return fail(created.error());
        }
        *swapchain = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_device_resize_swapchain(bfr_device* device, bfr_swapchain swapchain, uint32_t width, uint32_t height,
                                       bfr_swapchain_images* images) {
    return guarded([&] {
        if (device == nullptr || images == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and images must not be null");
        }
        const auto resized = device->device->resizeSwapchain(fromC(swapchain), width, height);
        if (!resized) {
            return fail(resized.error());
        }
        *images = {.width = resized->width, .height = resized->height, .count = resized->count};
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_device_destroy_swapchain(bfr_device* device, bfr_swapchain swapchain) {
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

bfr_result bfr_get_caps(bfr_instance* instance, uint32_t adapter, bfr_caps* caps) {
    return guarded([&] {
        if (instance == nullptr || caps == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and caps must not be null");
        }
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        const bfr::rhi::Adapter& listed = instance->rhi->adapters()[adapter];
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
            const auto found = instance->rhi->formatCaps(adapter, bfr::rhi::kFormats.at(i));
            if (!found) {
                return fail(found.error());
            }
            out[i] = toC(*found);
        }
        *count = written;
        return written < total ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_surface_caps(bfr_instance* instance, uint32_t adapter, const bfr_surface_desc* desc,
                                bfr_surface_caps* caps) {
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
        const auto found = instance->rhi->surfaceCaps(adapter, *window);
        if (!found) {
            return fail(found.error());
        }
        *caps = toC(*found);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_get_surface_formats(bfr_instance* instance, uint32_t adapter, const bfr_surface_desc* desc,
                                   uint32_t* count, bfr_surface_format* formats) {
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
        const auto found = instance->rhi->surfaceFormats(adapter, *window);
        if (!found) {
            return fail(found.error());
        }
        if (formats == nullptr) {
            *count = static_cast<std::uint32_t>(found->size());
            return BFR_RESULT_SUCCESS;
        }
        return writeList(std::span<const bfr::rhi::SurfaceFormat>{*found}, count, formats);
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

bfr_result bfr_get_native(bfr_instance* instance, uint32_t adapter, void** native) {
    return guarded([&] {
        if (instance == nullptr || native == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and native must not be null");
        }
        *native = nullptr;
        if (const auto failed = checkAdapter(*instance, adapter)) {
            return *failed;
        }
        const auto found = instance->rhi->native(adapter);
        if (!found) {
            return fail(found.error());
        }
        *native = *found;
        return BFR_RESULT_SUCCESS;
    });
}

} // extern "C"
