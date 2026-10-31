// The C API in bfr.h, over the C++ implementation. No exception crosses it:
// each entry point turns one into a result.

#include "bfr/bfr.h"
#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "bfr/rhi/swapchain.h"

#include <algorithm>
#include <array>
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

// The swapchain desc asks for, each 0 as its documented default. Nothing when a value is one BFR
// doesn't have; error says which.
std::optional<bfr::rhi::SwapchainDesc> swapchainIn(const bfr_swapchain_desc& desc, std::string& error) {
    using namespace bfr::rhi;
    SwapchainDesc swapchain;
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
            error = std::format("present mode {} is not one BFR has", static_cast<std::int64_t>(desc.present_mode));
            return std::nullopt;
    }
    if (const auto format = static_cast<std::int64_t>(desc.format); format != 0) {
        const auto* found = std::ranges::find_if(kFormats, [format](Format known) {
            return static_cast<std::int64_t>(known) == format;
        });
        if (found == kFormats.end()) {
            error = std::format("format {} is not one BFR has", format);
            return std::nullopt;
        }
        swapchain.format = *found;
    }
    if (const auto colorSpace = static_cast<std::int64_t>(desc.color_space); colorSpace != 0) {
        if (colorSpace < static_cast<std::int64_t>(ColorSpace::Srgb) ||
            colorSpace > static_cast<std::int64_t>(ColorSpace::DisplayNative)) {
            error = std::format("color space {} is not one BFR has", colorSpace);
            return std::nullopt;
        }
        swapchain.colorSpace = static_cast<ColorSpace>(colorSpace);
    }
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
            error = std::format("alpha mode {} is not one BFR has", static_cast<std::int64_t>(desc.alpha_mode));
            return std::nullopt;
    }
    return swapchain;
}

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

bfr_command_list toC(bfr::rhi::CommandListHandle handle) {
    return {.bits = handle.bits};
}

bfr_token toC(bfr::rhi::Token token) {
    return {.queue = token.queue, .value = token.value};
}

template <class T, std::size_t kInline = 16>
class SmallBuffer {
public:
    template <class From, class Convert>
    [[nodiscard]] std::span<const T> fill(std::span<const From> from, Convert convert) {
        T* out = inline_.data();
        if (from.size() > kInline) {
            heap_.resize(from.size());
            out = heap_.data();
        }
        std::ranges::transform(from, out, convert);
        return {out, from.size()};
    }

private:
    std::array<T, kInline> inline_{};
    std::vector<T> heap_;
};

bfr::rhi::Token fromC(const bfr_token& token) {
    return {.queue = token.queue, .value = token.value};
}

bfr::rhi::CommandListHandle fromC(bfr_command_list handle) {
    return bfr::rhi::CommandListHandle::fromBits(handle.bits);
}

// The rendering desc asks for, each 0 as its documented default. Nothing when
// a value is one BFR doesn't have; error says which.
std::optional<bfr::rhi::RenderingDesc> renderingIn(const bfr_rendering_desc& desc, std::string& error) {
    using namespace bfr::rhi;
    RenderingDesc rendering;
    switch (desc.load_op) {
        case BFR_LOAD_OP_LOAD:
            rendering.loadOp = LoadOp::Load;
            break;
        case BFR_LOAD_OP_CLEAR:
            rendering.loadOp = LoadOp::Clear;
            break;
        case BFR_LOAD_OP_DONT_CARE:
            rendering.loadOp = LoadOp::DontCare;
            break;
        case BFR_LOAD_OP_FORCE32:
        default:
            error = std::format("load_op {} is not one BFR has", static_cast<std::int64_t>(desc.load_op));
            return std::nullopt;
    }
    switch (desc.store_op) {
        case BFR_STORE_OP_STORE:
            rendering.storeOp = StoreOp::Store;
            break;
        case BFR_STORE_OP_DONT_CARE:
            rendering.storeOp = StoreOp::DontCare;
            break;
        case BFR_STORE_OP_FORCE32:
        default:
            error = std::format("store_op {} is not one BFR has", static_cast<std::int64_t>(desc.store_op));
            return std::nullopt;
    }
    rendering.clearColor = std::to_array(desc.clear_color);
    return rendering;
}

// The window in window, as the library keeps it: monostate for all zero,
// which means no window. Fails for a window system this platform lacks, or a
// window system with no window.
std::optional<NativeWindow> windowIn(const bfr_window& window, std::string& error) {
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
            error = std::format("window system {} is not one BFR has", static_cast<std::int64_t>(window.system));
            return std::nullopt;
    }
    if (window.window == nullptr) {
        error = "a window needs its window value set";
        return std::nullopt;
    }
#if defined(_WIN32)
    return bfr::core::window::Win32{.instance = nullptr, .window = window.window};
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
    return value ? BFR_TRUE : BFR_FALSE;
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
        return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, std::format("adapter {} of {}", adapter, count));
    }
    return std::nullopt;
}

// unusable one.
std::optional<NativeWindow> surfaceWindow(const bfr_swapchain_query& desc, bfr_result& failed) {
    std::string error;
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
    const std::string& error = lastError();
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
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT,
                        std::format("backend {} is not one BFR has", static_cast<std::int64_t>(desc->backend)));
        }
        const auto least = severityFrom(desc->min_severity);
        if (!least) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, std::format("min_severity {} is not a severity",
                                                                       static_cast<std::int64_t>(desc->min_severity)));
        }
        bfr::rhi::InstanceDesc rhiDesc{.backend = *backend};
        if (desc->message_callback != nullptr) {
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
        std::string error;
        const auto window = windowIn(desc->window, error);
        if (!window) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        auto created = instance->instance->createDevice(desc->adapter, *window,
                                                        {.compute = desc->compute_queues, .copy = desc->copy_queues});
        if (!created) {
            return fail(created.error());
        }
        // The caller owns it until bfr_destroy_device.
        *device = std::make_unique<bfr_device>(
                      bfr_device{.instance = instance->instance, .device = std::move(*created)})
                      .release();
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
        std::string error;
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

bfr_result bfr_acquire(bfr_device* device, bfr_swapchain swapchain) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto acquired = device->device->acquire(fromC(swapchain));
        return acquired ? BFR_RESULT_SUCCESS : fail(acquired.error());
    });
}

bfr_result bfr_present(bfr_device* device, bfr_swapchain swapchain) {
    return guarded([&] {
        if (device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device must not be null");
        }
        const auto presented = device->device->present(fromC(swapchain));
        return presented ? BFR_RESULT_SUCCESS : fail(presented.error());
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
        const auto created = device->device->createCommandPool(desc->queue);
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
        std::string error;
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
        SmallBuffer<bfr::rhi::CommandListHandle> lists;
        SmallBuffer<bfr::rhi::Token> waits;
        const bfr::rhi::SubmitDesc submitDesc{
            .queue = desc->queue,
            .commandLists = lists.fill(std::span{desc->command_lists, desc->command_list_count},
                                       [](bfr_command_list list) { return fromC(list); }),
            .waits = waits.fill(std::span{desc->waits, desc->wait_count},
                                [](const bfr_token& wait) { return fromC(wait); }),
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

bfr_result bfr_wait(bfr_device* device, const bfr_token* token) {
    return guarded([&] {
        if (device == nullptr || token == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and token must not be null");
        }
        const auto waited = device->device->wait(fromC(*token));
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
        *complete = *polled ? BFR_TRUE : BFR_FALSE;
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

} // extern "C"
