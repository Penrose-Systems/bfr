#include "vulkan/vulkan_sampler.h"

#include "bfr/core/error.h"
#include "bfr/rhi/sampler.h"
#include "rhi/bindless_slots.h"
#include "rhi/enum_check.h"
#include "rhi/sampler_border.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_pipeline.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

VkFilter filterFor(Filter filter) {
    return filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

VkSamplerMipmapMode mipmapModeFor(Filter filter) {
    return filter == Filter::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode addressFor(AddressMode mode) {
    switch (mode) {
        case AddressMode::MirrorRepeat:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case AddressMode::ClampToEdge:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case AddressMode::ClampToBorder:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case AddressMode::MirrorClampToEdge:
            return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        case AddressMode::Repeat:
        default:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

VkBorderColor borderFor(BorderPreset preset, bool integer) {
    switch (preset) {
        case BorderPreset::TransparentBlack:
            return integer ? VK_BORDER_COLOR_INT_TRANSPARENT_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        case BorderPreset::OpaqueBlack:
            return integer ? VK_BORDER_COLOR_INT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        case BorderPreset::OpaqueWhite:
            return integer ? VK_BORDER_COLOR_INT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        case BorderPreset::Custom:
        default:
            return integer ? VK_BORDER_COLOR_INT_CUSTOM_EXT : VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
    }
}

} // namespace

VulkanSampler::VulkanSampler(const VolkDeviceTable& table, VkDevice device, VkSampler sampler, std::uint32_t slot)
    : table_(&table), device_(device), sampler_(sampler), slot_(slot) {}

VulkanSampler::VulkanSampler(VulkanSampler&& other) noexcept
    : table_(other.table_),
      device_(other.device_),
      sampler_(std::exchange(other.sampler_, VK_NULL_HANDLE)),
      slot_(other.slot_) {}

VulkanSampler::~VulkanSampler() {
    if (sampler_ != VK_NULL_HANDLE) {
        table_->vkDestroySampler(device_, sampler_, nullptr);
    }
}

core::Result<VulkanSampler> createVulkanSampler(const VolkDeviceTable& table, VkDevice device, const SamplerDesc& desc,
                                                std::uint32_t slot) {
    if (desc.lodAverage) {
        return core::failure(core::ErrorCode::RhiUnsupported, "Vulkan has no LOD averaging");
    }
    const BorderPreset preset = borderPreset(desc);
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = filterFor(desc.magFilter);
    info.minFilter = filterFor(desc.minFilter);
    info.mipmapMode = mipmapModeFor(desc.mipFilter);
    info.addressModeU = addressFor(desc.addressU);
    info.addressModeV = addressFor(desc.addressV);
    info.addressModeW = addressFor(desc.addressW);
    info.mipLodBias = desc.lodBias;
    info.anisotropyEnable = desc.maxAnisotropy != 0 ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy = static_cast<float>(desc.maxAnisotropy);
    info.compareEnable = desc.reduction == SamplerReduction::Compare ? VK_TRUE : VK_FALSE;
    info.compareOp = compareFor(desc.compareOp);
    info.minLod = desc.minLod;
    info.maxLod = desc.maxLod;
    info.borderColor = borderFor(preset, desc.borderInteger);
    info.unnormalizedCoordinates = desc.unnormalizedCoordinates ? VK_TRUE : VK_FALSE;

    const void** tail = &info.pNext;
    VkSamplerReductionModeCreateInfo reduction{};
    if (desc.reduction == SamplerReduction::Minimum || desc.reduction == SamplerReduction::Maximum) {
        reduction.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
        reduction.reductionMode = desc.reduction == SamplerReduction::Minimum ? VK_SAMPLER_REDUCTION_MODE_MIN
                                                                              : VK_SAMPLER_REDUCTION_MODE_MAX;
        *tail = &reduction;
        tail = &reduction.pNext;
    }
    // Given without a format, which customBorderColorWithoutFormat allows.
    VkSamplerCustomBorderColorCreateInfoEXT custom{};
    if (preset == BorderPreset::Custom) {
        custom.sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT;
        custom.format = VK_FORMAT_UNDEFINED;
        if (desc.borderInteger) {
            std::ranges::copy(desc.borderColorInteger, std::begin(custom.customBorderColor.uint32));
        }
        else {
            std::ranges::copy(desc.borderColor, std::begin(custom.customBorderColor.float32));
        }
        *tail = &custom;
    }

    VkSampler sampler = VK_NULL_HANDLE;
    if (const VkResult result = table.vkCreateSampler(device, &info, nullptr, &sampler); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateSampler", .result = result});
    }
    return VulkanSampler{table, device, sampler, slot};
}

// The device's sampler calls, beside the samplers they make.

core::Result<SamplerHandle> VulkanDevice::createSampler(const SamplerDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    auto slot = bindless_.takeSampler();
    if (!slot) {
        return std::unexpected(std::move(slot.error()));
    }
    auto made = createVulkanSampler(table_, device_, desc, *slot);
    if (!made) {
        bindless_.giveSampler(*slot);
        return std::unexpected(std::move(made.error()));
    }
    VkSampler handle = made->sampler();
    const SamplerHandle sampler = samplers_.emplace(std::move(*made));
    if (sampler.isNull()) {
        bindless_.giveSampler(*slot);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} samplers",
                             desc_.maxBindlessSamplers);
    }
    layout_.writeSampler(*slot, handle);
    return sampler;
}

core::Result<std::uint32_t> VulkanDevice::samplerSlot(SamplerHandle sampler) {
    const VulkanSampler* found = samplers_.find(sampler);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    return found->slot();
}

core::Result<void> VulkanDevice::destroySampler(SamplerHandle sampler) {
    if (sampler.isNull()) {
        return {};
    }
    if (samplers_.find(sampler) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    // Retired against every submit made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!samplers_.retire(sampler, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sampler handle is stale");
    }
    collectRetired();
    return {};
}

} // namespace bfr::rhi::vulkan
