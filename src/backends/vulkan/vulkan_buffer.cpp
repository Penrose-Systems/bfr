#include "vulkan/vulkan_buffer.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/texture.h"
#include "core/slot_chains.h"
#include "rhi/bindless_slots.h"
#include "rhi/buffer_check.h"
#include "rhi/enum_check.h"
#include "rhi/name_check.h"
#include "rhi/ray_tracing_check.h"
#include "vulkan/vma.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_result.h"
#include "vulkan/vulkan_sampler.h"
#include "vulkan/vulkan_sparse.h"
#include "vulkan/vulkan_texture.h"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

VkBufferUsageFlags usageFor(const BufferUsage& usage, const BufferAddressing& addressing) {
    const auto flag = [](bool set, VkBufferUsageFlagBits bit) {
        return set ? VkBufferUsageFlags{bit} : 0U;
    };
    return flag(usage.copySrc, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) |
           flag(usage.copyDst, VK_BUFFER_USAGE_TRANSFER_DST_BIT) |
           flag(usage.uniform, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) |
           flag(usage.storage, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
           flag(usage.index, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) |
           flag(usage.indirect, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) |
           // Builds read their inputs and scratch by address; scratch is a
           // storage buffer (VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12260, -12261).
           flag(usage.accelerationStructureInput,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) |
           // Micromap builds read theirs by address too (VUID-vkCmdBuildMicromapsEXT-data-07510).
           flag(usage.accelerationStructureInput && addressing.micromapInputs,
                VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT) |
           flag(usage.accelerationStructureScratch, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
           // Cluster operations write structures to it by address
           // (VUID-vkCmdBuildClusterAccelerationStructureIndirectNV-pCommandInfos-12308, -12382).
           flag(usage.accelerationStructureStorage, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) |
           // Traces read binding tables, and indirect traces their
           // arguments, by address (VUID-vkCmdTraceRaysIndirect2KHR-indirectDeviceAddress-03633).
           flag(usage.shaderBindingTable, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR) |
           // Every buffer has an address where the device gives them, as
           // NVRHI's do (vulkan-buffer.cpp:81-82): builds, traces, cluster
           // records and shaders read by it.
           flag(addressing.addresses, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

VkDeviceSize alignmentFor(const BufferUsage& usage, const BufferAddressing& addressing) {
    return std::max({usage.accelerationStructureScratch ? addressing.scratch : 0,
                     usage.accelerationStructureInput ? addressing.input : 0,
                     usage.shaderBindingTable ? addressing.table : 0,
                     usage.accelerationStructureData ? addressing.data : 0,
                     usage.accelerationStructureStorage ? addressing.storage : 0});
}

// The memory properties each BufferMemory requires, and the ones it would
// like. The CPU never flushes, so mapped memory is coherent. Readback is read
// by the CPU, so cached memory is asked for but not required: every device has
// coherent host memory, and not every one has it cached.
struct MemoryProperties {
    VkMemoryPropertyFlags required = 0;
    VkMemoryPropertyFlags preferred = 0;
};

MemoryProperties propertiesFor(BufferMemory memory) {
    switch (memory) {
        case BufferMemory::Device:
            return {.required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .preferred = 0};
        case BufferMemory::Upload:
            return {.required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    .preferred = 0};
        case BufferMemory::Readback:
            return {.required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    .preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
    }
    return {};
}

// A buffer's create info, shared by the queue families in families when
// there is more than one, so any of the device's queues may use it without
// an ownership transfer. It points into families.
VkBufferCreateInfo bufferInfo(VkDeviceSize size, VkBufferUsageFlags usage, std::span<const std::uint32_t> families) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    if (families.size() > 1) {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
        info.pQueueFamilyIndices = families.data();
    }
    else {
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    return info;
}

VkDeviceAddress deviceAddressOf(const VolkDeviceTable& table, VkDevice device, VkBuffer buffer) {
    VkBufferDeviceAddressInfo named{};
    named.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    named.buffer = buffer;
    return table.vkGetBufferDeviceAddress(device, &named);
}

// buffer's device address, 0 for one made without the usage that gives it
// one. A sparse buffer has one before any memory is bound
// (VUID-VkBufferDeviceAddressInfo-buffer-02600 asks only non-sparse ones).
VkDeviceAddress addressFor(const VolkDeviceTable& table, VkDevice device, VkBuffer buffer, VkBufferUsageFlags usage) {
    return (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0 ? deviceAddressOf(table, device, buffer) : 0;
}

} // namespace

core::Result<VmaAllocator> createAllocator(VkInstance instance, VkPhysicalDevice adapter, VkDevice device,
                                           std::uint32_t apiVersion, bool deviceAddresses) {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo info{};
    info.physicalDevice = adapter;
    info.device = device;
    info.instance = instance;
    info.vulkanApiVersion = apiVersion;
    info.pVulkanFunctions = &functions;
    if (deviceAddresses) {
        info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    VmaAllocator allocator = VK_NULL_HANDLE;
    if (const VkResult result = vmaCreateAllocator(&info, &allocator); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaCreateAllocator", .result = result});
    }
    return allocator;
}

VulkanBuffer::VulkanBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, BufferMemory memory,
                           std::span<std::byte> mapped, BufferSlotsHeld slots, VkDeviceAddress address,
                           VkDeviceSize size)
    : allocator_(allocator),
      buffer_(buffer),
      allocation_(allocation),
      memory_(memory),
      mapped_(mapped),
      slots_(slots),
      address_(address),
      size_(size) {}

VulkanBuffer::VulkanBuffer(VmaAllocator allocator, VkBuffer buffer, BufferSlotsHeld slots,
                           const VkMemoryRequirements& sparse, VkDeviceAddress address, VkDeviceSize size)
    : allocator_(allocator),
      buffer_(buffer),
      allocation_(VK_NULL_HANDLE),
      memory_(BufferMemory::Device),
      slots_(slots),
      sparse_(true),
      tileBytes_(sparse.alignment),
      memoryTypes_(sparse.memoryTypeBits),
      address_(address),
      size_(size) {}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : allocator_(other.allocator_),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      memory_(other.memory_),
      mapped_(std::exchange(other.mapped_, {})),
      slots_(other.slots_),
      sparse_(other.sparse_),
      tileBytes_(other.tileBytes_),
      memoryTypes_(other.memoryTypes_),
      address_(other.address_),
      size_(other.size_) {}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
}

core::Result<VulkanBuffer> placeVulkanBuffer(const VolkDeviceTable& table, VkDevice device, VmaAllocator allocator,
                                             VkDeviceSize size, VkBufferUsageFlags usage, BufferMemory memory,
                                             std::span<const std::uint32_t> families, VkDeviceSize alignment,
                                             BufferSlotsHeld slots) {
    VkBufferCreateInfo info = bufferInfo(size, usage, families);
    VmaAllocationCreateInfo allocation{};
    const MemoryProperties properties = propertiesFor(memory);
    allocation.requiredFlags = properties.required;
    allocation.preferredFlags = properties.preferred;
    if (memory != BufferMemory::Device) {
        allocation.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation made = VK_NULL_HANDLE;
    VmaAllocationInfo placed{};
    if (const VkResult result = vmaCreateBufferWithAlignment(allocator, &info, &allocation, alignment, &buffer, &made,
                                                             &placed);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaCreateBufferWithAlignment", .result = result});
    }
    std::span<std::byte> mapped;
    if (placed.pMappedData != nullptr) {
        mapped = {static_cast<std::byte*>(placed.pMappedData), static_cast<std::size_t>(size)};
    }
    return VulkanBuffer{allocator, buffer, made, memory, mapped, slots, addressFor(table, device, buffer, usage), size};
}

core::Result<VulkanReplayableBuffer> placeReplayableVulkanBuffer(const VolkDeviceTable& table, VkDevice device,
                                                                 VmaAllocator allocator, VkDeviceSize size,
                                                                 VkBufferUsageFlags usage,
                                                                 std::span<const std::uint32_t> families,
                                                                 const VulkanBufferReplay& replay) {
    VkBufferOpaqueCaptureAddressCreateInfo bufferAt{};
    bufferAt.sType = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO;
    bufferAt.opaqueCaptureAddress = replay.buffer;
    VkBufferCreateInfo info = bufferInfo(size, usage, families);
    info.flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
    info.pNext = &bufferAt;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (const VkResult result = table.vkCreateBuffer(device, &info, nullptr, &buffer); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateBuffer", .result = result});
    }
    // The buffer's memory is its own, at its own opaque address, with the
    // capture-replay flag (VUID-vkBindBufferMemory-bufferDeviceAddressCaptureReplay-09200).
    // VMA makes no buffer with an address on an allocator that adds no flags
    // of its own, so it places memory for the buffer made here.
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.buffer = buffer;
    VkMemoryOpaqueCaptureAddressAllocateInfo memoryAt{};
    memoryAt.sType = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO;
    memoryAt.pNext = &dedicated;
    memoryAt.opaqueCaptureAddress = replay.memory;
    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.pNext = &memoryAt;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
    VkMemoryRequirements requirements{};
    table.vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VmaAllocationCreateInfo allocation{};
    allocation.requiredFlags = propertiesFor(BufferMemory::Device).required;
    // One memory type, the same each time, so a replay allocates as the
    // capture did, and VMA reports that type's result rather than trying another.
    std::uint32_t type = 0;
    if (const VkResult result = vmaFindMemoryTypeIndex(allocator, requirements.memoryTypeBits, &allocation, &type);
        result != VK_SUCCESS) {
        table.vkDestroyBuffer(device, buffer, nullptr);
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaFindMemoryTypeIndex", .result = result});
    }
    allocation.memoryTypeBits = 1U << type;
    VmaAllocation made = VK_NULL_HANDLE;
    VmaAllocationInfo placed{};
    if (const VkResult result = vmaAllocateDedicatedMemory(allocator, &requirements, &allocation, &flags, &made,
                                                           &placed);
        result != VK_SUCCESS) {
        table.vkDestroyBuffer(device, buffer, nullptr);
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaAllocateDedicatedMemory", .result = result});
    }
    if (const VkResult result = vmaBindBufferMemory(allocator, made, buffer); result != VK_SUCCESS) {
        vmaDestroyBuffer(allocator, buffer, made);
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vmaBindBufferMemory", .result = result});
    }
    const VkDeviceAddress address = deviceAddressOf(table, device, buffer);
    VkBufferDeviceAddressInfo named{};
    named.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    named.buffer = buffer;
    VkDeviceMemoryOpaqueCaptureAddressInfo memory{};
    memory.sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO;
    memory.memory = placed.deviceMemory;
    VulkanReplayableBuffer replayable{
        .buffer = VulkanBuffer{allocator, buffer, made, BufferMemory::Device, {}, {}, address, size},
        .made = {.buffer = table.vkGetBufferOpaqueCaptureAddress(device, &named),
                 .memory = table.vkGetDeviceMemoryOpaqueCaptureAddress(device, &memory)}};
    return replayable;
}

core::Result<VulkanBuffer> createVulkanBuffer(const VolkDeviceTable& table, VkDevice device, VmaAllocator allocator,
                                              const BufferDesc& desc, std::span<const std::uint32_t> families,
                                              const BufferAddressing& addressing, BufferSlotsHeld slots) {
    if (desc.sparse && desc.tileBytes != 0) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "Vulkan sets a sparse buffer's tile size itself; tileBytes is 0");
    }
    if (!desc.sparse) {
        return placeVulkanBuffer(table, device, allocator, desc.size, usageFor(desc.usage, addressing), desc.memory,
                                 families, alignmentFor(desc.usage, addressing), slots);
    }
    // Bound tile by tile, partly resident, and aliased where asked; no
    // memory is placed now.
    VkBufferCreateInfo info = bufferInfo(desc.size, usageFor(desc.usage, addressing), families);
    info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                 (desc.sparseAliased ? VkBufferCreateFlags{VK_BUFFER_CREATE_SPARSE_ALIASED_BIT} : 0U);
    VkBuffer buffer = VK_NULL_HANDLE;
    if (const VkResult result = table.vkCreateBuffer(device, &info, nullptr, &buffer); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "vkCreateBuffer", .result = result});
    }
    VkMemoryRequirements requirements{};
    table.vkGetBufferMemoryRequirements(device, buffer, &requirements);
    return VulkanBuffer{allocator, buffer, slots, requirements, addressFor(table, device, buffer, info.usage),
                        desc.size};
}

// The device's buffer calls, beside the buffers they make.

core::Result<VulkanBuffer*> VulkanDevice::bufferFor(BufferHandle buffer) {
    VulkanBuffer* found = buffers_.find(buffer);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    return found;
}

void VulkanDevice::collectRetired() {
    const std::uint64_t completed = submissions_.completed([this](std::uint32_t queue) {
        // A queue that can't say what it passed has passed nothing yet.
        const auto passed = completedValue(*queues_[queue]);
        return passed ? *passed : 0;
    });
    // A collected resource's slots go back to their tables, so no pending work
    // can see a slot reused.
    buffers_.collect(completed, [this](const VulkanBuffer& buffer) { bindless_.giveBuffer(buffer.slots()); });
    textures_.collect(completed, [this](const VulkanTexture& texture) { bindless_.giveTexture(texture.slots()); });
    samplers_.collect(completed, [this](const VulkanSampler& sampler) { bindless_.giveSampler(sampler.slot()); });
    timestampPools_.collect(completed);
    queryPools_.collect(completed);
    sparseMemories_.collect(completed);
    propertyPools_.collect(completed);
    pipelines_.collect(completed);
    accelerationStructures_.collect(completed, [this](const VulkanAccelerationStructure& structure) {
        bindless_.giveAccelerationStructure(structure.type(), structure.slot());
    });
}

core::Result<void> VulkanDevice::runGarbageCollection() {
    // collectRetired takes a queue that cannot say what it passed as having
    // passed nothing; here the caller hears of it.
    for (const auto& queue : queues_) {
        if (auto completed = completedValue(*queue); !completed) {
            return std::unexpected(std::move(completed.error()));
        }
    }
    collectRetired();
    return {};
}

core::Result<BufferHandle> VulkanDevice::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkObjectName(features_, desc.name); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so buffers the GPU is done with are freed here, never on the
    // frame path: freeing one frees the allocator's memory.
    collectRetired();
    auto slot = bindless_.takeBuffer(desc);
    if (!slot) {
        return std::unexpected(std::move(slot.error()));
    }
    auto made = createVulkanBuffer(table_, device_, allocator_, desc, families_, addressing_, *slot);
    if (!made) {
        bindless_.giveBuffer(*slot);
        return std::unexpected(std::move(made.error()));
    }
    VkBuffer handle = made->buffer();
    if (auto named = nameObject(VK_OBJECT_TYPE_BUFFER, handle, desc.name); !named) {
        bindless_.giveBuffer(*slot);
        return std::unexpected(std::move(named.error()));
    }
    const BufferHandle buffer = buffers_.emplace(std::move(*made));
    if (buffer.isNull()) {
        bindless_.giveBuffer(*slot);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", desc_.maxBuffers);
    }
    if (slot->storage != core::SlotChains::kEnd) {
        layout_.writeStorageBuffer(slot->storage, handle);
    }
    if (slot->uniform != core::SlotChains::kEnd) {
        layout_.writeUniformBuffer(slot->uniform, handle);
    }
    return buffer;
}

core::Result<BufferSlots> VulkanDevice::bufferSlots(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reported((*found)->slots());
}

core::Result<std::span<std::byte>> VulkanDevice::mapBuffer(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*found)->memory() == BufferMemory::Device) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the buffer is in device memory, which the CPU does not map");
    }
    return (*found)->mapped();
}

core::Result<void> VulkanDevice::copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
                                            std::span<const BufferCopy> regions) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = bufferFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = bufferFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    for (const BufferCopy& region : regions) {
        const VkBufferCopy copy{.srcOffset = region.srcOffset, .dstOffset = region.dstOffset, .size = region.size};
        table_.vkCmdCopyBuffer((*list)->buffer, (*from)->buffer(), (*to)->buffer(), 1, &copy);
    }
    return {};
}

core::Result<std::uint64_t> VulkanDevice::bufferAddress(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (auto checked = checkBufferAddress(features_.bufferAddress); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return (*found)->address();
}

core::Result<void> VulkanDevice::destroyBuffer(BufferHandle buffer) {
    if (buffer.isNull()) {
        return {};
    }
    if (buffers_.find(buffer) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    // Retired against every submit made so far, on every queue, so it lives
    // until the GPU has run all of them.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!buffers_.retire(buffer, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    collectRetired();
    return {};
}

core::Result<BufferTiling> VulkanDevice::bufferTiling(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (!(*found)->sparse()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer is not sparse");
    }
    return BufferTiling{.tileBytes = (*found)->tileBytes()};
}

core::Result<void> VulkanDevice::addBufferBinds(VulkanQueue& queue, const TileMappingDesc& desc,
                                                const VulkanSparseMemory* memory) {
    if (desc.buffer.isNull()) {
        if (!desc.bufferRegions.empty()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "buffer regions were given with no buffer");
        }
        return {};
    }
    auto found = bufferFor(desc.buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const VulkanBuffer& buffer = **found;
    for (const BufferTileRegion& region : desc.bufferRegions) {
        if (auto checked = checkKnown(region.mapping, "tile mapping"); !checked) {
            return checked;
        }
        const bool maps = region.mapping == TileMapping::Map;
        if (maps && memory == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a region maps, but no memory was given");
        }
        // The buffer's regions are at most maxTileMappings, which the array holds.
        std::ignore = queue.bufferBinds.push(
            VkSparseMemoryBind{.resourceOffset = region.tile * buffer.tileBytes(),
                               .size = region.tiles * buffer.tileBytes(),
                               .memory = maps ? memory->memory() : VK_NULL_HANDLE,
                               .memoryOffset = maps ? memory->offset() + region.memoryTile * buffer.tileBytes() : 0,
                               .flags = 0});
    }
    return {};
}

} // namespace bfr::rhi::vulkan
