#pragma once

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan_core.h>

// This layer's link in a create info's chain. The loader passes the chain as
// const, yet each layer must advance its link for the next layer.
template <class LinkInfo>
LinkInfo* linkInfo(const void* chain, VkStructureType type) {
    for (const void* next = chain; next != nullptr;) {
        auto* link = static_cast<LinkInfo*>(const_cast<void*>(next));
        if (link->sType == type && link->function == VK_LAYER_LINK_INFO) {
            return link;
        }
        next = link->pNext;
    }
    return nullptr;
}
