#pragma once

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::rhi::vulkan {

/// Fills items with what a Vulkan listing, called once to count and once to
/// fill, returns, keeping items' storage; or gives the failed function and its
/// result. VK_INCOMPLETE means the list grew between the calls, so it lists
/// again.
template <class Item, class Listing>
std::expected<void, std::string> listInto(std::vector<Item>& items, std::string_view function, Listing listing) {
    VkResult result = VK_INCOMPLETE;
    while (result == VK_INCOMPLETE) {
        std::uint32_t count = 0;
        result = listing(&count, nullptr);
        if (result != VK_SUCCESS) {
            break;
        }
        items.resize(count);
        result = listing(&count, items.data());
        items.resize(count);
    }
    if (result != VK_SUCCESS) {
        return std::unexpected(std::format("{}: {}", function, string_VkResult(result)));
    }
    return {};
}

/// What listInto fills, in a new list.
template <class Item, class Listing>
std::expected<std::vector<Item>, std::string> listAll(std::string_view function, Listing listing) {
    std::vector<Item> items;
    return listInto(items, function, listing).transform([&items] { return std::move(items); });
}

} // namespace bfr::rhi::vulkan
