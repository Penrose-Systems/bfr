#pragma once

#include <array>
#include <string_view>

namespace bfr::rhi {

/// A name a debugger or profiler shows on a region of a list's or a queue's
/// work, or on one point in it. The name is read during the call only. color
/// is red, green, blue and alpha, 0 to 1, passed to the backend as given.
struct DebugLabel {
    std::string_view name;
    std::array<float, 4> color{};
};

} // namespace bfr::rhi
