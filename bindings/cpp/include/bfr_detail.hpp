#pragma once

// What the generated bfr.hpp is built on. Nothing here is part of the API.

#include "bfr/bfr.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace bfr::detail {

/// Keeps alive what a C object may still call into, such as a callback's
/// state, for as long as any object holding it lives.
using KeepAlive = std::shared_ptr<const void>;

/// Holds object until the last copy of the result is gone, then destroys it
/// with destroy and lets go of inner, which goes after it. A handle that
/// needs its object copies the result, so the object outlives the handle.
template <class Object>
[[nodiscard]] KeepAlive share(Object* object, void (*destroy)(Object*), KeepAlive inner) {
    return KeepAlive{object, [destroy, inner = std::move(inner)](Object* owned) {
                         destroy(owned);
                     }};
}

/// Whether Extension may be passed after Desc; bfr.hpp says which may.
template <class Extension, class Desc>
inline constexpr bool kExtends = false;

[[nodiscard]] inline std::string_view text(bfr_string string) {
    return {string.data, string.length};
}

[[nodiscard]] inline bfr_string toC(std::string_view text) {
    return {.data = text.data(), .length = text.size()};
}

/// The calling thread's last error, copied before another call replaces it.
[[nodiscard]] inline std::string lastError() {
    return std::string{text(bfr_get_last_error())};
}

/// A C array as a std::array, each element converted by convert.
template <class To, class From, std::size_t Size, class Convert>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays): the C struct holds one
[[nodiscard]] std::array<To, Size> arrayFrom(const From (&values)[Size], Convert convert) {
    std::array<To, Size> converted{};
    std::ranges::transform(values, converted.begin(), convert);
    return converted;
}

/// A C array built from a range for one call: in place up to kInline
/// elements, on the heap past that.
template <class T, std::size_t kInline = 16>
class SmallBuffer {
public:
    template <class Range, class Convert>
    [[nodiscard]] const T* fill(const Range& range, Convert convert) {
        const std::size_t size = std::ranges::size(range);
        if (size == 0) {
            return nullptr;
        }
        T* out = inline_.data();
        if (size > kInline) {
            heap_.resize(size);
            out = heap_.data();
        }
        std::ranges::transform(range, out, convert);
        return out;
    }

private:
    std::array<T, kInline> inline_{};
    std::vector<T> heap_;
};

/// Calls call, keeping anything it throws from leaving: a callback runs on a
/// C stack, often a driver's thread.
template <class Call>
void callQuietly(Call&& call) noexcept {
#if defined(__cpp_exceptions)
    try {
        std::forward<Call>(call)();
    } catch (...) { // NOLINT(bugprone-empty-catch): nowhere to report it
    }
#else
    std::forward<Call>(call)();
#endif
}

/// Extension structs in C form, each linked to the next, on the stack of the
/// call that hands head() to the C API. They stay at one address, so a Chain
/// neither copies nor moves.
template <class... Links>
class Chain {
public:
    explicit Chain(const Links&... links) : links_{links...} { link(std::index_sequence_for<Links...>{}); }
    ~Chain() = default;
    Chain(const Chain&) = delete;
    Chain& operator=(const Chain&) = delete;
    Chain(Chain&&) = delete;
    Chain& operator=(Chain&&) = delete;

    /// The first link, or null when there are none.
    [[nodiscard]] const bfr_chain* head() const { return at<0>(); }

private:
    template <std::size_t Index>
    [[nodiscard]] const bfr_chain* at() const {
        if constexpr (Index < sizeof...(Links)) {
            return &std::get<Index>(links_).chain;
        }
        else {
            return nullptr;
        }
    }

    template <std::size_t... Index>
    void link(std::index_sequence<Index...> /*indices*/) {
        ((std::get<Index>(links_).chain.next = at<Index + 1>()), ...);
    }

    std::tuple<Links...> links_;
};

} // namespace bfr::detail
