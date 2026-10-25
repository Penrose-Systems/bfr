// h.h compiles here only because <cstddef> comes first; the header check must
// still reject it.
// clang-format off
#include <cstddef>
#include "h.h"
// clang-format on

namespace bfr::h {

std::size_t size() {
    return 0;
}

} // namespace bfr::h
