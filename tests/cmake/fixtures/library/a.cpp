constexpr long kCpp23 = 202100L;
static_assert(__cplusplus >= kCpp23, "the fixture builds as C++23");

#include "a.h"

namespace bfr::a {

int value() {
    return 0;
}

} // namespace bfr::a
