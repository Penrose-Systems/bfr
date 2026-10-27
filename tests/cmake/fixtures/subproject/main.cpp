#include "error.h"

int main() {
    return bfr::core::name(bfr::core::ErrorCode::RhiDeviceFailed).empty() ? 1 : 0;
}
