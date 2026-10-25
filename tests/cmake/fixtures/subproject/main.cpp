#include "error.h"

int main() {
    return bfr::core::name(bfr::core::ErrorCode::ConfigLoadFailed).empty() ? 1 : 0;
}
