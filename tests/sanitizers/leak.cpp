// Drops the only pointer to a heap block; LeakSanitizer must report it.
#include <cstdlib>

int main() {
    [[maybe_unused]] void* volatile block = std::malloc(sizeof(int));
    block = nullptr;
    return 0;
}
