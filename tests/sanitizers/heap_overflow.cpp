// Reads one element past a heap array; AddressSanitizer must stop it.
#include <cstddef>
#include <memory>

int main(int argc, char**) {
    auto values = std::make_unique<int[]>(1);
    return values[static_cast<std::size_t>(argc)];
}
