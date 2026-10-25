// Overflows a signed int; UndefinedBehaviorSanitizer must stop it.
#include <limits>

int main(int argc, char**) {
    int value = std::numeric_limits<int>::max();
    return value + argc;
}
