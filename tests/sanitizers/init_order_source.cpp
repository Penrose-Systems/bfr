// The global init_order.cpp reads; its initialiser runs at startup, not at
// compile time.
#include <string>

int sourceValue = std::stoi("41");
