// Initialises a global from a global in another file. C++ leaves the order of
// the two initialisers unspecified, so AddressSanitizer must report the read.
extern int sourceValue;

int readValue = sourceValue + 1;

int main() {
    return readValue;
}
