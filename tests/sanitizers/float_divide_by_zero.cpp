// Divides a float by zero; UndefinedBehaviorSanitizer must stop it.
int main(int argc, char**) {
    float zero = static_cast<float>(argc - 1);
    return 1.0f / zero > 0.0f ? 0 : 1;
}
