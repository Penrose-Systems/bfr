// Two threads write one int with nothing ordering them; ThreadSanitizer must
// report the race.
#include <thread>

int main() {
    int shared = 0;
    std::thread writer([&shared] { ++shared; });
    ++shared;
    writer.join();
    return 0;
}
