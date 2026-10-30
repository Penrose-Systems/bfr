// Calls the shared library from C: the entry points link by their C names and
// report errors through bfr_get_last_error.
#include "bfr/bfr.h"

#include <stdio.h>

int main(void) {
    bfr_instance* instance = NULL;
    if (bfr_create_instance(NULL, &instance) != BFR_RESULT_ERROR_INVALID_ARGUMENT) {
        fputs("expected BFR_RESULT_ERROR_INVALID_ARGUMENT\n", stderr);
        return 1;
    }
    const bfr_string error = bfr_get_last_error();
    if (error.length == 0) {
        fputs("expected a message from bfr_get_last_error\n", stderr);
        return 1;
    }
    bfr_destroy_instance(instance);
    return 0;
}
