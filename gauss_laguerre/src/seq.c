// seq.c
#include "../include/seq.h"
#include "../include/common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void run_sequential(int argc, char *argv[]) {
    double a, b;
    int    n;
    char   filename[256];

    // --- parse arguments or prompt ---
    if (argc >= 2) {
        n = atoi(argv[1]);
    } else {
        printf("Enter the value of N: ");
        scanf("%d", &n);
    }

    if (argc >= 3) {
        a = atof(argv[2]);
    } else {
        printf("Enter the left endpoint A: ");
        scanf("%lf", &a);
    }

    if (argc >= 4) {
        b = atof(argv[3]);
    } else {
        printf("Enter the right endpoint B: ");
        scanf("%lf", &b);
    }

    if (argc >= 5) {
        strncpy(filename, argv[4], sizeof(filename)-1);
        filename[sizeof(filename)-1] = '\0';
    } else {
        printf("Enter the root filename: ");
        scanf("%s", filename);
    }

    // Prepare region array
    double *r = malloc(2 * sizeof(double));
    if (!r) { perror("malloc"); exit(1); }
    r[0] = a; r[1] = b;

    // --- sequential computation & timing ---
    clock_t t_start = clock();

    double *x = ccn_compute_points_new(n);
    double *w = nc_compute_new(n, -1.0, +1.0, x);
    rescale(a, b, n, x, w);

    clock_t t_end = clock();
    double elapsed = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("Sequential time: %f seconds\n", elapsed);

    // Write output files
    rule_write(n, filename, x, w, r);

    // Cleanup
    free(r);
    free(x);
    free(w);
}

int main(int argc, char *argv[]) {
    run_sequential(argc, argv);
    return 0;
}
