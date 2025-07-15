#include "cuda.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

// Color codes for test output
const char *red   = "\033[1;31m";
const char *green = "\033[1;32m";
const char *blue  = "\033[1;36m";
const char *clear = "\033[0m";

void run_cuda(int argc, char *argv[]) {
    double a, b;
    int    n;
    char   filename[256];
    double *r, *x, *w, *xx, *ww;
    cudaEvent_t start, stop;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // User header
    printf("\nCCN_RULE CUDA variant\n\n");
    printf("Compute nested Clenshaw–Curtis quadrature of order N\n\n");

    // Parse or prompt for arguments
    if (argc >= 2)      n = atoi(argv[1]);
    else { printf("Enter N: "); scanf("%d", &n); }

    if (argc >= 3)      a = atof(argv[2]);
    else { printf("Enter A: "); scanf("%lf", &a); }

    if (argc >= 4)      b = atof(argv[3]);
    else { printf("Enter B: "); scanf("%lf", &b); }

    if (argc >= 5) {
        strncpy(filename, argv[4], sizeof(filename)-1);
        filename[sizeof(filename)-1] = '\0';
    } else {
        printf("Enter root filename: ");
        scanf("%s", filename);
    }

    printf("\n  N = %d\n  A = %g\n  B = %g\n  FILENAME = \"%s\"\n\n",
           n, a, b, filename);

    // Allocate region endpoints array
    r = (double*)malloc(2 * sizeof(double));
    if (!r) { perror("malloc"); exit(1); }
    r[0] = a; r[1] = b;

    // Sequential (CPU) computation
    cudaEventRecord(start, 0);
    x = ccn_compute_points_new(n);
    w = nc_compute_new(n, -1.0, +1.0, x);
    rescale(a, b, n, x, w);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time_seq;
    cudaEventElapsedTime(&time_seq, start, stop);
    printf("%sSequential (CPU) time: %f ms%s\n", blue, time_seq, clear);

    // Parallel (GPU) computation
    cudaEventRecord(start, 0);
    xx = ccn_compute_points_new(n);
    ww = nc_compute_new_cuda(n, a, b, xx);
    // Rescale abscissas on host
    for (int i = 0; i < n; i++) {
        xx[i] = ((a + b) + (b - a) * xx[i]) / 2.0;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time_par;
    cudaEventElapsedTime(&time_par, start, stop);
    printf("%sParallel (GPU)   time: %f ms%s\n", blue, time_par, clear);
    printf("%sSpeedup:           %f×%s\n\n", blue, time_seq / time_par, clear);

    // Test for correctness
    {
        int flag = 0;
        for (int i = 0; i < n; i++) {
            if (fabs(ww[i] - w[i]) > 1e-6 || fabs(xx[i] - x[i]) > 1e-12) {
                flag = 1;
                break;
            }
        }
        if (flag) printf("%sTest FAILED%s\n", red, clear);
        else      printf("%sTest PASSED%s\n", green, clear);
    }

    // Write output files
    rule_write(n, filename, xx, ww, r);

    // Cleanup
    free(r);
    free(x); free(w);
    free(xx);
    cudaFreeHost(ww);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    run_cuda(argc, argv);
    return 0;
}
