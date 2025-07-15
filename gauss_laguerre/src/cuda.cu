#include "../include/common.h"
#include "../include/cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cufft.h>

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

__global__ void extract_and_linearscale(cufftDoubleComplex *Y, double *w, int n, double a, double b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double scale = 2.0 / n;
    if (i == 0 || i == n - 1) scale *= 0.5;

    w[i] = scale * Y[i].x * ((b - a) * 0.5);
}

extern "C"
double *nc_compute_new_cuda(int n, double a, double b, double x[]) {
    int Nfft = 2 * n;
    double *d_y;
    cufftDoubleComplex *d_Y;
    double *h_w;

    cudaMallocHost(&h_w, n * sizeof(double));
    cudaMalloc(&d_y, Nfft * sizeof(double));
    cudaMalloc(&d_Y, (n + 1) * sizeof(cufftDoubleComplex));

    double *h_y = (double *)malloc((n + 1) * sizeof(double));
    h_y[0] = 1.0;
    h_y[n] = (n & 1) ? -1.0 : 1.0;
    for (int k = 1; k < n; ++k)
        h_y[k] = x[k] + x[n - k];

    cudaMemcpy(d_y, h_y, (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y + n + 1, h_y + 1, (n - 1) * sizeof(double), cudaMemcpyHostToDevice);
    free(h_y);

    cufftHandle plan;
    cufftPlan1d(&plan, Nfft, CUFFT_D2Z, 1);
    cufftExecD2Z(plan, d_y, d_Y);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    extract_and_linearscale<<<blocks, threads>>>(d_Y, h_w, n, a, b);
    cudaDeviceSynchronize();

    cufftDestroy(plan);
    cudaFree(d_y);
    cudaFree(d_Y);

    return h_w;
}

void run_cuda(int argc, char *argv[]) {
    int n;
    double a, b;
    char filename[256];

    if (argc >= 2) n = atoi(argv[1]); else { printf("Enter N: "); scanf("%d", &n); }
    if (argc >= 3) a = atof(argv[2]); else { printf("Enter A: "); scanf("%lf", &a); }
    if (argc >= 4) b = atof(argv[3]); else { printf("Enter B: "); scanf("%lf", &b); }
    if (argc >= 5) strncpy(filename, argv[4], 255); else { printf("Enter root filename: "); scanf("%s", filename); }
    filename[255] = '\0';

    double *r = (double *)malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
        double *x = ccn_compute_points_new(n);
        double *w = nc_compute_new(n, -1.0, +1.0, x);
        rescale(a, b, n, x, w);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    double time_seq = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_nsec - t1.tv_nsec) / 1e6;

    double *xx = ccn_compute_points_new(n);
    double *ww_warmup = nc_compute_new_cuda(n, a, b, xx);
    cudaFreeHost(ww_warmup);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
        double *ww = nc_compute_new_cuda(n, a, b, xx);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float time_par;
    cudaEventElapsedTime(&time_par, start, stop);

    for (int i = 0; i < n; i++)
        xx[i] = ((a + b) + (b - a) * xx[i]) * 0.5;

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(ww[i] - w[i]) > 1e-6 || fabs(xx[i] - x[i]) > 1e-12) {
            ok = 0;
            break;
        }
    }

    printf("%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
    printf("%s  Sequential time: %s%.3fms %s\n", BOLD, BLUE, time_seq, CLEAR);
    printf("%s  Parallel time:   %s%.3fms %s\n", BOLD, BLUE, time_par, CLEAR);
    printf("%s  Speedup:         %s%.2fx %s\n", BOLD, BLUE, time_seq / time_par, CLEAR);
    rule_write(n, filename, xx, ww, r);
    printf("\n");

    free(r);
    free(x); free(w); free(xx); cudaFreeHost(ww);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    run_cuda(argc, argv);
    return 0;
}
