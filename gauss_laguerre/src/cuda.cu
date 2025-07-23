#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cufft.h>

#include "../include/common.hpp"

__global__ void extract_and_linearscale(cufftDoubleComplex *Y, double *w, int n, double a, double b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double scale = 2.0 / n;
    if (i == 0 || i == n - 1) scale *= 0.5;

    w[i] = scale * Y[i].x * ((b - a) * 0.5);
}

double *nc_compute_new(int n, double a, double b, double x[]) {
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

int main(int argc, char *argv[]) {
	double a, b;
    int n;
    char out_prefix[256];
    if (argc >= 2) n = atoi(argv[1]); else { printf("Enter N: "); scanf("%d", &n); }
    if (argc >= 3) a = atof(argv[2]); else { printf("Enter A: "); scanf("%lf", &a); }
    if (argc >= 4) b = atof(argv[3]); else { printf("Enter B: "); scanf("%lf", &b); }
    if (argc >= 5) strncpy(out_prefix, argv[4], 255); else { printf("Enter root filename: "); scanf("%s", out_prefix); }
    out_prefix[255] = '\0';

    char base[256];
    getOutputBase(out_prefix, base, sizeof(base));

    double *x_ref = (double *)malloc(n * sizeof(double));
	double *w_ref = (double *)malloc(n * sizeof(double));
	if (!loadSequentialResult(base, n, "x", x_ref) || !loadSequentialResult(base, n, "w", w_ref)) {
    	fprintf(stderr, "Failed to load precomputed x or w files.\n");
    	exit(EXIT_FAILURE);
	}

    double seq_time = 0.0;
    if (!loadSequentialTiming(base, &seq_time)) {
        fprintf(stderr, "No times found in sequential timing file for %s\n", base);
        exit(EXIT_FAILURE);
    }

	double *r = (double *)malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

    double *x_calc = ccn_compute_points_new(n);
    double *w_warmup = nc_compute_new(n, a, b, x_calc);
    cudaFreeHost(w_warmup);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    double *w_calc = nc_compute_new(n, a, b, x_calc);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float par_time;
    cudaEventElapsedTime(&par_time, start, stop);
	par_time /= 1000.0;

    for (int i = 0; i < n; i++)
        x_calc[i] = ((a + b) + (b - a) * x_calc[i]) * 0.5;

    int ok = compareResults(x_ref, x_calc, n, 1e-6) && compareResults(w_ref, w_calc, n, 1e-6);

	printf("\n");
    printf("%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
    printf("%s  Sequential time: %s%.6fs %s\n", BOLD, BLUE, seq_time, CLEAR);
    printf("%s  Parallel time:   %s%.6fs %s\n", BOLD, BLUE, par_time, CLEAR);
    printf("%s  Speedup:         %s%.3fx %s\n", BOLD, BLUE, seq_time / par_time, CLEAR);
	printf("\n");
    rule_write(n, out_prefix, x_calc, w_calc, r);
	appendTiming(out_prefix, par_time);

    free(r);
	free(x_ref); free(w_ref);
	free(x_calc); cudaFreeHost(w_calc);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
