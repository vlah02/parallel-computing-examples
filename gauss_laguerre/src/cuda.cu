// cuda.cu
#include "../include/common.h"
#include "../include/cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cufft.h>

// GPU kernel: extract FFT result and linearly scale to [a,b]
__global__ void extract_and_linearscale(
    cufftDoubleComplex *Y,
    double             *w,
    int                 n,
    double              a,
    double              b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double c = Y[i].x;
    double scale = 2.0 / (double)n;
    if (i == 0 || i == n-1) scale *= 0.5;
    double wi = scale * c;
    w[i] = wi * ((b - a) * 0.5);
}

// Host wrapper for the CUDA weight computation
extern "C"
double *nc_compute_new_cuda(int n, double a, double b, double x[]) {
    double *h_w;
    cudaMallocHost(&h_w, n * sizeof(double));

    double *h_y = (double*)malloc((n+1) * sizeof(double));
    h_y[0] = 1.0;
    h_y[n] = (n & 1) ? -1.0 : +1.0;
    for (int k = 1; k < n; ++k) {
        h_y[k] = x[k] + x[n-k];
    }

    int Nfft = 2 * n;
    double *d_y;
    cufftDoubleComplex *d_Y;
    cudaMalloc(&d_y,   Nfft * sizeof(double));
    cudaMalloc(&d_Y, (n+1) * sizeof(cufftDoubleComplex));

    cudaMemcpy(d_y, h_y, (n+1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y + n + 1, h_y + 1, (n-1)*sizeof(double), cudaMemcpyHostToDevice);
    free(h_y);

    cufftHandle plan;
    cufftPlan1d(&plan, Nfft, CUFFT_D2Z, 1);
    cufftExecD2Z(plan, d_y, d_Y);

    int block = 256, grid = (n + block - 1) / block;
    extract_and_linearscale<<<grid, block>>>(d_Y, h_w, n, a, b);
    cudaDeviceSynchronize();

    cufftDestroy(plan);
    cudaFree(d_y);
    cudaFree(d_Y);

    return h_w;
}

void run_cuda(int argc, char *argv[]) {
    double a, b;
    int    n;
    char   filename[256];

    // Parse or prompt for arguments
    if (argc >= 2)      n = atoi(argv[1]);
    else { printf("Enter N: "); scanf("%d",&n); }
    if (argc >= 3)      a = atof(argv[2]);
    else { printf("Enter A: "); scanf("%lf",&a); }
    if (argc >= 4)      b = atof(argv[3]);
    else { printf("Enter B: "); scanf("%lf",&b); }
    if (argc >= 5) {
        strncpy(filename, argv[4], sizeof(filename)-1);
        filename[sizeof(filename)-1] = '\0';
    } else {
        printf("Enter root filename: ");
        scanf("%s", filename);
    }

    // Setup region endpoints array
    double *r = (double*)malloc(2*sizeof(double));
    if (!r) { perror("malloc"); exit(1); }
    r[0] = a; r[1] = b;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- sequential (CPU) portion timed on GPU event clock ---
    cudaEventRecord(start, 0);

    double *x = ccn_compute_points_new(n);
    double *w = nc_compute_new(n, -1.0, +1.0, x);
    rescale(a, b, n, x, w);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time_seq;
    cudaEventElapsedTime(&time_seq, start, stop);
    printf("Sequential (CPU) time: %f ms\n", time_seq);

    // --- parallel (GPU) portion ---
    cudaEventRecord(start, 0);

    double *xx = ccn_compute_points_new(n);
    double *ww = nc_compute_new_cuda(n, a, b, xx);

    // ** Scale abscissas from [-1,1] to [a,b] **
    for (int i = 0; i < n; i++) {
        xx[i] = ((a + b) + (b - a) * xx[i]) * 0.5;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time_par;
    cudaEventElapsedTime(&time_par, start, stop);
    printf("Parallel (GPU)   time: %f ms\n", time_par);
    printf("Speedup:           %f×\n", time_seq / time_par);

    // --- correctness test ---
    {
        int flag = 0;
        for (int i = 0; i < n; i++) {
            if (fabs(ww[i] - w[i]) > 1e-6 || fabs(xx[i] - x[i]) > 1e-12) {
                flag = 1;
                break;
            }
        }
        if (flag)  printf("Test FAILED\n");
        else       printf("Test PASSED\n");
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
