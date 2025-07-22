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
    int n;
    double a, b;
    char out_prefix[256];

    if (argc >= 2) n = atoi(argv[1]); else { printf("Enter N: "); scanf("%d", &n); }
    if (argc >= 3) a = atof(argv[2]); else { printf("Enter A: "); scanf("%lf", &a); }
    if (argc >= 4) b = atof(argv[3]); else { printf("Enter B: "); scanf("%lf", &b); }
    if (argc >= 5) strncpy(out_prefix, argv[4], 255); else { printf("Enter root filename: "); scanf("%s", out_prefix); }
    out_prefix[255] = '\0';

    const char *basename = strrchr(out_prefix, '/');
    basename = (basename == NULL) ? out_prefix : basename + 1;

    char xfile[300], wfile[300], tfile[300];
    snprintf(xfile, sizeof(xfile), "output/seq/%s_x.txt", basename);
    snprintf(wfile, sizeof(wfile), "output/seq/%s_w.txt", basename);
    snprintf(tfile, sizeof(tfile), "output/seq/%s_time.txt", basename);

    double *x = (double *)malloc(n * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));

    FILE *fx = fopen(xfile, "r");
    FILE *fw = fopen(wfile, "r");
    FILE *ft = fopen(tfile, "r");
    if (!fx || !fw || !ft) {
        fprintf(stderr, "Failed to load precomputed sequential files.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) fscanf(fx, "%lf", &x[i]);
    for (int i = 0; i < n; i++) fscanf(fw, "%lf", &w[i]);

	double time_seq = 0.0;
	int time_count = 0;
	double tval;
	while (fscanf(ft, "%lf", &tval) == 1) {
    	time_seq += tval;
    	time_count++;
	}
	if (time_count == 0) {
    	fprintf(stderr, "No times found in file %s\n", tfile);
    	exit(EXIT_FAILURE);
	}
	time_seq /= time_count;

    fclose(fx); fclose(fw); fclose(ft);

    double *xx = ccn_compute_points_new(n);
    double *ww_warmup = nc_compute_new(n, a, b, xx);
    cudaFreeHost(ww_warmup);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    double *ww = nc_compute_new(n, a, b, xx);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float time_par;
    cudaEventElapsedTime(&time_par, start, stop);
	time_par /= 1000.0;

    for (int i = 0; i < n; i++)
        xx[i] = ((a + b) + (b - a) * xx[i]) * 0.5;

	double *r = (double *)malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(ww[i] - w[i]) > 1e-6 || fabs(xx[i] - x[i]) > 1e-12) {
            ok = 0;
            break;
        }
    }

    printf("%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
    printf("%s  Sequential time: %s%.3fs %s\n", BOLD, BLUE, time_seq, CLEAR);
    printf("%s  Parallel time:   %s%.3fs %s\n", BOLD, BLUE, time_par, CLEAR);
    printf("%s  Speedup:         %s%.2fx %s\n", BOLD, BLUE, time_seq / time_par, CLEAR);
    rule_write(n, out_prefix, xx, ww, r);
	printf("\n");

    char time_out[300];
    snprintf(time_out, sizeof(time_out), "%s_time.txt", out_prefix);
    FILE *fout = fopen(time_out, "a");
    if (fout) {
        fprintf(fout, "%.6f\n", time_par);
        fclose(fout);
    } else {
        perror("fopen for CUDA time");
    }

    free(r); free(x); free(w); free(xx); cudaFreeHost(ww);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
