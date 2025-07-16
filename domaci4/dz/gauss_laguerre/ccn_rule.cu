#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

// CUDA kernel: extract real part and apply Clenshaw–Curtis linear scaling
__global__ void extract_and_linearscale(cufftDoubleComplex *Y, double *w, int n, double a, double b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double scale = 2.0 / n;
    if (i == 0 || i == n - 1) scale *= 0.5;

    w[i] = scale * Y[i].x * ((b - a) * 0.5);
}

// Sequential CC node computation
double *ccn_compute_points_new(int n) {
    double *x = (double *)malloc(n * sizeof(double));
    double r8_pi = 3.141592653589793;
    int m = 3, d = 2, tu, td, i, k;

    if (n >= 1) x[0] = 0.5;
    if (n >= 2) x[1] = 1.0;
    if (n >= 3) x[2] = 0.0;

    while (m < n) {
        tu = d + 1;
        td = d - 1;
        k = (d < n - m) ? d : (n - m);

        for (i = 1; i <= k; i++) {
            x[m + i - 1] = (i % 2 ? tu : td) / 2.0 / (double)k;
            if (i % 2) tu += 2; else td -= 2;
        }
        m += k;
        d *= 2;
    }

    for (i = 0; i < n; i++) x[i] = cos(x[i] * r8_pi);
    if (n >= 1) x[0] = 0.0;
    if (n >= 2) x[1] = -1.0;
    if (n >= 3) x[2] = +1.0;

    return x;
}

// Sequential Newton–Cotes weights
double *nc_compute_new(int n, double x_min, double x_max, double x[]) {
    double *w = (double *)malloc(n * sizeof(double));
    double *d = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        memset(d, 0, n * sizeof(double));
        d[i] = 1.0;

        for (int j = 2; j <= n; j++)
            for (int k = j; k <= n; k++)
                d[n + j - k - 1] = (d[n + j - k - 2] - d[n + j - k - 1]) /
                                   (x[n - k] - x[n + j - k - 1]);

        for (int j = 1; j <= n - 1; j++)
            for (int k = 1; k <= n - j; k++)
                d[n - k - 1] -= x[n - k - j] * d[n - k];

        double yvala = d[n - 1] / n;
        for (int j = n - 2; j >= 0; j--) yvala = yvala * x_min + d[j] / (j + 1);
        yvala *= x_min;

        double yvalb = d[n - 1] / n;
        for (int j = n - 2; j >= 0; j--) yvalb = yvalb * x_max + d[j] / (j + 1);
        yvalb *= x_max;

        w[i] = yvalb - yvala;
    }
    free(d);
    return w;
}

// CUDA version of NC weights
double *nc_compute_new_cuda(int n, double a, double b, double x[]) {
    int Nfft = 2 * n;
    double *d_y, *h_w, *h_y = (double *)malloc((n + 1) * sizeof(double));
    cufftDoubleComplex *d_Y;

    h_y[0] = 1.0;
    h_y[n] = (n & 1) ? -1.0 : 1.0;
    for (int k = 1; k < n; ++k) h_y[k] = x[k] + x[n - k];

    cudaMallocHost(&h_w, n * sizeof(double));
    cudaMalloc(&d_y, Nfft * sizeof(double));
    cudaMalloc(&d_Y, (n + 1) * sizeof(cufftDoubleComplex));

    cudaMemcpy(d_y, h_y, (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y + n + 1, h_y + 1, (n - 1) * sizeof(double), cudaMemcpyHostToDevice);
    free(h_y);

    cufftHandle plan;
    cufftPlan1d(&plan, Nfft, CUFFT_D2Z, 1);
    cufftExecD2Z(plan, d_y, d_Y);

    int threads = 256, blocks = (n + threads - 1) / threads;
    extract_and_linearscale<<<blocks, threads>>>(d_Y, h_w, n, a, b);
    cudaDeviceSynchronize();

    cufftDestroy(plan);
    cudaFree(d_y);
    cudaFree(d_Y);

    return h_w;
}

// Rescale from [-1,1] to [a,b]
void rescale(double a, double b, int n, double x[], double w[]) {
    for (int i = 0; i < n; i++) x[i] = ((a + b) + (b - a) * x[i]) / 2.0;
    for (int i = 0; i < n; i++) w[i] = (b - a) * w[i] / 2.0;
}

// Write matrix to file
void r8mat_write(char *filename, int m, int n, double table[]) {
    FILE *fp = fopen(filename, "wt");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++)
            fprintf(fp, "  %24.16g", table[i + j * m]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// Write rule to 3 files
void rule_write(int order, char *filename, double x[], double w[], double r[]) {
    char f_r[80], f_w[80], f_x[80];
    snprintf(f_r, sizeof(f_r), "%s_r.txt", filename);
    snprintf(f_w, sizeof(f_w), "%s_w.txt", filename);
    snprintf(f_x, sizeof(f_x), "%s_x.txt", filename);

    printf("  Creating quadrature files for \"%s\"\n", filename);
    r8mat_write(f_w, 1, order, w);
    r8mat_write(f_x, 1, order, x);
    r8mat_write(f_r, 1, 2, r);
}

// Unified entry point
int main(int argc, char *argv[]) {
    int n;
    double a, b;
    char filename[256];

    if (argc >= 2) n = atoi(argv[1]); else { printf("Enter N: "); scanf("%d", &n); }
    if (argc >= 3) a = atof(argv[2]); else { printf("Enter A: "); scanf("%lf", &a); }
    if (argc >= 4) b = atof(argv[3]); else { printf("Enter B: "); scanf("%lf", &b); }
    if (argc >= 5) strncpy(filename, argv[4], 255); else { printf("Enter filename: "); scanf("%s", filename); }
    filename[255] = '\0';

    double *r = (double *)malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

    // Sequential
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double *x = ccn_compute_points_new(n);
    double *w = nc_compute_new(n, -1.0, +1.0, x);
    rescale(a, b, n, x, w);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double time_seq = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_nsec - t1.tv_nsec) / 1e6;

    // CUDA warmup
    double *xx = ccn_compute_points_new(n);
    double *ww_warmup = nc_compute_new_cuda(n, a, b, xx);
    cudaFreeHost(ww_warmup);

    // CUDA timing
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

    free(r); free(x); free(w); free(xx); cudaFreeHost(ww);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
