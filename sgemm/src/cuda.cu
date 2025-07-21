#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "../include/common.hpp"

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

#define TILE_WIDTH 32

__global__ void kernelSgemm(
    const float *A, int lda,
    const float *BT, int ldb,
          float *C,  int ldc,
    float alpha, float beta,
    int m, int n, int k
) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y,   tx = threadIdx.x;
    int mm = by * TILE_WIDTH + ty;
    int nn = bx * TILE_WIDTH + tx;
    float acc = 0.0f;

    int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_WIDTH + tx;
        sharedA[ty][tx] = (mm < m && aCol < k)
            ? A[mm + aCol * lda]
            : 0.0f;

        int bRow = t * TILE_WIDTH + ty;
        sharedB[ty][tx] = (nn < n && bRow < k)
            ? BT[nn + bRow * ldb]
            : 0.0f;

        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; ++j) {
            acc += sharedA[ty][j] * sharedB[j][tx];
        }
        __syncthreads();
    }

    if (mm < m && nn < n) {
        C[mm + nn * ldc] = beta * C[mm + nn * ldc] + alpha * acc;
    }
}

void cudaSgemm(
    int    m, int n, int k,
    float  alpha,
    const float *A, int lda,
    const float *BT, int ldb,
    float  beta,
    float  *C,  int ldc,
    float  *time_ms
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *dA, *dBT, *dC;
    cudaMalloc(&dA,  m * k * sizeof(float));
    cudaMalloc(&dBT, n * k * sizeof(float));
    cudaMalloc(&dC,  m * n * sizeof(float));

    cudaMemcpy(dA,  A,   m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, BT,  n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((n + TILE_WIDTH -1)/TILE_WIDTH,
              (m + TILE_WIDTH -1)/TILE_WIDTH);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    cudaEventRecord(start, 0);
    kernelSgemm<<<grid,block>>>(dA, lda, dBT, ldb, dC, ldc,
                                alpha, beta, m, n, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(C, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaFree(dA);
    cudaFree(dBT);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s A.txt BT.txt output_root\n", argv[0]);
        return 1;
    }

    std::vector<float> matA, matBT;
    int m, k1, n, k2;
    if (!readColMajorMatrixFile(argv[1], m, k1, matA)
     || !readColMajorMatrixFile(argv[2], n, k2, matBT)
     || k1 != k2) {
        fprintf(stderr, "Error loading inputs or mismatched dims\n");
        return 1;
    }
    int k = k1;

    const char *root = argv[3];
    const char *slash = strrchr(root, '/');
    const char *name  = slash ? slash+1 : root;
    char base[256];
    strncpy(base, name, sizeof(base)-1);
    base[sizeof(base)-1] = '\0';
    if (char *dot = strchr(base, '.')) *dot = '\0';

    char seqtime[512];
    snprintf(seqtime, sizeof(seqtime),
             "output/seq/%s.txt_time.txt", base);
    FILE *fs = fopen(seqtime, "r");
    if (!fs) {
        fprintf(stderr, "Error: cannot open \"%s\"\n", seqtime);
        return 1;
    }
    double sum = 0, tval;
    int cnt = 0;
    while (fscanf(fs, "%lf", &tval) == 1) {
        sum += tval;
        cnt++;
    }
    fclose(fs);
    if (cnt == 0) {
        fprintf(stderr, "Error: no entries in \"%s\"\n", seqtime);
        return 1;
    }
    double cpu_sec = sum / cnt;

    std::vector<float> C_seq;
    int rm, rn;
    char seqout[512];
    snprintf(seqout, sizeof(seqout), "output/seq/%s.txt", base);
    if (!readColMajorMatrixFile(seqout, rm, rn, C_seq)
     || rm != m || rn != n) {
        fprintf(stderr, "Error: cannot load \"%s\" or size mismatch\n", seqout);
        return 1;
    }

    std::vector<float> C_gpu(m*n, 0.0f);
    float gpu_ms = 0.0f;
    cudaSgemm(
        m, n, k,
        1.0f,
        matA .data(), m,
        matBT.data(), n,
        0.0f,
        C_gpu.data(), m,
        &gpu_ms
    );
    double gpu_sec = gpu_ms * 1e-3;

    bool ok = true;
    for (int i = 0; i < m*n; i++) {
        if (fabsf(C_seq[i] - C_gpu[i]) > 1e-3f) {
            ok = false;
            break;
        }
    }

    printf("%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
    printf("%s  Sequential time: %s%.6f s %s\n", BOLD, BLUE, cpu_sec, CLEAR);
    printf("%s  Parallel time:   %s%.6f s %s\n", BOLD, BLUE, gpu_sec, CLEAR);
    printf("%s  Speedup:         %s%.2fx %s\n", BOLD, BLUE, cpu_sec / gpu_sec, CLEAR);
    printf("\n");

    writeColMajorMatrixFile(root, m, n, C_gpu);
    {
      char tf[512];
      snprintf(tf, sizeof(tf), "%s_time.txt", root);
      if (FILE *f = fopen(tf,"a")) {
        fprintf(f,"%.6f\n", gpu_sec);
        fclose(f);
      }
    }

    return ok ? 0 : 1;
}
