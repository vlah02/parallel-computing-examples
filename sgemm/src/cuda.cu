// src/cuda.cu

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cuda_runtime.h>

#include "../include/cuda.h"
#include "../include/common.h"

#define TILE_WIDTH 32

// --------------------------------------------------
// Tile‐based SGEMM kernel (column‐major, BT stored as B^T)
// --------------------------------------------------
__global__ void kernelSgemm(
    const float *A, int lda,
    const float *BT, int ldb,
    float       *C, int ldc,
    float alpha, float beta,
    int m, int n, int k
) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y,   tx = threadIdx.x;

    int mm = by * TILE_WIDTH + ty;  // row in C, row in A
    int nn = bx * TILE_WIDTH + tx;  // col in C, row in B^T
    float acc = 0.0f;

    int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + tx;  // iterate columns of A
        if (mm < m && aCol < k)
            sharedA[ty][tx] = A[mm + aCol * lda];
        else
            sharedA[ty][tx] = 0.0f;

        int bRow = t * TILE_WIDTH + ty;  // iterate columns of B^T (rows of B)
        if (nn < n && bRow < k)
            // <<< FIXED HERE >>>
            // load BT[row = nn, col = bRow] (= B[bRow][nn])
            sharedB[ty][tx] = BT[nn + bRow * ldb];
        else
            sharedB[ty][tx] = 0.0f;

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

// --------------------------------------------------
// Host‐side wrapper
// --------------------------------------------------
void cudaSgemm(
    char   transa, char transb,
    int    m, int n, int k,
    float  alpha,
    const float *A, int lda,
    const float *BT, int ldb,
    float  beta,
    float  *C,  int ldc,
    float  *time_ms
) {
    // we only support transa='N', transb='T'
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *dA, *dBT, *dC;
    cudaMalloc(&dA,   m * k * sizeof(float));
    cudaMalloc(&dBT,  n * k * sizeof(float));
    cudaMalloc(&dC,   m * n * sizeof(float));

    cudaMemcpy(dA,  A,    m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, BT,   n * k * sizeof(float), cudaMemcpyHostToDevice);

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

// --------------------------------------------------
// Simple CPU SGEMM for correctness/timing
// --------------------------------------------------
static void basicSgemm(
    int    m, int n, int k,
    float  alpha,
    const float *A, int lda,
    const float *BT, int ldb,
    float  beta,
    float  *C,  int ldc
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++)
                acc += A[i + t*lda] * BT[j + t*ldb];
            C[i + j*ldc] = beta*C[i + j*ldc] + alpha*acc;
        }
    }
}

// --------------------------------------------------
// Host main
// --------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s A.txt BT.txt output_root\n", argv[0]);
        return 1;
    }

    // 1) Load A (m×k) and Bᵀ (n×k)
    std::vector<float> matA, matBT;
    int m,k1,n,k2;
    if (!readColMajorMatrixFile(argv[1], m, k1, matA)
     || !readColMajorMatrixFile(argv[2], n, k2, matBT)
     || k1 != k2) {
        fprintf(stderr, "Error loading inputs or incompatible dims\n");
        return 1;
    }
    int k = k1;

    // 2) Allocate C_cpu and C_gpu
    std::vector<float> C_cpu(m*n, 0.0f),
                       C_gpu(m*n, 0.0f);

    // 3) CPU timing
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    basicSgemm(m,n,k, 1.0f, matA.data(),m,
                     matBT.data(),n, 0.0f, C_cpu.data(),m);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double cpu_sec = (t1.tv_sec-t0.tv_sec)
                   + (t1.tv_nsec-t0.tv_nsec)*1e-9;

    // 4) GPU timing
    float gpu_ms = 0.0f;
    cudaSgemm('N','T', m,n,k, 1.0f,
              matA.data(),m,
              matBT.data(),n,
              0.0f, C_gpu.data(),m,
              &gpu_ms);
    double gpu_sec = gpu_ms*1e-3;

    // 5) Check correctness
    bool ok = true;
    for (int i = 0; i < m*n; i++) {
        if (fabsf(C_cpu[i] - C_gpu[i]) > 1e-3f) {
            ok = false; break;
        }
    }

    // 6) Report
    printf("Test %s\n", ok ? "PASSED" : "FAILED");
    printf("  CPU time: %.6f s\n", cpu_sec);
    printf("  GPU time: %.6f s\n", gpu_sec);
    printf("  Speedup:  %.2fx\n", cpu_sec / gpu_sec);

    // 7) Write result and timing
    writeColMajorMatrixFile(argv[3], m, n, C_gpu);
    {
      char tf[512];
      snprintf(tf,sizeof(tf), "%s_time.txt", argv[3]);
      FILE *f = fopen(tf,"a");
      if (f) {
          fprintf(f,"%.6f\n", gpu_sec);
          fclose(f);
      }
    }

    return ok ? 0 : 1;
}
