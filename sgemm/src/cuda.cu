#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "../include/common.hpp"

#define TILE_WIDTH 32

__global__ void kernel(
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

void sgemm(
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
    kernel<<<grid,block>>>(dA, lda, dBT, ldb, dC, ldc, alpha, beta, m, n, k);
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
        std::cerr << "Usage: " << argv[0] << " A.txt BT.txt output_root" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string fnameA = argv[1], fnameBT = argv[2], out_root = argv[3];

    std::vector<float> matA, matBT;
    int m, k1, n, k2;

    if (!readColMajorMatrixFile(fnameA, m, k1, matA) ||
        !readColMajorMatrixFile(fnameBT, n, k2, matBT) ||
        k1 != k2)
    {
        std::cerr << "Error loading inputs or mismatched dims" << std::endl;
        exit(EXIT_FAILURE);
    }
    int k = k1;

    std::vector<float> C_gpu(m*n, 0.0f);

    float gpu_ms = 0.0f;
    sgemm(
        m, n, k,
        1.0f,
        matA .data(), m,
        matBT.data(), n,
        0.0f,
        C_gpu.data(), m,
        &gpu_ms
    );
    double gpu_sec = gpu_ms * 1e-3;

    std::string base = getOutputBase(out_root);

    double cpu_sec = 0;
    if (!loadSequentialTiming(base, cpu_sec)) {
        std::cerr << "Error: cannot load sequential timing for \"" << base << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<float> C_seq;
    int rm, rn;
    if (!loadSequentialResult(base, rm, rn, C_seq) || rm != m || rn != n) {
        std::cerr << "Error: cannot load sequential result or size mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }

    bool ok = compareResults(C_seq, C_gpu);

    std::cout << BOLD << "  Test " << (ok ? (std::string(GREEN) + "PASSED") : (std::string(RED) + "FAILED")) << CLEAR << std::endl;
    std::cout << BOLD << "  Sequential time: " << BLUE << cpu_sec << " s " << CLEAR << std::endl;
    std::cout << BOLD << "  Parallel time:   " << BLUE << gpu_sec << " s " << CLEAR << std::endl;
    std::cout << BOLD << "  Speedup:         " << BLUE << (cpu_sec / gpu_sec) << "x " << CLEAR << std::endl;
    std::cout << std::endl;

    writeColMajorMatrixFile(out_root, m, n, C_gpu);
    appendTiming(out_root, gpu_sec);

    return 0;
}
