#include "../include/common.hpp"
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void kernel(
    const float* matA, int lda,
    const float* matBT, int ldb,
          float* matC,  int ldc,
    float alpha, float beta,
    int numRowsA, int numColsB, int sharedDim
) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float acc = 0.0f;

    int numTiles = (sharedDim + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_WIDTH + tx;
        sharedA[ty][tx] = (row < numRowsA && aCol < sharedDim)
            ? matA[row + aCol * lda]
            : 0.0f;

        int bRow = t * TILE_WIDTH + ty;
        sharedB[ty][tx] = (col < numColsB && bRow < sharedDim)
            ? matBT[col + bRow * ldb]
            : 0.0f;

        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; ++j) {
            acc += sharedA[ty][j] * sharedB[j][tx];
        }
        __syncthreads();
    }

    if (row < numRowsA && col < numColsB) {
        matC[row + col * ldc] = beta * matC[row + col * ldc] + alpha * acc;
    }
}

void sgemm(
    int numRowsA, int numColsB, int sharedDim,
    float alpha,
    const float* matA, int lda,
    const float* matBT, int ldb,
    float beta,
    float* matC, int ldc,
    float* timeMs
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* dA;
    float* dBT;
    float* dC;
    cudaMalloc(&dA,  numRowsA * sharedDim * sizeof(float));
    cudaMalloc(&dBT, numColsB * sharedDim * sizeof(float));
    cudaMalloc(&dC,  numRowsA * numColsB * sizeof(float));

    cudaMemcpy(dA,  matA,  numRowsA * sharedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, matBT, numColsB * sharedDim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((numColsB + TILE_WIDTH - 1) / TILE_WIDTH,
              (numRowsA + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    cudaEventRecord(start, 0);
    kernel<<<grid, block>>>(
        dA, numRowsA, dBT, numColsB, dC, numRowsA,
        alpha, beta,
        numRowsA, numColsB, sharedDim
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(matC, dC, numRowsA * numColsB * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(timeMs, start, stop);

    cudaFree(dA);
    cudaFree(dBT);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " A.txt BT.txt output_root" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string matAFile = argv[1];
    std::string matBTFile = argv[2];
    std::string outputRoot = argv[3];

    std::vector<float> matA, matBT;
    int numRowsA, numColsA, numColsB, numRowsB;

    if (!readColMajorMatrixFile(matAFile, numRowsA, numColsA, matA) ||
        !readColMajorMatrixFile(matBTFile, numColsB, numRowsB, matBT) ||
        numColsA != numRowsB)
    {
        std::cerr << "Error loading inputs or mismatched dims" << std::endl;
        exit(EXIT_FAILURE);
    }
    int sharedDim = numColsA;

    std::vector<float> matC(numRowsA * numColsB, 0.0f);

    float timeMs = 0.0f;
    sgemm(
        numRowsA, numColsB, sharedDim,
        1.0f,
        matA.data(), numRowsA,
        matBT.data(), numColsB,
        0.0f,
        matC.data(), numRowsA,
        &timeMs
    );
    double timeParallel = timeMs * 1e-3;

    std::string base = getOutputBase(outputRoot);

    double timeSequential = 0;
    if (!loadSequentialTiming(base, timeSequential)) {
        std::cerr << "Error: cannot load sequential timing for \"" << base << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<float> matSequential;
    int seqRows, seqCols;
    if (!loadSequentialResult(base, seqRows, seqCols, matSequential) ||
        seqRows != numRowsA || seqCols != numColsB)
    {
        std::cerr << "Error: cannot load sequential result or size mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }

    bool ok = compareResults(matSequential, matC);

    std::cout << BOLD << "  Test " << (ok ? (std::string(GREEN) + "PASSED") : (std::string(RED) + "FAILED")) << CLEAR << std::endl;
    std::cout << BOLD << "  Sequential time: " << BLUE << timeSequential << " s " << CLEAR << std::endl;
    std::cout << BOLD << "  Parallel time:   " << BLUE << timeParallel << " s " << CLEAR << std::endl;
    std::cout << BOLD << "  Speedup:         " << BLUE << (timeSequential / timeParallel) << "x " << CLEAR << std::endl;
    std::cout << std::endl;

    writeColMajorMatrixFile(outputRoot, numRowsA, numColsB, matC);
    appendTiming(outputRoot, timeParallel);

    return 0;
}
