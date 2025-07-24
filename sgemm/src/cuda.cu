#include "../include/common.hpp"
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#define TILE_WIDTH 16
#define BLOCK_WIDTH 8

__global__ void kernel(
    const float* __restrict__ matA, int lda,
    const float* __restrict__ matBT, int ldb,
          float* matC,  int ldc,
    float alpha, float beta,
    int numRowsA, int numColsB, int sharedDim
) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty * 2;
    int col = blockIdx.x * TILE_WIDTH + tx * 2;

    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int t = 0; t < (sharedDim + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        for (int i = 0; i < 2; ++i) {
            int globalRowA = row + i;
            int globalColA = t * TILE_WIDTH + tx * 2;
            for (int j = 0; j < 2; ++j) {
                int tileCol = tx * 2 + j;
                int globalColA2 = t * TILE_WIDTH + tileCol;
                sharedA[ty * 2 + i][tileCol] =
                    (globalRowA < numRowsA && globalColA2 < sharedDim) ?
                        matA[globalRowA + globalColA2 * lda] : 0.0f;
            }
        }
        for (int i = 0; i < 2; ++i) {
            int globalRowB = t * TILE_WIDTH + ty * 2 + i;
            int globalColB = col;
            for (int j = 0; j < 2; ++j) {
                int tileCol = tx * 2 + j;
                int globalColB2 = col + j;
                sharedB[ty * 2 + i][tileCol] =
                    (globalColB2 < numColsB && globalRowB < sharedDim) ?
                        matBT[globalColB2 + globalRowB * ldb] : 0.0f;
            }
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    acc[i][j] += sharedA[ty * 2 + i][k] * sharedB[k][tx * 2 + j];
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            if (row + i < numRowsA && col + j < numColsB)
                matC[(row + i) + (col + j) * ldc] =
                    beta * matC[(row + i) + (col + j) * ldc] + alpha * acc[i][j];
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
    nvtxRangePushA("CUDA malloc");
    float* dA;
    float* dBT;
    float* dC;
    cudaMalloc(&dA,  numRowsA * sharedDim * sizeof(float));
    cudaMalloc(&dBT, numColsB * sharedDim * sizeof(float));
    cudaMalloc(&dC,  numRowsA * numColsB * sizeof(float));
    nvtxRangePop();

    nvtxRangePushA("Host to Device copy");
    cudaMemcpy(dA,  matA,  numRowsA * sharedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, matBT, numColsB * sharedDim * sizeof(float), cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid((numColsB + TILE_WIDTH - 1) / TILE_WIDTH, (numRowsA + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    nvtxRangePushA("SGEMM kernel launch");
    kernel<<<grid, block>>>(
        dA, numRowsA, dBT, numColsB, dC, numRowsA,
        alpha, beta,
        numRowsA, numColsB, sharedDim
    );
    cudaDeviceSynchronize();
    nvtxRangePop();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(timeMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    nvtxRangePushA("Device to Host copy");
    cudaMemcpy(matC, dC, numRowsA * numColsB * sizeof(float), cudaMemcpyDeviceToHost);
    nvtxRangePop();

    cudaFree(dA);
    cudaFree(dBT);
    cudaFree(dC);
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
    nvtxRangePushA("Total SGEMM");
    sgemm(
        numRowsA, numColsB, sharedDim,
        1.0f,
        matA.data(), numRowsA,
        matBT.data(), numColsB,
        0.0f,
        matC.data(), numRowsA,
        &timeMs
    );
    nvtxRangePop();
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
