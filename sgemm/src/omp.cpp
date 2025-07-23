#include "../include/common.hpp"
#include <omp.h>

void sgemm(
    char transA, char transB,
    int  numRowsA, int numColsB, int sharedDim,
    float alpha,
    const float* matA, int lda,
    const float* matBT, int ldb,
    float beta,
    float* matC, int ldc
) {
    if ((transA != 'N' && transA != 'n') ||
        (transB != 'T' && transB != 't'))
    {
        std::cerr << "sgemm: unsupported transpose options\n";
        return;
    }
    int numThreads = omp_get_max_threads();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk = numRowsA / numThreads;
        int rem   = numRowsA % numThreads;
        int start = chunk * tid + (rem > tid ? tid : rem);
        int end   = start + chunk + (rem > tid ? 1 : 0);

        for (int i = start; i < end; ++i) {
            for (int j = 0; j < numColsB; ++j) {
                float acc = 0.0f;
                for (int t = 0; t < sharedDim; ++t) {
                    acc += matA[i + t*lda] * matBT[j + t*ldb];
                }
                matC[i + j*ldc] = beta * matC[i + j*ldc] + alpha * acc;
            }
        }
    }
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
        std::cerr << "Error reading inputs or mismatched dims" << std::endl;
        exit(EXIT_FAILURE);
    }
    int sharedDim = numColsA;

    std::vector<float> matC(numRowsA * numColsB, 0.0f);

    double t0 = omp_get_wtime();
    sgemm(
        'N', 'T',
        numRowsA, numColsB, sharedDim,
        1.0f,
        matA.data(), numRowsA,
        matBT.data(), numColsB,
        0.0f,
        matC.data(), numRowsA
    );
    double t1 = omp_get_wtime();
    double timeParallel = t1 - t0;

    std::string base = getOutputBase(outputRoot);

    double timeSequential = 0;
    if (!loadSequentialTiming(base, timeSequential)) {
        std::cerr << "Error: cannot load sequential timing for \"" << base << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<float> matSequential;
    int seqRows, seqCols;
    if (!loadSequentialResult(base, seqRows, seqCols, matSequential) || seqRows != numRowsA || seqCols != numColsB) {
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
