#include "../include/common.hpp"
#include <chrono>

void sgemm(
    char transA, char transB,
    int numRowsA, int numColsB, int sharedDim,
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
    for (int i = 0; i < numRowsA; ++i) {
        for (int j = 0; j < numColsB; ++j) {
            float acc = 0.0f;
            for (int t = 0; t < sharedDim; ++t) {
                acc += matA[i + t*lda] * matBT[j + t*ldb];
            }
            matC[i + j*ldc] = beta * matC[i + j*ldc] + alpha * acc;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " A.txt BT.txt output_root" << std::endl;
        return 1;
    }

    std::string matAFile   = argv[1];
    std::string matBTFile  = argv[2];
    std::string outputRoot = argv[3];

    int numRowsA, numColsA, numColsB, numRowsB;
    std::vector<float> matA, matBT;

    if (!readColMajorMatrixFile(matAFile, numRowsA, numColsA, matA)) {
        std::cerr << "Failed to load A from " << matAFile << std::endl;
        return 1;
    }
    if (!readColMajorMatrixFile(matBTFile, numColsB, numRowsB, matBT)) {
        std::cerr << "Failed to load Bᵀ from " << matBTFile << std::endl;
        return 1;
    }
    if (numColsA != numRowsB) {
        std::cerr << "Dimension mismatch: numColsA=" << numColsA << " vs numRowsB=" << numRowsB << std::endl;
        return 1;
    }
    int sharedDim = numColsA;

    std::vector<float> matC(numRowsA * numColsB, 0.0f);

    auto t0 = std::chrono::steady_clock::now();

    sgemm(
        'N', 'T',
        numRowsA, numColsB, sharedDim,
        1.0f,
        matA.data(), numRowsA,
        matBT.data(), numColsB,
        0.0f,
        matC.data(), numRowsA
    );

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << BOLD << "  Sequential time: " << BLUE << elapsed << " s " << CLEAR << std::endl;
    writeColMajorMatrixFile(outputRoot, numRowsA, numColsB, matC);

    appendTiming(outputRoot, elapsed);

    std::cout << std::endl;
    return 0;
}
