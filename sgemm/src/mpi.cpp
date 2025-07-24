#include "../include/common.hpp"
#include <mpi.h>

#define MASTER 0

void sgemm(
    char transA, char transB,
    int numRowsA, int numColsB, int sharedDim,
    float alpha,
    const float* matA, int lda,
    const float* matBT, int ldb,
    float beta,
    float* matC, int ldc,
    int rank, int size
) {
    if ((transA != 'N' && transA != 'n') ||
        (transB != 'T' && transB != 't')) {
        if (rank == MASTER)
            std::cerr << "sgemm: unsupported transpose option" << std::endl;
        return;
    }

    int colsPerProc = numColsB / size;
    int rem = numColsB % size;
    int colStart = colsPerProc * rank + (rem > rank ? rank : rem);
    int localNumCols = colsPerProc + (rem > rank ? 1 : 0);

    std::vector<float> localC(numRowsA * localNumCols, 0.0f);

    constexpr int BLOCK_SIZE = 16;
    for (int jj = 0; jj < localNumCols; jj += BLOCK_SIZE) {
        int j_max = std::min(jj + BLOCK_SIZE, localNumCols);
        for (int ii = 0; ii < numRowsA; ii += BLOCK_SIZE) {
            int i_max = std::min(ii + BLOCK_SIZE, numRowsA);
            for (int j = jj; j < j_max; ++j) {
                int globalJ = colStart + j;
                for (int i = ii; i < i_max; ++i) {
                    float acc = 0.0f;
                    for (int t = 0; t < sharedDim; ++t) {
                        acc += matA[i + t * lda] * matBT[globalJ + t * ldb];
                    }
                    localC[i + j * numRowsA] = beta * localC[i + j * numRowsA] + alpha * acc;
                }
            }
        }
    }

    std::vector<int> recvcounts(size), displs(size);
    int ofs = 0;
    for (int r = 0; r < size; ++r) {
        int cpc = numColsB / size;
        int remc = numColsB % size;
        int lnc = cpc + (remc > r ? 1 : 0);
        recvcounts[r] = numRowsA * lnc;
        displs[r] = ofs;
        ofs += recvcounts[r];
    }

    MPI_Gatherv(
        localC.data(), numRowsA * localNumCols, MPI_FLOAT,
        matC, recvcounts.data(), displs.data(), MPI_FLOAT,
        MASTER, MPI_COMM_WORLD
    );
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == MASTER)
            std::cerr << "Usage: " << argv[0] << " A.txt BT.txt output_root" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string matAFile = argv[1];
    std::string matBTFile = argv[2];
    std::string outputRoot = argv[3];

    std::vector<float> matA, matBT;
    int numRowsA, numColsA, numColsB, numRowsB;

    if (rank == MASTER) {
        if (!readColMajorMatrixFile(matAFile,  numRowsA,  numColsA, matA) ||
            !readColMajorMatrixFile(matBTFile, numColsB, numRowsB, matBT) ||
            numColsA != numRowsB)
        {
            std::cerr << "Error reading inputs or mismatched dims" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&numRowsA,  1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&numColsA,  1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&numColsB,  1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&numRowsB,  1, MPI_INT, MASTER, MPI_COMM_WORLD);

    int sharedDim = numColsA;

    if (rank != MASTER) {
        matA.resize(numRowsA * sharedDim);
        matBT.resize(numColsB * sharedDim);
    }
    MPI_Bcast(matA.data(),  numRowsA * sharedDim, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(matBT.data(), numColsB * sharedDim, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    std::vector<float> matC(numRowsA * numColsB, 0.0f);

    double t0 = MPI_Wtime();
    sgemm('N','T',
        numRowsA, numColsB, sharedDim,
        1.0f,
        matA.data(), numRowsA,
        matBT.data(), numColsB,
        0.0f,
        matC.data(), numRowsA,
        rank, size
    );
    double t1 = MPI_Wtime();
    double timeParallel = t1 - t0;

    if (rank == MASTER) {
        std::string base = getOutputBase(outputRoot);

        double timeSequential = 0;
        if (!loadSequentialTiming(base, timeSequential)) {
            std::cerr << "Error: cannot load sequential timing for \"" << base << "\"" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<float> matSequential;
        int seqRows, seqCols;
        if (!loadSequentialResult(base, seqRows, seqCols, matSequential) ||
            seqRows != numRowsA || seqCols != numColsB)
        {
            std::cerr << "Error: cannot load sequential result or size mismatch" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        bool ok = compareResults(matSequential, matC);

        std::cout << BOLD << "  Test " << (ok ? (std::string(GREEN) + "PASSED") : (std::string(RED) + "FAILED")) << CLEAR << std::endl;
        std::cout << BOLD << "  Sequential time: " << BLUE << timeSequential << " s " << CLEAR << std::endl;
        std::cout << BOLD << "  Parallel time:   " << BLUE << timeParallel << " s " << CLEAR << std::endl;
        std::cout << BOLD << "  Speedup:         " << BLUE << (timeSequential / timeParallel) << "x " << CLEAR << std::endl;
        std::cout << std::endl;

        writeColMajorMatrixFile(outputRoot, numRowsA, numColsB, matC);
        appendTiming(outputRoot, timeParallel);
    }

    MPI_Finalize();
    return 0;
}
