#include "../include/common.hpp"
#include <mpi.h>

#define MASTER 0

void sgemm(
    char transa, char transb,
    int m, int n, int k,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc,
    int rank, int size
) {
    if ((transa != 'N' && transa != 'n') ||
        (transb != 'T' && transb != 't'))
    {
        if (rank == MASTER)
            std::cerr << "sgemm: unsupported transpose option" << std::endl;
        return;
    }

    int chunk = m / size;
    int rem   = m % size;
    int start = chunk * rank + (rem > rank ? rank : rem);
    int end   = start + chunk + (rem > rank ? 1 : 0);

    std::vector<float> localC(m * n, 0.0f);

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += A[i + t*lda] * B[j + t*ldb];
            }
            localC[i + j*ldc] = beta * localC[i + j*ldc]
                              + alpha * sum;
        }
    }

    MPI_Reduce(localC.data(), C, m * n, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
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

    std::string fnameA = argv[1], fnameBT = argv[2], out_root = argv[3];

    std::vector<float> matA, matBT;
    int m, k1, n, k2;

    if (rank == MASTER) {
        if (!readColMajorMatrixFile(fnameA,  m, k1, matA) ||
            !readColMajorMatrixFile(fnameBT, n, k2, matBT) ||
            k1 != k2)
        {
            std::cerr << "Error reading inputs or mismatched dims" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&m,  1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&k1, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&n,  1, MPI_INT, MASTER, MPI_COMM_WORLD);
    int k = k1;

    if (rank != MASTER) {
        matA.resize(m * k);
        matBT.resize(n * k);
    }
    MPI_Bcast(matA .data(), m * k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(matBT.data(), n * k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    std::vector<float> C(m * n, 0.0f);

    double t0 = MPI_Wtime();
    sgemm('N','T',
          m, n, k,
          1.0f,
          matA.data(),  m,
          matBT.data(), n,
          0.0f,
          C.data(),  m,
          rank, size);
    double t1 = MPI_Wtime();
    double mpi_sec = t1 - t0;

    if (rank == MASTER) {
        std::string base = getOutputBase(out_root);

        double cpu_sec = 0;
        if (!loadSequentialTiming(base, cpu_sec)) {
            std::cerr << "Error: cannot load sequential timing for \"" << base << "\"" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<float> C_seq;
        int rm, rn;
        if (!loadSequentialResult(base, rm, rn, C_seq) || rm != m || rn != n) {
            std::cerr << "Error: cannot load sequential result or size mismatch" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        bool ok = compareResults(C_seq, C);

        std::cout << BOLD << "  Test " << (ok ? (std::string(GREEN) + "PASSED") : (std::string(RED) + "FAILED")) << CLEAR << std::endl;
        std::cout << BOLD << "  Sequential time: " << BLUE << cpu_sec << " s " << CLEAR << std::endl;
        std::cout << BOLD << "  Parallel time:   " << BLUE << mpi_sec << " s " << CLEAR << std::endl;
        std::cout << BOLD << "  Speedup:         " << BLUE << (cpu_sec / mpi_sec) << "x " << CLEAR << std::endl;
        std::cout << std::endl;

        writeColMajorMatrixFile(out_root, m, n, C);
        appendTiming(out_root, mpi_sec);
    }

    MPI_Finalize();
    return 0;
}
