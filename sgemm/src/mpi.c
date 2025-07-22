#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstring>
#include "../include/common.hpp"

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
            fprintf(stderr, "sgemm: unsupported transpose option\n");
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

    MPI_Reduce(localC.data(), C, m*n,
               MPI_FLOAT, MPI_SUM, MASTER,
               MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == MASTER)
            fprintf(stderr, "Usage: %s A.txt BT.txt output_root\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

	std::vector<float> matA, matBT;
    int m, k1, n, k2;

    if (rank == MASTER) {
        if (!readColMajorMatrixFile(argv[1],  m, k1, matA) ||
            !readColMajorMatrixFile(argv[2], n, k2, matBT) ||
             k1 != k2)
        {
            fprintf(stderr, "Error reading inputs or mismatched dims\n");
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
    MPI_Bcast(matA .data(), m*k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(matBT.data(), n*k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    std::vector<float> C_vec(m * n, 0.0f);

    double t0 = MPI_Wtime();
    sgemm('N','T',
          m, n, k,
          1.0f,
          matA.data(),  m,
          matBT.data(), n,
          0.0f,
          C_vec.data(),  m,
          rank, size);
    double t1 = MPI_Wtime();
    double mpi_sec = t1 - t0;

    const char *root = argv[3];

    if (rank == MASTER) {
        char base[256];
        getOutputBase(root, base, sizeof(base));

        double cpu_sec = 0;
        if (!loadSequentialTiming(base, cpu_sec)) {
            fprintf(stderr, "Error: cannot load sequential timing for \"%s\"\n", base);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<float> C_seq;
        int rm, rn;
        if (!loadSequentialResult(base, rm, rn, C_seq) || rm != m || rn != n) {
            fprintf(stderr, "Error: cannot load sequential result or size mismatch\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        bool ok = compareResults(C_seq, C_vec);

        printf("%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
        printf("%s  Sequential time: %s%.6f s %s\n", BOLD, BLUE, cpu_sec, CLEAR);
        printf("%s  Parallel time:   %s%.6f s %s\n", BOLD, BLUE, mpi_sec, CLEAR);
        printf("%s  Speedup:         %s%.2fx %s\n", BOLD, BLUE, cpu_sec/mpi_sec, CLEAR);
        printf("\n");

        writeColMajorMatrixFile(root, m, n, C_vec);
        appendTiming(root, mpi_sec);
    }

    MPI_Finalize();
    return 0;
}
