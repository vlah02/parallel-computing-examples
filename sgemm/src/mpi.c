#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "../include/common.h"
#include "../include/mpi.h"

#define MASTER 0

void parallelSgemm(char transa, char transb, int m, int n, int k, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc, int rank, int size) {
    if (transa != 'N' && transa != 'n') {
        if (rank == MASTER) fprintf(stderr, "Unsupported transa\n");
        return;
    }
    if (transb != 'T' && transb != 't') {
        if (rank == MASTER) fprintf(stderr, "Unsupported transb\n");
        return;
    }

    int chunk_size = m / size;
    int leftover = m % size;
    int start = chunk_size * rank + (leftover > rank ? rank : leftover);
    int end = start + chunk_size + (leftover > rank ? 1 : 0);

    float *localC = (float *)calloc(m * n, sizeof(float));
    if (!localC) { perror("calloc localC"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int mm = start; mm < end; ++mm) {
        for (int nn = 0; nn < n; ++nn) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                float a = A[mm + i * lda];
                float b = B[nn + i * ldb];
                sum += a * b;
            }
            localC[mm + nn * ldc] = beta * localC[mm + nn * ldc] + alpha * sum;
        }
    }

    MPI_Reduce(localC, C, m * n, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    free(localC);
}

void run_mpi(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == MASTER)
            fprintf(stderr, "Usage: %s matrixA.txt matrixBT.txt output_root\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    const char *fileA = argv[1];
    const char *fileBT = argv[2];
    const char *root = argv[3];

    int m, k, n;
    float *A = NULL, *BT = NULL;
    float *C = NULL;

    if (rank == MASTER) {
        A = load_matrix(fileA, &m, &k);
        BT = load_matrix(fileBT, &n, &k);
    }

    MPI_Bcast(&m, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank != MASTER) {
        A = (float *)malloc(m * k * sizeof(float));
        BT = (float *)malloc(n * k * sizeof(float));
    }
    C = (float *)calloc(m * n, sizeof(float));
    if (!A || !BT || !C) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    MPI_Bcast(A, m * k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(BT, n * k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    double t_start = MPI_Wtime();
    parallelSgemm('N', 'T', m, n, k, 1.0f, A, m, BT, n, 0.0f, C, m, rank, size);
    double t_end = MPI_Wtime();

    if (rank == MASTER) {
        printf("MPI time: %.6f seconds with %d processes\n", t_end - t_start, size);

        write_matrix(root, m, n, C);

        char timefile[256];
        snprintf(timefile, sizeof(timefile), "%s_time.txt", root);
        FILE *f = fopen(timefile, "a");
        if (f) {
            fprintf(f, "%.6f\n", t_end - t_start);
            fclose(f);
        } else {
            perror("fopen");
        }
    }

    free(A);
    free(BT);
    free(C);
    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    run_mpi(argc, argv);
    return 0;
}
