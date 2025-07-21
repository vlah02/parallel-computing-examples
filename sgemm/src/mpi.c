#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <vector>
#include <cmath>

#include "../include/common.hpp"

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

#define MASTER 0

void parallelSgemm(char transa, char transb,
                   int m, int n, int k,
                   float alpha,
                   const float *A, int lda,
                   const float *B, int ldb,
                   float beta,
                   float *C, int ldc,
                   int rank, int size)
{
    if ((transa != 'N' && transa != 'n') ||
        (transb != 'T' && transb != 't'))
    {
        if (rank == MASTER)
            fprintf(stderr, "parallelSgemm: unsupported transpose option\n");
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

int main(int argc, char *argv[])
{
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

    const char *fileA  = argv[1];
    const char *fileBT = argv[2];
    const char *root   = argv[3];

    int m, k1, n, k2;
    std::vector<float> A_vec, BT_vec;
    if (rank == MASTER) {
        if (!readColMajorMatrixFile(fileA,  m, k1, A_vec) ||
            !readColMajorMatrixFile(fileBT, n, k2, BT_vec) ||
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
        A_vec .resize(m * k);
        BT_vec.resize(n * k);
    }
    MPI_Bcast(A_vec .data(), m*k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(BT_vec.data(), n*k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    std::vector<float> C_vec(m * n, 0.0f);
    double t0 = MPI_Wtime();
    parallelSgemm('N','T',
                  m, n, k,
                  1.0f,
                  A_vec.data(),  m,
                  BT_vec.data(), n,
                  0.0f,
                  C_vec.data(),  m,
                  rank, size);
    double t1 = MPI_Wtime();
    double mpi_sec = t1 - t0;

    if (rank == MASTER) {
        const char *slash = strrchr(root, '/');
        const char *name  = slash ? slash+1 : root;
        char base[256];
        strncpy(base, name, sizeof(base)-1);
        base[sizeof(base)-1] = '\0';
        if (char *dot = strchr(base, '.')) *dot = '\0';

        char seqtime[512];
        snprintf(seqtime, sizeof(seqtime), "output/seq/%s.txt_time.txt", base);
        FILE *fs = fopen(seqtime, "r");
        if (!fs) {
            fprintf(stderr, "Error: cannot open \"%s\"\n", seqtime);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double sum = 0, tv;
        int cnt = 0;
        while (fscanf(fs, "%lf", &tv) == 1) {
            sum += tv;
            cnt++;
        }
        fclose(fs);
        if (cnt == 0) {
            fprintf(stderr, "Error: no entries in \"%s\"\n", seqtime);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double cpu_sec = sum / cnt;

        std::vector<float> C_seq;
        int rm, rn;
        char seqout[512];
        snprintf(seqout, sizeof(seqout),
                 "output/seq/%s.txt", base);
        if (!readColMajorMatrixFile(seqout, rm, rn, C_seq)
         || rm != m || rn != n)
        {
            fprintf(stderr, "Error: cannot load \"%s\" or size mismatch\n", seqout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        bool ok = true;
        for (int i = 0; i < m*n; i++) {
            if (fabsf(C_seq[i] - C_vec[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }

        printf("%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
        printf("%s  Sequential time: %s%.6f s %s\n", BOLD, BLUE, cpu_sec, CLEAR);
        printf("%s  Parallel time:   %s%.6f s %s\n", BOLD, BLUE, mpi_sec, CLEAR);
        printf("%s  Speedup:         %s%.2fx %s\n", BOLD, BLUE, cpu_sec/mpi_sec, CLEAR);
        printf("\n");

        writeColMajorMatrixFile(root, m, n, C_vec);
        char tf[512];
        snprintf(tf, sizeof(tf), "%s_time.txt", root);
        if (FILE *f = fopen(tf, "a")) {
            fprintf(f, "%.6f\n", mpi_sec);
            fclose(f);
        } else {
            perror("fopen timefile");
        }
    }

    MPI_Finalize();
    return 0;
}
