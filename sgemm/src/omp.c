#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../include/common.h"
#include "../include/omp.h"

void parallelSgemm(char transa, char transb, int m, int n, int k, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc) {
#pragma omp parallel
    {
        int num_of_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        int chunk_size = m / num_of_threads;
        int leftover_size = m % num_of_threads;
        int start = chunk_size * thread_id + (leftover_size > thread_id ? thread_id : leftover_size);
        int end = start + chunk_size + (leftover_size > thread_id ? 1 : 0);

        for (int mm = start; mm < end; ++mm) {
            for (int nn = 0; nn < n; ++nn) {
                float c = 0.0f;
                for (int i = 0; i < k; ++i) {
                    float a = A[mm + i * lda];
                    float b = B[nn + i * ldb];
                    c += a * b;
                }
                C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
            }
        }
    }
}

void run_omp(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s matA.txt matB_T.txt result_basename\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int matArow, matAcol, matBrow, matBcol;
    char *fileA = argv[1];
    char *fileB = argv[2];
    char *output_base = argv[3];

    float *A = NULL, *BT = NULL, *C = NULL;
    read_matrix(fileA, &matArow, &matAcol, &A);
    read_matrix(fileB, &matBcol, &matBrow, &BT);  // B is transposed, so cols are rows

    C = (float *)malloc(matArow * matBcol * sizeof(float));
    if (!C) { perror("malloc C"); exit(EXIT_FAILURE); }

    double t_start = omp_get_wtime();
    parallelSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f,
                  A, matArow, BT, matBcol, 0.0f, C, matArow);
    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

    char result_file[256], time_file[256];
    snprintf(result_file, sizeof(result_file), "%s_result.txt", output_base);
    snprintf(time_file, sizeof(time_file), "%s_time.txt", output_base);

    write_matrix(result_file, matArow, matBcol, C);
    append_time(time_file, elapsed);

    printf("OMP execution time: %.6f seconds\n", elapsed);

    free(A);
    free(BT);
    free(C);
}

int main(int argc, char *argv[]) {
    run_omp(argc, argv);
    return 0;
}
