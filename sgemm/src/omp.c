#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <omp.h>
#include "../include/common.hpp"

void sgemm(
    char transa, char transb,
    int  m,      int  n,      int  k,
    float alpha,
    const float *A, int lda,
    const float *BT, int ldb,
    float beta,
    float *C,       int ldc
) {
    if ((transa != 'N' && transa != 'n') ||
        (transb != 'T' && transb != 't'))
    {
        fprintf(stderr, "sgemm: unsupported transpose options\n");
        return;
    }
    int num_threads = omp_get_max_threads();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk = m / num_threads;
        int rem   = m % num_threads;
        int start = chunk * tid + (rem > tid ? tid : rem);
        int end   = start + chunk + (rem > tid ? 1 : 0);

        for (int i = start; i < end; ++i) {
            for (int j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int t = 0; t < k; ++t) {
                    acc += A[i + t*lda] * BT[j + t*ldb];
                }
                C[i + j*ldc] = beta*C[i + j*ldc] + alpha*acc;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s A.txt BT.txt output_root\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    std::vector<float> matA, matBT;
    int m, k1, n, k2;

    if (!readColMajorMatrixFile(argv[1],  m,  k1, matA) ||
        !readColMajorMatrixFile(argv[2], n,  k2, matBT) ||
         k1 != k2)
    {
        fprintf(stderr, "Error reading inputs or mismatched dims\n");
        exit(EXIT_FAILURE);
    }
    int k = k1;

	std::vector<float> C_omp(m*n, 0.0f);

    double t0 = omp_get_wtime();
    sgemm(
        'N','T',
        m, n, k,
        1.0f,
        matA .data(), m,
        matBT.data(), n,
        0.0f,
        C_omp.data(), m
    );
    double t1 = omp_get_wtime();
    double omp_sec = t1 - t0;

	const char *root = argv[3];
    char base[256];
    getOutputBase(root, base, sizeof(base));

    double cpu_sec = 0;
    if (!loadSequentialTiming(base, cpu_sec)) {
        fprintf(stderr, "Error: cannot load sequential timing for \"%s\"\n", base);
        exit(EXIT_FAILURE);
    }

    std::vector<float> C_seq;
    int rm, rn;
    if (!loadSequentialResult(base, rm, rn, C_seq) || rm != m || rn != n) {
        fprintf(stderr, "Error: cannot load sequential result or size mismatch\n");
        exit(EXIT_FAILURE);
    }

    bool ok = compareResults(C_seq, C_omp);

    printf("%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
    printf("%s  Sequential time: %s%.6f s %s\n", BOLD, BLUE, cpu_sec, CLEAR);
    printf("%s  Parallel time:   %s%.6f s %s\n", BOLD, BLUE, omp_sec, CLEAR);
    printf("%s  Speedup:         %s%.2fx %s\n", BOLD, BLUE, cpu_sec / omp_sec, CLEAR);
    printf("\n");

    writeColMajorMatrixFile(root, m, n, C_omp);
    appendTiming(root, omp_sec);

    return 0;
}
