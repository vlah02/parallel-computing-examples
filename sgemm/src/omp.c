#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <omp.h>

#include "../include/common.hpp"

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

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

void run_omp(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s A.txt BT.txt output_root\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *fileA  = argv[1];
    const char *fileBT = argv[2];
    const char *root   = argv[3];

    std::vector<float> matA, matBT;
    int m, k1, n, k2;
    if (!readColMajorMatrixFile(fileA,  m,  k1, matA) ||
        !readColMajorMatrixFile(fileBT, n,  k2, matBT) ||
         k1 != k2)
    {
        fprintf(stderr, "Error reading inputs or mismatched dims\n");
        exit(EXIT_FAILURE);
    }
    int k = k1;

    const char *slash = strrchr(root, '/');
    const char *name  = slash ? slash+1 : root;
    char base[256];
    strncpy(base, name, sizeof(base)-1);
    base[sizeof(base)-1] = '\0';
    if (char *dot = strchr(base, '.')) *dot = '\0';

    char seqtime[512];
    snprintf(seqtime, sizeof(seqtime),
             "output/seq/%s.txt_time.txt", base);
    FILE *fs = fopen(seqtime, "r");
    if (!fs) {
        fprintf(stderr, "Error: cannot open \"%s\"\n", seqtime);
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
    }
    double cpu_sec = sum / cnt;

    std::vector<float> C_seq;
    int rm, rn;
    char seqout[512];
    snprintf(seqout, sizeof(seqout), "output/seq/%s.txt", base);
    if (!readColMajorMatrixFile(seqout, rm, rn, C_seq) ||
         rm != m || rn != n)
    {
        fprintf(stderr, "Error: cannot load \"%s\" or size mismatch\n", seqout);
        exit(EXIT_FAILURE);
    }

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

    bool ok = true;
    for (int i = 0; i < m*n; i++) {
        if (fabsf(C_seq[i] - C_omp[i]) > 1e-3f) {
            ok = false;
            break;
        }
    }

    printf("%s  Test %s%s\n",            BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
    printf("%s  Sequential time: %s%.6f s %s\n", BOLD, BLUE, cpu_sec, CLEAR);
    printf("%s  Parallel time:   %s%.6f s %s\n", BOLD, BLUE, omp_sec, CLEAR);
    printf("%s  Speedup:         %s%.2fx %s\n\n", BOLD, BLUE, cpu_sec / omp_sec, CLEAR);

    writeColMajorMatrixFile(root, m, n, C_omp);
    char tf[256];
    snprintf(tf, sizeof(tf), "%s_time.txt", root);
    FILE *f = fopen(tf, "a");
    if (f) {
        fprintf(f, "%.6f\n", omp_sec);
        fclose(f);
    } else {
        perror("fopen timefile");
    }
}

int main(int argc, char *argv[]) {
    run_omp(argc, argv);
    return 0;
}