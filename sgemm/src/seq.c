#include <vector>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include "../include/common.hpp"

void sgemm(
    char transa, char transb,
    int  m,      int  n,      int  k,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C,       int ldc
) {
    if ((transa != 'N' && transa != 'n') ||
        (transb != 'T' && transb != 't'))
    {
        fprintf(stderr, "sgemm: unsupported transpose options\n");
        return;
    }
    for (int mm = 0; mm < m; ++mm) {
        for (int nn = 0; nn < n; ++nn) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) {
                acc += A[ mm + i*lda ]
                     * B[ nn + i*ldb ];
            }
            C[ mm + nn*ldc ] = beta * C[ mm + nn*ldc ]
                              + alpha * acc;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <A.txt> <BT.txt> <output_root>\n", argv[0]);
        exit(1);
    }

    const char *fileA   = argv[1];
    const char *fileBT  = argv[2];
    const char *outroot = argv[3];

    int m, k, n;
    std::vector<float> A, BT;

    if (!readColMajorMatrixFile(fileA,  m, k, A))  {
        fprintf(stderr, "Failed to load A from %s\n", fileA);
        exit(1);
    }
    if (!readColMajorMatrixFile(fileBT, n, k, BT)) {
        fprintf(stderr, "Failed to load Bᵀ from %s\n", fileBT);
        exit(1);
    }

    std::vector<float> C(m * n, 0.0f);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    sgemm(
        'N','T',
        m, n, k,
        1.0f,
        A.data(),  m,
        BT.data(), n,
        0.0f,
        C.data(),  m
    );

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed =
        (t1.tv_sec  - t0.tv_sec)
      + (t1.tv_nsec - t0.tv_nsec)*1e-9;

    printf("%s  Sequential time: %s%.6f s %s\n", BOLD, BLUE, elapsed, CLEAR);
    writeColMajorMatrixFile(outroot, m, n, C);
    printf("\n");

    char base[256];
    getOutputBase(outroot, base, sizeof(base));
    appendTiming(outroot, elapsed);

    return 0;
}
