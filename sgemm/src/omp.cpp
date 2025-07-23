#include "../include/common.hpp"
#include <omp.h>

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
        std::cerr << "sgemm: unsupported transpose options\n";
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
        std::cerr << "Usage: " << argv[0] << " A.txt BT.txt output_root" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string fnameA = argv[1], fnameBT = argv[2], out_root = argv[3];

    std::vector<float> matA, matBT;
    int m, k1, n, k2;

    if (!readColMajorMatrixFile(fnameA,  m,  k1, matA) ||
        !readColMajorMatrixFile(fnameBT, n,  k2, matBT) ||
         k1 != k2)
    {
        std::cerr << "Error reading inputs or mismatched dims" << std::endl;
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

    std::string base = getOutputBase(out_root);

    double cpu_sec = 0;
    if (!loadSequentialTiming(base, cpu_sec)) {
        std::cerr << "Error: cannot load sequential timing for \"" << base << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<float> C_seq;
    int rm, rn;
    if (!loadSequentialResult(base, rm, rn, C_seq) || rm != m || rn != n) {
        std::cerr << "Error: cannot load sequential result or size mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }

    bool ok = compareResults(C_seq, C_omp);

    std::cout << BOLD << "  Test " << (ok ? (std::string(GREEN) + "PASSED") : (std::string(RED) + "FAILED")) << CLEAR << std::endl;
    std::cout << BOLD << "  Sequential time: " << BLUE << cpu_sec << " s " << CLEAR << std::endl;
    std::cout << BOLD << "  Parallel time:   " << BLUE << omp_sec << " s " << CLEAR << std::endl;
    std::cout << BOLD << "  Speedup:         " << BLUE << (cpu_sec / omp_sec) << "x " << CLEAR << std::endl;
    std::cout << std::endl;

    writeColMajorMatrixFile(out_root, m, n, C_omp);
    appendTiming(out_root, omp_sec);

    return 0;
}
