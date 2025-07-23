#include "../include/common.hpp"
#include <chrono>

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
        std::cerr << "sgemm: unsupported transpose options\n";
        return;
    }
    for (int mm = 0; mm < m; ++mm) {
        for (int nn = 0; nn < n; ++nn) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) {
                acc += A[mm + i*lda] * B[nn + i*ldb];
            }
            C[mm + nn*ldc] = beta * C[mm + nn*ldc] + alpha * acc;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <A.txt> <BT.txt> <output_root>\n";
        return 1;
    }

    std::string fileA   = argv[1];
    std::string fileBT  = argv[2];
    std::string outroot = argv[3];

    int m, k, n, k2;
    std::vector<float> A, BT;

    if (!readColMajorMatrixFile(fileA,  m,  k,  A)) {
        std::cerr << "Failed to load A from " << fileA << std::endl;
        return 1;
    }
    if (!readColMajorMatrixFile(fileBT, n, k2, BT)) {
        std::cerr << "Failed to load Bᵀ from " << fileBT << std::endl;
        return 1;
    }
    if (k != k2) {
        std::cerr << "Dimension mismatch: k=" << k << " vs k2=" << k2 << std::endl;
        return 1;
    }

    std::vector<float> C(m * n, 0.0f);

    auto t0 = std::chrono::steady_clock::now();

    sgemm(
        'N','T',
        m, n, k,
        1.0f,
        A.data(),  m,
        BT.data(), n,
        0.0f,
        C.data(),  m
    );

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << BOLD << "  Sequential time: " << BLUE << elapsed << " s " << CLEAR << std::endl;
    writeColMajorMatrixFile(outroot, m, n, C);

    std::string base = getOutputBase(outroot);
    appendTiming(outroot, elapsed);

    std::cout << std::endl;
    return 0;
}
