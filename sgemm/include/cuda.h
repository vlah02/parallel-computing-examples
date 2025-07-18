#ifndef CUDA_SGEMM_H
#define CUDA_SGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

    void cudaSgemm(char transa, char transb, int m, int n, int k, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc, float *time);

#ifdef __cplusplus
}
#endif

#endif
