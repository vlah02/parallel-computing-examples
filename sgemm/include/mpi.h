#ifndef MPI_SGEMM_H
#define MPI_SGEMM_H

void parallelSgemm(char transa, char transb, int m, int n, int k, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc, int rank, int size);

#endif // MPI_SGEMM_H
