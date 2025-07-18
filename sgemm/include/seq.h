#ifndef SEQ_H
#define SEQ_H

void basicSgemm(
    char transa, char transb,
    int  m,      int  n,      int  k,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C,       int ldc
);

void run_sequential(int argc, char *argv[]);

#endif // SEQ_H
