// cuda.h
#ifndef CUDA_H
#define CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

    double *nc_compute_new_cuda(int n, double a, double b, double x[]);

#ifdef __cplusplus
}
#endif

#endif // CUDA_H
