#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

#ifdef __cplusplus
extern "C" {
#endif

    double *ccn_compute_points_new(int n);
    int     i4_min(int a, int b);
    void    rescale(double a, double b, int n, double x[], double w[]);
    void    r8mat_write(const char *output_filename, int m, int n, const double *table);
    void    rule_write(int order, const char *filename, const double *x, const double *w, const double *r);

    void    get_output_base(const char *root, char *base, size_t len);

    bool    load_sequential_result(const char *base, int n, const char *kind, double *arr);
    bool    load_sequential_timing(const char *base, double *cpu_sec);

    bool    compare_results(const double *a, const double *b, int n, double tol);

    bool    append_timing(const char *root, double time_sec);

#ifdef __cplusplus
}
#endif
