// common.h
#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

    double *ccn_compute_points_new(int n);
    int     i4_min(int i1, int i2);
    double *nc_compute_new(int n, double x_min, double x_max, double x[]);
    void    rescale(double a, double b, int n, double x[], double w[]);
    void    r8mat_write(char *output_filename, int m, int n, double table[]);
    void    rule_write(int order, char *filename, double x[], double w[], double r[]);

#ifdef __cplusplus
}
#endif

#endif // COMMON_H
