#ifndef OMP_H
#define OMP_H

/* compute weights in parallel via OpenMP */
double *nc_compute_new_parallel(int n, double x_min, double x_max, double x[]);
double *nc_compute_new_tasks(int n, double x_min, double x_max, double x[]);

#endif // OMP_H
