#ifndef MPI_WRAPPER_H
#define MPI_WRAPPER_H

double *nc_compute_new_parallel(int n, double x_min, double x_max, double x[], int rank, int size);
void run_mpi(int argc, char *argv[]);

#endif
