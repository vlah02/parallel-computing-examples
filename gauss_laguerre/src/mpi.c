#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "../include/common.h"
#include "../include/mpi.h"    // your wrapper

#define MASTER 0

double *nc_compute_new_parallel(int n, double x_min, double x_max, double x[],
                                int rank, int size) {
    double *d;
    int i, j, k;
    double *w;
    double yvala, yvalb;

    int chunk_size = n / size;
    int local_count = chunk_size;
    int starting_index = rank * chunk_size;
    int end_index = starting_index + chunk_size;

    d = (double *)malloc(n * sizeof(double));
    w = (double *)malloc(local_count * sizeof(double));

    for (i = starting_index; i < end_index; i++) {
        for (j = 0; j < n; j++) {
            d[j] = 0.0;
        }
        d[i] = 1.0;

        for (j = 2; j <= n; j++) {
            for (k = j; k <= n; k++) {
                d[n + j - k - 1] = (d[n + j - k - 2] - d[n + j - k - 1]) /
                                   (x[n - k] - x[n + j - k - 1]);
            }
        }

        for (j = 1; j <= n - 1; j++) {
            for (k = 1; k <= n - j; k++) {
                d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k];
            }
        }

        yvala = d[n - 1] / (double)(n);
        for (j = n - 2; j >= 0; j--) {
            yvala = yvala * x_min + d[j] / (double)(j + 1);
        }
        yvala *= x_min;

        yvalb = d[n - 1] / (double)(n);
        for (j = n - 2; j >= 0; j--) {
            yvalb = yvalb * x_max + d[j] / (double)(j + 1);
        }
        yvalb *= x_max;

        w[i - starting_index] = yvalb - yvala;
    }

    free(d);

    double *ww = NULL;
    if (rank == MASTER) {
        ww = (double *)malloc(n * sizeof(double));
    }

    MPI_Gather(w, local_count, MPI_DOUBLE, ww, local_count, MPI_DOUBLE, MASTER,
               MPI_COMM_WORLD);

    free(w);

    return ww;
}

void run_mpi(int argc, char *argv[]) {
    double a, b;
    char filename[255];
    int n, rank, size;
    double *r, *w, *ww, *x, *xx;
    double x_min, x_max;
    double t1_seq, t2_seq, seq_time;
    double t1_par, t2_par, par_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        printf("\nCCN_RULE MPI variant\n\n");
        printf("  Compute nested Clenshaw–Curtis quadrature of order N\n\n");

        if (argc >= 2)      n = atoi(argv[1]);
        else { printf("Enter N: "); scanf("%d", &n); }

        if (argc >= 3)      a = atof(argv[2]);
        else { printf("Enter A: "); scanf("%lf", &a); }

        if (argc >= 4)      b = atof(argv[3]);
        else { printf("Enter B: "); scanf("%lf", &b); }

        if (argc >= 5) {
            strncpy(filename, argv[4], sizeof(filename)-1);
            filename[sizeof(filename)-1] = '\0';
        } else {
            printf("Enter root filename: ");
            scanf("%s", filename);
        }

        printf("\n  N = %d   A = %g   B = %g   OUTPUT ROOT = \"%s\"\n\n",
               n, a, b, filename);
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&n, 1, MPI_INT,    MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // region endpoints array
    r = malloc(2 * sizeof(double));
    r[0] = a;  r[1] = b;

    // --- sequential on MASTER only ---
    if (rank == MASTER) {
        t1_seq = MPI_Wtime();
        x = ccn_compute_points_new(n);
        x_min = -1.0;  x_max = +1.0;
        w = nc_compute_new(n, x_min, x_max, x);
        rescale(a, b, n, x, w);
        t2_seq = MPI_Wtime();
        seq_time = t2_seq - t1_seq;
    }

    // --- parallel piece ---
    MPI_Barrier(MPI_COMM_WORLD);
    t1_par = MPI_Wtime();

    // build or receive the abscissas
    if (rank == MASTER) {
        xx = ccn_compute_points_new(n);
    } else {
        xx = malloc(n * sizeof(double));
    }
    MPI_Bcast(xx, n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // compute weights in parallel
    ww = nc_compute_new_parallel(n, -1.0, +1.0, xx, rank, size);

    // rescale on MASTER
    if (rank == MASTER) {
        rescale(a, b, n, xx, ww);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2_par = MPI_Wtime();
    par_time = t2_par - t1_par;

    // MASTER prints results/tests
    if (rank == MASTER) {
        int ok = 1;
        for (int i = 0; i < n; i++)
            if (fabs(x[i] - xx[i]) > 1e-6) { ok = 0; break; }

        printf("%s\n", ok ? "Test PASSED" : "Test FAILED");
        rule_write(n, filename, xx, ww, r);
        printf(" Sequential time: %f s\n", seq_time);
        printf(" Parallel time:   %f s\n", par_time);
        if (par_time > 0.0)
            printf(" Speedup:         %.2f×\n", seq_time / par_time);
    }

    // cleanup
    free(r);
    if (rank == MASTER) {
        free(x);
        free(w);
    }
    free(xx);
    free(ww);

    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    run_mpi(argc, argv);
    return 0;
}
