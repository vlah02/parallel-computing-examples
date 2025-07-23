#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "../include/common.hpp"

#define MASTER 0

double *nc_compute_new(int n, double x_min, double x_max, double x[], int rank, int size) {
    int chunk_size = n / size;
    int local_count = chunk_size;
    int starting_index = rank * chunk_size;
    int end_index = starting_index + chunk_size;

    double *d = (double *)malloc(n * sizeof(double));
    double *w = (double *)malloc(local_count * sizeof(double));
    double yvala, yvalb;

    for (int i = starting_index; i < end_index; i++) {
        for (int j = 0; j < n; j++) d[j] = 0.0;
        d[i] = 1.0;

        for (int j = 2; j <= n; j++) {
            for (int k = j; k <= n; k++) {
                d[n + j - k - 1] = (d[n + j - k - 2] - d[n + j - k - 1]) /
                                   (x[n - k] - x[n + j - k - 1]);
            }
        }

        for (int j = 1; j <= n - 1; j++) {
            for (int k = 1; k <= n - j; k++) {
                d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k];
            }
        }

        yvala = d[n - 1] / n;
        for (int j = n - 2; j >= 0; j--) yvala = yvala * x_min + d[j] / (j + 1);
        yvala *= x_min;

        yvalb = d[n - 1] / n;
        for (int j = n - 2; j >= 0; j--) yvalb = yvalb * x_max + d[j] / (j + 1);
        yvalb *= x_max;

        w[i - starting_index] = yvalb - yvala;
    }

    free(d);

    double *w_calc = NULL;
    if (rank == MASTER) w_calc = (double *)malloc(n * sizeof(double));

    MPI_Gather(w, local_count, MPI_DOUBLE, w_calc, local_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    free(w);
    return w_calc;
}

int main(int argc, char *argv[]) {
    double a, b;
	int n;
	char out_prefix[256];
	int rank, size;

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == MASTER) {
        if (argc >= 2) n = atoi(argv[1]); else { printf("Enter N: "); scanf("%d", &n); }
        if (argc >= 3) a = atof(argv[2]); else { printf("Enter A: "); scanf("%lf", &a); }
        if (argc >= 4) b = atof(argv[3]); else { printf("Enter B: "); scanf("%lf", &b); }
    	if (argc >= 5) strncpy(out_prefix, argv[4], 255); else { printf("Enter root filename: "); scanf("%s", out_prefix); }
        out_prefix[255] = '\0';
    }

    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(out_prefix, 256, MPI_CHAR, MASTER, MPI_COMM_WORLD);

    char base[256];
    getOutputBase(out_prefix, base, sizeof(base));

    double *x_ref = (double *)malloc(n * sizeof(double));
	double *w_ref = (double *)malloc(n * sizeof(double));
    if (rank == MASTER) {
		if (!loadSequentialResult(base, n, "x", x_ref) || !loadSequentialResult(base, n, "w", w_ref)) {
    		fprintf(stderr, "Failed to load precomputed x or w files.\n");
    		exit(EXIT_FAILURE);
		}
    }

    double seq_time = 0.0;
    if (!loadSequentialTiming(base, &seq_time)) {
        if (rank == MASTER) fprintf(stderr, "No times found in sequential timing file for %s\n", base);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
	
	double *r = (double *)malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

	double t0 = MPI_Wtime();
	double* x_calc;
    if (rank == MASTER) x_calc = ccn_compute_points_new(n);
    else x_calc = (double *)malloc(n * sizeof(double));
    MPI_Bcast(x_calc, n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    double* w_calc = nc_compute_new(n, -1.0, +1.0, x_calc, rank, size);

    if (rank == MASTER) {
        rescale(a, b, n, x_calc, w_calc);
        double t1 = MPI_Wtime();
        double par_time = t1 - t0;

		int ok = compareResults(x_ref, x_calc, n, 1e-6) && compareResults(w_ref, w_calc, n, 1e-6);

		printf("\n");
        printf("  %sTest %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
        printf("  %sSequential time: %s%.6fs %s\n", BOLD, BLUE, seq_time, CLEAR);
        printf("  %sParallel time:   %s%.6fs %s\n", BOLD, BLUE, par_time, CLEAR);
        printf("  %sSpeedup:         %s%.3fx %s\n", BOLD, BLUE, seq_time / par_time, CLEAR);
		printf("\n");
        rule_write(n, out_prefix, x_calc, w_calc, r);
		appendTiming(out_prefix, par_time);
    }

	free(r);
	free(x_ref); free(w_ref);
	free(x_calc); free(w_calc);
    MPI_Finalize();
    return 0;
}
