#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "../include/common.h"
#include "../include/mpi.h"

#define MASTER 0

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

double *nc_compute_new_parallel(int n, double x_min, double x_max, double x[], int rank, int size) {
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

    double *ww = NULL;
    if (rank == MASTER) ww = (double *)malloc(n * sizeof(double));

    MPI_Gather(w, local_count, MPI_DOUBLE, ww, local_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    free(w);
    return ww;
}

void run_mpi(int argc, char *argv[]) {
    double a, b, *r, *w, *ww, *x, *xx;
    char out_prefix[256];
    int n, rank, size;
    double seq_time;
    double t1_par, t2_par, par_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        if (argc >= 2) n = atoi(argv[1]);
        else { printf("Enter N: "); scanf("%d", &n); }

        if (argc >= 3) a = atof(argv[2]);
        else { printf("Enter A: "); scanf("%lf", &a); }

        if (argc >= 4) b = atof(argv[3]);
        else { printf("Enter B: "); scanf("%lf", &b); }

        if (argc >= 5) {
            strncpy(out_prefix, argv[4], sizeof(out_prefix) - 1);
            out_prefix[sizeof(out_prefix) - 1] = '\0';
        } else {
            printf("Enter root filename: ");
            scanf("%s", out_prefix);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(out_prefix, 256, MPI_CHAR, MASTER, MPI_COMM_WORLD);

    const char *basename = strrchr(out_prefix, '/');
    basename = (basename == NULL) ? out_prefix : basename + 1;

    char xfile[300], wfile[300], tfile[300];
    snprintf(xfile, sizeof(xfile), "output/seq/%s_x.txt", basename);
    snprintf(wfile, sizeof(wfile), "output/seq/%s_w.txt", basename);
    snprintf(tfile, sizeof(tfile), "output/seq/%s_time.txt", basename);

    r = malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

    x = malloc(n * sizeof(double));
    w = malloc(n * sizeof(double));
    FILE *fx = fopen(xfile, "r");
    FILE *fw = fopen(wfile, "r");
    FILE *ft = fopen(tfile, "r");

    if (!fx || !fw || !ft) {
        if (rank == MASTER) fprintf(stderr, "Failed to open sequential files.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) fscanf(fx, "%lf", &x[i]);
    for (int i = 0; i < n; i++) fscanf(fw, "%lf", &w[i]);

    seq_time = 0.0;
    int time_count = 0;
    double tval;
    while (fscanf(ft, "%lf", &tval) == 1) {
        seq_time += tval;
        time_count++;
    }
    if (time_count == 0) {
        if (rank == MASTER) fprintf(stderr, "No times found in %s\n", tfile);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    seq_time /= time_count;

    fclose(fx); fclose(fw); fclose(ft);

    t1_par = MPI_Wtime();

    if (rank == MASTER) xx = ccn_compute_points_new(n);
    else xx = malloc(n * sizeof(double));

    MPI_Bcast(xx, n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    ww = nc_compute_new_parallel(n, -1.0, +1.0, xx, rank, size);

    if (rank == MASTER) {
        rescale(a, b, n, xx, ww);
        t2_par = MPI_Wtime();
        par_time = t2_par - t1_par;

        int ok = 1;
        for (int i = 0; i < n; i++) {
            if (fabs(x[i] - xx[i]) > 1e-6 || fabs(w[i] - ww[i]) > 1e-6) {
                ok = 0;
                break;
            }
        }

        printf("\n%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
        printf("%s  Sequential time: %s%.3fs %s\n", BOLD, BLUE, seq_time, CLEAR);
        printf("%s  Parallel time:   %s%.3fs %s\n", BOLD, BLUE, par_time, CLEAR);
        printf("%s  Speedup:         %s%.2fx %s\n", BOLD, BLUE, seq_time / par_time, CLEAR);
        rule_write(n, out_prefix, xx, ww, r);
        printf("\n");

        char time_out[300];
        snprintf(time_out, sizeof(time_out), "%s_time.txt", out_prefix);
        FILE *fout = fopen(time_out, "a");
        if (fout) {
            fprintf(fout, "%.6f\n", par_time);
            fclose(fout);
        } else {
            perror("fopen for MPI time");
        }
    }

    free(r); free(x); free(w); free(xx); free(ww);
    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    run_mpi(argc, argv);
    return 0;
}
