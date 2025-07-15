#include "../include/common.h"
#include "../include/omp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

double *nc_compute_new_tasks(int n, double x_min, double x_max, double x[]) {
    double *d;
    double *w = (double *)malloc(n * sizeof(double));
    int j, k;
    double yvala, yvalb;

#pragma omp parallel private(j, k, yvala, yvalb, d)
    {
        d = (double *)malloc(n * sizeof(double));
#pragma omp for
        for (int i = 0; i < n; i++) {
#pragma omp task shared(d, w)
            {
                for (int j = 0; j < n; j++) d[j] = 0.0;
                d[i] = 1.0;

                for (int j = 2; j <= n; j++) {
                    for (int k = j; k <= n; k++)
                        d[n + j - k - 1] = (d[n + j - k - 1 - 1] - d[n + j - k - 1]) /
                                           (x[n + 1 - k - 1] - x[n + j - k - 1]);
                }

                for (int j = 1; j <= n - 1; j++) {
                    for (int k = 1; k <= n - j; k++)
                        d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k];
                }

                double yvala = d[n - 1] / (double)(n);
                for (int j = n - 2; 0 <= j; j--)
                    yvala = yvala * x_min + d[j] / (double)(j + 1);
                yvala = yvala * x_min;

                double yvalb = d[n - 1] / (double)(n);
                for (int j = n - 2; 0 <= j; j--)
                    yvalb = yvalb * x_max + d[j] / (double)(j + 1);
                yvalb = yvalb * x_max;

                w[i] = yvalb - yvala;
            }
        }
#pragma omp taskwait
        free(d);
    }
    return w;
}

double *nc_compute_new_parallel(int n, double x_min, double x_max, double x[]) {
    double *d;
    int i;
    int j;
    int k;
    double *w;
    double yvala;
    double yvalb;

    w = (double *)malloc(n * sizeof(double));

#pragma omp parallel private(j, k, yvala, yvalb, d)
    {
        d = (double *)malloc(n * sizeof(double));

#pragma omp for schedule(guided)
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                d[j] = 0.0;
            }
            d[i] = 1.0;

            for (j = 2; j <= n; j++) {
                for (k = j; k <= n; k++) {
                    d[n + j - k - 1] = (d[n + j - k - 1 - 1] - d[n + j - k - 1]) /
                                       (x[n + 1 - k - 1] - x[n + j - k - 1]);
                }
            }

            for (j = 1; j <= n - 1; j++) {
                for (k = 1; k <= n - j; k++) {
                    d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k];
                }
            }

            yvala = d[n - 1] / (double)(n);
            for (j = n - 2; 0 <= j; j--) {
                yvala = yvala * x_min + d[j] / (double)(j + 1);
            }
            yvala = yvala * x_min;

            yvalb = d[n - 1] / (double)(n);
            for (j = n - 2; 0 <= j; j--) {
                yvalb = yvalb * x_max + d[j] / (double)(j + 1);
            }
            yvalb = yvalb * x_max;

            w[i] = yvalb - yvala;
        }

        free(d);
    }

    return w;
}

int main(int argc, char *argv[]) {
    double a, b;
    int    n;
    char   filename[256];

    // --- banner ---
    printf("\nCCN_RULE OpenMP variant\n\n");

    // --- parse arguments or prompt ---
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

    printf("  N = %d,  A = %g,  B = %g,  root = \"%s\"\n\n",
           n, a, b, filename);

    // region endpoints
    double *r = malloc(2 * sizeof(double));
    if (!r) { perror("malloc"); exit(1); }
    r[0] = a; r[1] = b;

    // --- sequential run ---
    double t0 = omp_get_wtime();
      double *x  = ccn_compute_points_new(n);
      double *w  = nc_compute_new(n, -1.0, +1.0, x);
      rescale(a, b, n, x, w);
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;
    printf("Sequential time: %f seconds\n", seq_time);

    // --- parallel run ---
    omp_set_num_threads( omp_get_max_threads() );
    double t2 = omp_get_wtime();
      double *x2 = ccn_compute_points_new(n);
      //double *w2 = nc_compute_new_parallel(n, -1.0, +1.0, x2);
      double *w2 = nc_compute_new_tasks(n, -1.0, +1.0, x2);
      rescale(a, b, n, x2, w2);
    double t3 = omp_get_wtime();
    double par_time = t3 - t2;
    printf("Parallel time:   %f seconds\n", par_time);

    // speedup
    printf("Speedup:         %f×\n\n", seq_time / par_time);

    // --- correctness test ---
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(w2[i] - w[i]) > 1e-6
         || fabs(x2[i] - x[i]) > 1e-12) {
            ok = 0;
            break;
        }
    }
    if (ok) printf("Test PASSED\n\n");
    else    printf("Test FAILED\n\n");

    // --- write outputs ---
    rule_write(n, filename, x2, w2, r);

    // cleanup
    free(r);
    free(x);  free(w);
    free(x2); free(w2);

    return 0;
}
