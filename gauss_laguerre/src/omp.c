#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "../include/common.h"
#include "../include/omp.h"

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

double *nc_compute_new_tasks(int n, double x_min, double x_max, double x[]) {
    double *w = (double *)malloc(n * sizeof(double));
#pragma omp parallel
    {
        double *d = (double *)malloc(n * sizeof(double));
#pragma omp for schedule(guided)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) d[j] = 0.0;
            d[i] = 1.0;

            for (int j = 2; j <= n; j++) {
                for (int k = j; k <= n; k++)
                    d[n + j - k - 1] = (d[n + j - k - 2] - d[n + j - k - 1]) /
                                       (x[n - k] - x[n + j - k - 1]);
            }

            for (int j = 1; j <= n - 1; j++) {
                for (int k = 1; k <= n - j; k++)
                    d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k];
            }

            double yvala = d[n - 1] / (double)(n);
            for (int j = n - 2; j >= 0; j--)
                yvala = yvala * x_min + d[j] / (double)(j + 1);
            yvala *= x_min;

            double yvalb = d[n - 1] / (double)(n);
            for (int j = n - 2; j >= 0; j--)
                yvalb = yvalb * x_max + d[j] / (double)(j + 1);
            yvalb *= x_max;

            w[i] = yvalb - yvala;
        }
        free(d);
    }
    return w;
}

int main(int argc, char *argv[]) {
    double a, b;
    int n;
    char out_prefix[256];

    if (argc >= 2) n = atoi(argv[1]); else { printf("Enter N: "); scanf("%d", &n); }
    if (argc >= 3) a = atof(argv[2]); else { printf("Enter A: "); scanf("%lf", &a); }
    if (argc >= 4) b = atof(argv[3]); else { printf("Enter B: "); scanf("%lf", &b); }
    if (argc >= 5) strncpy(out_prefix, argv[4], 255); else { printf("Enter root filename: "); scanf("%s", out_prefix); }
    out_prefix[255] = '\0';

    const char *basename = strrchr(out_prefix, '/');
    basename = (basename == NULL) ? out_prefix : basename + 1;

    // Load sequential reference data
    char xfile[300], wfile[300], tfile[300];
    snprintf(xfile, sizeof(xfile), "output/seq/%s_x.txt", basename);
    snprintf(wfile, sizeof(wfile), "output/seq/%s_w.txt", basename);
    snprintf(tfile, sizeof(tfile), "output/seq/%s_time.txt", basename);

    double *x = (double *)malloc(n * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));
    double time_seq;

    FILE *fx = fopen(xfile, "r");
    FILE *fw = fopen(wfile, "r");
    FILE *ft = fopen(tfile, "r");
    if (!fx || !fw || !ft) {
        fprintf(stderr, "Failed to load precomputed sequential files.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) fscanf(fx, "%lf", &x[i]);
    for (int i = 0; i < n; i++) fscanf(fw, "%lf", &w[i]);
    fscanf(ft, "%lf", &time_seq);
    fclose(fx); fclose(fw); fclose(ft);

    double *r = malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

    omp_set_num_threads(omp_get_max_threads());
    double t0 = omp_get_wtime();
    double *x2 = ccn_compute_points_new(n);
    double *w2 = nc_compute_new_tasks(n, -1.0, +1.0, x2);
    rescale(a, b, n, x2, w2);
    double t1 = omp_get_wtime();
    double par_time = t1 - t0;

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(w2[i] - w[i]) > 1e-6 || fabs(x2[i] - x[i]) > 1e-12) {
            ok = 0;
            break;
        }
    }

    printf("\n%s  Test %s%s\n", BOLD, ok ? GREEN "PASSED" : RED "FAILED", CLEAR);
    printf("  %sSequential time: %s%.3fs %s\n", BOLD, BLUE, time_seq, CLEAR);
    printf("  %sParallel time:   %s%.3fs %s\n", BOLD, BLUE, par_time, CLEAR);
    printf("  %sSpeedup:         %s%.2fx %s\n", BOLD, BLUE, time_seq / par_time, CLEAR);
    rule_write(n, out_prefix, x2, w2, r);
    printf("\n");

    char time_out[300];
    snprintf(time_out, sizeof(time_out), "%s_time.txt", out_prefix);
    FILE *fout = fopen(time_out, "w");
    if (fout) {
        fprintf(fout, "%.6f\n", par_time);
        fclose(fout);
    } else {
        perror("fopen for OMP time");
    }

    free(r);
    free(x);  free(w);
    free(x2); free(w2);
    return 0;
}
