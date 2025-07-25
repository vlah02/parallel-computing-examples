#include "../include/common.h"
#include <omp.h>
#include <fftw3.h>

double* nc_compute_new_fft(int n, double x_min, double x_max, double x[]) {
    double* w = (double*)malloc(n * sizeof(double));
    double* c = (double*)fftw_malloc(sizeof(double) * n);
    double* d = (double*)fftw_malloc(sizeof(double) * n);

    for (int k = 0; k < n; ++k) c[k] = 1.0;
    c[0] = c[n-1] = 0.5;

    fftw_plan p = fftw_plan_r2r_1d(n, c, d, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    double scale = 2.0 / (n-1) * ((x_max - x_min) / 2.0);

#pragma omp parallel for
    for (int k = 0; k < n; ++k)
        w[k] = d[k] * scale;

    fftw_free(c);
    fftw_free(d);
    return w;
}

double *nc_compute_new(int n, double x_min, double x_max, double x[]) {
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

    char base[256];
    getOutputBase(out_prefix, base, sizeof(base));

	double *x_ref = (double *)malloc(n * sizeof(double));
	double *w_ref = (double *)malloc(n * sizeof(double));
	if (!loadSequentialResult(base, n, "x", x_ref) || !loadSequentialResult(base, n, "w", w_ref)) {
    	fprintf(stderr, "Failed to load precomputed x or w files.\n");
    	exit(EXIT_FAILURE);
	}

    double seq_time = 0.0;
    if (!loadSequentialTiming(base, &seq_time)) {
        fprintf(stderr, "No times found in sequential timing file for %s\n", base);
        exit(EXIT_FAILURE);
    }

    double *r = (double *)malloc(2 * sizeof(double));
    r[0] = a; r[1] = b;

    omp_set_num_threads(omp_get_max_threads());
    double t0 = omp_get_wtime();
    double *x_calc = ccn_compute_points_new(n);
    double *w_calc = nc_compute_new(n, -1.0, +1.0, x_calc);
    rescale(a, b, n, x_calc, w_calc);
    double t1 = omp_get_wtime();
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

    free(r);
	free(x_ref); free(w_ref);
	free(x_calc); free(w_calc);
	return 0;
}
