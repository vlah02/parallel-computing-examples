#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/common.hpp"

double *nc_compute_new(int n, double x_min, double x_max, double x[]) {
    double *d = (double*)malloc(n * sizeof(double));
    double *w = (double*)malloc(n * sizeof(double));
    if (!d || !w) { perror("malloc"); exit(1); }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) d[j] = 0.0;
        d[i] = 1.0;

        for (int j = 2; j <= n; j++) {
            for (int k = j; k <= n; k++) {
                int idx = n + j - k - 1;
                d[idx] = (d[idx-1] - d[idx]) / (x[n+1-k-1] - x[idx]);
            }
        }
        for (int j = 1; j <= n-1; j++) {
            for (int k = 1; k <= n-j; k++) {
                d[n-k-1] -= x[n-k-j] * d[n-k];
            }
        }

        double yvala = d[n-1]/n, yvalb = yvala;
        for (int j = n-2; j >= 0; j--) {
            yvala = yvala * x_min + d[j]/(j+1);
            yvalb = yvalb * x_max + d[j]/(j+1);
        }
        w[i] = yvalb * x_max - yvala * x_min;
    }

    free(d);
    return w;
}

int main(int argc, char *argv[]) {
    double a, b;
    int    n;
    char   filename[256];

    if (argc >= 2) {
        n = atoi(argv[1]);
    } else {
        printf("Enter the value of N: ");
        scanf("%d", &n);
    }

    if (argc >= 3) {
        a = atof(argv[2]);
    } else {
        printf("Enter the left endpoint A: ");
        scanf("%lf", &a);
    }

    if (argc >= 4) {
        b = atof(argv[3]);
    } else {
        printf("Enter the right endpoint B: ");
        scanf("%lf", &b);
    }

    if (argc >= 5) {
        strncpy(filename, argv[4], sizeof(filename)-1);
        filename[sizeof(filename)-1] = '\0';
    } else {
        printf("Enter the root filename: ");
        scanf("%s", filename);
    }

    double *r = (double*)malloc(2 * sizeof(double));
    if (!r) { perror("malloc"); exit(1); }
    r[0] = a; r[1] = b;

    clock_t t_start = clock();

    double *x = ccn_compute_points_new(n);
    double *w = nc_compute_new(n, -1.0, +1.0, x);
    rescale(a, b, n, x, w);

    clock_t t_end = clock();
    double elapsed = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("\n  Sequential time: %f seconds\n", elapsed);
    rule_write(n, filename, x, w, r);
    printf("\n");

	char time_filename[300];
	snprintf(time_filename, sizeof(time_filename), "%s_time.txt", filename);

	FILE *f = fopen(time_filename, "a");
	if (!f) {
    	perror("fopen");
	} else {
    	fprintf(f, "%.6f\n", elapsed);
    	fclose(f);
	}

    free(r);
    free(x);
    free(w);

    return 0;
}
