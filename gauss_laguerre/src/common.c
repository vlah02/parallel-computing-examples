#include "../include/seq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//
//  Write an M×N matrix of doubles to a text file.
//
void r8mat_write(char *output_filename, int m, int n, double table[]) {
    FILE *output = fopen(output_filename, "wt");
    if (!output) {
        fprintf(stderr, "R8MAT_WRITE - Fatal error: could not open \"%s\".\n",
                output_filename);
        exit(1);
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            fprintf(output, "  %24.16g", table[i + j * m]);
        }
        fprintf(output, "\n");
    }
    fclose(output);
}

//
//  Scale abscissas and weights from [-1,1]→[a,b].
//
void rescale(double a, double b, int n, double x[], double w[]) {
    for (int i = 0; i < n; i++) {
        x[i] = ((a + b) + (b - a) * x[i]) / 2.0;
    }
    for (int i = 0; i < n; i++) {
        w[i] = (b - a) * w[i] / 2.0;
    }
}

//
//  Write the three output files for quadrature.
//
void rule_write(int order, char *filename, double x[], double w[], double r[]) {
    char fn_r[80], fn_w[80], fn_x[80];
    strcpy(fn_r, filename); strcat(fn_r, "_r.txt");
    strcpy(fn_w, filename); strcat(fn_w, "_w.txt");
    strcpy(fn_x, filename); strcat(fn_x, "_x.txt");

    printf("  Creating files \"%s\", \"%s\", \"%s\"\n",
           fn_w, fn_x, fn_r);

    r8mat_write(fn_w, 1, order, w);
    r8mat_write(fn_x, 1, order, x);
    r8mat_write(fn_r, 1, 2, r);
}

//
//  Minimum of two integers.
//
int i4_min(int i1, int i2) {
    return (i1 < i2 ? i1 : i2);
}

//
//  Compute the nested Clenshaw–Curtis abscissas.
//
double *ccn_compute_points_new(int n) {
    const double r8_pi = 3.141592653589793;
    double *x = malloc(n * sizeof(double));
    if (!x) { perror("malloc"); exit(1); }

    if (n >= 1) x[0] = 0.5;
    if (n >= 2) x[1] = 1.0;
    if (n >= 3) x[2] = 0.0;

    int m = 3, d = 2;
    while (m < n) {
        int tu = d+1, td = d-1;
        int k = i4_min(d, n-m);

        for (int i = 1; i <= k; i++) {
            x[m + i - 1] = (i % 2
                ? (tu / 2.0) / k
                : (td / 2.0) / k
            );
            if (i % 2) tu += 2; else td -= 2;
        }
        m += k;
        d *= 2;
    }

    for (int i = 0; i < n; i++) {
        x[i] = cos(x[i] * r8_pi);
    }
    if (n >= 1) x[0] = 0.0;
    if (n >= 2) x[1] = -1.0;
    if (n >= 3) x[2] = +1.0;

    return x;
}

//
//  Compute weights sequentially.
//
double *nc_compute_new(int n, double x_min, double x_max, double x[]) {
    double *d = malloc(n * sizeof(double));
    double *w = malloc(n * sizeof(double));
    if (!d || !w) { perror("malloc"); exit(1); }

    for (int i = 0; i < n; i++) {
        // build divided‐difference table d[]
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

        // integrate basis to get weight[i]
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
