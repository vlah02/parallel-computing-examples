#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/common.hpp"

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

void rescale(double a, double b, int n, double x[], double w[]) {
    for (int i = 0; i < n; i++) {
        x[i] = ((a + b) + (b - a) * x[i]) / 2.0;
    }
    for (int i = 0; i < n; i++) {
        w[i] = (b - a) * w[i] / 2.0;
    }
}

void rule_write(int order, char *filename, double x[], double w[], double r[]) {
    char fn_r[80], fn_w[80], fn_x[80];
    strcpy(fn_r, filename); strcat(fn_r, "_r.txt");
    strcpy(fn_w, filename); strcat(fn_w, "_w.txt");
    strcpy(fn_x, filename); strcat(fn_x, "_x.txt");

//    printf("  Creating files \"%s\", \"%s\", \"%s\"\n", fn_w, fn_x, fn_r);

    r8mat_write(fn_w, 1, order, w);
    r8mat_write(fn_x, 1, order, x);
    r8mat_write(fn_r, 1, 2, r);
}

int i4_min(int i1, int i2) {
    return (i1 < i2 ? i1 : i2);
}

double *ccn_compute_points_new(int n) {
    const double r8_pi = 3.141592653589793;
    double *x = (double*)malloc(n * sizeof(double));
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

void getOutputBase(const char *root, char *base, size_t len) {
    const char *slash = strrchr(root, '/');
    const char *name = slash ? slash+1 : root;
    strncpy(base, name, len-1);
    base[len-1] = '\0';
   	char *dot = strchr(base, '.');
    if (dot) *dot = '\0';
}

bool loadSequentialResult(const char *base, int n, const char *kind, double *arr) {
    char filename[512];
    snprintf(filename, sizeof(filename), "output/seq/%s_%s.txt", base, kind);
    FILE *f = fopen(filename, "r");
    if (!f) return false;

    for (int i = 0; i < n; ++i) {
        if (fscanf(f, "%lf", &arr[i]) != 1) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

bool loadSequentialTiming(const char *base, double *cpu_sec) {
    char seqtime[512];
    snprintf(seqtime, sizeof(seqtime), "output/seq/%s_time.txt", base);
    FILE *fs = fopen(seqtime, "r");
    if (!fs) return false;
    double sum = 0, tv;
    int cnt = 0;
    while (fscanf(fs, "%lf", &tv) == 1) {
        sum += tv;
        cnt++;
    }
    fclose(fs);
    if (cnt == 0) return false;
    *cpu_sec = sum / cnt;
    return true;
}

bool compareResults(const double *a, const double *b, int n, double tol) {
    for (int i = 0; i < n; ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

bool appendTiming(const char *root, double time_sec) {
    char tf[256];
    snprintf(tf, sizeof(tf), "%s_time.txt", root);
    FILE *f = fopen(tf, "a");
    if (!f) {
        perror("fopen timefile");
        return false;
    }
    fprintf(f, "%.6f\n", time_sec);
    fclose(f);
    return true;
}