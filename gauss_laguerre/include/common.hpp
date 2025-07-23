#pragma once

#include <stdbool.h>

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

double *ccn_compute_points_new(int n);
int     i4_min(int i1, int i2);
void    rescale(double a, double b, int n, double x[], double w[]);
void    r8mat_write(char *output_filename, int m, int n, double table[]);
void    rule_write(int order, char *filename, double x[], double w[], double r[]);

void getOutputBase(const char *root, char *base, size_t len);

bool loadSequentialResult(const char *base, int n, const char *kind, double *arr);
bool loadSequentialTiming(const char *base, double *cpu_sec);

bool compareResults(const double *a, const double *b, int n, double tol);

bool appendTiming(const char *root, double time_sec);