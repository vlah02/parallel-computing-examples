#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mpi.h"

#define MASTER 0

int main(int argc, char *argv[]);
double *ccn_compute_points_new(int n);
int i4_min(int i1, int i2);
double *nc_compute_new(int n, double x_min, double x_max, double x[]);
double *nc_compute_new_parallel(int n, double x_min, double x_max, double x[],
                                int rank, int size);
void r8mat_write(char *output_filename, int m, int n, double table[]);
void rescale(double a, double b, int n, double x[], double w[]);
void rule_write(int order, char *filename, double x[], double w[], double r[]);

int main(int argc, char *argv[]) {
  double a;
  double b;
  char filename[255];
  int n;
  double *r;
  double *w, *ww;
  double *x, *xx;
  double x_max;
  double x_min;
  int rank, size;

  double t1_seq, t2_seq, seq_time;
  double t1_par, t2_par, par_time;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == MASTER) {
    printf("\n");
    printf("CCN_RULE\n");
    printf("  C version\n");
    printf("\n");
    printf("  Compute one of a family of nested Clenshaw Curtis rules\n");
    printf("  for approximating\n");
    printf("    Integral ( -1 <= x <= +1 ) f(x) dx\n");
    printf("  of order N.\n");
    printf("\n");
    printf("  The user specifies N, A, B and FILENAME.\n");
    printf("\n");
    printf("  N is the number of points.\n");
    printf("  A is the left endpoint.\n");
    printf("  B is the right endpoint.\n");
    printf("  FILENAME is used to generate 3 files:\n");
    printf("    filename_w.txt - the weight file\n");
    printf("    filename_x.txt - the abscissa file.\n");
    printf("    filename_r.txt - the region file.\n");

    if (1 < argc) {
      n = atoi(argv[1]);
    } else {
      printf("\n");
      printf("  Enter the value of N (1 or greater)\n");
      scanf("%d", &n);
    }

    if (2 < argc) {
      a = atof(argv[2]);
    } else {
      printf("\n");
      printf("  Enter the left endpoint A:\n");
      scanf("%lf", &a);
    }

    if (3 < argc) {
      b = atof(argv[3]);
    } else {
      printf("\n");
      printf("  Enter the right endpoint B:\n");
      scanf("%lf", &b);
    }

    if (4 < argc) {
      strcpy(filename, argv[4]);
    } else {
      printf("\n");
      printf("  Enter FILENAME, the \"root name\" of the quadrature files.\n");
      scanf("%s", filename);
    }

    printf("\n");
    printf("  N = %d\n", n);
    printf("  A = %g\n", a);
    printf("  B = %g\n", b);
    printf("  FILENAME = \"%s\".\n", filename);
  }

  MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&a, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  r = (double *)malloc(2 * sizeof(double));
  r[0] = a;
  r[1] = b;

  // Sekvencijalna obrada
  if (rank == MASTER) {
    t1_seq = MPI_Wtime();

    x = ccn_compute_points_new(n);
    x_min = -1.0;
    x_max = +1.0;
    w = nc_compute_new(n, x_min, x_max, x);
    rescale(a, b, n, x, w);

    t2_seq = MPI_Wtime();
    seq_time = t2_seq - t1_seq;
  }

  // paralelna obrada
  MPI_Barrier(MPI_COMM_WORLD);
  t1_par = MPI_Wtime();

  if (rank == MASTER) {
    xx = ccn_compute_points_new(n);
  } else {
    xx = (double *)malloc(n * sizeof(double));
  }
  MPI_Bcast(xx, n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

  x_min = -1.0;
  x_max = +1.0;
  ww = nc_compute_new_parallel(n, x_min, x_max, xx, rank, size);

  if (rank == MASTER) {
    rescale(a, b, n, xx, ww);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t2_par = MPI_Wtime();
  par_time = t2_par - t1_par;

  if (rank == MASTER) {
    int isSame = 1;
    for (int i = 0; i < n; i++) {
      if (fabs(x[i] - xx[i]) > 0.000001) {
        isSame = 0;
        break;
      }
    }

    if (isSame) {
      printf("Test PASSED\n");
    } else {
      printf("Test FAILED\n");
    }

    rule_write(n, filename, xx, ww, r);

    printf("Sekvencijalno vreme: %f sekundi\n", seq_time);
    printf("Paralelno vreme: %f sekundi\n", par_time);
    if (par_time > 0.0) {
      double speedup = seq_time / par_time;
      printf("Ubrzanje: %.2f puta\n", speedup);
    }
  }

  free(r);
  if (rank == MASTER) {
    free(w);
    free(x);
  }
  free(ww);
  free(xx);

  if (rank == MASTER) {
    printf("\n");
    printf("CCN_RULE:\n");
    printf("  Normal end of execution.\n");
    printf("\n");
  }

  MPI_Finalize();
  return 0;
}

double *ccn_compute_points_new(int n) {
  int d;
  int i;
  int k;
  int m;
  double r8_pi = 3.141592653589793;
  int td;
  int tu;
  double *x;

  x = (double *)malloc(n * sizeof(double));

  if (1 <= n) {
    x[0] = 0.5;
  }

  if (2 <= n) {
    x[1] = 1.0;
  }

  if (3 <= n) {
    x[2] = 0.0;
  }

  m = 3;
  d = 2;

  while (m < n) {
    tu = d + 1;
    td = d - 1;

    k = i4_min(d, n - m);

    for (i = 1; i <= k; i++) {
      if ((i % 2) == 1) {
        x[m + i - 1] = tu / 2.0 / (double)(k);
        tu = tu + 2;
      } else {
        x[m + i - 1] = td / 2.0 / (double)(k);
        td = td - 2;
      }
    }
    m = m + k;
    d = d * 2;
  }

  for (i = 0; i < n; i++) {
    x[i] = cos(x[i] * r8_pi);
  }
  x[0] = 0.0;

  if (2 <= n) {
    x[1] = -1.0;
  }

  if (3 <= n) {
    x[2] = +1.0;
  }

  return x;
}

// double *nc_compute_new_parallel(int n, double x_min, double x_max, double
// x[],
//                                 int rank, int size) {
//   double *d;
//   int i;
//   int j;
//   int k;
//   double *w;
//   double yvala;
//   double yvalb;
//
//   int chunk_size = n / size;
//   int leftover_size = n % size;
//   int starting_index =
//       chunk_size * rank + (leftover_size > rank ? rank : leftover_size);
//   int end_index = starting_index + chunk_size + (leftover_size > rank ? 1 :
//   0);
//
//   d = (double *)malloc(n * sizeof(double));
//   w = (double *)malloc((end_index - starting_index) * sizeof(double));
//
//   for (i = starting_index; i < end_index; i++) {
//     for (j = 0; j < n; j++) {
//       d[j] = 0.0;
//     }
//     d[i] = 1.0;
//
//     for (j = 2; j <= n; j++) {
//       for (k = j; k <= n; k++) {
//         d[n + j - k - 1] = (d[n + j - k - 1 - 1] - d[n + j - k - 1]) /
//                            (x[n + 1 - k - 1] - x[n + j - k - 1]);
//       }
//     }
//
//     for (j = 1; j <= n - 1; j++) {
//       for (k = 1; k <= n - j; k++) {
//         d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k];
//       }
//     }
//
//     yvala = d[n - 1] / (double)(n);
//     for (j = n - 2; 0 <= j; j--) {
//       yvala = yvala * x_min + d[j] / (double)(j + 1);
//     }
//     yvala = yvala * x_min;
//
//     yvalb = d[n - 1] / (double)(n);
//     for (j = n - 2; 0 <= j; j--) {
//       yvalb = yvalb * x_max + d[j] / (double)(j + 1);
//     }
//     yvalb = yvalb * x_max;
//
//     w[i - starting_index] = yvalb - yvala;
//   }
//
//   free(d);
//
//   double *ww = (double *)malloc(n * sizeof(double));
//
//   //MPI_Gather(w, end_index - starting_index, MPI_DOUBLE, ww, n, MPI_DOUBLE,
//              //MASTER, MPI_COMM_WORLD);
//   MPI_Reduce(w, ww, n, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
//   free(w);
//   return ww;
// }

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

int i4_min(int i1, int i2) {
  int value;

  if (i1 < i2) {
    value = i1;
  } else {
    value = i2;
  }
  return value;
}

double *nc_compute_new(int n, double x_min, double x_max, double x[]) {
  double *d;
  int i;
  int j;
  int k;
  double *w;
  double yvala;
  double yvalb;

  d = (double *)malloc(n * sizeof(double));
  w = (double *)malloc(n * sizeof(double));

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

  return w;
}

void r8mat_write(char *output_filename, int m, int n, double table[]) {
  int i;
  int j;
  FILE *output;

  output = fopen(output_filename, "wt");

  if (!output) {
    fprintf(stderr, "\n");
    fprintf(stderr, "R8MAT_WRITE - Fatal error!\n");
    fprintf(stderr, "  Could not open the output file.\n");
    exit(1);
  }

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      fprintf(output, "  %24.16g", table[i + j * m]);
    }
    fprintf(output, "\n");
  }

  fclose(output);

  return;
}

void rescale(double a, double b, int n, double x[], double w[]) {
  int i;

  for (i = 0; i < n; i++) {
    x[i] = ((a + b) + (b - a) * x[i]) / 2.0;
  }
  for (i = 0; i < n; i++) {
    w[i] = (b - a) * w[i] / 2.0;
  }
  return;
}

void rule_write(int order, char *filename, double x[], double w[], double r[]) {
  char filename_r[80];
  char filename_w[80];
  char filename_x[80];

  strcpy(filename_r, filename);
  strcat(filename_r, "_r.txt");
  strcpy(filename_w, filename);
  strcat(filename_w, "_w.txt");
  strcpy(filename_x, filename);
  strcat(filename_x, "_x.txt");

  printf("\n");
  printf("  Creating quadrature files.\n");
  printf("\n");
  printf("  Root file name is     \"%s\".\n", filename);
  printf("\n");
  printf("  Weight file will be   \"%s\".\n", filename_w);
  printf("  Abscissa file will be \"%s\".\n", filename_x);
  printf("  Region file will be   \"%s\".\n", filename_r);

  r8mat_write(filename_w, 1, order, w);
  r8mat_write(filename_x, 1, order, x);
  r8mat_write(filename_r, 1, 2, r);

  return;
}