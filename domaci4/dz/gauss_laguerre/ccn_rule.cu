#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cufft.h>

const char *red   = "\033[1;31m";
const char *green = "\033[1;32m";
const char *blue  = "\033[1;36m";
const char *clear = "\033[0m";

int main(int argc, char *argv[]);
double *ccn_compute_points_new(int n);
int i4_min(int i1, int i2);
double *nc_compute_new(int n, double x_min, double x_max, double x[]);
double *nc_compute_new_cuda(int n, double x_min, double x_max, double x[]);
void r8mat_write(char *output_filename, int m, int n, double table[]);
void rescale(double a, double b, int n, double x[], double w[]);
void rule_write(int order, char *filename, double x[], double w[], double r[]);

int main(int argc, char *argv[]) {
  double a, b;
  char filename[255];
  int n;
  double *r;
  double *w, *ww;
  double *x, *xx;
  float time_seq, time_parallel;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("\nCCN_RULE\n  C version\n\n");
  printf("  Compute one of a family of nested Clenshaw Curtis rules\n");
  printf("  for approximating Integral ( -1 <= x <= +1 ) f(x) dx of order N.\n\n");
  printf("  N is the number of points.\n  A is the left endpoint.\n  B is the right endpoint.\n");
  printf("  FILENAME is the root name for output files.\n\n");

  if (argc > 1)      n = atoi(argv[1]);
  else { printf("Enter N: "); scanf("%d",&n); }
  if (argc > 2)      a = atof(argv[2]);
  else { printf("Enter A: "); scanf("%lf",&a); }
  if (argc > 3)      b = atof(argv[3]);
  else { printf("Enter B: "); scanf("%lf",&b); }
  if (argc > 4)      strcpy(filename, argv[4]);
  else { printf("Enter FILENAME: "); scanf("%s",filename); }

  printf("\n  N = %d\n  A = %g\n  B = %g\n  FILENAME = \"%s\"\n\n",
         n, a, b, filename);

  r = (double*)malloc(2 * sizeof(double));
  r[0] = a;  r[1] = b;

  cudaEventRecord(start,0);
  x = ccn_compute_points_new(n);
  w = nc_compute_new(n, -1.0, +1.0, x);
  rescale(a, b, n, x, w);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_seq, start, stop);
  printf("%sVreme sekvencijalne implementacije %f sekundi%s\n",
         blue, time_seq/1000, clear);

  cudaEventRecord(start,0);
  xx = ccn_compute_points_new(n);
  ww = nc_compute_new_cuda(n, a, b, xx);
  for (int i = 0; i < n; i++) {
    xx[i] = ((a + b) + (b - a) * xx[i]) / 2.0;
  }
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_parallel, start, stop);
  printf("%sVreme paralelne implementacije %f sekundi%s\n",
         blue, time_parallel/1000, clear);
  printf("%sUbrzanje = %f%s\n\n",
         blue, time_seq / time_parallel, clear);

  int flag = 0;
  for (int i = 0; i < n; i++) {
    if (fabs(ww[i] - w[i]) > 1e-6 || fabs(xx[i] - x[i]) > 1e-12) {
      flag = 1;
      break;
    }
  }
  if (flag) printf("%sTest FAILED%s\n", red, clear);
  else      printf("%sTest PASSED%s\n", green, clear);

  rule_write(n, filename, xx, ww, r);

  free(r);
  free(w);
  free(x);
  cudaFreeHost(ww);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
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

int i4_min(int i1, int i2) {
  int value;

  if (i1 < i2) {
    value = i1;
  } else {
    value = i2;
  }
  return value;
}
__global__ void extract_and_linearscale(
    cufftDoubleComplex *Y,
    double             *w,
    int                 n,
    double              a,
    double              b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double c = Y[i].x;
  double scale = 2.0 / (double)n;
  if (i == 0 || i == n-1) scale *= 0.5;
  double wi = scale * c;
  w[i] = wi * ((b - a) * 0.5);
}

double *nc_compute_new_cuda(int n, double a, double b, double x[]) {
  double *h_w;
  cudaMallocHost(&h_w, n * sizeof(double));

  double *h_y = (double*)malloc((n+1) * sizeof(double));
  h_y[0] = 1.0;
  h_y[n] = (n & 1) ? -1.0 : +1.0;
  for (int k = 1; k < n; ++k) {
    h_y[k] = x[k] + x[n-k];
  }

  int Nfft = 2 * n;
  double *d_y;
  cufftDoubleComplex *d_Y;
  cudaMalloc(&d_y,   Nfft * sizeof(double));
  cudaMalloc(&d_Y, (n+1) * sizeof(cufftDoubleComplex));

  cudaMemcpy(d_y, h_y, (n+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y + n + 1, h_y + 1, (n-1)*sizeof(double), cudaMemcpyHostToDevice);
  free(h_y);

  cufftHandle plan;
  cufftPlan1d(&plan, Nfft, CUFFT_D2Z, 1);
  cufftExecD2Z(plan, d_y, d_Y);

  int block = 256, grid = (n + block - 1) / block;
  extract_and_linearscale<<<grid, block>>>(d_Y, h_w, n, a, b);
  cudaGetLastError();
  cudaDeviceSynchronize();

  cufftDestroy(plan);
  cudaFree(d_y);
  cudaFree(d_Y);

  return h_w;
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