#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <vector>

std::string red = "\033[1;31m";
std::string green = "\033[1;32m";
std::string blue = "\033[1;36m";
std::string clear = "\033[0m";

#define NUM_OF_GPU_THREADS 1024
#define TILE_WIDTH 32

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col,
                            std::vector<float> &v) {
  std::cerr << "Opening file:" << fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if (!f.good()) {
    return false;
  }

  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;

  float data;
  std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
  while (f.good()) {
    f >> data;
    v.push_back(data);
  }
  v.pop_back();  // remove the duplicated last element
  return false;
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col,
                             std::vector<float> &v) {
  std::cerr << "Opening file:" << fn << " for write." << std::endl;
  std::fstream f(fn, std::fstream::out);
  if (!f.good()) {
    return false;
  }

  // Read # of rows and cols
  f << nr_row << " " << nr_col << " ";

  std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
  for (int i = 0; i < v.size(); ++i) {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;
}

/*
 * Base C implementation of MM
 */

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha,
                const float *A, int lda, const float *B, int ldb, float beta,
                float *C, int ldc, float *time) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  cudaEvent_t start = cudaEvent_t();
  cudaEvent_t stop = cudaEvent_t();
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda];
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(time, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/*
 * Base C implementation of MM
 */
__global__ void kernelSgemm(float *A, int lda, float *B, int ldb, float *C,
                            int ldc, float alpha, float beta, int m, int n,
                            int k) {
  __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int mm = by * TILE_WIDTH + ty;
  int nn = bx * TILE_WIDTH + tx;
  float c = 0.0;

  for (int i = 0; i < (k + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
    if (mm < m && i * TILE_WIDTH + tx < k) {
      sharedA[ty][tx] = A[(i * TILE_WIDTH + tx) * lda + mm];
    } else {
      sharedA[ty][tx] = 0.0f;
    }

    if (nn < n && i * TILE_WIDTH + ty < k) {
      sharedB[ty][tx] = B[(i * TILE_WIDTH + ty) * ldb + nn];
    } else {
      sharedB[ty][tx] = 0.0f;
    }
    __syncthreads();

    for (int j = 0; j < TILE_WIDTH; j++) {
      float a = sharedA[ty][j];
      float b = sharedB[j][tx];
      c += a * b;
    }
    __syncthreads();
  }
  if (mm < m && nn < n) {
    C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
  }
}

void cudaSgemm(char transa, char transb, int m, int n, int k, float alpha,
               const float *A, int lda, const float *B, int ldb, float beta,
               float *C, int ldc, float *time) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  cudaEvent_t start = cudaEvent_t();
  cudaEvent_t stop = cudaEvent_t();
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  float *devA, *devB, *devC;
  cudaMalloc((void **)&devA, m * k * sizeof(float));
  cudaMalloc((void **)&devB, n * k * sizeof(float));
  cudaMalloc((void **)&devC, m * n * sizeof(float));

  cudaMemcpy(devA, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devB, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH,
               (m + TILE_WIDTH - 1) / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  kernelSgemm<<<dimGrid, dimBlock>>>(devA, lda, devB, ldb, devC, ldc, alpha,
                                     beta, m, n, k);

  cudaMemcpy(C, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(time, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
}

int main(int argc, char *argv[]) {
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  if (argc != 4) {
    fprintf(stderr, "Expecting three input filenames\n");
    exit(-1);
  }

  /* Read in data */
  // load A
  {
    // peek A’s dimensions so we can reserve up front
    std::ifstream f1(argv[1]);
    f1 >> matArow >> matAcol;
    f1.close();
    matA.reserve( (size_t)matArow * (size_t)matAcol );
  }
  readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

  // load B^T
  {
    std::ifstream f2(argv[2]);
    int rows, cols;
    f2 >> rows >> cols;
    f2.close();
    matBT.reserve((size_t)rows * (size_t)cols);
  }
  readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);
  matBT.resize((size_t)matBcol * (size_t)matBrow);

  // allocate space for C
  std::vector<float> matC(matArow * matBcol);
  std::vector<float> matD(matArow * matBcol);

  float seq_time, cuda_time;

  // Use standard sgemm interface
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow,
             &matBT.front(), matBcol, 0.0f, &matC.front(), matArow, &seq_time);

  std::cout << blue << "Vreme sekvencijalne implementacije " << seq_time / 1000
            << " sekundi" << clear << std::endl;

  cudaSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow,
            &matBT.front(), matBcol, 0.0f, &matD.front(), matArow, &cuda_time);

  std::cout << blue << "Vreme paralelne implementacije " << cuda_time / 1000
            << " sekundi" << clear << std::endl;

  std::cout << blue << "Ubrzanje = " << seq_time / cuda_time << clear
            << std::endl;

  bool flag = false;
  for (int i = 0; i < matArow * matBcol; i++) {
    if (abs(matC[i] - matD[i]) > 0.01) {
      flag = true;
      break;
    }
  }

  if (flag) {
    std::cout << red << "Test FAILED" << clear << std::endl;
  } else {
    std::cout << green << "Test PASSED" << clear << std::endl;
  }

  writeColMajorMatrixFile(argv[3], matArow, matBcol, matD);

  return 0;
}