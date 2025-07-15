#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "mpi.h"

std::string red = "\033[1;31m";
std::string green = "\033[1;32m";
std::string blue = "\033[1;36m";
std::string clear = "\033[0m";

#define MASTER 0

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

  return true;
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

  float data;
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
                float *C, int ldc) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

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
}

void parallelSgemm(char transa, char transb, int m, int n, int k, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc, int rank, int size) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  std::vector<float> CC(m * n);

  int chunk_size = m / size;
  int leftover_size = m % size;
  int starting_index = chunk_size * rank + (leftover_size > rank ? rank : 0);
  int end_index = starting_index + chunk_size + (leftover_size > rank ? 1 : 0);

  for (int mm = starting_index; mm < end_index; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda];
        float b = B[nn + i * ldb];
        c += a * b;
      }
      CC[mm + nn * ldc] = CC[mm + nn * ldc] * beta + alpha * c;
    }
  }
  MPI::COMM_WORLD.Reduce(&CC.front(), C, m * n, MPI::FLOAT, MPI::SUM, MASTER);
}

int main(int argc, char *argv[]) {
  int matArow, matAcol;
  int matBrow, matBcol;
  int rank, size;
  double start, end;
  double t1;

  if (argc != 4) {
    fprintf(stderr, "Expecting three input filenames\n");
    exit(-1);
  }

  MPI::Init();

  std::vector<float> matA, matBT;

  rank = MPI::COMM_WORLD.Get_rank();
  size = MPI::COMM_WORLD.Get_size();

  if (rank == MASTER) {
    /* Read in data */
    // load A
    readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

    // load B^T
    readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);
  }
  MPI::COMM_WORLD.Bcast(&matArow, 1, MPI_INT, MASTER);
  MPI::COMM_WORLD.Bcast(&matAcol, 1, MPI_INT, MASTER);
  MPI::COMM_WORLD.Bcast(&matBrow, 1, MPI_INT, MASTER);
  MPI::COMM_WORLD.Bcast(&matBcol, 1, MPI_INT, MASTER);

  if (rank != MASTER) {
    matA = std::vector<float>(matArow * matAcol);
    matBT = std::vector<float>(matBcol * matBrow);
  }

  MPI::COMM_WORLD.Bcast(&matA.front(), matArow * matAcol, MPI_FLOAT, MASTER);
  MPI::COMM_WORLD.Bcast(&matBT.front(), matBcol * matBrow, MPI_FLOAT, MASTER);

  // allocate space for C
  std::vector<float> matC(matArow * matBcol);
  std::vector<float> matD(matArow * matBcol);

  if (rank == MASTER) {
    // Use standard sgemm interface
    start = MPI::Wtime();
    basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(),
               matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
    end = MPI::Wtime();

    t1 = end - start;

    std::cout << blue << "Brzina izvrsavanja sekvencijalnog programa "
              << (end - start) << clear << std::endl;
  }

  if (rank == MASTER) {
    start = MPI::Wtime();
  }

  parallelSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(),
                matArow, &matBT.front(), matBcol, 0.0f, &matD.front(), matArow,
                rank, size);
  if (rank == MASTER) {
    end = MPI::Wtime();
    std::cout << blue << "Brzina izvrsavanja paralelnog programa na " << size
              << " procesora" << (end - start) << clear << std::endl;
    std::cout << blue << "Ukupno ubrzanje = " << t1 / (end - start) << clear
              << std::endl;
  }

  if (rank == MASTER) {
    bool isSame = true;

    for (int i = 0; i < matArow; i++) {
      for (int j = 0; j < matBcol; j++) {
        if (matC[i] != matD[i]) {
          isSame = false;
          break;
        }
      }
    }

    if (isSame) {
      std::cout << green << "TEST PASSED\n" << clear;
    } else {
      std::cout << red << "TEST FAILED\n" << clear;
    }

    writeColMajorMatrixFile(argv[3], matArow, matBcol, matD);
  }

  MPI::Finalize();

  return 0;
}