#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "model.h"

#define MASTER 0
#define WORKTAG 1
#define DIETAG 2

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2, int n2,
              int doSelf, long long *data_bins, int nbins, float *binb) {
  int i, j, k;
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

  for (i = 0; i < ((doSelf) ? n1 - 1 : n1); i++) {
    const register float xi = data1[i].x;
    const register float yi = data1[i].y;
    const register float zi = data1[i].z;

    for (j = ((doSelf) ? i + 1 : 0); j < n2; j++) {
      register float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

      // run binary search
      register int min = 0;
      register int max = nbins;
      register int k, indx;

      while (max > min + 1) {
        k = (min + max) / 2;
        if (dot >= binb[k])
          max = k;
        else
          min = k;
      };

      if (dot >= binb[min]) {
        data_bins[min] += 1; /*k = min;*/
      } else if (dot < binb[max]) {
        data_bins[max + 1] += 1; /*k = max+1;*/
      } else {
        data_bins[max] += 1; /*k = max;*/
      }
    }
  }

  return 0;
}

int doComputeParallel(struct cartesian *data1, int n1, struct cartesian *data2,
                      int n2, int doSelf, long long *data_bins, int nbins,
                      float *binb, int rank, int size) {
  int i, j, k;
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

  int sentinel = ((doSelf) ? n1 - 1 : n2);

  int chunk_size = sentinel / size;
  int leftover_size = sentinel % size;
  int starting_index =
      chunk_size * rank + (leftover_size > rank ? rank : leftover_size);
  int end_index = starting_index + chunk_size + (leftover_size > rank ? 1 : 0);

  long long *local_bins = malloc((nbins + 2) * sizeof(long long));

  for (int i = 0; i < nbins + 2; i++) {
    local_bins[i] = data_bins[i];
  }

  for (i = starting_index; i < end_index; i++) {
    const register float xi = data1[i].x;
    const register float yi = data1[i].y;
    const register float zi = data1[i].z;

    for (j = ((doSelf) ? i + 1 : 0); j < n2; j++) {
      register float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

      // run binary search
      register int min = 0;
      register int max = nbins;
      register int k, indx;

      while (max > min + 1) {
        k = (min + max) / 2;
        if (dot >= binb[k])
          max = k;
        else
          min = k;
      };

      if (dot >= binb[min]) {
        local_bins[min] += 1; /*k = min;*/
      } else if (dot < binb[max]) {
        local_bins[max + 1] += 1; /*k = max+1;*/
      } else {
        local_bins[max] += 1; /*k = max;*/
      }
    }
  }

  MPI_Reduce(local_bins, data_bins, nbins + 2, MPI_LONG_LONG, MPI_SUM, MASTER,
             MPI_COMM_WORLD);

  return 0;
}

void master(int n1, int n2, int doSelf, long long *data_bins, int nbins,
            int rank, int size);
void slave(struct cartesian *data1, int n1, struct cartesian *data2, int n2,
           int doSelf, long long *data_bins, int nbins, float *binb, int rank,
           int size);

int doComputeMasterSlave(struct cartesian *data1, int n1,
                         struct cartesian *data2, int n2, int doSelf,
                         long long *data_bins, int nbins, float *binb, int rank,
                         int size) {
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }
  if (rank == MASTER) {
    master(n1, n2, doSelf, data_bins, nbins, rank, size);
  } else {
    slave(data1, n1, data2, n2, doSelf, data_bins, nbins, binb, rank, size);
  }
}

int min(int num1, int num2) { return num1 < num2 ? num1 : num2; }

void master(int n1, int n2, int doSelf, long long *data_bins, int nbins,
            int rank, int size) {
  int sentinel = ((doSelf) ? n1 - 1 : n1);
  int array_index = 0;
  int confirm;
  long long *buffer = malloc(sizeof(long long) * (nbins + 2));
  MPI_Status status;

  for (int i = 1; i < min(size, sentinel + 1); i++) {
    MPI_Send(&array_index, 1, MPI_INT, i, WORKTAG, MPI_COMM_WORLD);
    array_index++;
  }
  while (array_index < sentinel) {
    MPI_Recv(&confirm, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
             &status);

    MPI_Send(&array_index, 1, MPI_INT, status.MPI_SOURCE, WORKTAG,
             MPI_COMM_WORLD);

    array_index++;
  }

  for (int i = 1; i < min(size, sentinel + 1); i++) {
    MPI_Recv(&confirm, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
             &status);
  }

  for (int i = 1; i < min(size, sentinel + 1); i++) {
    MPI_Send(0, 0, MPI_INT, i, DIETAG, MPI_COMM_WORLD);
    MPI_Recv(buffer, nbins + 2, MPI_LONG_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD,
             &status);

    for (int j = 0; j < nbins + 2; j++) {
      data_bins[j] += buffer[j];
    }
  }

  free(buffer);
}

void slave(struct cartesian *data1, int n1, struct cartesian *data2, int n2,
           int doSelf, long long *data_bins, int nbins, float *binb, int rank,
           int size) {
  long long *local_bins = calloc((nbins + 2), sizeof(long long));

  while (1) {
    int index;
    MPI_Status status;
    MPI_Recv(&index, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == DIETAG) {
      MPI_Send(local_bins, nbins + 2, MPI_LONG_LONG, MASTER, 0, MPI_COMM_WORLD);
      free(local_bins);
      return;
    }

    const register float xi = data1[index].x;
    const register float yi = data1[index].y;
    const register float zi = data1[index].z;

    for (int j = ((doSelf) ? index + 1 : 0); j < n2; j++) {
      register float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

      // run binary search
      register int min = 0;
      register int max = nbins;
      register int k, indx;

      while (max > min + 1) {
        k = (min + max) / 2;
        if (dot >= binb[k])
          max = k;
        else
          min = k;
      };

      if (dot >= binb[min]) {
        local_bins[min] += 1; /*k = min;*/
      } else if (dot < binb[max]) {
        local_bins[max + 1] += 1; /*k = max+1;*/
      } else {
        local_bins[max] += 1; /*k = max;*/
      }
    }

    MPI_Send(0, 0, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
  }
}