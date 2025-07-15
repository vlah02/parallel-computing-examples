#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "model.h"

#define NUM_OF_THREADS 1024

__global__ void doComputeKernel(struct cartesian* data1, int n1,
                                struct cartesian* data2, int n2, int doSelf,
                                long long* data_bins, int nbins, float* binb) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;

  if (i < ((doSelf) ? n1 - 1 : n1)) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    for (j = ((doSelf) ? i + 1 : 0); j < n2; j++) {
      float a      = __fmul_rn(xi, data2[j].x);
      float b      = __fmul_rn(yi, data2[j].y);
      float c      = __fmul_rn(zi, data2[j].z);
      float sum_ab = __fadd_rn(a, b);
      float dot    = __fadd_rn(sum_ab, c);

      // run binary search
      int min = 0;
      int max = nbins;
      int k;

      while (max > min + 1) {
        k = (min + max) / 2;
        if (dot >= binb[k])
          max = k;
        else
          min = k;
      }

      if (dot >= binb[min]) {
        atomicAdd((unsigned long long*)&data_bins[min], 1);
        // data_bins[min] += 1; /*k = min;*/
      } else if (dot < binb[max]) {
        atomicAdd((unsigned long long*)&data_bins[max + 1], 1);
        // data_bins[max + 1] += 1; /*k = max+1;*/
      } else {
        atomicAdd((unsigned long long*)&data_bins[max], 1);
        // data_bins[max] += 1; /*k = max;*/
      }
    }
  }
}

int doComputeCuda(struct cartesian* data1, int n1, struct cartesian* data2,
                  int n2, int doSelf, long long* data_bins, int nbins,
                  float* binb) {
  struct cartesian* devData1;
  struct cartesian* devData2;
  float* binb_dev;
  long long* dev_data_bins;

  cudaMalloc((void**)&binb_dev, (nbins + 2) * sizeof(float));
  cudaMalloc((void**)&dev_data_bins, (nbins + 2) * sizeof(long long));
  cudaMalloc((void**)&devData1, n1 * sizeof(struct cartesian));
  if (!doSelf) {
    cudaMalloc((void**)&devData2, n2 * sizeof(struct cartesian));
  }

  cudaMemcpy(devData1, data1, n1 * sizeof(struct cartesian),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_data_bins, data_bins, (nbins + 2) * sizeof(long long),
             cudaMemcpyHostToDevice);
  cudaMemcpy(binb_dev, binb, (nbins + 2) * sizeof(float),
             cudaMemcpyHostToDevice);
  if (doSelf) {
    devData2 = devData1;
    n2 = n1;
  } else {
    cudaMemcpy(devData2, data2, n2 * sizeof(struct cartesian),
               cudaMemcpyHostToDevice);
  }

  doComputeKernel<<<(n1 + NUM_OF_THREADS - 1) / NUM_OF_THREADS,
                    NUM_OF_THREADS>>>(devData1, n1, devData2, n2, doSelf,
                                      dev_data_bins, nbins, binb_dev);
  cudaDeviceSynchronize();
  cudaMemcpy(data_bins, dev_data_bins, (nbins + 2) * sizeof(long long),
             cudaMemcpyDeviceToHost);

  cudaFree(devData1);
  cudaFree(devData2);
  cudaFree(dev_data_bins);
  cudaFree(binb_dev);

  return 0;
}

int doCompute(struct cartesian* data1, int n1, struct cartesian* data2, int n2,
              int doSelf, long long* data_bins, int nbins, float* binb) {
  int i, j;
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

  for (i = 0; i < ((doSelf) ? n1 - 1 : n1); i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    for (j = ((doSelf) ? i + 1 : 0); j < n2; j++) {
      float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

      // run binary search
      int min = 0;
      int max = nbins;
      int k;

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