#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>
#include <unistd.h>

#include "args.h"
#include "model.h"

#define MASTER 0

const char *red = "\033[1;31m";
const char *green = "\033[1;32m";
const char *blue = "\033[1;36m";
const char *clear = "\033[0m";

int main(int argc, char **argv) {
  struct pb_TimerSet timers;
  struct pb_Parameters *params;
  int rf, k, nbins, npd, npr;
  float *binb;
  long long *DD, *RRS, *DRS;
  long long *DDDD, *RRSRRS, *DRSDRS;
  size_t memsize;
  struct cartesian *data, *random;
  FILE *outfile;
  options args;
  cudaEvent_t start = cudaEvent_t();
  cudaEvent_t end = cudaEvent_t();
  float sequential_time;
  float paralel_time;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  pb_InitializeTimerSet(&timers);
  params = pb_ReadParameters(&argc, argv);

  parse_args(argc, argv, &args);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  nbins = (int)floor(bins_per_dec * (log10(max_arcmin) - log10(min_arcmin)));
  memsize = (nbins + 2) * sizeof(long long);

  // memory for bin boundaries
  binb = (float *)malloc((nbins + 1) * sizeof(float));
  if (binb == NULL) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(-1);
  }
  for (k = 0; k < nbins + 1; k++) {
    binb[k] =
        cos(pow(10, log10(min_arcmin) + k * 1.0 / bins_per_dec) / 60.0 * D2R);
  }

  // memory for DD
  DD = (long long *)malloc(memsize);
  DDDD = (long long *)malloc(memsize);
  if (DD == NULL || DDDD == NULL) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(-1);
  }
  bzero(DD, memsize);
  bzero(DDDD, memsize);

  // memory for RR
  RRS = (long long *)malloc(memsize);
  RRSRRS = (long long *)malloc(memsize);
  if (RRS == NULL || RRSRRS == NULL) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(-1);
  }
  bzero(RRS, memsize);
  bzero(RRSRRS, memsize);

  // memory for DR
  DRS = (long long *)malloc(memsize);
  DRSDRS = (long long *)malloc(memsize);
  if (DRS == NULL || DRSDRS == NULL) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(-1);
  }
  bzero(DRS, memsize);
  bzero(DRSDRS, memsize);

  // memory for input data
  data = (struct cartesian *)malloc(args.npoints * sizeof(struct cartesian));
  if (data == NULL) {
    fprintf(stderr, "Unable to allocate memory for % data points (#1)\n",
            args.npoints);
    return (0);
  }

  random = (struct cartesian *)malloc(args.npoints * sizeof(struct cartesian));
  if (random == NULL) {
    fprintf(stderr, "Unable to allocate memory for % data points (#2)\n",
            args.npoints);
    return (0);
  }

  printf("Min distance: %f arcmin\n", min_arcmin);
  printf("Max distance: %f arcmin\n", max_arcmin);
  printf("Bins per dec: %i\n", bins_per_dec);
  printf("Total bins  : %i\n", nbins);

  // read data file
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  npd = readdatafile(params->inpFiles[0], data, args.npoints);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if (npd != args.npoints) {
    fprintf(stderr, "Error: read %i data points out of %i\n", npd,
            args.npoints);
    return (0);
  }

  // sekvencijalna implementacija

  cudaEventRecord(start, 0);

  doCompute(data, npd, NULL, 0, 1, DD, nbins, binb);

  for (rf = 0; rf < args.random_count; rf++) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    npr = readdatafile(params->inpFiles[rf + 1], random, args.npoints);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    if (npr != args.npoints) {
      fprintf(stderr, "Error: read %i random points out of %i in file %s\n",
              npr, args.npoints, params->inpFiles[rf + 1]);
      return (0);
    }

    // compute RR
    doCompute(random, npr, NULL, 0, 1, RRS, nbins, binb);

    // compute DR
    doCompute(data, npd, random, npr, 0, DRS, nbins, binb);
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&sequential_time, start, end);

  printf("%sVreme sekvencijalne implementacije = %lf sekundi\n%s", blue,
         sequential_time / 1000, clear);
  // paralelna impelmentacija

  cudaEventRecord(start, 0);
  // compute DD
  doComputeCuda(data, npd, NULL, 0, 1, DDDD, nbins, binb);

  // loop through random data files
  for (rf = 0; rf < args.random_count; rf++) {
    // read random file
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    npr = readdatafile(params->inpFiles[rf + 1], random, args.npoints);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    if (npr != args.npoints) {
      fprintf(stderr, "Error: read %i random points out of %i in file %s\n",
              npr, args.npoints, params->inpFiles[rf + 1]);
      return (0);
    }

    // compute RR
    doComputeCuda(random, npr, NULL, 0, 1, RRSRRS, nbins, binb);

    // compute DR
    doComputeCuda(data, npd, random, npr, 0, DRSDRS, nbins, binb);
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&paralel_time, start, end);

  // compute and output results
  int isSame = 1;

  for (int i = 0; i < nbins + 2; i++) {
    if (DD[i] != DDDD[i] || RRS[i] != RRSRRS[i] || DRS[i] != DRSDRS[i]) {
      isSame = 0;
      break;
    }
  }
  if (isSame) {
    printf("%sTest PASSED\n%s", green, clear);
  } else {
    printf("%sTest FAILED\n%s", red, clear);
  }

  printf("%sVreme paralelne implementacije = %lf sekundi\n%s", blue,
         paralel_time / 1000, clear);
  printf("%sUbrzanje je = %lf\n%s", blue, sequential_time / paralel_time,
         clear);

  if ((outfile = fopen(params->outFile, "w")) == NULL) {
    fprintf(stderr,
            "Unable to open output file %s for writing, assuming stdout\n",
            params->outFile);
    outfile = stdout;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  for (k = 0; k < nbins + 2; k++) {
    fprintf(outfile, "%lld\n%lld\n%lld\n", DDDD[k], DRSDRS[k], RRSRRS[k]);
  }
  // for (k = 0; k < nbins + 2; k++) {
  //   printf("%f\n", binb[k]);
  // }
  if (outfile != stdout) fclose(outfile);

  // free memory
  free(data);
  free(random);
  free(binb);
  free(DD);
  free(RRS);
  free(DRS);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
}