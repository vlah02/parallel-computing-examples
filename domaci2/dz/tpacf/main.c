#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>
#include <unistd.h>

#include "args.h"
#include "model.h"
#include "mpi.h"

#define MASTER 0

const char *red = "\033[1;31m";
const char *green = "\033[1;32m";
const char *blue = "\033[1;36m";
const char *clear = "\033[0m";

int main(int argc, char **argv) {
  struct pb_TimerSet timers;
  struct pb_Parameters *params;
  int rf, k, nbins, npd, npr;
  float *binb, w;
  long long *DD, *RRS, *DRS;
  long long *DDDD, *RRSRRS, *DRSDRS;
  size_t memsize;
  struct cartesian *data, *random;
  FILE *outfile;
  int rank, size;
  options args;
  double start, end;
  double sequential_time;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == MASTER) {
    pb_InitializeTimerSet(&timers);
    params = pb_ReadParameters(&argc, argv);

    parse_args(argc, argv, &args);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }

  MPI_Bcast(&args.random_count, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&args.npoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

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
  if (rank == MASTER) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    npd = readdatafile(params->inpFiles[0], data, args.npoints);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    if (npd != args.npoints) {
      fprintf(stderr, "Error: read %i data points out of %i\n", npd,
              args.npoints);
      return (0);
    }
  }

  // sekvencijalna implementacija

  if (rank == MASTER) {
    start = MPI_Wtime();

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

    end = MPI_Wtime();

    sequential_time = end - start;

    printf("%sVreme sekvencijalne implementacije = %lf sekundi\n%s", blue,
           sequential_time, clear);
  }
  // paralelna impelmentacija

  MPI_Datatype cartesian_type;

  MPI_Type_contiguous(3, MPI_FLOAT, &cartesian_type);
  MPI_Type_commit(&cartesian_type);

  MPI_Bcast(&npd, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(data, args.npoints, cartesian_type, MASTER, MPI_COMM_WORLD);

  if (rank == MASTER) {
    start = MPI_Wtime();
  }

  // compute DD
  // doComputeParallel(data, npd, NULL, 0, 1, DDDD, nbins, binb, rank, size);
  doComputeMasterSlave(data, npd, NULL, 0, 1, DDDD, nbins, binb, rank, size);
  // loop through random data files
  for (rf = 0; rf < args.random_count; rf++) {
    // read random file
    if (rank == MASTER) {
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      npr = readdatafile(params->inpFiles[rf + 1], random, args.npoints);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

      if (npr != args.npoints) {
        fprintf(stderr, "Error: read %i random points out of %i in file %s\n",
                npr, args.npoints, params->inpFiles[rf + 1]);
        return (0);
      }
    }
    MPI_Bcast(&npr, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(random, args.npoints, cartesian_type, MASTER, MPI_COMM_WORLD);

    // compute RR
    // doComputeParallel(random, npr, NULL, 0, 1, RRSRRS, nbins, binb, rank,
    // size);
    doComputeMasterSlave(random, npr, NULL, 0, 1, RRSRRS, nbins, binb, rank,
                         size);
    // compute DR
    // doComputeParallel(data, npd, random, npr, 0, DRSDRS, nbins, binb, rank,
    //                   size);
    doComputeMasterSlave(data, npd, random, npr, 0, DRSDRS, nbins, binb, rank,
                         size);
  }

  MPI_Type_free(&cartesian_type);

  // compute and output results
  if (rank == MASTER) {
    int isSame = 1;

    for (int i = 0; i < nbins + 2; i++) {
      if (DD[i] != DDDD[i] || RRS[i] != RRSRRS[i] || DRS[i] != DRSDRS[i]) {
        printf("DD = %lld %lld\n", DD[i], DDDD[i]);
        printf("RRS = %lld %lld\n", RRS[i], RRSRRS[i]);
        printf("DRS = %lld %lld\n", DRS[i], DRSDRS[i]);
        printf("i = %d\n", i);
        isSame = 0;
        break;
      }
    }

    if (isSame) {
      printf("%sTest PASSED\n%s", green, clear);
    } else {
      printf("%sTest FAILED\n%s", red, clear);
    }

    end = MPI_Wtime();

    printf("%sVreme paralelne implementacije na %d procesora = %lf sekundi\n%s",
           blue, size, end - start, clear);
    printf("%sUbrzanje je = %lf\n%s", blue, sequential_time / (end - start),
           clear);

    if ((outfile = fopen(params->outFile, "w")) == NULL) {
      fprintf(stderr,
              "Unable to open output file %s for writing, assuming stdout\n",
              params->outFile);
      outfile = stdout;
    }

    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    for (k = 1; k < nbins + 1; k++) {
      fprintf(outfile, "%lld\n%lld\n%lld\n", DDDD[k], DRSDRS[k], RRSRRS[k]);
    }

    if (outfile != stdout) fclose(outfile);
  }
  // free memory
  free(data);
  free(random);
  free(binb);
  free(DD);
  free(RRS);
  free(DRS);

  if (rank == MASTER) {
    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
    pb_FreeParameters(params);
  }

  MPI_Finalize();
}