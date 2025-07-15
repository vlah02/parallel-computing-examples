#!/usr/bin/env bash
set -euo pipefail

orders=(99 999 2999)

for variant in seq mpi omp cuda; do
  mkdir -p output/$variant
done

#echo "==> Running sequential version"
#for N in "${orders[@]}"; do
#  printf "  N=%-4d  " "$N"
#  ./bin/seq $N -100 +100 output/seq/ccn_o${N}
#done
#
#echo "==> Running MPI version"
#for N in "${orders[@]}"; do
#  printf "  N=%-4d  " "$N"
#  mpirun -np 8 ./bin/mpi $N -100 +100 output/mpi/ccn_o${N}
#done
#
#echo "==> Running OpenMP version (8 threads)"
#export OMP_NUM_THREADS=8
#for N in "${orders[@]}"; do
#  printf "  N=%-4d  " "$N"
#  ./bin/omp $N -100 +100 output/omp/ccn_o${N}
#done

echo "==> Running CUDA version"
for N in "${orders[@]}"; do
  printf "  N=%-4d  " "$N"
  ./bin/cuda $N -100 +100 output/cuda/ccn_o${N}
done

echo "All done!"
