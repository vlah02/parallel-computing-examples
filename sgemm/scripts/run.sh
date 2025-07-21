#!/usr/bin/env bash
set -euo pipefail

runs=${1:-1}
sizes=(small medium)

make clean
make seq
make mpi
make cuda

mkdir -p output
for variant in seq mpi omp cuda; do
  mkdir -p output/$variant
done

echo

for ((i = 1; i <= runs; i++)); do
  for size in "${sizes[@]}"; do
    echo "==> Running sequential on ${size}"
    ./bin/seq \
      data/${size}/input/matrix1.txt \
      data/${size}/input/matrix2t.txt \
      output/seq/result_${size}.txt
  done
done

for size in "${sizes[@]}"; do
  echo "==> Running MPI on ${size}"
  mpirun -np 8 ./bin/mpi \
    data/${size}/input/matrix1.txt \
    data/${size}/input/matrix2t.txt \
    output/mpi/result_${size}.txt
done

for size in "${sizes[@]}"; do
  echo "==> Running CUDA on ${size}"
  ./bin/cuda \
    data/${size}/input/matrix1.txt \
    data/${size}/input/matrix2t.txt \
    output/cuda/result_${size}.txt
done

echo "All runs complete."
