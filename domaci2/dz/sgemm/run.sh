#!/bin/bash

NUM_PROCS=${1:-8}

mpic++ -lm -o binc/sgemm sgemm.cc
mpirun -np $NUM_PROCS ./binc/sgemm data/small/input/matrix1.txt data/small/input/matrix2t.txt output/result_small.txt
mpirun -np $NUM_PROCS ./binc/sgemm data/medium/input/matrix1.txt data/medium/input/matrix2t.txt output/result_medium.txt
