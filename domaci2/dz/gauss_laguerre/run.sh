#!/bin/bash

NUM_PROCS=${1:-8}

mpic++ -lm -o binc/ccn_rule ccn_rule.c
mpirun -np $NUM_PROCS ./binc/ccn_rule 99 -10 +10 output/ccn_o99
mpirun -np $NUM_PROCS ./binc/ccn_rule 999 -100 +100 output/ccn_o999
mpirun -np $NUM_PROCS ./binc/ccn_rule 2999 -100 +100 output/ccn_o2999
