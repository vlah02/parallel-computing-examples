#!/bin/bash

BIN="./bin/cuda"
ARGS="2999 -100 100 output/cuda/ccn_o2999/ccn_o2999"

mkdir -p output/cuda/profiler

echo "Running Nsight Systems profile..."
nsys profile --trace=cuda,nvtx --stats=true -o cuda_profile $BIN $ARGS > output/cuda/profiler/cuda_profile.txt 2>&1
rm -f cuda_profile.nsys-rep cuda_profile.sqlite

echo "Running Nsight Compute profile..."
sudo ncu --set full --target-processes all --export cuda_kernel_report $BIN $ARGS
ncu --import cuda_kernel_report.ncu-rep --page details > output/cuda/profiler/cuda_kernel_report.txt
rm -f cuda_kernel_report.ncu-rep

echo "All runs complete!"
