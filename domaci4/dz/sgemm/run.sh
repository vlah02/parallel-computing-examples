#!/bin/bash

nvcc -o binc/sgemm sgemm.cu -lm
./binc/sgemm data/small/input/matrix1.txt data/small/input/matrix2t.txt output/result_small.txt
./binc/sgemm data/medium/input/matrix1.txt data/medium/input/matrix2t.txt output/result_medium.txt
