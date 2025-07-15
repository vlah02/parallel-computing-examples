#!/bin/bash

/usr/local/cuda/bin/nvcc -o binc/ccn_rule ccn_rule.cu -lm -lcufft

./binc/ccn_rule 99 -10 +10 output/ccn_o99
./binc/ccn_rule 999 -100 +100 output/ccn_o999
./binc/ccn_rule 2999 -100 +100 output/ccn_o2999
