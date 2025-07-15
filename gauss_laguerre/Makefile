# Makefile

# Tools
CC      := gcc
MPI_CC  := mpicc
NVCC    := nvcc

# Flags
OMPFLAGS     := -fopenmp
MPI_LDFLAGS  := -lm
CUDA_LDFLAGS := -lm -lcufft

# Output directory
BIN_DIR := bin

.PHONY: all clean seq mpi omp cuda

all: seq mpi omp cuda

# Ensure bin/ exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Sequential
seq: | $(BIN_DIR)
	$(CC) src/seq.c   src/common.c -o $(BIN_DIR)/seq   -lm

# MPI
mpi: | $(BIN_DIR)
	$(MPI_CC) src/mpi.c  src/common.c -o $(BIN_DIR)/mpi  $(MPI_LDFLAGS)

# OpenMP
omp: | $(BIN_DIR)
	$(CC) $(OMPFLAGS) src/omp.c   src/common.c -o $(BIN_DIR)/omp   -lm

# CUDA
cuda: | $(BIN_DIR)
	$(NVCC) -O3 src/cuda.cu src/common.c -o $(BIN_DIR)/cuda $(CUDA_LDFLAGS)

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf output/*

