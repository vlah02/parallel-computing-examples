# Tools
CXX     := g++
MPI_CC  := mpicc
NVCC    := nvcc

# Flags
OMPFLAGS     := -fopenmp
MPI_LDFLAGS  := -lm
CUDA_LDFLAGS := -lm -lcufft
CUDA_WARNFLAGS := -diag-suppress=1650 -Xcompiler "-Wno-unused-result"

# Output directory
BIN_DIR := bin

.PHONY: all clean seq mpi omp cuda

all: seq mpi omp cuda

# Ensure bin/ exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Sequential
seq: | $(BIN_DIR)
	$(CXX) src/seq.c src/common.cpp -o $(BIN_DIR)/seq -lm

# MPI
mpi: | $(BIN_DIR)
	$(MPI_CC) src/mpi.c src/common.c -o $(BIN_DIR)/mpi $(MPI_LDFLAGS)

# OpenMP
omp: | $(BIN_DIR)
	$(CXX) $(OMPFLAGS) src/omp.c src/common.c -o $(BIN_DIR)/omp -lm

# CUDA
cuda: | $(BIN_DIR)
	$(NVCC) -O3 $(CUDA_WARNFLAGS) src/cuda.cu src/common.cpp -o $(BIN_DIR)/cuda $(CUDA_LDFLAGS)

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf output/*
