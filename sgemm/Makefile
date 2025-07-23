CXX     := g++
MPI_CC  := mpicxx
NVCC    := nvcc

OMPFLAGS     := -fopenmp
MPI_LDFLAGS  := -lm
CUDA_LDFLAGS := -lm -lcufft
CUDA_WARNFLAGS := -diag-suppress=1650 -Xcompiler "-Wno-unused-result"

BIN_DIR := bin

.PHONY: all clean seq mpi omp cuda

all: seq mpi omp cuda

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

seq: | $(BIN_DIR)
	$(CXX) src/seq.cpp src/common.cpp -o $(BIN_DIR)/seq -lm

mpi: | $(BIN_DIR)
	$(MPI_CC) src/mpi.cpp src/common.cpp -o $(BIN_DIR)/mpi $(MPI_LDFLAGS)

omp: | $(BIN_DIR)
	$(CXX) $(OMPFLAGS) src/omp.cpp src/common.cpp -o $(BIN_DIR)/omp -lm

cuda: | $(BIN_DIR)
	$(NVCC) -O3 $(CUDA_WARNFLAGS) src/cuda.cu src/common.cpp -o $(BIN_DIR)/cuda $(CUDA_LDFLAGS)

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf output/*
