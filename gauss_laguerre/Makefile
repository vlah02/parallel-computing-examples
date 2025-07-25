GNU := gcc
MPI := mpicc
CUDA := nvcc
OMP := gcc -fopenmp

SEQ_LDFLAGS := -lm
OMP_LDFLAGS := -lm -lfftw3
MPI_LDFLAGS := -lm -lfftw3
CUDA_LDFLAGS := -lm -lcufft -diag-suppress=1650 -Xcompiler "-Wno-unused-result"

CMN_DIR := src/common.c
SRC_DIR := src
BIN_DIR := bin

.PHONY: all clean seq mpi omp cuda

all: seq mpi omp cuda

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

seq: | $(BIN_DIR)
	$(GNU) $(SRC_DIR)/seq.c $(CMN_DIR) -o $(BIN_DIR)/seq $(SEQ_LDFLAGS)

mpi: | $(BIN_DIR)
	$(MPI) $(SRC_DIR)/mpi.c $(CMN_DIR) -o $(BIN_DIR)/mpi $(MPI_LDFLAGS)

omp: | $(BIN_DIR)
	$(OMP) $(SRC_DIR)/omp.c $(CMN_DIR) -o $(BIN_DIR)/omp $(OMP_LDFLAGS)

cuda: | $(BIN_DIR)
	$(CUDA) $(SRC_DIR)/cuda.cu $(CMN_DIR) -o $(BIN_DIR)/cuda $(CUDA_LDFLAGS)

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf output/*

