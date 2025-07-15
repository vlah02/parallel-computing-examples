# C compiler
CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = -fopenmp -O2

prime: sgemm.cu 
	$(CC) sgemm.cu -o sgemm

clean:
	rm -f sgemm result*
