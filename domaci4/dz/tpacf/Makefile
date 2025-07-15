# C compiler
CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = -O2 -diag-suppress 2464
LD_FLAGS = -lm

tpacf: args.cu main.cu model_compute_cpu.cu model_io.cu utils.cu
	$(CC) $(CC_FLAGS) args.cu main.cu model_compute_cpu.cu model_io.cu utils.cu -o tpacf $(LD_FLAGS) 

clean:
	rm -f tpacf *.txt
