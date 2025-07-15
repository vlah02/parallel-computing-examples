# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 
LD_FLAGS = -lm

tpacf: 
	$(CC) $(CC_FLAGS) args.c main.c model_compute_cpu.c model_io.c utils.c -o tpacf $(LD_FLAGS) 

clean:
	rm -f tpacf *.txt
