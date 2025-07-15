# Makefile for compiling and linking ccn_rule with OpenMP

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O2 -fopenmp  # OpenMP flag for compiling

# Linker flags
LDFLAGS = -fopenmp           # OpenMP flag for linking

# Libraries to link
LIBS = -lm

# Target executable
TARGET = ./binc/ccn_rule

# Object files
OBJS = ccn_rule.o

# Default target
all: $(TARGET)

# Compile source files into object files
%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@
	@if [ $$? -ne 0 ]; then echo "Compile error."; exit 1; fi

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LIBS) $(LDFLAGS) -o $(TARGET)
	@if [ $$? -ne 0 ]; then echo "Load error."; exit 1; fi
	rm -f $(OBJS)
	chmod ugo+x $(TARGET)

# Clean up object files
clean:
	rm -f $(OBJS)

# Clean everything, including the executable and any txt files
cleanall: clean
	rm -f $(TARGET)
	rm -r -f *.txt

# Phony targets
.PHONY: all clean cleanall
