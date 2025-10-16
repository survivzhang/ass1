# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 2
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)

# Check if mpicc is available
MPICC_CHECK := $(shell which mpicc 2>/dev/null)
ifeq ($(MPICC_CHECK),)
    $(info mpicc not found. Please load modules first:)
    $(info   On Kaya: module load gcc && module load openmpi)
    $(info   On Setonix: module load gcc && module load cray-mpich)
    $(info   Or use: ./build.sh)
    $(error mpicc not found. Load modules manually or use ./build.sh)
endif

# Use MPI compiler
CC = mpicc
CFLAGS = -fopenmp -O3 -Wall

# Source files
SOURCES = conv_stride_test.c conv2d.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = conv_stride_test

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(CFLAGS)

# Compile object files
%.o: %.c conv2d.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Show loaded modules
modules:
	@echo "Current modules:"
	@module list 2>/dev/null || echo "No modules loaded or module command not available"

# Show compiler info
info:
	@echo "Compiler: $(CC)"
	@echo "mpicc location: $(shell which mpicc 2>/dev/null || echo 'Not found')"
	@echo "CFLAGS: $(CFLAGS)"

# Prevent make from treating file names as targets
.PHONY: all clean modules info