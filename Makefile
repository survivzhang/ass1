# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 1 & 2
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)

# Compiler and flags
CC = gcc
MPICC = mpicc
# Common flags for both assignments: OpenMP support, optimization, warnings
CFLAGS = -fopenmp -O3 -Wall
MPIFLAGS = $(CFLAGS)  # MPI version uses same flags

# Source files
SOURCES = main.c conv2d.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = conv_test

# MPI source files
MPI_SOURCES = conv_stride_test.c conv2d.c
MPI_OBJECTS = conv_stride_test.o conv2d_mpi.o
MPI_TARGET = conv_stride_test

# Default target - build both
all: $(TARGET) $(MPI_TARGET)

# Build Assignment 1 executable (OpenMP only)
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(CFLAGS)

# Build Assignment 2 executable (MPI + OpenMP)
$(MPI_TARGET): $(MPI_OBJECTS)
	$(MPICC) $(MPI_OBJECTS) -o $(MPI_TARGET) $(MPIFLAGS)

# Compile object files for Assignment 1
%.o: %.c conv2d.h
	$(CC) $(CFLAGS) -c $< -o $@

# Compile object files for Assignment 2
conv_stride_test.o: conv_stride_test.c conv2d.h
	$(MPICC) $(MPIFLAGS) -c conv_stride_test.c -o conv_stride_test.o

conv2d_mpi.o: conv2d.c conv2d.h
	$(MPICC) $(MPIFLAGS) -c conv2d.c -o conv2d_mpi.o

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(MPI_OBJECTS) $(TARGET) $(MPI_TARGET)

# Prevent make from treating file names as targets
.PHONY: all clean