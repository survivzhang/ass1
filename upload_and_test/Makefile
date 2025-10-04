# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 2
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)

# Use MPI compiler for everything
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

# Prevent make from treating file names as targets
.PHONY: all clean