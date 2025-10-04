# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 1
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)

# Compiler and flags
CC = gcc
CFLAGS = -fopenmp

# Source files
SOURCES = main.c conv2d.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = conv_test

# Default target
all: $(TARGET)

# Build the main executable
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