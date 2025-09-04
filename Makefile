# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 1

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp -std=c99
LDFLAGS = -lm -fopenmp

# Source files
SOURCES = main.c conv2d.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = conv_test

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile object files
%.o: %.c conv2d.h
	$(CC) $(CFLAGS) -c $< -o $@

# Debug build
debug: CFLAGS = -Wall -Wextra -g -fopenmp -std=c99 -DDEBUG
debug: clean $(TARGET)

# Performance build (maximum optimization)
performance: CFLAGS = -Wall -Wextra -O3 -march=native -fopenmp -std=c99 -DNDEBUG
performance: clean $(TARGET)

# Test with provided examples (create some test files first)
test: $(TARGET)
	@echo "Creating test input files..."
	@echo "4 4" > f_test.txt
	@echo "0.889 0.364 0.073 0.536" >> f_test.txt
	@echo "0.507 0.886 0.843 0.360" >> f_test.txt
	@echo "0.103 0.280 0.713 0.827" >> f_test.txt
	@echo "0.663 0.131 0.508 0.830" >> f_test.txt
	@echo ""
	@echo "3 3" > g_test.txt
	@echo "0.485 0.529 0.737" >> g_test.txt
	@echo "0.638 0.168 0.338" >> g_test.txt
	@echo "0.894 0.182 0.314" >> g_test.txt
	@echo ""
	@echo "Running test with example files..."
	./$(TARGET) -f f_test.txt -g g_test.txt -c
	@echo ""
	@echo "Running performance test with random arrays..."
	./$(TARGET) -H 100 -W 100 -h 5 -w 5 -c

# Quick benchmark
benchmark: $(TARGET)
	@echo "Benchmarking different array sizes..."
	@for size in 100 200 500 1000; do \
		echo "Testing $$size x $$size array with 5x5 kernel:"; \
		./$(TARGET) -H $$size -W $$size -h 5 -w 5 -c; \
		echo ""; \
	done

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install (copy to system directory - requires sudo)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build the main executable (default)"
	@echo "  debug      - Build with debug flags"
	@echo "  performance- Build with maximum optimization"
	@echo "  test       - Run basic functionality tests"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install to system directory"
	@echo "  help       - Show this help message"

# Prevent make from treating file names as targets
.PHONY: all debug performance test benchmark clean install help