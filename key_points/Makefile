# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 1

# Compiler and flags
CC = gcc
CFLAGS = -fopenmp
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


# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install (copy to system directory - requires sudo)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Performance analysis with different thread counts
analyze: $(TARGET)
	@echo "Running performance analysis with f0.txt and g0.txt..."
	@if [ -f f0.txt ] && [ -f g0.txt ]; then \
		./$(TARGET) -f f0.txt -g g0.txt -a; \
	else \
		echo "f0.txt or g0.txt not found, using random data..."; \
		./$(TARGET) -H 100 -W 100 -h 3 -w 3 -a; \
	fi

# Performance analysis with large random data
analyze-large: $(TARGET)
	@echo "Running performance analysis with large random data..."
	./$(TARGET) -H 500 -W 500 -h 5 -w 5 -a

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build the main executable (default)"
	@echo "  debug      - Build with debug flags"
	@echo "  performance- Build with maximum optimization"
	@echo "  analyze    - Run performance analysis with f0.txt/g0.txt"
	@echo "  analyze-large - Run performance analysis with large random data"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install to system directory"
	@echo "  help       - Show this help message"

# Prevent make from treating file names as targets
.PHONY: all debug performance analyze analyze-large clean install help