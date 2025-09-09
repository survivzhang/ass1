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
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install to system directory"
	@echo "  help       - Show this help message"

# Prevent make from treating file names as targets
.PHONY: all debug performance clean install help