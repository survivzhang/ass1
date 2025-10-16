# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 2
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)

# Use full path to mpicc (Kaya system)
KAYA_MPICC := /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicc

# Check if Kaya mpicc exists
ifneq ($(wildcard $(KAYA_MPICC)),)
    # On Kaya, use full path directly
    CC = $(KAYA_MPICC)
    CFLAGS = -fopenmp -O3 -Wall
    LDFLAGS = -fopenmp
else
    # Try system mpicc
    MPICC_CHECK := $(shell which mpicc 2>/dev/null)
    ifneq ($(MPICC_CHECK),)
        CC = mpicc
        CFLAGS = -fopenmp -O3 -Wall
        LDFLAGS = -fopenmp
    else
        # Use gcc as final fallback
        CC = gcc
        CFLAGS = -fopenmp -O3 -Wall
        
        # Try pkg-config
        MPI_PKG_INCLUDE := $(shell pkg-config --cflags mpi 2>/dev/null)
        MPI_PKG_LIBS := $(shell pkg-config --libs mpi 2>/dev/null)
        
        ifneq ($(MPI_PKG_INCLUDE),)
            CFLAGS += $(MPI_PKG_INCLUDE)
            LDFLAGS = $(MPI_PKG_LIBS) -fopenmp
        else
            # Manually add common MPI paths
            MPI_INCLUDE := -I/usr/include/mpi -I/usr/local/include
            MPI_LIBS := -L/usr/lib -L/usr/local/lib -lmpi
            CFLAGS += $(MPI_INCLUDE)
            LDFLAGS = $(MPI_LIBS) -fopenmp
        endif
    endif
endif

# Source files
SOURCES = conv_stride_test.c conv2d.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = conv_stride_test

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile object files
%.o: %.c conv2d.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Show compiler info
info:
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"

.PHONY: all clean info