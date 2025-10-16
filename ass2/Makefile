# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 2
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)

# Try different approaches to find MPI
MPICC_CHECK := $(shell which mpicc 2>/dev/null)
ifneq ($(MPICC_CHECK),)
    # mpicc found, use it
    CC = mpicc
    CFLAGS = -fopenmp -O3 -Wall
    LDFLAGS = 
else
    # mpicc not found, try gcc with common MPI paths
    CC = gcc
    CFLAGS = -fopenmp -O3 -Wall
    
    # Try to find MPI using pkg-config first
    MPI_PKG_INCLUDE := $(shell pkg-config --cflags mpi 2>/dev/null)
    MPI_PKG_LIBS := $(shell pkg-config --libs mpi 2>/dev/null)
    
    ifneq ($(MPI_PKG_INCLUDE),)
        # pkg-config found MPI
        CFLAGS += $(MPI_PKG_INCLUDE)
        LDFLAGS = $(MPI_PKG_LIBS) -lomp
    else
        # Auto-detect MPI paths
        MPI_H_PATH := $(shell find /usr /opt /usr/local -name "mpi.h" 2>/dev/null | head -1)
        MPI_LIB_PATH := $(shell find /usr /opt /usr/local -name "libmpi.so*" 2>/dev/null | head -1 | xargs dirname)
        MPI_BIN_PATH := $(shell find /usr /opt /usr/local -name "mpirun" 2>/dev/null | head -1 | xargs dirname)
        
        ifneq ($(MPI_H_PATH),)
            MPI_INCLUDE := -I$(dir $(MPI_H_PATH))
            CFLAGS += $(MPI_INCLUDE)
        endif
        
        ifneq ($(MPI_LIB_PATH),)
            MPI_LIBS := -L$(MPI_LIB_PATH) -Wl,-rpath,$(MPI_LIB_PATH) -lmpi
            LDFLAGS = $(MPI_LIBS)
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
	$(CC) $(OBJECTS) -o $(TARGET) $(CFLAGS) $(LDFLAGS)


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
	@echo "mpicc found: $(if $(MPICC_CHECK),yes,no)"

# Test if MPI headers can be found
test-mpi:
	@echo "Testing MPI header availability..."
	@echo '#include <mpi.h>' | $(CC) $(CFLAGS) -x c - -c -o /dev/null 2>/dev/null && echo "MPI headers found!" || echo "MPI headers not found"

# Prevent make from treating file names as targets
.PHONY: all clean info test-mpi