#!/bin/bash
# Build script for 2D Convolution Assignment
# CITS3402/CITS5507 - Assignment 2
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)

echo "=========================================="
echo "Building 2D Convolution with Stride"
echo "=========================================="

# Detect system and try to load modules
echo "System: $(hostname)"
echo "Checking available modules..."

# Try different GCC versions (based on Kaya availability)
echo "Trying GCC modules..."
if module load gcc/12.4.0 2>/dev/null; then
    echo "Loaded: gcc/12.4.0"
elif module load gcc/14.3.0 2>/dev/null; then
    echo "Loaded: gcc/14.3.0"
elif module load gcc/11.5.0 2>/dev/null; then
    echo "Loaded: gcc/11.5.0"
elif module load gcc 2>/dev/null; then
    echo "Loaded: gcc (default)"
else
    echo "Warning: Could not load GCC module, using system GCC"
fi

# Try different MPI implementations (based on Kaya availability)
echo "Trying MPI modules..."
if module load openmpi/5.0.5 2>/dev/null; then
    echo "Loaded: openmpi/5.0.5"
elif module load mpi/2021.8.0 2>/dev/null; then
    echo "Loaded: mpi/2021.8.0"
elif module load mpi/latest 2>/dev/null; then
    echo "Loaded: mpi/latest"
elif module load cray-mpich 2>/dev/null; then
    echo "Loaded: cray-mpich"
else
    echo "Error: Could not load any MPI module"
    echo ""
    echo "Available modules on this system:"
    module avail 2>&1 | head -20
    echo ""
    echo "Please check available modules with:"
    echo "  module spider gcc"
    echo "  module spider mpi"
    echo "  module spider openmpi"
    exit 1
fi

echo ""
echo "Current modules:"
module list

echo ""
echo "Building..."
make clean
make

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Build successful!"
    echo "Executable: conv_stride_test"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Build failed!"
    echo "Check module loading and try again"
    echo "=========================================="
    exit 1
fi
