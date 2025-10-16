#!/bin/bash
# Wrapper script to load modules and run make
# CITS3402/CITS5507 - Assignment 2

echo "Loading modules and building..."

# Load modules based on system
if [[ $(hostname) == *"kaya"* ]]; then
    echo "Loading Kaya modules..."
    module load gcc/12.4.0 2>/dev/null || module load gcc 2>/dev/null
    module load openmpi/5.0.5 2>/dev/null || module load openmpi 2>/dev/null
elif [[ $(hostname) == *"setonix"* ]]; then
    echo "Loading Setonix modules..."
    module load gcc/12.2.0 2>/dev/null || module load gcc 2>/dev/null
    module load cray-mpich 2>/dev/null
else
    echo "Loading default modules..."
    module load gcc 2>/dev/null
    module load openmpi 2>/dev/null || module load cray-mpich 2>/dev/null
fi

# Check if mpicc is now available
if ! which mpicc >/dev/null 2>&1; then
    echo "Error: mpicc still not found after loading modules"
    echo "Please load modules manually:"
    echo "  module load gcc"
    echo "  module load openmpi"
    exit 1
fi

echo "Modules loaded successfully. Building..."
make "$@"
