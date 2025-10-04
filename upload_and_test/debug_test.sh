#!/bin/bash
# Quick debug test script

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=1

echo "Testing conv_stride_test with simple case..."
echo "Current directory: $(pwd)"
echo "Executable exists: $(ls -l conv_stride_test 2>&1)"

cd "conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 1 -sH 1"
echo "Input file:"
head -3 f0.txt
echo "Kernel file:"
head -3 g0.txt

echo ""
echo "Running serial mode with debugging..."
srun -n 1 ../conv_stride_test -f f0.txt -g g0.txt -sH 1 -sW 1 -o test_output.txt -m serial

echo "Exit code: $?"
echo "Output file exists: $(ls -l test_output.txt 2>&1)"
