#!/bin/bash

#SBATCH --job-name=conv_test_simple
#SBATCH --output=conv_test.out
#SBATCH --error=conv_test.err
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jzguo99@outlook.com


echo "=== 2D Convolution Performance Test ==="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Test parameters
MATRIX_HEIGHT=1000
MATRIX_WIDTH=2000
KERNEL_HEIGHT=3
KERNEL_WIDTH=5

echo ""
echo "=== Test Configuration ==="
echo "Matrix size: ${MATRIX_HEIGHT}x${MATRIX_WIDTH}"
echo "Kernel size: ${KERNEL_HEIGHT}x${KERNEL_WIDTH}"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"

# Calculate maximum threads (CPU count * 2)
MAX_THREADS=$((SLURM_CPUS_PER_TASK * 2))

# Use built-in performance analysis
echo ""
echo "=== Running Built-in Performance Analysis ==="
echo "Using program's built-in analysis function (-a parameter)"

# Run the built-in performance analysis and save to file
./conv_test -H $MATRIX_HEIGHT -W $MATRIX_WIDTH -h $KERNEL_HEIGHT -w $KERNEL_WIDTH -a > performance_results.txt