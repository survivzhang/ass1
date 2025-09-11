#!/bin/bash

#SBATCH --job-name=conv_test_single
#SBATCH --output=conv_test_single.out
#SBATCH --error=conv_test_single.err
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jzguo99@outlook.com

echo "=== 2D Convolution Single Thread Performance Test ==="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Test parameters
MATRIX_SIZE=100000
KERNEL_SIZE=3

echo ""
echo "=== Single Thread Test Configuration ==="
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Using single thread only"

# Force single thread
export OMP_NUM_THREADS=1

echo ""
echo "=== Running Single Thread Convolution ==="

# Run the convolution test and capture the computation time from program output
output=$(./conv_test -H $MATRIX_SIZE -W $MATRIX_SIZE -h $KERNEL_SIZE -w $KERNEL_SIZE \
    -f input_test_single.txt -g kernel_test_single.txt -o output_test_single.txt -p 2>&1)

# Extract computation time from program output
single_comp_time=$(echo "$output" | grep "Parallel computation time:" | awk '{print $4}')

echo ""
echo "=== Single Thread Performance Results ==="
echo "Matrix Size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel Size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Single Thread Computing Time: ${single_comp_time}s"
echo "Baseline established for future parallel comparisons"

# Send output file as email attachment
echo ""
echo "Sending single thread results via email..."
mail -s "Single Thread Convolution Test Results - Job $SLURM_JOB_ID" -a conv_test_single.out jzguo99@outlook.com < /dev/null

echo ""
echo "Single thread test completed successfully!"
