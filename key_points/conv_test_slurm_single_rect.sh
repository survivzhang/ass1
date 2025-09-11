#!/bin/bash

#SBATCH --job-name=conv_test_single_rect
#SBATCH --output=conv_test_single_rect.out
#SBATCH --error=conv_test_single_rect.err
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzcnhy@gmail.com

echo "=== 2D Convolution Single Thread Performance Test (Rectangular Matrix) ==="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Test parameters for rectangular matrix
MATRIX_HEIGHT=10000
MATRIX_WIDTH=1000
KERNEL_HEIGHT=999
KERNEL_WIDTH=999

echo ""
echo "=== Single Thread Test Configuration (Rectangular) ==="
echo "Matrix size: ${MATRIX_HEIGHT}x${MATRIX_WIDTH} (Height x Width)"
echo "Kernel size: ${KERNEL_HEIGHT}x${KERNEL_WIDTH} (Height x Width)"
echo "Using single thread only"

# Force single thread
export OMP_NUM_THREADS=1

echo ""
echo "=== Running Single Thread Convolution (Rectangular) ==="

# Run the convolution test and capture the computation time from program output
output=$(./conv_test -H $MATRIX_HEIGHT -W $MATRIX_WIDTH -h $KERNEL_HEIGHT -w $KERNEL_WIDTH \
    -f input_test_single_rect.txt -g kernel_test_single_rect.txt -o output_test_single_rect.txt -p 2>&1)

# Extract computation time from program output
single_comp_time=$(echo "$output" | grep "Parallel computation time:" | awk '{print $4}')

echo ""
echo "=== Single Thread Performance Results (Rectangular) ==="
echo "Matrix Size: ${MATRIX_HEIGHT}x${MATRIX_WIDTH} (Height x Width)"
echo "Kernel Size: ${KERNEL_HEIGHT}x${KERNEL_WIDTH} (Height x Width)"
echo "Single Thread Computing Time: ${single_comp_time}s"
echo "Baseline established for future parallel comparisons with rectangular matrices"

# Send output file as email attachment
echo ""
echo "Sending single thread rectangular results via email..."
mail -s "Single Thread Rectangular Convolution Test Results - Job $SLURM_JOB_ID" -a conv_test_single_rect.out zzcnhy@gmail.com < /dev/null

echo ""
echo "Single thread rectangular test completed successfully!"
