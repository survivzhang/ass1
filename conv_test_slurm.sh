#!/bin/bash

#SBATCH --job-name=conv_test_simple
#SBATCH --output=conv_test.out
#SBATCH --error=conv_test.err
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzcnhy@gmail.com

# Load any required modules (adjust as needed for your system)
# module load gcc/9.3.0
# module load openmpi/4.0.3

echo "=== 2D Convolution Performance Test ==="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Compile the program
echo "Compiling conv_test..."
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Test parameters
MATRIX_SIZE=1000
KERNEL_SIZE=99

echo ""
echo "=== Test Configuration ==="
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"

# Test 1: Single-threaded performance
echo ""
echo "=== Test 1: Single-threaded Performance ==="
export OMP_NUM_THREADS=1
echo "Running with 1 thread..."

./conv_test -H $MATRIX_SIZE -W $MATRIX_SIZE -h $KERNEL_SIZE -w $KERNEL_SIZE \
    -f input_single.txt -g kernel_single.txt -o output_single.txt -s

echo "Single-threaded test completed."

# Test 2: Multi-threaded performance with different thread counts
echo ""
echo "=== Test 2: Multi-threaded Performance (2, 4, 6, 8, 16 threads) ==="

# Test different thread counts
for threads in 2 4 6 8 16; do
    echo ""
    echo "--- Testing with $threads threads ---"
    export OMP_NUM_THREADS=$threads
    echo "Running with $threads threads..."
    
    ./conv_test -H $MATRIX_SIZE -W $MATRIX_SIZE -h $KERNEL_SIZE -w $KERNEL_SIZE \
        -f input_${threads}threads.txt -g kernel_${threads}threads.txt -o output_${threads}threads.txt -p
    
    echo "Test with $threads threads completed."
done

echo "All multi-threaded tests completed."


echo ""
echo "=== All Tests Completed ==="
echo "Job finished at: $(date)"

# Clean up generated files (optional)
echo "Cleaning up temporary files..."
rm -f input_*.txt kernel_*.txt output_*.txt

echo "Job completed successfully!"
