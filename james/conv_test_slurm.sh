#!/bin/bash

#SBATCH --job-name=conv_test_simple
#SBATCH --output=conv_test.out
#SBATCH --error=conv_test.err
#SBATCH --cpus-per-task=64
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jzguo99@outlook.com

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
MATRIX_SIZE=100
KERNEL_SIZE=3

echo ""
echo "=== Test Configuration ==="
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"

# Comprehensive thread performance testing
echo ""
echo "=== Comprehensive Thread Performance Analysis ==="
echo "Testing from 1 to $SLURM_CPUS_PER_TASK threads"

# Array to store timing results for analysis
declare -a execution_times
declare -a speedup_values
declare -a efficiency_values

# Test all thread counts from 1 to maximum
for threads in $(seq 1 $SLURM_CPUS_PER_TASK); do
    echo ""
    echo "--- Testing with $threads thread(s) ---"
    export OMP_NUM_THREADS=$threads
    
    # Start timing
    start_time=$(date +%s.%N)
    
    # Run the convolution test (all using same files for comparison)
    ./conv_test -H $MATRIX_SIZE -W $MATRIX_SIZE -h $KERNEL_SIZE -w $KERNEL_SIZE \
        -f input_test.txt -g kernel_test.txt -o output_test.txt -p 2>/dev/null
    
    # End timing
    end_time=$(date +%s.%N)
    
    # Calculate execution time
    exec_time=$(echo "$end_time - $start_time" | bc -l)
    execution_times[$threads]=$exec_time
    
    echo "Threads: $threads, Execution Time: ${exec_time}s"
    
    # Calculate speedup and efficiency (using single-thread as baseline)
    if [ $threads -eq 1 ]; then
        baseline_time=$exec_time
        speedup_values[1]=1.0
        efficiency_values[1]=1.0
        echo "Speedup: 1.0x, Efficiency: 100.0%"
    else
        speedup=$(echo "scale=3; $baseline_time / $exec_time" | bc -l)
        efficiency=$(echo "scale=3; ($speedup / $threads) * 100" | bc -l)
        speedup_values[$threads]=$speedup
        efficiency_values[$threads]=$efficiency
        echo "Speedup: ${speedup}x, Efficiency: ${efficiency}%"
    fi
done

echo ""
echo "=== Performance Analysis Summary ==="
echo "Matrix Size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel Size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Maximum Threads: $SLURM_CPUS_PER_TASK"
echo ""
printf "%-8s %-12s %-10s %-12s\n" "Threads" "Time(s)" "Speedup" "Efficiency(%)"
printf "%-8s %-12s %-10s %-12s\n" "-------" "--------" "-------" "------------"

for threads in $(seq 1 $SLURM_CPUS_PER_TASK); do
    printf "%-8s %-12.3f %-10.3f %-12.2f\n" "$threads" "${execution_times[$threads]}" "${speedup_values[$threads]}" "${efficiency_values[$threads]}"
done

# Find optimal thread count (best efficiency > 80% or maximum speedup)
echo ""
echo "=== Optimization Analysis ==="
optimal_threads=1
best_speedup=1.0
best_efficiency_threads=1

for threads in $(seq 1 $SLURM_CPUS_PER_TASK); do
    current_speedup=${speedup_values[$threads]}
    current_efficiency=${efficiency_values[$threads]}
    
    # Check if this gives better speedup
    if (( $(echo "$current_speedup > $best_speedup" | bc -l) )); then
        best_speedup=$current_speedup
        optimal_threads=$threads
    fi
    
    # Check if efficiency is still good (>80%)
    if (( $(echo "$current_efficiency > 80" | bc -l) )); then
        best_efficiency_threads=$threads
    fi
done

echo "Optimal thread count for maximum speedup: $optimal_threads threads (${best_speedup}x speedup)"
echo "Best thread count with >80% efficiency: $best_efficiency_threads threads"
echo "Baseline (1 thread) execution time: ${baseline_time}s"
echo "Best execution time: ${execution_times[$optimal_threads]}s"

echo "All performance tests completed."


echo ""
echo "=== All Tests Completed ==="
echo "Job finished at: $(date)"

# Clean up generated files (optional)
echo "Cleaning up temporary files..."
rm -f input_test.txt kernel_test.txt output_test.txt

echo "Job completed successfully!"
