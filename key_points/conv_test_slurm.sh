#!/bin/bash

#SBATCH --job-name=conv_test_simple
#SBATCH --output=conv_test.out
#SBATCH --error=conv_test.err
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --mem=1G
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

# Test parameters
MATRIX_SIZE=10000
KERNEL_SIZE=3

echo ""
echo "=== Test Configuration ==="
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"

# Calculate maximum threads (CPU count * 2)
MAX_THREADS=$((SLURM_CPUS_PER_TASK * 2))

# Comprehensive thread performance testing
echo ""
echo "=== Comprehensive Thread Performance Analysis ==="
echo "Testing from 1 to $MAX_THREADS threads (CPU count * 2)"

# Array to store timing results for analysis
declare -a computing_times
declare -a speedup_values
declare -a efficiency_values
declare -a expected_times

# Test all thread counts from 1 to maximum
for threads in $(seq 1 $MAX_THREADS); do
    export OMP_NUM_THREADS=$threads
    
    # Run the convolution test and capture the computation time from program output
    output=$(./conv_test -H $MATRIX_SIZE -W $MATRIX_SIZE -h $KERNEL_SIZE -w $KERNEL_SIZE \
        -f input_test.txt -g kernel_test.txt -o output_test.txt -p 2>&1)
    
    # Extract computation time from program output
    comp_time=$(echo "$output" | grep "Parallel computation time:" | awk '{print $4}')
    computing_times[$threads]=$comp_time
    
    # Calculate speedup, efficiency and expected time (using single-thread as baseline)
    if [ $threads -eq 1 ]; then
        baseline_time=$comp_time
        speedup_values[1]=1.0
        efficiency_values[1]=1.0
        expected_times[1]=$comp_time
    else
        speedup=$(echo "scale=5; $baseline_time / $comp_time" | bc -l)
        efficiency=$(echo "scale=5; ($speedup / $threads) * 100" | bc -l)
        expected_time=$(echo "scale=5; $baseline_time / $threads" | bc -l)
        speedup_values[$threads]=$speedup
        efficiency_values[$threads]=$efficiency
        expected_times[$threads]=$expected_time
    fi
done

echo ""
echo "=== Performance Analysis Summary ==="
echo "Matrix Size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel Size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Maximum Threads: $MAX_THREADS"
echo ""
printf "%-8s %-12s %-12s %-10s %-12s\n" "Threads" "Computing(s)" "Expected(s)" "Speedup" "Efficiency(%)"
printf "%-8s %-12s %-12s %-10s %-12s\n" "-------" "--------" "--------" "-------" "------------"

for threads in $(seq 1 $MAX_THREADS); do
    printf "%-8s %-12.5f %-12.5f %-10.5f %-12.5f\n" "$threads" "${computing_times[$threads]}" "${expected_times[$threads]}" "${speedup_values[$threads]}" "${efficiency_values[$threads]}"
done

# Find optimal thread count (best efficiency > 80% or maximum speedup)
echo ""
echo "=== Optimization Analysis ==="
optimal_threads=1
best_speedup=1.0
best_efficiency_threads=1

for threads in $(seq 1 $MAX_THREADS); do
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
echo "Baseline (1 thread) computing time: ${baseline_time}s"
echo "Best computing time: ${computing_times[$optimal_threads]}s"

# Send output file as email attachment
echo "Sending results via email..."
mail -s "Convolution Performance Test Results - Job $SLURM_JOB_ID" -a conv_test.out jzguo99@outlook.com < /dev/null


