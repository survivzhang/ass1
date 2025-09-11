#!/bin/bash

#SBATCH --job-name=conv_test_33_64
#SBATCH --output=conv_test_33_64.out
#SBATCH --error=conv_test_33_64.err
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=24070858@student.uwa.edu.au

echo "=== 2D Convolution Performance Test (Threads 33-64) ==="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Test parameters
MATRIX_SIZE=1000
KERNEL_SIZE=999

# Use single thread baseline from previous test (1000x999 kernel)
BASELINE_TIME=2409.242719

echo ""
echo "=== Test Configuration ==="
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Testing threads: 33-64"
echo "Baseline time (1 thread): ${BASELINE_TIME}s"

# Array to store timing results for analysis
declare -a computing_times
declare -a speedup_values
declare -a efficiency_values
declare -a expected_times

echo ""
echo "=== Thread Performance Analysis (33-64 threads) ==="

# Test thread counts from 33 to 64
for threads in $(seq 33 64); do
    export OMP_NUM_THREADS=$threads
    
    # Run the convolution test and capture the computation time from program output
    output=$(./conv_test -H $MATRIX_SIZE -W $MATRIX_SIZE -h $KERNEL_SIZE -w $KERNEL_SIZE \
        -f input_test_33_64.txt -g kernel_test_33_64.txt -o output_test_33_64.txt -p 2>&1)
    
    # Extract computation time from program output
    comp_time=$(echo "$output" | grep "Parallel computation time:" | awk '{print $4}')
    computing_times[$threads]=$comp_time
    
    # Calculate speedup, efficiency and expected time using baseline
    speedup=$(echo "scale=5; $BASELINE_TIME / $comp_time" | bc -l)
    efficiency=$(echo "scale=5; ($speedup / $threads) * 100" | bc -l)
    expected_time=$(echo "scale=5; $BASELINE_TIME / $threads" | bc -l)
    speedup_values[$threads]=$speedup
    efficiency_values[$threads]=$efficiency
    expected_times[$threads]=$expected_time
    
    echo "Completed thread $threads: ${comp_time}s (${speedup}x speedup, ${efficiency}% efficiency)"
done

echo ""
echo "=== Performance Analysis Summary (Threads 33-64) ==="
echo "Matrix Size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Kernel Size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "Thread Range: 33-64"
echo "Baseline: ${BASELINE_TIME}s"
echo ""
printf "%-8s %-12s %-12s %-10s %-12s\n" "Threads" "Computing(s)" "Expected(s)" "Speedup" "Efficiency(%)"
printf "%-8s %-12s %-12s %-10s %-12s\n" "-------" "--------" "--------" "-------" "------------"

for threads in $(seq 33 64); do
    printf "%-8s %-12.5f %-12.5f %-10.5f %-12.5f\n" "$threads" "${computing_times[$threads]}" "${expected_times[$threads]}" "${speedup_values[$threads]}" "${efficiency_values[$threads]}"
done

# Find optimal thread count in 33-64 range
echo ""
echo "=== Optimization Analysis (Threads 33-64) ==="
optimal_threads=33
best_speedup=0.0
best_efficiency_threads=33

for threads in $(seq 33 64); do
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

echo "Optimal thread count for maximum speedup (33-64): $optimal_threads threads (${best_speedup}x speedup)"
echo "Best thread count with >80% efficiency (33-64): $best_efficiency_threads threads"
echo "Best computing time (33-64): ${computing_times[$optimal_threads]}s"

# Send output file as email attachment
echo ""
echo "Sending results (33-64 threads) via email..."
mail -s "Convolution Performance Test Results 33-64 Threads - Job $SLURM_JOB_ID" -a conv_test_33_64.out 24070858@student.uwa.edu.au < /dev/null

echo ""
echo "Thread range 33-64 test completed successfully!"
