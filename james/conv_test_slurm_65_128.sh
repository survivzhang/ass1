#!/bin/bash

#SBATCH --job-name=conv_test_65_128
#SBATCH --output=conv_test_65_128.out
#SBATCH --error=conv_test_65_128.err
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jzguo99@outlook.com

echo "=== 2D Convolution Performance Test (Threads 65-128) ==="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Test parameters - will be read from input files
# Use single thread baseline from previous test
BASELINE_TIME=330.950020

echo ""
echo "=== Test Configuration ==="
echo "Input file: input.txt"
echo "Kernel file: kernel.txt"
echo "Testing threads: 65-128"
echo "Baseline time (1 thread): ${BASELINE_TIME}s"

# Array to store timing results for analysis
declare -a computing_times
declare -a speedup_values
declare -a efficiency_values
declare -a expected_times

echo ""
echo "=== Thread Performance Analysis (65-128 threads) ==="

# Test thread counts from 65 to 128
for threads in $(seq 65 128); do
    export OMP_NUM_THREADS=$threads
    
    # Run the convolution test and capture the computation time from program output
    output=$(./conv_test -f input.txt -g kernel.txt -p 2>&1)
    
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
done

echo ""
echo "=== Performance Analysis Summary (Threads 65-128) ==="
echo "Input file: input.txt"
echo "Kernel file: kernel.txt"
echo "Thread Range: 65-128"
echo "Baseline: ${BASELINE_TIME}s"
echo ""
printf "%-8s %-12s %-12s %-10s %-12s\n" "Threads" "Computing(s)" "Expected(s)" "Speedup" "Efficiency(%)"
printf "%-8s %-12s %-12s %-10s %-12s\n" "-------" "--------" "--------" "-------" "------------"

for threads in $(seq 65 128); do
    printf "%-8s %-12.5f %-12.5f %-10.5f %-12.5f\n" "$threads" "${computing_times[$threads]}" "${expected_times[$threads]}" "${speedup_values[$threads]}" "${efficiency_values[$threads]}"
done

# Find optimal thread count in 65-128 range
echo ""
echo "=== Optimization Analysis (Threads 65-128) ==="
optimal_threads=65
best_speedup=0.0
best_efficiency_threads=65

for threads in $(seq 65 128); do
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

echo "Optimal thread count for maximum speedup (65-128): $optimal_threads threads (${best_speedup}x speedup)"
echo "Best thread count with >80% efficiency (65-128): $best_efficiency_threads threads"
echo "Best computing time (65-128): ${computing_times[$optimal_threads]}s"

# Send output file as email attachment
echo ""
echo "Sending results (65-128 threads) via email..."
mail -s "Convolution Performance Test Results 65-128 Threads - Job $SLURM_JOB_ID" -a conv_test_65_128.out jzguo99@outlook.com < /dev/null

echo ""
echo "Thread range 65-128 test completed successfully!"
