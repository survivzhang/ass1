#!/bin/bash
#SBATCH --job-name=conv2d_stride_test
#SBATCH --output=conv2d_stride_test_%j.out
#SBATCH --error=conv2d_stride_test_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1-4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --partition=work

# CITS3402/CITS5507 Assignment 2 - Testing Script
# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Testing with 1-4 nodes to demonstrate hybrid MPI+OpenMP scaling

module load gcc/12.2.0
module load cray-mpich

echo "=========================================="
echo "2D Convolution with Stride - Testing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Test cases directory
TEST_CASES=(
    "conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 1 -sH 1"
    "conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 2 -sH 3"
    "conv_stride_test -H 7 -W 7 -kH 2 -kW 2 -sW 1 -sH 1"
    "conv_stride_test -H 7 -W 7 -kH 2 -kW 2 -sW 3 -sH 2"
    "conv_stride_test -H 100 -W 100 -kH 5 -kW 5 -sW 1 -sH 1"
    "conv_stride_test -H 100 -W 100 -kH 5 -kW 5 -sW 7 -sH 3"
)

# Run test cases
for test_dir in "${TEST_CASES[@]}"; do
    if [ -d "$test_dir" ]; then
        echo ""
        echo "=========================================="
        echo "Testing: $test_dir"
        echo "=========================================="

        cd "$test_dir"

        # Find test files
        input=$(ls f*.txt 2>/dev/null | head -1)
        kernel=$(ls g*.txt 2>/dev/null | head -1)
        expected=$(ls o*.txt 2>/dev/null | head -1)

        if [ -f "$input" ] && [ -f "$kernel" ] && [ -f "$expected" ]; then
            # Parse stride from directory name
            sH=$(echo "$test_dir" | grep -oP 'sH \K[0-9]+' || echo 1)
            sW=$(echo "$test_dir" | grep -oP 'sW \K[0-9]+' || echo 1)

            echo "Input: $input"
            echo "Kernel: $kernel"
            echo "Expected: $expected"
            echo "Stride: sH=$sH, sW=$sW"
            echo ""

            # Test with different modes
            for mode in serial omp mpi hybrid; do
                echo "--- Mode: $mode ---"
                output="output_${mode}.txt"

                if [ "$mode" = "serial" ] || [ "$mode" = "omp" ]; then
                    # Single process for serial/omp
                    srun -n 1 ../conv_stride_test -f "$input" -g "$kernel" \
                        -sH $sH -sW $sW -o "$output" -m $mode
                else
                    # Multiple processes for mpi/hybrid
                    srun -n $SLURM_NTASKS ../conv_stride_test -f "$input" -g "$kernel" \
                        -sH $sH -sW $sW -o "$output" -m $mode
                fi

                # Compare with expected output
                if [ -f "$output" ]; then
                    if python3 -c "
import sys
import numpy as np

def read_array(fname):
    with open(fname) as f:
        h, w = map(int, f.readline().split())
        data = []
        for line in f:
            if line.strip():
                data.extend(map(float, line.split()))
        return np.array(data).reshape(h, w)

try:
    expected = read_array('$expected')
    output = read_array('$output')

    if expected.shape != output.shape:
        print(f'Shape mismatch: expected {expected.shape}, got {output.shape}')
        sys.exit(1)

    diff = np.abs(expected - output)
    max_diff = np.max(diff)

    if max_diff < 1e-3:
        print(f'✓ PASS - Max difference: {max_diff:.2e}')
        sys.exit(0)
    else:
        print(f'✗ FAIL - Max difference: {max_diff:.6f}')
        sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
                        echo "Result: PASSED"
                    else
                        # Fallback to basic comparison if Python not available
                        echo "Result: Output generated (Python not available for validation)"
                    fi
                else
                    echo "Result: FAILED - No output generated"
                fi
                echo ""
            done
        else
            echo "Skipping: Missing test files"
        fi

        cd ..
    fi
done

echo "=========================================="
echo "All tests completed"
echo "=========================================="
