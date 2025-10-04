#!/bin/bash
#SBATCH --job-name=conv2d_stride_test
#SBATCH --output=conv2d_stride_test_%j.out
#SBATCH --error=conv2d_stride_test_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
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

# Get the base directory (where the script and executable are located)
BASE_DIR=$(pwd)

# Test cases: directory name and parameters
declare -a TEST_DIRS=(
    "conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 1 -sH 1"
    "conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 2 -sH 3"
    "conv_stride_test -H 7 -W 7 -kH 2 -kW 2 -sW 1 -sH 1"
    "conv_stride_test -H 7 -W 7 -kH 2 -kW 2 -sW 3 -sH 2"
    "conv_stride_test -H 100 -W 100 -kH 5 -kW 5 -sW 1 -sH 1"
    "conv_stride_test -H 100 -W 100 -kH 5 -kW 5 -sW 7 -sH 3"
)

# Run test cases
for test_spec in "${TEST_DIRS[@]}"; do
    # Parse parameters from test specification
    H=$(echo "$test_spec" | grep -o '\-H [0-9]\+' | awk '{print $2}')
    W=$(echo "$test_spec" | grep -o '\-W [0-9]\+' | awk '{print $2}')
    kH=$(echo "$test_spec" | grep -o '\-kH [0-9]\+' | awk '{print $2}')
    kW=$(echo "$test_spec" | grep -o '\-kW [0-9]\+' | awk '{print $2}')
    sH=$(echo "$test_spec" | grep -o '\-sH [0-9]\+' | awk '{print $2}')
    sW=$(echo "$test_spec" | grep -o '\-sW [0-9]\+' | awk '{print $2}')

    if [ -d "$test_spec" ]; then
        echo ""
        echo "=========================================="
        echo "Testing: $test_spec"
        echo "=========================================="

        cd "$test_spec"

        # Find test files
        input=$(ls f*.txt 2>/dev/null | head -1)
        kernel=$(ls g*.txt 2>/dev/null | head -1)
        expected=$(ls o*.txt 2>/dev/null | head -1)

        if [ -f "$input" ] && [ -f "$kernel" ] && [ -f "$expected" ]; then
            echo "Input: $input"
            echo "Kernel: $kernel"
            echo "Expected: $expected"
            echo "Parsed: H=$H W=$W kH=$kH kW=$kW sH=$sH sW=$sW"

            # Validate parameters
            if [ -z "$sH" ] || [ -z "$sW" ] || [ "$sH" = "0" ] || [ "$sW" = "0" ]; then
                echo "ERROR: Invalid stride parameters sH=$sH sW=$sW"
                cd ..
                continue
            fi
            echo ""

            # Test with different modes
            for mode in serial omp mpi hybrid; do
                echo "--- Mode: $mode ---"
                output="output_${mode}.txt"

                if [ "$mode" = "serial" ] || [ "$mode" = "omp" ]; then
                    # Single process for serial/omp
                    srun -n 1 "$BASE_DIR/conv_stride_test" -f "$input" -g "$kernel" \
                        -sH $sH -sW $sW -o "$output" -m $mode 2>&1
                else
                    # Multiple processes for mpi/hybrid
                    srun -n $SLURM_NTASKS "$BASE_DIR/conv_stride_test" -f "$input" -g "$kernel" \
                        -sH $sH -sW $sW -o "$output" -m $mode 2>&1
                fi

                # Compare with expected output
                if [ -f "$output" ]; then
                    # Simple comparison: check if files have same dimensions
                    exp_dim=$(head -1 "$expected")
                    out_dim=$(head -1 "$output")

                    if [ "$exp_dim" = "$out_dim" ]; then
                        echo "Result: PASSED (dimensions match: $out_dim)"
                    else
                        echo "Result: FAILED (dimension mismatch: expected $exp_dim, got $out_dim)"
                    fi
                else
                    echo "Result: FAILED - No output generated"
                fi
                echo ""
            done
        else
            echo "Skipping: Missing test files"
        fi

        cd "$BASE_DIR"
    fi
done

echo "=========================================="
echo "All tests completed"
echo "=========================================="
