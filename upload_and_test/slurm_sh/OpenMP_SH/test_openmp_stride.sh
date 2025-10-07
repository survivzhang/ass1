#!/bin/bash
#SBATCH --job-name=openmp_stride
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --output=openmp_stride_%j.out
#SBATCH --error=openmp_stride_%j.err
#SBATCH --partition=work
#SBATCH --account=courses0101

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# OpenMP Test - Single Node with varying thread counts and stride values
# Testing the effect of stride on computational cost

module load gcc/12.2.0

# Change to working directory
cd /scratch/courses0101/zzhang5/upload_and_test || exit 1

echo "=========================================="
echo "OpenMP Stride Test - Single Node"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Testing different stride values with different thread counts"
echo "=========================================="

# Test configurations: Matrix sizes
CONFIGS=(
    "100 100 3 3"      # 100x100, kernel 3x3
    "1000 1000 3 3"    # 1000x1000, kernel 3x3
    "10000 10000 3 3"  # 10000x10000, kernel 3x3
    "1000 1000 10 10"  # 1000x1000, kernel 10x10
    "1000 1000 100 100" # 1000x1000, kernel 100x100
)

CONFIG_NAMES=(
    "100x100 matrix, kernel 3x3"
    "1000x1000 matrix, kernel 3x3"
    "10000x10000 matrix, kernel 3x3"
    "1000x1000 matrix, kernel 10x10"
    "1000x1000 matrix, kernel 100x100"
)

# Stride values to test
STRIDES=(
    "2 2"    # stride (2,2)
    "3 3"    # stride (3,3)
    "5 5"    # stride (5,5)
    "10 10"  # stride (10,10)
)

STRIDE_NAMES=(
    "stride (2,2)"
    "stride (3,3)"
    "stride (5,5)"
    "stride (10,10)"
)

# Thread counts to test (1 = serial)
THREADS=(1 2 4 8 16 32 64 96)

# Test each thread count
for threads in "${THREADS[@]}"; do
    echo ""
    echo "=========================================="
    if [ $threads -eq 1 ]; then
        echo "Testing with $threads thread (Serial)"
    else
        echo "Testing with $threads threads (OpenMP)"
    fi
    echo "=========================================="

    export OMP_NUM_THREADS=$threads

    # Test each matrix configuration
    for i in "${!CONFIGS[@]}"; do
        config=(${CONFIGS[$i]})
        H=${config[0]}
        W=${config[1]}
        kH=${config[2]}
        kW=${config[3]}

        echo ""
        echo "--- ${CONFIG_NAMES[$i]} ---"

        # Test each stride value
        for s in "${!STRIDES[@]}"; do
            stride=(${STRIDES[$s]})
            sH=${stride[0]}
            sW=${stride[1]}

            echo ""
            if [ $threads -eq 1 ]; then
                echo "[$threads thread] ${CONFIG_NAMES[$i]}, ${STRIDE_NAMES[$s]}"
                ./conv_stride_test -H $H -W $W -kH $kH -kW $kW -sH $sH -sW $sW -m serial
            else
                echo "[$threads threads] ${CONFIG_NAMES[$i]}, ${STRIDE_NAMES[$s]}"
                ./conv_stride_test -H $H -W $W -kH $kH -kW $kW -sH $sH -sW $sW -m omp
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
