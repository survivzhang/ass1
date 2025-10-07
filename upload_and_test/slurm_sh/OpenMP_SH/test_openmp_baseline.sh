#!/bin/bash
#SBATCH --job-name=openmp_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --output=openmp_baseline_%j.out
#SBATCH --error=openmp_baseline_%j.err
#SBATCH --partition=work
#SBATCH --account=courses0101

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# OpenMP Test - Single Node with varying thread counts (baseline: stride 1,1)

module load gcc/12.2.0

# Change to working directory
cd /scratch/courses0101/zzhang5/upload_and_test || exit 1

echo "=========================================="
echo "OpenMP Baseline Test - Single Node"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Testing stride (1,1) with different thread counts"
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
        if [ $threads -eq 1 ]; then
            echo "[$threads thread] ${CONFIG_NAMES[$i]}"
            ./conv_stride_test -H $H -W $W -kH $kH -kW $kW -sH 1 -sW 1 -m serial
        else
            echo "[$threads threads] ${CONFIG_NAMES[$i]}"
            ./conv_stride_test -H $H -W $W -kH $kH -kW $kW -sH 1 -sW 1 -m omp
        fi
    done
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
