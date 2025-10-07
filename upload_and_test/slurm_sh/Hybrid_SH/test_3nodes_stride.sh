#!/bin/bash
#SBATCH --job-name=hybrid_3nodes_stride
#SBATCH --nodes=3
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=hybrid_3nodes_stride_%j.out
#SBATCH --error=hybrid_3nodes_stride_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid MPI+OpenMP Test - 3 Nodes with different stride values
# Testing the effect of stride on computational and communication cost

module load gcc/12.2.0
module load cray-mpich

echo "=========================================="
echo "Hybrid MPI+OpenMP Stride Test - 3 Nodes"
echo "=========================================="
echo "Job started at: $(date)"
echo "Nodes: 3"
echo "Mode: Hybrid (MPI + OpenMP)"
echo "Testing different stride values"
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

# Core counts to test
CORES=(3 6 12 24 48 96)

# Test each core count
for cores in "${CORES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing with $cores cores (Hybrid)"
    echo "=========================================="

    # Calculate optimal MPI tasks and OpenMP threads
    if [ $cores -eq 3 ]; then
        NTASKS=3
        THREADS=1
    elif [ $cores -eq 6 ]; then
        NTASKS=3
        THREADS=2
    elif [ $cores -eq 12 ]; then
        NTASKS=6
        THREADS=2
    elif [ $cores -eq 24 ]; then
        NTASKS=6
        THREADS=4
    elif [ $cores -eq 48 ]; then
        NTASKS=12
        THREADS=4
    elif [ $cores -eq 96 ]; then
        NTASKS=12
        THREADS=8
    fi

    export OMP_NUM_THREADS=$THREADS
    echo "MPI Processes: $NTASKS, OpenMP Threads per process: $THREADS"

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
            echo "[$cores cores] ${CONFIG_NAMES[$i]}, ${STRIDE_NAMES[$s]}"
            srun -n $NTASKS ./conv_stride_test -H $H -W $W -kH $kH -kW $kW -sH $sH -sW $sW -m hybrid
        done
    done
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
