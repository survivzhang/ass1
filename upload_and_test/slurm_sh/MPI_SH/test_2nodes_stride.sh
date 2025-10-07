#!/bin/bash
#SBATCH --job-name=pure_mpi_2nodes_stride
#SBATCH --nodes=2
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=pure_mpi_2nodes_stride_%j.out
#SBATCH --error=pure_mpi_2nodes_stride_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Pure MPI Test - 2 Nodes with different stride values
# Testing the effect of stride on computational and communication cost

module load gcc/12.2.0
module load cray-mpich

# Disable OpenMP - Pure MPI only
export OMP_NUM_THREADS=1

echo "=========================================="
echo "Pure MPI Stride Test - 2 Nodes"
echo "=========================================="
echo "Job started at: $(date)"
echo "Nodes: 2"
echo "Mode: Pure MPI (no OpenMP)"
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
CORES=(2 4 8 16 32 64 96)

# Test each core count
for cores in "${CORES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing with $cores cores (Pure MPI)"
    echo "=========================================="

    # Pure MPI: number of tasks = number of cores
    NTASKS=$cores

    echo "MPI Processes: $NTASKS (1 process per core, no OpenMP)"

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
            srun -n $NTASKS ./conv_stride_test -H $H -W $W -kH $kH -kW $kW -sH $sH -sW $sW -m mpi
        done
    done
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
