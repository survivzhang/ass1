#!/bin/bash
#SBATCH --job-name=hybrid_all_dist
#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=output/hybrid_all_dist_%j.out
#SBATCH --error=output/hybrid_all_dist_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid Distribution Test - Testing all MPI×OpenMP distributions for 96 cores

module load gcc/12.2.0
module load cray-mpich

echo "=========================================="
echo "Hybrid Distribution Test - All Combinations"
echo "20000x20000 Matrix, Kernel 200x200"
echo "Testing various MPI×OpenMP distributions"
echo "=========================================="
echo "Job started at: $(date)"
echo "=========================================="

# Distribution configurations: (MPI_PROCESSES, OPENMP_THREADS)
DISTRIBUTIONS=(
    "2 48"
    "3 32"
    "4 24"
    "6 16"
    "8 12"
    "12 8"
    "16 6"
    "24 4"
    "32 3"
    "48 2"
)

DIST_NAMES=(
    "2×48"
    "3×32"
    "4×24"
    "6×16"
    "8×12"
    "12×8"
    "16×6"
    "24×4"
    "32×3"
    "48×2"
)

# Test each distribution
for i in "${!DISTRIBUTIONS[@]}"; do
    dist=(${DISTRIBUTIONS[$i]})
    NTASKS=${dist[0]}
    THREADS=${dist[1]}

    echo ""
    echo "=========================================="
    echo "Testing Distribution: ${DIST_NAMES[$i]}"
    echo "MPI Processes: $NTASKS, OpenMP Threads: $THREADS"
    echo "Total cores: $(($NTASKS * $THREADS))"
    echo "=========================================="

    export OMP_NUM_THREADS=$THREADS

    srun -n $NTASKS ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m hybrid
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
