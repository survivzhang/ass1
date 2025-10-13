#!/bin/bash
#SBATCH --job-name=hybrid_20k_s2_call
#SBATCH --nodes=2
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=output/hybrid_20k_s2_call_%j.out
#SBATCH --error=output/hybrid_20k_s2_call_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid MPI+OpenMP Test - 2 Nodes, all core counts, stride (2,2)

module load gcc/12.2.0
module load cray-mpich

echo "=========================================="
echo "Hybrid MPI+OpenMP Test - 2 Nodes, stride (2,2)"
echo "20000x20000 Matrix, Kernel 200x200"
echo "Testing all core counts"
echo "=========================================="
echo "Job started at: $(date)"
echo "=========================================="

# Core counts to test
CORES=(4 8 16 32 64 96)

for cores in "${CORES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing with $cores cores (Hybrid)"
    echo "=========================================="

    # Calculate optimal MPI tasks and OpenMP threads
    if [ $cores -eq 4 ]; then
        NTASKS=2
        THREADS=2
    elif [ $cores -eq 8 ]; then
        NTASKS=4
        THREADS=2
    elif [ $cores -eq 16 ]; then
        NTASKS=4
        THREADS=4
    elif [ $cores -eq 32 ]; then
        NTASKS=8
        THREADS=4
    elif [ $cores -eq 64 ]; then
        NTASKS=8
        THREADS=8
    elif [ $cores -eq 96 ]; then
        NTASKS=12
        THREADS=8
    fi

    export OMP_NUM_THREADS=$THREADS
    echo "MPI Processes: $NTASKS, OpenMP Threads per process: $THREADS"

    srun -n $NTASKS ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 2 -sW 2 -m hybrid
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
