#!/bin/bash
#SBATCH --job-name=mpi_20k_s2_call
#SBATCH --nodes=2
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=output/mpi_20k_s2_call_%j.out
#SBATCH --error=output/mpi_20k_s2_call_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Pure MPI Test - 2 Nodes, all core counts, stride (2,2)

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=1

echo "=========================================="
echo "Pure MPI Test - 2 Nodes, stride (2,2)"
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
    echo "Testing with $cores cores (MPI processes)"
    echo "=========================================="

    srun -n $cores ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 2 -sW 2 -m mpi
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
