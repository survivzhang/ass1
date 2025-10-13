#!/bin/bash
#SBATCH --job-name=hybrid_3x32
#SBATCH --nodes=2
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=output/hybrid_3x32_%j.out
#SBATCH --error=output/hybrid_3x32_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid Distribution Test - 3 MPI processes × 32 OpenMP threads = 96 cores

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=32

echo "=========================================="
echo "Hybrid Distribution Test: 3×32"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 3, OpenMP Threads: 32"
echo "Total cores: 96"
echo "=========================================="

echo "[3×32] 20000x20000 matrix, kernel 200x200"
srun -n 3 ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m hybrid

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
