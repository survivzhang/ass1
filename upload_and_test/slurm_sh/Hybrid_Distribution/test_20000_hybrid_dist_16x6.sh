#!/bin/bash
#SBATCH --job-name=hybrid_16x6
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=output/hybrid_16x6_%j.out
#SBATCH --error=output/hybrid_16x6_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid Distribution Test - 16 MPI processes × 6 OpenMP threads = 96 cores

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=6

echo "=========================================="
echo "Hybrid Distribution Test: 16×6"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 16, OpenMP Threads: 6"
echo "Total cores: 96"
echo "=========================================="

echo "[16×6] 20000x20000 matrix, kernel 200x200"
srun -n 16 ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m hybrid

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
