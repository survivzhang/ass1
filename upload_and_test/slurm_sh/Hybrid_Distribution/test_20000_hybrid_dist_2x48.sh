#!/bin/bash
#SBATCH --job-name=hybrid_2x48
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=output/hybrid_2x48_%j.out
#SBATCH --error=output/hybrid_2x48_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid Distribution Test - 2 MPI processes × 48 OpenMP threads = 96 cores

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=48

echo "=========================================="
echo "Hybrid Distribution Test: 2×48"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 2, OpenMP Threads: 48"
echo "Total cores: 96"
echo "=========================================="

echo "[2×48] 20000x20000 matrix, kernel 200x200"
srun -n 2 ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m hybrid

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
