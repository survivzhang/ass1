#!/bin/bash
#SBATCH --job-name=hybrid_20000_2n_c16
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=output/hybrid_20000_2n_c16_%j.out
#SBATCH --error=output/hybrid_20000_2n_c16_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid MPI+OpenMP Test - 2 Nodes, 16 cores, 20000x20000 matrix, kernel 200x200

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=4

echo "=========================================="
echo "Hybrid MPI+OpenMP Test - 2 Nodes, 16 cores"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 4, OpenMP Threads: 4"
echo "=========================================="

srun -n 4 ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m hybrid

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
