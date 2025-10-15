#!/bin/bash
#SBATCH --job-name=hybrid_20000_4n_c16
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=hout/hybrid_20000_4n_c16_%j.out
#SBATCH --error=hout/hybrid_20000_4n_c16_%j.err
#SBATCH --partition=work

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Hybrid MPI+OpenMP Test - 4 Nodes, 16 cores, 20000x20000 matrix, kernel 200x200

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=2

echo "=========================================="
echo "Hybrid MPI+OpenMP Test - 4 Nodes, 16 cores"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 8, OpenMP Threads: 2 (Total: 16 cores)"
echo "Memory allocated: 96GB"
echo "=========================================="
echo "Memory usage before execution:"
echo "Node memory: $(free -h)"
echo "=========================================="

echo "Starting hybrid execution..."
srun -n 8 ../../../conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m hybrid
echo ""
echo "=========================================="
echo "Memory usage after execution:"
echo "Node memory: $(free -h)"
echo "=========================================="

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
