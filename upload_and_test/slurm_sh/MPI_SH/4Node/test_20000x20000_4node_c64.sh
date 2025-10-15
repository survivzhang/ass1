#!/bin/bash
#SBATCH --job-name=mpi_20000_4n_c64
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=../output/mpi_20000_4n_c64_%j.out
#SBATCH --error=../output/mpi_20000_4n_c64_%j.err
#SBATCH --partition=work

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Pure MPI Test - 4 Nodes, 64 cores, 20000x20000 matrix, kernel 200x200

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=1

echo "=========================================="
echo "Pure MPI Test - 4 Nodes, 64 cores"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 64"
echo "Memory allocated: 96GB"
echo "=========================================="
echo "Memory usage before execution:"
echo "Node memory: $(free -h)"
echo "=========================================="

echo "Starting MPI execution..."
srun -n 64 ../../../conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m mpi
echo ""
echo "=========================================="
echo "Memory usage after execution:"
echo "Node memory: $(free -h)"
echo "=========================================="

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
