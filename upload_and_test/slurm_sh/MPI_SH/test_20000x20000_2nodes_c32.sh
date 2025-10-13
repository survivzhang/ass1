#!/bin/bash
#SBATCH --job-name=mpi_20000_2n_c32
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=output/mpi_20000_2n_c32_%j.out
#SBATCH --error=output/mpi_20000_2n_c32_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Pure MPI Test - 2 Nodes, 32 cores, 20000x20000 matrix, kernel 200x200

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=1

echo "=========================================="
echo "Pure MPI Test - 2 Nodes, 32 cores"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 32"
echo "=========================================="

srun -n 32 ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m mpi

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
