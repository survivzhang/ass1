#!/bin/bash
#SBATCH --job-name=mpi_20k_s10_c2
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=output/mpi_20k_s10_c2_%j.out
#SBATCH --error=output/mpi_20k_s10_c2_%j.err
#SBATCH --partition=cits3402

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Pure MPI Test - 2 Nodes, 2 cores, stride (10,10)

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=1

echo "=========================================="
echo "Pure MPI Test - 2 Nodes, 2 cores, stride (10,10)"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo "Job started at: $(date)"
echo "MPI Processes: 2"
echo "=========================================="

srun -n 2 ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 10 -sW 10 -m mpi

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
