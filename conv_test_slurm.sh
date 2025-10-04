#!/bin/bash
#SBATCH --job-name=conv2d_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=conv2d_mpi_%j.out
#SBATCH --error=conv2d_mpi_%j.err
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zic.zhang@outlook.com

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# CITS3402/CITS5507 Assignment 2 - MPI+OpenMP 2D Convolution
# Using 2 nodes × 4 tasks/node × 8 threads/task = 64 total cores

module load openmpi/4.1.5
module load gcc/12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "MPI+OpenMP 2D Convolution with Stride"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "MPI Processes: $SLURM_NTASKS"
echo "OpenMP Threads per process: $OMP_NUM_THREADS"
echo "Total cores: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "=========================================="

# Test different configurations
echo ""
echo "Test 1: Small matrix (1000x1000, kernel 3x3, stride 1x1)"
mpirun -np $SLURM_NTASKS ./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m hybrid

echo ""
echo "Test 2: Medium matrix with stride (2000x2000, kernel 5x5, stride 2x2)"
mpirun -np $SLURM_NTASKS ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 2 -sW 2 -m hybrid

echo ""
echo "Test 3: Large matrix (4000x4000, kernel 7x7, stride 1x1)"
mpirun -np $SLURM_NTASKS ./conv_stride_test -H 4000 -W 4000 -kH 7 -kW 7 -sH 1 -sW 1 -m hybrid

echo ""
echo "=========================================="
echo "Comparing different modes (size 2000x2000, kernel 5x5)"
echo "=========================================="
echo ""

echo "Serial:"
mpirun -np 1 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m serial

echo ""
echo "OpenMP only:"
mpirun -np 1 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m omp

echo ""
echo "MPI only:"
mpirun -np $SLURM_NTASKS ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m mpi

echo ""
echo "Hybrid MPI+OpenMP:"
mpirun -np $SLURM_NTASKS ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="