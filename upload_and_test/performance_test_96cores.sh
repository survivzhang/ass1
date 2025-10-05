#!/bin/bash
#SBATCH --job-name=conv2d_perf96
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=conv2d_perf96_%j.out
#SBATCH --error=conv2d_perf96_%j.err
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zic.zhang@outlook.com

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# CITS3402/CITS5507 Assignment 2 - Performance Analysis
# Configuration: 4 nodes × 4 processes/node × 6 threads = 96 total cores

module load gcc/12.2.0
module load cray-mpich

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "Assignment 2 Performance Tests - 96 Cores"
echo "=========================================="
echo "Job started at: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI Processes: $SLURM_NTASKS"
echo "OpenMP Threads per process: $OMP_NUM_THREADS"
echo "Total cores: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "=========================================="

# Test matrix sizes
KERNEL=5

echo ""
echo "==========================================="
echo "Experiment 1: Mode Comparison (4000x4000)"
echo "==========================================="

echo ""
echo "1.1 Serial (baseline):"
srun -n 1 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m serial

echo ""
echo "1.2 OpenMP only (1 process, 96 threads):"
export OMP_NUM_THREADS=96
srun -n 1 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m omp

echo ""
echo "1.3 MPI only (16 processes, 1 thread each):"
export OMP_NUM_THREADS=1
srun -n 96 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m mpi

echo ""
echo "1.4 Hybrid (16 processes × 6 threads = 96 cores):"
export OMP_NUM_THREADS=6
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "==========================================="
echo "Experiment 2: Hybrid Configuration Tests"
echo "Testing different process × thread = 96 cores"
echo "==========================================="

echo ""
echo "2.1 Config: 4 processes × 24 threads"
export OMP_NUM_THREADS=24
srun -n 4 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "2.2 Config: 8 processes × 12 threads"
export OMP_NUM_THREADS=12
srun -n 8 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "2.3 Config: 12 processes × 8 threads"
export OMP_NUM_THREADS=8
srun -n 12 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "2.4 Config: 16 processes × 6 threads"
export OMP_NUM_THREADS=6
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "2.5 Config: 24 processes × 4 threads"
export OMP_NUM_THREADS=4
srun -n 24 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "2.6 Config: 32 processes × 3 threads"
export OMP_NUM_THREADS=3
srun -n 32 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "==========================================="
echo "Experiment 3: Stride Impact Analysis"
echo "==========================================="

export OMP_NUM_THREADS=6

echo ""
echo "3.1 Stride 1x1 (no stride):"
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "3.2 Stride 2x2:"
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 2 -sW 2 -m hybrid

echo ""
echo "3.3 Stride 3x3:"
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 3 -sW 3 -m hybrid

echo ""
echo "3.4 Stride 4x4:"
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 4 -sW 4 -m hybrid

echo ""
echo "==========================================="
echo "Experiment 4: Strong Scaling (Fixed Size 8000x8000)"
echo "==========================================="

export OMP_NUM_THREADS=6

echo ""
echo "4.1 Using 6 cores (1 process × 6 threads):"
srun -n 1 ./conv_stride_test -H 8000 -W 8000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "4.2 Using 12 cores (2 processes × 6 threads):"
srun -n 2 ./conv_stride_test -H 8000 -W 8000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "4.3 Using 24 cores (4 processes × 6 threads):"
srun -n 4 ./conv_stride_test -H 8000 -W 8000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "4.4 Using 48 cores (8 processes × 6 threads):"
srun -n 8 ./conv_stride_test -H 8000 -W 8000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "4.5 Using 96 cores (16 processes × 6 threads):"
srun -n 16 ./conv_stride_test -H 8000 -W 8000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "==========================================="
echo "Experiment 5: Different Matrix Sizes"
echo "==========================================="

export OMP_NUM_THREADS=6

echo ""
echo "5.1 Small: 2000x2000"
srun -n 16 ./conv_stride_test -H 2000 -W 2000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "5.2 Medium: 4000x4000"
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "5.3 Large: 6000x6000"
srun -n 16 ./conv_stride_test -H 6000 -W 6000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "5.4 X-Large: 8000x8000"
srun -n 16 ./conv_stride_test -H 8000 -W 8000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "5.5 XX-Large: 10000x10000"
srun -n 16 ./conv_stride_test -H 10000 -W 10000 -kH $KERNEL -kW $KERNEL -sH 1 -sW 1 -m hybrid

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
