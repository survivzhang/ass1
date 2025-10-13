#!/bin/bash
#SBATCH --job-name=openmp_20000_t8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=output/openmp_20000_t8_%j.out
#SBATCH --error=output/openmp_20000_t8_%j.err
#SBATCH --partition=work
#SBATCH --account=courses0101

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# OpenMP Test - 20000x20000 matrix, kernel 200x200 with 8 threads

module load gcc/12.2.0

cd /scratch/courses0101/zzhang5/upload_and_test || exit 1

echo "=========================================="
echo "20000x20000 Matrix, Kernel 200x200 - 8 threads"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

export OMP_NUM_THREADS=8

echo "[8 threads] 20000x20000 matrix, kernel 200x200"
./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 1 -sW 1 -m omp

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
