#!/bin/bash
#SBATCH --job-name=omp_20k_s5_t1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=output/omp_20k_s5_t1_%j.out
#SBATCH --error=output/omp_20k_s5_t1_%j.err
#SBATCH --partition=work
#SBATCH --account=courses0101

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# OpenMP Test - 20000x20000 matrix, kernel 200x200, stride (5,5), 1 thread

module load gcc/12.2.0

cd /scratch/courses0101/zzhang5/upload_and_test || exit 1

echo "=========================================="
echo "20000x20000 Matrix, Kernel 200x200, Stride (5,5) - 1 thread"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

export OMP_NUM_THREADS=1

echo "[1 thread] 20000x20000 matrix, kernel 200x200, stride (5,5)"
./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 5 -sW 5 -m serial

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
