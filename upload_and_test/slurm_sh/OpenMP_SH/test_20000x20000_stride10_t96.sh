#!/bin/bash
#SBATCH --job-name=omp_20k_s10_tall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --output=output/omp_20k_s10_tall_%j.out
#SBATCH --error=output/omp_20k_s10_tall_%j.err
#SBATCH --partition=work
#SBATCH --account=courses0101

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# OpenMP Test - 20000x20000 matrix, kernel 200x200, stride (10,10), all threads

module load gcc/12.2.0

cd /scratch/courses0101/zzhang5/upload_and_test || exit 1

echo "=========================================="
echo "20000x20000 Matrix, Kernel 200x200, Stride (10,10)"
echo "Testing all thread counts"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Thread counts to test
THREADS=(2 4 8 16 32 64 96)

for threads in "${THREADS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing with $threads threads"
    echo "=========================================="

    export OMP_NUM_THREADS=$threads

    echo "[$threads threads] 20000x20000 matrix, kernel 200x200, stride (10,10)"
    ./conv_stride_test -H 20000 -W 20000 -kH 200 -kW 200 -sH 10 -sW 10 -m omp
done

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
