#!/bin/bash

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Submit stride variation tests for 20000x20000 matrix
# Tests with strides: (2,2), (3,3), (5,5), (10,10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting Stride Variation Tests"
echo "20000x20000 Matrix, Kernel 200x200"
echo "Strides: (2,2), (3,3), (5,5), (10,10)"
echo "=========================================="
echo ""

submit_count=0

# OpenMP stride tests
echo "Submitting OpenMP stride tests..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_20000x20000_stride*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename $script)"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# MPI stride tests
echo "Submitting MPI stride tests..."
for script in "$SCRIPT_DIR/MPI_SH"/test_20000x20000_2nodes_stride*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename $script)"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# Hybrid stride tests
echo "Submitting Hybrid stride tests..."
for script in "$SCRIPT_DIR/Hybrid_SH"/test_20000x20000_2nodes_stride*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename $script)"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

echo "=========================================="
echo "Total stride tests submitted: $submit_count"
echo "=========================================="
echo "Check status: squeue -u \$USER"
