#!/bin/bash

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Submit baseline tests (stride 1,1 only) for 20000x20000 matrix
# This submits only the main performance tests, excluding stride variations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting Baseline Tests (Stride 1,1)"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo ""

submit_count=0

# OpenMP baseline tests (all thread counts, stride 1,1)
echo "Submitting OpenMP baseline tests..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_20000x20000_t*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename $script)"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# MPI baseline tests (all core counts, stride 1,1)
echo "Submitting MPI baseline tests..."
for script in "$SCRIPT_DIR/MPI_SH"/test_20000x20000_2nodes_c*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename $script)"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# Hybrid baseline tests (all core counts, stride 1,1)
echo "Submitting Hybrid baseline tests..."
for script in "$SCRIPT_DIR/Hybrid_SH"/test_20000x20000_2nodes_c*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename $script)"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

echo "=========================================="
echo "Total baseline tests submitted: $submit_count"
echo "=========================================="
echo "Check status: squeue -u \$USER"
