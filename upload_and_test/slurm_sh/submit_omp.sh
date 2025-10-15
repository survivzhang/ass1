#!/bin/bash

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Submit all OpenMP tests for 20000x20000 matrix
# This includes baseline (stride 1,1) and all stride variations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting All OpenMP Tests"
echo "20000x20000 Matrix, Kernel 200x200"
echo "=========================================="
echo ""

submit_count=0

# OpenMP baseline tests (all thread counts, stride 1,1)
echo "Submitting OpenMP baseline tests (stride 1,1)..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_20000x20000_t*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename "$script")"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# OpenMP stride variation tests (stride 2,2 with t1 and t96)
echo "Submitting OpenMP stride 2,2 tests..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_20000x20000_stride2_t*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename "$script")"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# OpenMP stride variation tests (stride 3,3 with t1 and t96)
echo "Submitting OpenMP stride 3,3 tests..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_20000x20000_stride3_t*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename "$script")"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# OpenMP stride variation tests (stride 5,5 with t1 and t96)
echo "Submitting OpenMP stride 5,5 tests..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_20000x20000_stride5_t*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename "$script")"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# OpenMP stride variation tests (stride 10,10 with t1 and t96)
echo "Submitting OpenMP stride 10,10 tests..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_20000x20000_stride10_t*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename "$script")"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

# Submit any other OpenMP test scripts (like test_openmp_baseline.sh, test_openmp_stride.sh)
echo "Submitting other OpenMP test scripts..."
for script in "$SCRIPT_DIR/OpenMP_SH"/test_openmp_*.sh; do
    if [ -f "$script" ]; then
        echo "  $(basename "$script")"
        sbatch "$script"
        submit_count=$((submit_count + 1))
        sleep 0.5
    fi
done
echo ""

echo "=========================================="
echo "Total OpenMP tests submitted: $submit_count"
echo "=========================================="
echo "Check status: squeue -u \$USER"
echo "View output: ls -ltr output/openmp_*.out"
echo "View errors: ls -ltr output/openmp_*.err"
echo "=========================================="
