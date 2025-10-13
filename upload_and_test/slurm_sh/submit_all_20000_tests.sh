#!/bin/bash

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Submit all 20000x20000 test scripts to SLURM
# Usage: bash submit_all_20000_tests.sh [category]
# Categories: openmp, mpi, hybrid, hybrid_dist, all (default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATEGORY="${1:-all}"

echo "=========================================="
echo "SLURM Batch Submission Script"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Category: $CATEGORY"
echo "=========================================="
echo ""

submit_count=0

# Function to submit scripts
submit_scripts() {
    local dir=$1
    local pattern=$2
    local description=$3

    echo "----------------------------------------"
    echo "Submitting: $description"
    echo "----------------------------------------"

    local count=0
    for script in "$SCRIPT_DIR/$dir"/$pattern; do
        if [ -f "$script" ]; then
            echo "Submitting: $(basename $script)"
            sbatch "$script"
            count=$((count + 1))
            submit_count=$((submit_count + 1))
            sleep 0.5  # Small delay to avoid overwhelming the scheduler
        fi
    done

    echo "Submitted $count scripts from $dir"
    echo ""
}

# Submit based on category
case $CATEGORY in
    openmp)
        echo "Submitting OpenMP tests only..."
        submit_scripts "OpenMP_SH" "test_20000*.sh" "OpenMP Tests"
        ;;

    mpi)
        echo "Submitting MPI tests only..."
        submit_scripts "MPI_SH" "test_20000*.sh" "Pure MPI Tests"
        ;;

    hybrid)
        echo "Submitting Hybrid tests only (excluding distribution tests)..."
        submit_scripts "Hybrid_SH" "test_20000*.sh" "Hybrid MPI+OpenMP Tests"
        ;;

    hybrid_dist)
        echo "Submitting Hybrid distribution tests only..."
        submit_scripts "Hybrid_Distribution" "test_20000*.sh" "Hybrid Distribution Tests"
        ;;

    all)
        echo "Submitting ALL 20000x20000 tests..."
        echo ""
        submit_scripts "OpenMP_SH" "test_20000*.sh" "OpenMP Tests"
        submit_scripts "MPI_SH" "test_20000*.sh" "Pure MPI Tests"
        submit_scripts "Hybrid_SH" "test_20000*.sh" "Hybrid MPI+OpenMP Tests"
        submit_scripts "Hybrid_Distribution" "test_20000*.sh" "Hybrid Distribution Tests"
        ;;

    *)
        echo "Error: Unknown category '$CATEGORY'"
        echo "Valid categories: openmp, mpi, hybrid, hybrid_dist, all"
        exit 1
        ;;
esac

echo "=========================================="
echo "Submission Complete"
echo "=========================================="
echo "Total scripts submitted: $submit_count"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Cancel all jobs with: scancel -u \$USER"
echo "=========================================="
