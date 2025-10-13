#!/bin/bash

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Submit hybrid distribution tests for 20000x20000 matrix
# Tests different MPI×OpenMP distributions for 96 cores

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-individual}"

echo "=========================================="
echo "Submitting Hybrid Distribution Tests"
echo "20000x20000 Matrix, Kernel 200x200"
echo "Testing MPI×OpenMP distributions for 96 cores"
echo "=========================================="
echo ""

submit_count=0

if [ "$MODE" = "all" ]; then
    echo "Submitting single job that tests all distributions..."
    script="$SCRIPT_DIR/Hybrid_Distribution/test_20000_hybrid_all_dist.sh"
    if [ -f "$script" ]; then
        echo "  $(basename $script)"
        sbatch "$script"
        submit_count=1
    fi
else
    echo "Submitting individual distribution tests..."
    for script in "$SCRIPT_DIR/Hybrid_Distribution"/test_20000_hybrid_dist_*.sh; do
        if [ -f "$script" ]; then
            echo "  $(basename $script)"
            sbatch "$script"
            submit_count=$((submit_count + 1))
            sleep 0.5
        fi
    done
fi

echo ""
echo "=========================================="
echo "Total jobs submitted: $submit_count"
echo "=========================================="
echo "Check status: squeue -u \$USER"
echo ""
echo "Note: Use 'bash $(basename $0) all' to submit"
echo "      a single job that tests all distributions."
