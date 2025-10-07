#!/bin/bash
# Fix all test scripts to add cd command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fixing all scripts to add working directory..."

# Add cd command after module load in all scripts
for script in "$SCRIPT_DIR"/MPI_SH/*.sh "$SCRIPT_DIR"/Hybrid_SH/*.sh; do
    if [ -f "$script" ]; then
        # Check if cd command already exists
        if ! grep -q "cd.*upload_and_test" "$script"; then
            # Add cd command after the last module load line
            sed -i '/^module load cray-mpich/a\
\
# Change to working directory\
cd /scratch/courses0101/zzhang5/upload_and_test || exit 1' "$script"
            echo "Fixed: $script"
        else
            echo "Already fixed: $script"
        fi
    fi
done

echo "All scripts fixed!"
