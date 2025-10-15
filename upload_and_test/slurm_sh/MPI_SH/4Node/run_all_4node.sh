#!/bin/bash

# Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
# Runner script to submit all 4-node MPI tests
# Output will be collected in MPIoutput_4node file

OUTPUT_FILE="MPIoutput_4node"

echo "=========================================="
echo "Submitting all 4-node MPI tests"
echo "=========================================="
echo "Started at: $(date)"
echo ""

# Clear or create output file
> $OUTPUT_FILE

echo "4-Node MPI Test Results - 20000x20000 Matrix, 200x200 Kernel" >> $OUTPUT_FILE
echo "Generated at: $(date)" >> $OUTPUT_FILE
echo "=========================================="  >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Array of core counts
CORES=(2 4 8 16 32 64 96)

# Submit each test
for cores in "${CORES[@]}"; do
    script="test_20000x20000_4node_c${cores}.sh"

    echo "Submitting: $script (${cores} cores)"

    # Submit the job and capture job ID
    job_id=$(sbatch $script | awk '{print $NF}')

    echo "  Job ID: $job_id" >> $OUTPUT_FILE
    echo "  Cores: $cores" >> $OUTPUT_FILE
    echo "  Script: $script" >> $OUTPUT_FILE
    echo "  Submitted at: $(date)" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
done

echo ""
echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo "Completed at: $(date)"
echo ""
echo "Job IDs and submission info written to: $OUTPUT_FILE"
echo ""
echo "To check job status: squeue -u \$USER"
echo "To view results: ls -lh ../output/mpi_20000_4n_*.out"
echo ""
echo "Output files will be in: ../output/"
echo "  - mpi_20000_4n_c2_*.out"
echo "  - mpi_20000_4n_c4_*.out"
echo "  - mpi_20000_4n_c8_*.out"
echo "  - mpi_20000_4n_c16_*.out"
echo "  - mpi_20000_4n_c32_*.out"
echo "  - mpi_20000_4n_c64_*.out"
echo "  - mpi_20000_4n_c96_*.out"
echo "=========================================="
