Hybrid Distribution Tests - 96 Cores (2 Nodes)
===============================================

This folder contains SLURM scripts to test different MPI×OpenMP distributions 
for the Hybrid mode using 96 total cores across 2 nodes.

Test Configuration:
- Matrix: 20000×20000
- Kernel: 200×200
- Stride: (1,1)
- Total Cores: 96 (2 nodes × 48 cores/node)

Distribution Scripts:
---------------------
1. test_20000_hybrid_dist_2x48.sh   - 2 MPI × 48 OpenMP
2. test_20000_hybrid_dist_3x32.sh   - 3 MPI × 32 OpenMP
3. test_20000_hybrid_dist_4x24.sh   - 4 MPI × 24 OpenMP
4. test_20000_hybrid_dist_6x16.sh   - 6 MPI × 16 OpenMP
5. test_20000_hybrid_dist_8x12.sh   - 8 MPI × 12 OpenMP
6. test_20000_hybrid_dist_12x8.sh   - 12 MPI × 8 OpenMP  (default in main folder)
7. test_20000_hybrid_dist_16x6.sh   - 16 MPI × 6 OpenMP
8. test_20000_hybrid_dist_24x4.sh   - 24 MPI × 4 OpenMP
9. test_20000_hybrid_dist_32x3.sh   - 32 MPI × 3 OpenMP
10. test_20000_hybrid_dist_48x2.sh  - 48 MPI × 2 OpenMP

All-in-One Script:
------------------
test_20000_hybrid_all_dist.sh - Runs all 10 distributions in a single job

Usage:
------
To test a specific distribution:
  sbatch test_20000_hybrid_dist_<config>.sh

To test all distributions:
  sbatch test_20000_hybrid_all_dist.sh

Output:
-------
All output files will be placed in the output/ folder with names like:
  output/hybrid_<config>_<jobid>.out
  output/hybrid_<config>_<jobid>.err

Purpose:
--------
These tests help determine the optimal balance between MPI processes and 
OpenMP threads for the hybrid parallelization approach on your specific 
hardware and problem size.
