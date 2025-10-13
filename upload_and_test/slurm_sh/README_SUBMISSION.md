# SLURM Test Submission Guide

Group Members: Jiazheng Guo (24070858), Zichen Zhang (24064091)

## Overview

This directory contains 57 SLURM test scripts for testing the 20000×20000 matrix with 200×200 kernel, organized into 4 folders:
- **OpenMP_SH**: 16 scripts
- **MPI_SH**: 15 scripts
- **Hybrid_SH**: 15 scripts
- **Hybrid_Distribution**: 11 scripts (testing different MPI×OpenMP distributions)

All output files will be saved to the `output/` folder.

---

## Quick Start

### Submit ALL tests at once:
```bash
bash submit_all_20000_tests.sh
```

### Submit tests by category:
```bash
bash submit_all_20000_tests.sh openmp       # OpenMP tests only
bash submit_all_20000_tests.sh mpi          # MPI tests only
bash submit_all_20000_tests.sh hybrid       # Hybrid tests only (main folder)
bash submit_all_20000_tests.sh hybrid_dist  # Hybrid distribution tests only
bash submit_all_20000_tests.sh all          # All tests (default)
```

---

## Selective Submission Scripts

### 1. Baseline Tests Only (stride 1,1)
Submit only the main performance tests without stride variations:
```bash
bash submit_baseline_tests.sh
```
This submits:
- OpenMP: threads 1, 2, 4, 8, 16, 32, 64, 96
- MPI: cores 2, 4, 8, 16, 32, 64, 96
- Hybrid: cores 2, 4, 8, 16, 32, 64, 96

### 2. Stride Variation Tests Only
Submit only stride tests (2,2), (3,3), (5,5), (10,10):
```bash
bash submit_stride_tests.sh
```

### 3. Hybrid Distribution Tests
Submit hybrid distribution tests (different MPI×OpenMP combinations):

**Option A - Individual jobs for each distribution:**
```bash
bash submit_hybrid_distributions.sh
```

**Option B - Single job testing all distributions:**
```bash
bash submit_hybrid_distributions.sh all
```

---

## Manual Submission

### Submit a specific test:
```bash
sbatch OpenMP_SH/test_20000x20000_t96.sh
```

### Submit specific categories manually:
```bash
# Submit all OpenMP tests
for script in OpenMP_SH/test_20000*.sh; do sbatch "$script"; sleep 0.5; done

# Submit all MPI tests
for script in MPI_SH/test_20000*.sh; do sbatch "$script"; sleep 0.5; done

# Submit all Hybrid tests
for script in Hybrid_SH/test_20000*.sh; do sbatch "$script"; sleep 0.5; done

# Submit all Hybrid distribution tests
for script in Hybrid_Distribution/test_20000*.sh; do sbatch "$script"; sleep 0.5; done
```

---

## Monitoring Jobs

### Check job status:
```bash
squeue -u $USER
```

### Check specific job details:
```bash
scontrol show job <job_id>
```

### View running jobs with more info:
```bash
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

---

## Managing Jobs

### Cancel all your jobs:
```bash
scancel -u $USER
```

### Cancel a specific job:
```bash
scancel <job_id>
```

### Cancel all jobs matching a pattern:
```bash
# Cancel all OpenMP jobs
squeue -u $USER -n omp_20k* -h -o "%i" | xargs -n 1 scancel

# Cancel all MPI jobs
squeue -u $USER -n mpi_20k* -h -o "%i" | xargs -n 1 scancel

# Cancel all Hybrid jobs
squeue -u $USER -n hybrid_20k* -h -o "%i" | xargs -n 1 scancel
```

---

## Output Files

All test outputs are saved in the `output/` folder with the format:
- `output/<test_name>_<job_id>.out` - Standard output
- `output/<test_name>_<job_id>.err` - Error output

### View output while job is running:
```bash
tail -f output/<test_name>_<job_id>.out
```

### Search for specific results:
```bash
# Find all completed jobs
grep "Job completed" output/*.out

# Find execution times
grep "Time:" output/*.out

# Find errors
grep -i "error" output/*.err
```

---

## Test Details

### OpenMP Tests
- **Baseline (stride 1,1)**: 8 individual scripts for each thread count
- **Stride tests**: 8 scripts (4 strides × 2: serial + threaded loop)

### MPI Tests
- **Baseline (stride 1,1)**: 7 individual scripts for each core count
- **Stride tests**: 8 scripts (4 strides × 2: minimal + core loop)

### Hybrid Tests
- **Baseline (stride 1,1)**: 7 individual scripts for each core count
- **Stride tests**: 8 scripts (4 strides × 2: minimal + core loop)

### Hybrid Distribution Tests
Testing different MPI×OpenMP distributions for 96 cores:
- 2×48, 3×32, 4×24, 6×16, 8×12, 12×8, 16×6, 24×4, 32×3, 48×2
- Plus 1 all-in-one script

---

## Recommended Submission Strategy

### Strategy 1: Progressive Testing (Recommended)
1. Start with baseline tests to establish performance profile:
   ```bash
   bash submit_baseline_tests.sh
   ```

2. Once baseline completes, test stride variations:
   ```bash
   bash submit_stride_tests.sh
   ```

3. Finally, explore hybrid distributions:
   ```bash
   bash submit_hybrid_distributions.sh all
   ```

### Strategy 2: Complete Testing
Submit everything at once (may queue for longer):
```bash
bash submit_all_20000_tests.sh
```

### Strategy 3: Focused Testing
Test specific parallelization approaches:
```bash
bash submit_all_20000_tests.sh openmp    # Focus on shared memory
bash submit_all_20000_tests.sh mpi       # Focus on distributed memory
bash submit_all_20000_tests.sh hybrid    # Focus on hybrid approach
```

---

## Time Estimates

- Each individual test: ~30-120 minutes (depends on configuration)
- All baseline tests: ~3-6 hours total walltime
- All stride tests: ~3-5 hours total walltime
- All hybrid distributions: ~2-4 hours total walltime
- Complete test suite: ~8-15 hours total walltime

Note: Actual runtime depends on cluster load and queue times.

---

## Troubleshooting

### Jobs not starting?
- Check partition availability: `sinfo`
- Check account status: `sacctmgr show user $USER`
- Verify time limits are acceptable

### Jobs failing immediately?
- Check error files in `output/` directory
- Verify working directory exists on compute nodes
- Ensure executable (`conv_stride_test`) is accessible

### Out of disk space?
- Clean old output files: `rm output/*`
- Check quota: `quota -s`

---

## Summary Scripts Reference

| Script | Purpose |
|--------|---------|
| `submit_all_20000_tests.sh` | Main script - submit by category or all |
| `submit_baseline_tests.sh` | Submit only stride (1,1) tests |
| `submit_stride_tests.sh` | Submit only stride variation tests |
| `submit_hybrid_distributions.sh` | Submit hybrid distribution tests |

All scripts support `-h` or `--help` for usage information.
