# SLURM Scripts Configuration Guide

## Summary of Changes

### Makefile
- **CFLAGS vs MPIFLAGS**: Now `MPIFLAGS = $(CFLAGS)` to avoid duplication
- Both use: `-fopenmp -O3 -Wall` (OpenMP, optimization, warnings)
- Separate flags allow future customization if needed

## SLURM Scripts Comparison

### 1. `conv_test_slurm.sh` - Main Testing

**Purpose:** Demonstrate hybrid MPI+OpenMP advantages

**Configuration:**
```bash
#SBATCH --nodes=2                 # 2 nodes for distributed computing
#SBATCH --ntasks-per-node=4       # 4 MPI processes per node
#SBATCH --cpus-per-task=8         # 8 OpenMP threads per MPI process
# Total: 2×4×8 = 64 cores
```

**Why multi-node?**
- Shows MPI distributing work **across nodes** (network communication)
- Each MPI process uses OpenMP **within node** (shared memory)
- Demonstrates true hybrid parallelism

**Tests:**
- Serial baseline
- OpenMP only (1 node, multi-threaded)
- MPI only (multi-node, single-threaded per process)
- Hybrid (multi-node + multi-threaded) ← **Best performance expected**

---

### 2. `test_conv_stride.sh` - Validation Testing

**Purpose:** Validate correctness with test cases

**Configuration:**
```bash
#SBATCH --nodes=1-4               # Request 1-4 nodes (flexible)
#SBATCH --ntasks=16               # 16 total MPI tasks
#SBATCH --cpus-per-task=4         # 4 OpenMP threads each
# Total: 16×4 = 64 cores
```

**Why 1-4 nodes?**
- Flexibility: SLURM allocates what's available
- Tests correctness regardless of node count
- `1-4` means "give me between 1 and 4 nodes"

---

### 3. `benchmark_conv.sh` - Performance Analysis

**Purpose:** Comprehensive scaling study for report

**Configuration:**
```bash
#SBATCH --nodes=4                 # Request exactly 4 nodes
#SBATCH --ntasks-per-node=8       # 8 tasks per node
#SBATCH --cpus-per-task=4         # 4 threads per task
# Total: 4×8×4 = 128 cores
```

**Why 4 nodes?**
- Maximum resources for scaling study
- Tests strong scaling: fixed problem, more resources
- Tests weak scaling: problem grows with resources
- Generates comprehensive data for report graphs

**What it tests:**
1. **OpenMP scaling**: 1, 2, 4, 8, 16 threads
2. **MPI scaling**: 1, 2, 4, 8, 16 processes
3. **Hybrid combinations**: e.g., 4 procs × 4 threads
4. **Stride effects**: Performance with different strides
5. **Strong scaling**: Same problem, more cores
6. **Weak scaling**: Larger problem per core

---

## Key Differences Explained

| Script | Nodes | Tasks | Threads | Purpose |
|--------|-------|-------|---------|---------|
| `conv_test_slurm.sh` | 2 | 8 | 8 | Compare modes |
| `test_conv_stride.sh` | 1-4 | 16 | 4 | Validate correctness |
| `benchmark_conv.sh` | 4 | 32 | 4 | Report data |

## Why Multi-Node Testing is Important

### Single Node (OpenMP only)
```
[Node 1: Thread1 Thread2 Thread3 Thread4]
```
- All threads share memory
- Fast communication (L3 cache)
- Limited by single node's cores

### Multi-Node (MPI)
```
[Node 1: Process1] <--network--> [Node 2: Process2]
```
- Processes communicate over network
- Slower than shared memory
- Can use MANY more nodes

### Multi-Node Hybrid (MPI + OpenMP)
```
[Node 1: Process1{Thr1 Thr2}] <--network--> [Node 2: Process2{Thr1 Thr2}]
```
- **Best of both worlds**
- MPI for inter-node (coarse-grained parallelism)
- OpenMP for intra-node (fine-grained parallelism)
- Reduces MPI communication overhead
- Better cache utilization

## Example: 64 Cores, Different Distributions

| Config | Nodes | Procs/Node | Threads/Proc | Communication |
|--------|-------|------------|--------------|---------------|
| OpenMP only | 1 | 1 | 64 | All shared memory ✓ |
| MPI only | 64 | 1 | 1 | Maximum network ✗ |
| Hybrid 8×8 | 8 | 8 | 1 | More network |
| Hybrid 4×16 | 4 | 1 | 16 | Balanced ✓✓ |
| Hybrid 2×32 | 2 | 1 | 32 | Less network ✓✓✓ |

**Optimal:** Balance processes and threads to minimize communication while maximizing parallelism.

## For Your Report

Your performance analysis should show:

1. **Speedup vs cores**: How performance improves with more resources
2. **Efficiency**: speedup / (cores × baseline_time)
3. **Scalability**: Does it scale linearly?
4. **Communication overhead**: MPI vs OpenMP vs Hybrid
5. **Stride impact**: How stride affects computation time

The multi-node setup will clearly show hybrid's advantages over pure MPI or pure OpenMP!
