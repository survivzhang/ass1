# CITS3402/CITS5507 Assignment 2 - MPI+OpenMP 2D Convolution with Stride

**Group Members:** Jiazheng Guo (24070858), Zichen Zhang (24064091)

## Overview

This project implements 2D convolution with stride support using:
- **Serial** implementation (baseline)
- **OpenMP** shared memory parallelism
- **MPI** distributed memory parallelism
- **Hybrid MPI+OpenMP** (main implementation for Assignment 2)

## Building the Project

### Building the Project:
```bash
# Load modules first, then build
module load gcc && module load openmpi
make clean
make

# Alternative: Use the build script
./build.sh
```

### System-Specific Module Loading:
```bash
# On Setonix:
module load gcc/12.2.0
module load cray-mpich

# On Kaya:
module load gcc/12.4.0
module load openmpi/5.0.5

# Then build:
make clean
make
```

**Note:** Module loading is **required** because:
- The code uses `#include <mpi.h>` and `#include <omp.h>`
- The Makefile uses `mpicc` compiler
- **Setonix**: Uses `cray-mpich` (confirmed in error logs)
- **Kaya**: Uses `openmpi/4.1.5` (different MPI implementation)
- Lmod automatically manages module versions on Setonix

This creates two executables:
- `conv_test` - Assignment 1 (OpenMP only)
- `conv_stride_test` - Assignment 2 (MPI+OpenMP with stride)

## Running the Code

### Basic Usage

```bash
# Serial mode (single thread)
./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m serial

# OpenMP only (single MPI process, multiple threads)
export OMP_NUM_THREADS=8
./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m omp

# MPI only (multiple processes, single thread each)
mpirun -np 4 ./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m mpi

# Hybrid MPI+OpenMP (you control the distribution)
export OMP_NUM_THREADS=4
mpirun -np 4 ./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m hybrid
```

### Hybrid MPI×OpenMP Distribution Examples

**Key Point:** You control the distribution by setting:
- `mpirun -np N` → Number of MPI processes
- `export OMP_NUM_THREADS=M` → Number of OpenMP threads per process
- **Total cores = N × M**

```bash
# Example 1: 2×16 distribution (2 MPI processes × 16 OpenMP threads = 32 cores)
export OMP_NUM_THREADS=16
mpirun -np 2 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# Example 2: 8×4 distribution (8 MPI processes × 4 OpenMP threads = 32 cores)
export OMP_NUM_THREADS=4
mpirun -np 8 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# Example 3: 16×2 distribution (16 MPI processes × 2 OpenMP threads = 32 cores)
export OMP_NUM_THREADS=2
mpirun -np 16 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# Example 4: 32×1 distribution (32 MPI processes × 1 OpenMP thread = 32 cores)
export OMP_NUM_THREADS=1
mpirun -np 32 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

### File I/O Examples

```bash
# Read from files and save output
mpirun -np 4 ./conv_stride_test -f input.txt -g kernel.txt -sH 2 -sW 2 -o output.txt -m hybrid

# Generate random data and save to files
mpirun -np 4 ./conv_stride_test -H 1000 -W 1000 -kH 5 -kW 5 -sH 1 -sW 1 -f my_input.txt -g my_kernel.txt -m hybrid
```

### SLURM Usage Examples

```bash
# Submit a SLURM job with specific MPI×OpenMP distribution
sbatch -J my_job \
  --nodes=2 \
  --ntasks=8 \
  --cpus-per-task=4 \
  --time=01:00:00 \
  --wrap="export OMP_NUM_THREADS=4; srun -n 8 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid"

# Interactive SLURM session
salloc --nodes=2 --ntasks=8 --cpus-per-task=4 --time=01:00:00
export OMP_NUM_THREADS=4
srun -n 8 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

### Performance Testing Examples

```bash
# Test different distributions with same total cores (32)
echo "Testing 2×16 distribution:"
export OMP_NUM_THREADS=16
mpirun -np 2 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

echo "Testing 4×8 distribution:"
export OMP_NUM_THREADS=8
mpirun -np 4 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

echo "Testing 8×4 distribution:"
export OMP_NUM_THREADS=4
mpirun -np 8 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

echo "Testing 16×2 distribution:"
export OMP_NUM_THREADS=2
mpirun -np 16 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

### Command Line Options

- `-H HEIGHT` - Input height
- `-W WIDTH` - Input width
- `-kH HEIGHT` - Kernel height
- `-kW WIDTH` - Kernel width
- `-sH STRIDE` - Vertical stride (default: 1)
- `-sW STRIDE` - Horizontal stride (default: 1)
- `-f FILE` - Input file
- `-g FILE` - Kernel file
- `-o FILE` - Output file
- `-t THREADS` - OpenMP threads per process
- `-m MODE` - Execution mode: `serial`, `omp`, `mpi`, `hybrid`

### Modes

1. **serial** - Single-threaded baseline
2. **omp** - OpenMP only (single MPI process)
3. **mpi** - MPI only (no OpenMP threading)
4. **hybrid** - MPI + OpenMP (recommended)

## SLURM Scripts

### 1. Main Testing Script: `conv_test_slurm.sh`

Basic performance testing with different configurations:

```bash
sbatch conv_test_slurm.sh
```

### 2. Test Case Validation: `test_conv_stride.sh`

Validates correctness against provided test cases:

```bash
sbatch test_conv_stride.sh
```

### 3. Performance Benchmark: `benchmark_conv.sh`

Comprehensive performance analysis:
- Serial baseline
- OpenMP scaling
- MPI scaling
- Hybrid scaling
- Stride effects
- Strong/weak scaling

```bash
sbatch benchmark_conv.sh
```

Results saved to CSV for analysis.

## Test Data

Test cases are in directories named like:
```
conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 1 -sH 1/
├── f0.txt         # Input
├── g0.txt         # Kernel
└── o0_sH_1_sW_1.txt  # Expected output
```

## File Format

Arrays stored as space-separated text:
```
3 4
0.884 0.915 0.259 0.937
0.189 0.448 0.337 0.033
0.122 0.169 0.316 0.111
```

First line: `height width`
Following lines: array values

## Output Size with Stride

For input size H×W with stride sH×sW:
- Output height: `ceil(H / sH)`
- Output width: `ceil(W / sW)`

## Implementation Details

### Data Decomposition
- Row-based decomposition of output array
- Each MPI process computes a block of output rows
- Halo regions communicated for overlapping input data

### Communication Strategy
- Initial broadcast of input and kernel to all processes
- Local computation with halo data
- Gather results back to all processes

### Hybrid Parallelism
- **MPI level**: Distributes output rows across processes
- **OpenMP level**: Parallelizes computation within each process
- Optimized for cache locality and load balancing

## Performance Tips

1. **Process/Thread Balance**: Total cores = MPI_processes × OpenMP_threads
   - Example: 16 cores = 4 processes × 4 threads

2. **Stride Effects**: Larger strides reduce output size and computation time

3. **Strong Scaling**: Fixed problem size, increase processes/threads

4. **Weak Scaling**: Scale problem size with resources

## Example Commands

```bash
# Test correctness on small example
mpirun -np 1 ./conv_stride_test \
  -f "conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 1 -sH 1/f0.txt" \
  -g "conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 1 -sH 1/g0.txt" \
  -sH 1 -sW 1 -o my_output.txt -m hybrid

# Performance test with 4 processes, 8 threads each
export OMP_NUM_THREADS=8
mpirun -np 4 ./conv_stride_test -H 4000 -W 4000 -kH 7 -kW 7 -sH 1 -sW 1 -m hybrid

# Test stride effect
mpirun -np 4 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 3 -sW 3 -m hybrid
```

## Troubleshooting

- **Build fails**: Check modules are loaded (`module list`)
- **MPI not found**: 
  - **Setonix**: Load `cray-mpich` module
  - **Kaya**: Load `openmpi/4.1.5` module
- **Wrong results**: Verify stride parameters match test case directory name
- **Slow performance**: Adjust process/thread ratio, check node allocation
- **Module errors**: 
  - **Setonix**: Has `cray-mpich` and `gcc/12.2.0` available (Lmod manages versions automatically)
  - **Kaya**: Has `openmpi/4.1.5` and `gcc/12.2.0` available

## Report Analysis Points

1. **Parallelization Strategy**: Row decomposition, hybrid parallelism
2. **Data Distribution**: Halo exchange, broadcast strategy
3. **Communication Cost**: MPI overhead, synchronization points
4. **Stride Impact**: Reduced computation, same communication
5. **Scalability**: Strong scaling, weak scaling analysis
6. **Cache Effects**: Memory layout, access patterns
7. **Load Balancing**: Work distribution across processes

## Quick Reference

### Common Commands

```bash
# Build
module load gcc && module load openmpi
make clean && make

# Test all modes
./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m serial
export OMP_NUM_THREADS=4; ./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m omp
mpirun -np 2 ./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m mpi
export OMP_NUM_THREADS=2; mpirun -np 2 ./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m hybrid

# Performance test (32 cores total)
export OMP_NUM_THREADS=16; mpirun -np 2 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
export OMP_NUM_THREADS=8; mpirun -np 4 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
export OMP_NUM_THREADS=4; mpirun -np 8 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
export OMP_NUM_THREADS=2; mpirun -np 16 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

### Key Points

- **`-m hybrid`** uses both MPI and OpenMP, but **you control the distribution**
- **`mpirun -np N`** sets number of MPI processes
- **`export OMP_NUM_THREADS=M`** sets OpenMP threads per process
- **Total cores = N × M**
- **Test different distributions** to find optimal performance
- **SLURM scripts** automate testing of multiple configurations

## Contact

For questions about this implementation, contact group members.
