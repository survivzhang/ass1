# CITS3402/CITS5507 Assignment 2 - Core Program

**Group Members:** Jiazheng Guo (24070858), Zichen Zhang (24064091)

## Overview

This is the core implementation of 2D convolution with stride support using:
- **Serial** implementation (baseline)
- **OpenMP** shared memory parallelism
- **MPI** distributed memory parallelism
- **Hybrid MPI+OpenMP** (main implementation for Assignment 2)

## Building the Project

The Makefile tries multiple approaches to find MPI:

```bash
make clean
make
```

### How the Makefile Works:
1. **First tries `mpicc`** (if modules are loaded)
2. **Falls back to `gcc`** with common MPI paths
3. **Tests MPI header availability** automatically

# Then build:
make clean && make

# Setting up MPI Environment :
```bash
# On Setonix: No modules needed, use srun directly
```

### Basic Usage

```bash
# Serial mode (single thread)
#KAYA
./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m serial

#KAYA
# OpenMP only (single MPI process, multiple threads)
export OMP_NUM_THREADS=8
./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m omp

# Setonix:
# MPI only (multiple processes, single thread each)
srun -n 4 ./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m mpi

# Setonix:
# Hybrid MPI+OpenMP (you control the distribution)
export OMP_NUM_THREADS=4
srun -n 4 --cpus-per-task=4 ./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m hybrid
```

### Hybrid MPI×OpenMP Distribution Examples

**Key Point:** You control the distribution by setting:
- `srun -n N` → Number of MPI processes
- `export OMP_NUM_THREADS=M` → Number of OpenMP threads per process
- **Total cores = N × M**

```bash
# Setonix Examples:
# Example 1: 2×16 distribution (2 MPI processes × 16 OpenMP threads = 32 cores)
export OMP_NUM_THREADS=16
srun -n 2 --cpus-per-task=16 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# Example 2: 8×4 distribution (8 MPI processes × 4 OpenMP threads = 32 cores)
export OMP_NUM_THREADS=4
srun -n 8 --cpus-per-task=4 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# Example 3: 16×2 distribution (16 MPI processes × 2 OpenMP threads = 32 cores)
export OMP_NUM_THREADS=2
srun -n 16 --cpus-per-task=2 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

### File I/O Examples

```bash
# Setonix File I/O Examples:
# Read from files and save output
srun -n 4 ./conv_stride_test -f input.txt -g kernel.txt -sH 2 -sW 2 -o output.txt -m hybrid

# Generate random data and save to files
srun -n 4 ./conv_stride_test -H 1000 -W 1000 -kH 5 -kW 5 -sH 1 -sW 1 -f my_input.txt -g my_kernel.txt -m hybrid
```

## Command Line Options

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

## Modes

1. **serial** - Single-threaded baseline
2. **omp** - OpenMP only (single MPI process)
3. **mpi** - MPI only (no OpenMP threading)
4. **hybrid** - MPI + OpenMP (recommended)

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

## Quick Reference

### Common Commands

```bash
# Build
make clean && make

# Test all modes
./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m serial
export OMP_NUM_THREADS=4; ./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m omp
# Setonix Quick Test:
srun -n 2 ./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m mpi
export OMP_NUM_THREADS=2; srun -n 2 --cpus-per-task=2 ./conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 1 -sW 1 -m hybrid
```

### Key Points

- **`-m hybrid`** uses both MPI and OpenMP, but **you control the distribution**
- **`srun -n N`** sets number of MPI processes
- **`export OMP_NUM_THREADS=M`** sets OpenMP threads per process
- **Total cores = N × M**
- **Test different distributions** to find optimal performance
