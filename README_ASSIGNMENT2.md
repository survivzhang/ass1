# CITS3402/CITS5507 Assignment 2 - MPI+OpenMP 2D Convolution with Stride

**Group Members:** Jiazheng Guo (24070858), Zichen Zhang (24064091)

## Overview

This project implements 2D convolution with stride support using:
- **Serial** implementation (baseline)
- **OpenMP** shared memory parallelism
- **MPI** distributed memory parallelism
- **Hybrid MPI+OpenMP** (main implementation for Assignment 2)

## Building the Project

On Kaya or Setonix:

```bash
# Load required modules
module load openmpi/4.1.5
module load gcc/12.2.0

# Build everything
make clean
make
```

This creates two executables:
- `conv_test` - Assignment 1 (OpenMP only)
- `conv_stride_test` - Assignment 2 (MPI+OpenMP with stride)

## Running the Code

### Basic Usage

```bash
# Run with MPI processes and OpenMP threads
mpirun -np 4 ./conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -m hybrid

# Read from files
mpirun -np 4 ./conv_stride_test -f input.txt -g kernel.txt -sH 2 -sW 2 -o output.txt
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
- **MPI not found**: Load OpenMPI module
- **Wrong results**: Verify stride parameters match test case directory name
- **Slow performance**: Adjust process/thread ratio, check node allocation

## Report Analysis Points

1. **Parallelization Strategy**: Row decomposition, hybrid parallelism
2. **Data Distribution**: Halo exchange, broadcast strategy
3. **Communication Cost**: MPI overhead, synchronization points
4. **Stride Impact**: Reduced computation, same communication
5. **Scalability**: Strong scaling, weak scaling analysis
6. **Cache Effects**: Memory layout, access patterns
7. **Load Balancing**: Work distribution across processes

## Contact

For questions about this implementation, contact group members.
