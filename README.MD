# 2D Convolution with OpenMP Parallelization

**CITS3402/CITS5507 Assignment 1**  
**Authors:** Jiazheng Guo(24070858),Zichen Zhang(24064091)
**Due:** Friday, 12th September 2025

## Building the Project

### Prerequisites
- GCC compiler with OpenMP support
- Make utility
- Linux/Unix environment (tested on Kaya HPC cluster)

### Compilation
```bash
# Standard build
make

# Debug build with debugging symbols
make debug

# Performance build with maximum optimization
make performance

# Clean build artifacts
make clean
```

## Usage

The program supports multiple execution modes through command-line arguments:

### Basic Usage
```bash
# Using existing input files
./conv_test -f input.txt -g kernel.txt

# With output file
./conv_test -f input.txt -g kernel.txt -o output.txt

# Generate random arrays
./conv_test -H 1000 -W 1000 -h 3 -w 3

# Generate and save arrays
./conv_test -H 1000 -W 1000 -h 3 -w 3 -f input.txt -g kernel.txt -o output.txt
```

### Command Line Options
- `-f FILE`: Input feature map file
- `-g FILE`: Input kernel file  
- `-o FILE`: Output file (optional)
- `-H HEIGHT`: Height for random input generation
- `-W WIDTH`: Width for random input generation
- `-h HEIGHT`: Kernel height for random generation
- `-w WIDTH`: Kernel width for random generation
- `-t THREADS`: Number of OpenMP threads
- `-s`: Use serial implementation only
- `-p`: Use parallel implementation only
- `-c`: Compare serial and parallel implementations
- `--help`: Show help message

### File Format

Arrays are stored in space-separated text files with the format:
```
height width
value1 value2 value3 ...
value4 value5 value6 ...
...
```

Example:
```
3 4
0.884 0.915 0.259 0.937
0.189 0.448 0.337 0.033
0.122 0.169 0.316 0.111
```

