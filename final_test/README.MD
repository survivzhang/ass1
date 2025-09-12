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
# Build the project
make

# Clean build artifacts
make clean
```

## Usage

The program supports multiple execution modes through command-line arguments:

### Usage Example
```bash
# Using existing input files
./conv_test -f input.txt -g kernel.txt

# With output file
./conv_test -f input.txt -g kernel.txt -o output.txt

# Generate random arrays
./conv_test -H 1000 -W 1000 -h 3 -w 3

# Generate and save arrays
./conv_test -H 1000 -W 1000 -h 3 -w 3 -f input.txt -g kernel.txt -o output.txt

# Use serial implementation only
./conv_test -f input.txt -g kernel.txt -s

# Use parallel implementation with specific thread count
./conv_test -f input.txt -g kernel.txt -p -t 8

# Compare serial and parallel implementations
./conv_test -f input.txt -g kernel.txt -c

# Performance analysis across different thread counts
./conv_test -f input.txt -g kernel.txt -a

# Show help message
./conv_test --help
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
- `-p`: Use parallel implementation only （default MAX number）
- `-c`: Compare serial and parallel implementations
- `-a`: Analyze performance across different thread counts
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

### Run in Kaya

Here's a step-by-step process to run the 2D convolution program on the Kaya HPC cluster:

#### Step 1: Build the Project
```bash
# Compile the program
make
```

#### Step 2: Modify Parameters in SLURM Script
Edit the `conv_test_slurm.sh` file to adjust test parameters:

```bash
# Test parameters (modify as needed)
MATRIX_HEIGHT=1000      # Input matrix height
MATRIX_WIDTH=1000       # Input matrix width  
KERNEL_HEIGHT=3         # Kernel height
KERNEL_WIDTH=3          # Kernel width
```

You can also modify SLURM job parameters:
```bash
#SBATCH --cpus-per-task=64     # Number of CPUs (adjust as needed)
#SBATCH --time=00:30:00        # Time limit
#SBATCH --mem=4G               # Memory allocation
#SBATCH --mail-user=your@email.com  # Your email for notifications
```

#### Step 3: Submit Job to SLURM
```bash
# Submit the job
sbatch conv_test_slurm.sh
```

#### Step 4: Monitor Job Status
```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job JOBID
```

#### Step 5: Check Results
```bash
# View output file
cat conv_test.out

# View error file (if any)
cat conv_test.err

# View performance results
cat performance_results.txt
```

#### Alternative: Interactive Testing
For quick testing, you can also run interactively:
```bash
# Request interactive session
salloc --cpus-per-task=16 --time=00:10:00 --partition=cits3402

# Run program directly
./conv_test -H 500 -W 500 -h 3 -w 3 -a
```
