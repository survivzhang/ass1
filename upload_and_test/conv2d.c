#include "conv2d.h"

/**
 * Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
 */

/**
 * Serial implementation of 2D convolution with "same" padding
 *
 * Memory layout: Arrays are stored as row-major order (array[row][col])
 * Cache considerations: Access patterns are optimized for spatial locality
 * by accessing consecutive memory locations in the innermost loops
 */
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **output) {
    // Use precise padding calculation for both odd and even kernels
    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;

    // For each output pixel
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;

            // Convolve with kernel
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    // Calculate input indices with padding
                    int input_i = i + ki - pad_top;
                    int input_j = j + kj - pad_left;

                    // Apply "same" padding (zero-padding outside boundaries)
                    if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                        // Direct convolution without kernel flipping (correlation)
                        sum += f[input_i][input_j] * g[ki][kj];
                    }
                    // If outside boundaries, the padded value is 0, so no contribution
                }
            }
            output[i][j] = sum;
        }
    }
}

/**
 * Serial implementation of 2D convolution with stride and "same" padding
 * Output size: ceil(H/sH) × ceil(W/sW)
 */
void conv2d_serial_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output) {
    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;

    int out_H = (H + sH - 1) / sH;  // ceil(H/sH)
    int out_W = (W + sW - 1) / sW;  // ceil(W/sW)

    // For each output pixel (with stride)
    for (int out_i = 0; out_i < out_H; out_i++) {
        for (int out_j = 0; out_j < out_W; out_j++) {
            float sum = 0.0f;

            // Map output position to input position
            int i = out_i * sH;
            int j = out_j * sW;

            // Convolve with kernel
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int input_i = i + ki - pad_top;
                    int input_j = j + kj - pad_left;

                    if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                        sum += f[input_i][input_j] * g[ki][kj];
                    }
                }
            }
            output[out_i][out_j] = sum;
        }
    }
}


/**
 * OpenMP implementation with stride support
 */
void conv2d_omp_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output) {
    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;

    int out_H = (H + sH - 1) / sH;
    int out_W = (W + sW - 1) / sW;

    #pragma omp parallel for schedule(dynamic, 16) collapse(2)
    for (int out_i = 0; out_i < out_H; out_i++) {
        for (int out_j = 0; out_j < out_W; out_j++) {
            float sum = 0.0f;
            int i = out_i * sH;
            int j = out_j * sW;

            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int input_i = i + ki - pad_top;
                    int input_j = j + kj - pad_left;

                    if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                        sum += f[input_i][input_j] * g[ki][kj];
                    }
                }
            }
            output[out_i][out_j] = sum;
        }
    }
}

/**
 * OpenMP blocked parallel implementation of 2D convolution
 *
 * Advanced parallelization strategy inspired by omp.cpp:
 * - Uses block-based parallelization for better cache utilization
 * - Dynamic scheduling with larger chunks for reduced overhead
 * - Optimized for large matrices with better load balancing
 */
void conv2d_omp_blocked(float **f, int H, int W, float **g, int kH, int kW, float **output) {
    // Use precise padding calculation for both odd and even kernels
    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;
    
    // Calculate optimal block size based on matrix dimensions, kernel size, and thread count
    int num_threads = omp_get_max_threads();
    int kernel_ops = kH * kW;  // Operations per output pixel
    int block_size;
    
    // Base block size based on matrix dimensions
    if (H < 100) {
        block_size = 8;
    } else if (H < 500) {
        block_size = 16;
    } else if (H < 2000) {
        block_size = 32;
    } else {
        block_size = 64;
    }
    
    // Adjust block size based on kernel complexity
    if (kernel_ops <= 9) {
        block_size = block_size;
    } else if (kernel_ops <= 25) {
        block_size = (block_size / 2 > 2) ? block_size / 2 : 2;
    } else if (kernel_ops <= 49) {
        block_size = (block_size / 3 > 1) ? block_size / 3 : 1;
    } else {
        block_size = (block_size / 4 > 1) ? block_size / 4 : 1;
    }
    
    // Ensure minimum block size
    if (block_size < 1) block_size = 1;
    
    // Adjust for thread count and add upper bound
    if (num_threads > 16) {
        block_size = block_size * 2;
    }

    // Prevent division issues when H is small
    int max_block = (num_threads * 2 > 0 && H / (num_threads * 2) > 0) ?
                    H / (num_threads * 2) : H;
    block_size = (block_size < max_block) ? block_size : max_block;
    if (block_size < 1) block_size = 1;
    
    // Parallelize over output rows with dynamic scheduling
    #pragma omp parallel for schedule(dynamic, block_size) \
        shared(f, g, output, H, W, kH, kW, pad_top, pad_left)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            
            // Convolve with kernel
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    // Calculate input indices with padding
                    int input_i = i + ki - pad_top;
                    int input_j = j + kj - pad_left;
                    
                    // Apply "same" padding (zero-padding outside boundaries)
                    if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                        // Direct convolution without kernel flipping (correlation)
                        sum += f[input_i][input_j] * g[ki][kj];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

/**
 * Allocate memory for a 2D array stored as array of pointers
 * This layout provides cache-friendly access patterns and flexibility
 * for different row sizes, though all rows have the same size in our case
 */
float** allocate_2d_array(int rows, int cols) {
    float **array = (float**)malloc(rows * sizeof(float*));
    if (!array) {
        fprintf(stderr, "Error: Failed to allocate memory for row pointers\n");
        return NULL;
    }
    
    // Allocate each row
    for (int i = 0; i < rows; i++) {
        array[i] = (float*)malloc(cols * sizeof(float));
        if (!array[i]) {
            fprintf(stderr, "Error: Failed to allocate memory for row %d\n", i);
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }
    }
    
    return array;
}

/**
 * Free memory allocated for 2D array
 */
void free_2d_array(float **array, int rows) {
    if (array) {
        for (int i = 0; i < rows; i++) {
            if (array[i]) {
                free(array[i]);
            }
        }
        free(array);
    }
}

/**
 * Read array from file following the specification:
 * First line: height width
 * Following lines: space-separated float values
 */
int read_array_from_file(const char *filename, float ***array, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    // Read dimensions
    if (fscanf(file, "%d %d", rows, cols) != 2) {
        fprintf(stderr, "Error: Cannot read array dimensions from %s\n", filename);
        fclose(file);
        return -1;
    }
    
    // Validate dimensions
    if (*rows <= 0 || *cols <= 0) {
        fprintf(stderr, "Error: Invalid array dimensions in %s: %dx%d\n", filename, *rows, *cols);
        fclose(file);
        return -1;
    }
    
    // Allocate memory
    *array = allocate_2d_array(*rows, *cols);
    if (!*array) {
        fclose(file);
        return -1;
    }
    
    // Read data
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fscanf(file, "%f", &((*array)[i][j])) != 1) {
                fprintf(stderr, "Error: Cannot read element [%d][%d] from %s\n", i, j, filename);
                free_2d_array(*array, *rows);
                fclose(file);
                return -1;
            }
        }
    }
    
    fclose(file);
    return 0;
}

/**
 * Write array to file following the specification
 */
int write_array_to_file(const char *filename, float **array, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return -1;
    }
    
    // Write dimensions
    fprintf(file, "%d %d\n", rows, cols);
    
    // Write data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.3f", array[i][j]);
            if (j < cols - 1) {
                fprintf(file, " ");
            }
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    return 0;
}

/**
 * Generate random array with values between 0 and 1
 * Simple single-threaded implementation (not timed in performance tests)
 */
void generate_random_array(float **array, int rows, int cols) {
    // Seed random number generator with current time
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}



/**
 * Performance analysis function to test different thread counts (1 to max threads)
 * Inspired by omp.cpp's performance testing approach
 */
void performance_analysis_threads(float **f, int H, int W, float **g, int kH, int kW) {
    int max_threads = omp_get_max_threads();
    printf("\n=== Thread Performance Analysis (1-%d threads) ===\n", max_threads);
    printf("Matrix size: %dx%d, Kernel size: %dx%d\n", H, W, kH, kW);
    printf("Testing thread counts from 1 to %d\n\n", max_threads);
    
    struct timespec start, end;
    double best_time = 1e9;
    int best_threads = 1;
    double single_thread_time = 0.0;
    
    for (int threads = 1; threads <= max_threads; threads++) {
        omp_set_num_threads(threads);
        
        // Allocate output array
        float **output = allocate_2d_array(H, W);
        if (!output) {
            fprintf(stderr, "Error allocating memory for performance test\n");
            continue;
        }
        
        // Warm up (not timed)
        conv2d_omp_blocked(f, H, W, g, kH, kW, output);
        
        // Measure pure computation time only
        clock_gettime(CLOCK_MONOTONIC, &start);
        conv2d_omp_blocked(f, H, W, g, kH, kW, output);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double runtime = get_time_diff(start, end);
        
        // Store single thread time as baseline
        if (threads == 1) {
            single_thread_time = runtime;
        }
        
        // Calculate efficiency: (single_thread_time / (runtime * threads)) * 100%
        double efficiency = (single_thread_time / (runtime * threads)) * 100.0;
        
        printf("Threads: %2d | Runtime: %.6f seconds | Efficiency: %.1f%%", 
               threads, runtime, efficiency);
        
        if (runtime < best_time) {
            best_time = runtime;
            best_threads = threads;
            printf(" <-- BEST");
        }
        printf("\n");
        
        free_2d_array(output, H);
    }
    
    printf("\nOptimal thread count: %d (%.6f seconds)\n", best_threads, best_time);
}


/**
 * Calculate time difference in seconds
 */
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

/**
 * MPI-only distributed memory implementation with stride
 *
 * Data decomposition: Row-based decomposition of output
 * Each process computes a contiguous block of output rows
 * Requires halo exchange for overlapping input regions
 */
void conv2d_mpi_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;

    int out_H = (H + sH - 1) / sH;
    int out_W = (W + sW - 1) / sW;

    // Distribute output rows among processes
    int rows_per_proc = (out_H + size - 1) / size;
    int local_start = rank * rows_per_proc;
    int local_end = (rank + 1) * rows_per_proc;
    if (local_end > out_H) local_end = out_H;
    int local_rows = local_end - local_start;

    // Compute local output only if this process has rows assigned
    if (local_rows > 0) {
        // Calculate input region needed (with halo)
        int input_start = local_start * sH - pad_top;
        int input_end = (local_end - 1) * sH + kH - pad_top;

        // Clamp to valid input range
        if (input_start < 0) input_start = 0;
        if (input_end > H) input_end = H;
        int input_rows = input_end - input_start;

        // Allocate local input buffer if needed
        float **local_f = NULL;
        if (size > 1) {
            local_f = allocate_2d_array(input_rows, W);

            // Gather required input rows
            for (int i = 0; i < input_rows; i++) {
                memcpy(local_f[i], f[input_start + i], W * sizeof(float));
            }
        } else {
            local_f = f;
        }

        // Compute local output
        for (int out_i = 0; out_i < local_rows; out_i++) {
            for (int out_j = 0; out_j < out_W; out_j++) {
                float sum = 0.0f;
                int i = (local_start + out_i) * sH;
                int j = out_j * sW;

                for (int ki = 0; ki < kH; ki++) {
                    for (int kj = 0; kj < kW; kj++) {
                        int input_i = i + ki - pad_top;
                        int input_j = j + kj - pad_left;

                        if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                            int local_i = input_i - input_start;
                            sum += local_f[local_i][input_j] * g[ki][kj];
                        }
                    }
                }
                output[local_start + out_i][out_j] = sum;
            }
        }

        if (size > 1) {
            free_2d_array(local_f, input_rows);
        }
    }

    // Gather results to all processes
    // All processes must participate in all MPI_Bcast calls
    if (size > 1) {
        for (int p = 0; p < size; p++) {
            int p_start = p * rows_per_proc;
            int p_end = (p + 1) * rows_per_proc;
            if (p_end > out_H) p_end = out_H;
            int p_rows = p_end - p_start;

            // All processes participate, even if p_rows is 0
            for (int i = 0; i < p_rows; i++) {
                MPI_Bcast(output[p_start + i], out_W, MPI_FLOAT, p, comm);
            }
        }
    }
}

/**
 * Hybrid MPI+OpenMP implementation with stride
 *
 * Two-level parallelism:
 * - MPI: Distribute output rows across processes
 * - OpenMP: Parallelize computation within each process
 *
 * This is the main function for Assignment 2
 */
void conv2d_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;

    int out_H = (H + sH - 1) / sH;
    int out_W = (W + sW - 1) / sW;

    // Distribute output rows among processes
    int rows_per_proc = (out_H + size - 1) / size;
    int local_start = rank * rows_per_proc;
    int local_end = (rank + 1) * rows_per_proc;
    if (local_end > out_H) local_end = out_H;
    int local_rows = local_end - local_start;

    // Compute local output only if this process has rows assigned
    if (local_rows > 0) {
        // Calculate input region needed (with halo)
        int input_start = local_start * sH - pad_top;
        int input_end = (local_end - 1) * sH + kH - pad_top;

        if (input_start < 0) input_start = 0;
        if (input_end > H) input_end = H;
        int input_rows = input_end - input_start;

        // Allocate local input buffer if needed
        float **local_f = NULL;
        if (size > 1) {
            local_f = allocate_2d_array(input_rows, W);

            for (int i = 0; i < input_rows; i++) {
                memcpy(local_f[i], f[input_start + i], W * sizeof(float));
            }
        } else {
            local_f = f;
        }

        // Compute local output with OpenMP parallelization
        #pragma omp parallel for schedule(dynamic, 16) collapse(2)
        for (int out_i = 0; out_i < local_rows; out_i++) {
            for (int out_j = 0; out_j < out_W; out_j++) {
                float sum = 0.0f;
                int i = (local_start + out_i) * sH;
                int j = out_j * sW;

                for (int ki = 0; ki < kH; ki++) {
                    for (int kj = 0; kj < kW; kj++) {
                        int input_i = i + ki - pad_top;
                        int input_j = j + kj - pad_left;

                        if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                            int local_i = input_i - input_start;
                            sum += local_f[local_i][input_j] * g[ki][kj];
                        }
                    }
                }
                output[local_start + out_i][out_j] = sum;
            }
        }

        if (size > 1) {
            free_2d_array(local_f, input_rows);
        }
    }

    // Gather results to all processes
    // All processes must participate in all MPI_Bcast calls
    if (size > 1) {
        for (int p = 0; p < size; p++) {
            int p_start = p * rows_per_proc;
            int p_end = (p + 1) * rows_per_proc;
            if (p_end > out_H) p_end = out_H;
            int p_rows = p_end - p_start;

            // All processes participate, even if p_rows is 0
            for (int i = 0; i < p_rows; i++) {
                MPI_Bcast(output[p_start + i], out_W, MPI_FLOAT, p, comm);
            }
        }
    }
}

/**
 * MPI-only implementation with detailed performance statistics
 */
void conv2d_mpi_stride_stats(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm, PerfStats *stats) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Initialize stats
    stats->total_time = 0.0;
    stats->computation_time = 0.0;
    stats->communication_time = 0.0;
    stats->broadcast_time = 0.0;
    stats->memory_copy_time = 0.0;
    stats->bytes_communicated = 0;
    stats->num_communications = 0;

    double t_start, t_comp_start, t_comm_start;
    t_start = MPI_Wtime();

    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;

    int out_H = (H + sH - 1) / sH;
    int out_W = (W + sW - 1) / sW;

    stats->output_elements = (long long)out_H * out_W;

    // Distribute output rows among processes
    int rows_per_proc = (out_H + size - 1) / size;
    int local_start = rank * rows_per_proc;
    int local_end = (rank + 1) * rows_per_proc;
    if (local_end > out_H) local_end = out_H;
    int local_rows = local_end - local_start;

    // Compute local output only if this process has rows assigned
    if (local_rows > 0) {
        // Calculate input region needed (with halo)
        int input_start = local_start * sH - pad_top;
        int input_end = (local_end - 1) * sH + kH - pad_top;

        // Clamp to valid input range
        if (input_start < 0) input_start = 0;
        if (input_end > H) input_end = H;
        int input_rows = input_end - input_start;

        // Allocate local input buffer if needed
        float **local_f = NULL;
        if (size > 1) {
            t_comm_start = MPI_Wtime();
            local_f = allocate_2d_array(input_rows, W);

            // Gather required input rows (memory copy)
            for (int i = 0; i < input_rows; i++) {
                memcpy(local_f[i], f[input_start + i], W * sizeof(float));
            }
            stats->memory_copy_time += MPI_Wtime() - t_comm_start;
            stats->bytes_communicated += (long long)input_rows * W * sizeof(float);
        } else {
            local_f = f;
        }

        // Compute local output
        t_comp_start = MPI_Wtime();
        for (int out_i = 0; out_i < local_rows; out_i++) {
            for (int out_j = 0; out_j < out_W; out_j++) {
                float sum = 0.0f;
                int i = (local_start + out_i) * sH;
                int j = out_j * sW;

                for (int ki = 0; ki < kH; ki++) {
                    for (int kj = 0; kj < kW; kj++) {
                        int input_i = i + ki - pad_top;
                        int input_j = j + kj - pad_left;

                        if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                            int local_i = input_i - input_start;
                            sum += local_f[local_i][input_j] * g[ki][kj];
                        }
                    }
                }
                output[local_start + out_i][out_j] = sum;
            }
        }
        stats->computation_time += MPI_Wtime() - t_comp_start;

        if (size > 1) {
            free_2d_array(local_f, input_rows);
        }
    }

    // Gather results to all processes
    if (size > 1) {
        t_comm_start = MPI_Wtime();
        for (int p = 0; p < size; p++) {
            int p_start = p * rows_per_proc;
            int p_end = (p + 1) * rows_per_proc;
            if (p_end > out_H) p_end = out_H;
            int p_rows = p_end - p_start;

            // All processes participate, even if p_rows is 0
            for (int i = 0; i < p_rows; i++) {
                MPI_Bcast(output[p_start + i], out_W, MPI_FLOAT, p, comm);
                stats->num_communications++;
                stats->bytes_communicated += (long long)out_W * sizeof(float);
            }
        }
        stats->broadcast_time = MPI_Wtime() - t_comm_start;
        stats->communication_time = stats->broadcast_time + stats->memory_copy_time;
    }

    stats->total_time = MPI_Wtime() - t_start;
}

/**
 * Hybrid MPI+OpenMP implementation with detailed performance statistics
 */
void conv2d_stride_stats(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm, PerfStats *stats) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Initialize stats
    stats->total_time = 0.0;
    stats->computation_time = 0.0;
    stats->communication_time = 0.0;
    stats->broadcast_time = 0.0;
    stats->memory_copy_time = 0.0;
    stats->bytes_communicated = 0;
    stats->num_communications = 0;

    double t_start, t_comp_start, t_comm_start;
    t_start = MPI_Wtime();

    int pad_top = (kH - 1) / 2;
    int pad_left = (kW - 1) / 2;

    int out_H = (H + sH - 1) / sH;
    int out_W = (W + sW - 1) / sW;

    stats->output_elements = (long long)out_H * out_W;

    // Distribute output rows among processes
    int rows_per_proc = (out_H + size - 1) / size;
    int local_start = rank * rows_per_proc;
    int local_end = (rank + 1) * rows_per_proc;
    if (local_end > out_H) local_end = out_H;
    int local_rows = local_end - local_start;

    // Compute local output only if this process has rows assigned
    if (local_rows > 0) {
        // Calculate input region needed (with halo)
        int input_start = local_start * sH - pad_top;
        int input_end = (local_end - 1) * sH + kH - pad_top;

        if (input_start < 0) input_start = 0;
        if (input_end > H) input_end = H;
        int input_rows = input_end - input_start;

        // Allocate local input buffer if needed
        float **local_f = NULL;
        if (size > 1) {
            t_comm_start = MPI_Wtime();
            local_f = allocate_2d_array(input_rows, W);

            for (int i = 0; i < input_rows; i++) {
                memcpy(local_f[i], f[input_start + i], W * sizeof(float));
            }
            stats->memory_copy_time += MPI_Wtime() - t_comm_start;
            stats->bytes_communicated += (long long)input_rows * W * sizeof(float);
        } else {
            local_f = f;
        }

        // Compute local output with OpenMP parallelization
        t_comp_start = MPI_Wtime();
        #pragma omp parallel for schedule(dynamic, 16) collapse(2)
        for (int out_i = 0; out_i < local_rows; out_i++) {
            for (int out_j = 0; out_j < out_W; out_j++) {
                float sum = 0.0f;
                int i = (local_start + out_i) * sH;
                int j = out_j * sW;

                for (int ki = 0; ki < kH; ki++) {
                    for (int kj = 0; kj < kW; kj++) {
                        int input_i = i + ki - pad_top;
                        int input_j = j + kj - pad_left;

                        if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                            int local_i = input_i - input_start;
                            sum += local_f[local_i][input_j] * g[ki][kj];
                        }
                    }
                }
                output[local_start + out_i][out_j] = sum;
            }
        }
        stats->computation_time += MPI_Wtime() - t_comp_start;

        if (size > 1) {
            free_2d_array(local_f, input_rows);
        }
    }

    // Gather results to all processes
    if (size > 1) {
        t_comm_start = MPI_Wtime();
        for (int p = 0; p < size; p++) {
            int p_start = p * rows_per_proc;
            int p_end = (p + 1) * rows_per_proc;
            if (p_end > out_H) p_end = out_H;
            int p_rows = p_end - p_start;

            // All processes participate, even if p_rows is 0
            for (int i = 0; i < p_rows; i++) {
                MPI_Bcast(output[p_start + i], out_W, MPI_FLOAT, p, comm);
                stats->num_communications++;
                stats->bytes_communicated += (long long)out_W * sizeof(float);
            }
        }
        stats->broadcast_time = MPI_Wtime() - t_comm_start;
        stats->communication_time = stats->broadcast_time + stats->memory_copy_time;
    }

    stats->total_time = MPI_Wtime() - t_start;
}
