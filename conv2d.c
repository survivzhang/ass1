#include "conv2d.h"

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
    
    // Calculate optimal block size based on matrix dimensions
    int block_size = (H > 100) ? 16 : 8;  // Larger blocks for bigger matrices
    
    // Parallelize over output rows with dynamic scheduling and larger chunks
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
 * Configure OpenMP settings for optimal performance
 * Inspired by omp.cpp's thread configuration approach
 */
void configure_omp_settings(int num_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    omp_set_schedule(omp_sched_dynamic, 1);
    printf("OpenMP configured with %d threads\n", num_threads);
    #endif
}

/**
 * Performance analysis function to test different thread counts (1 to max threads)
 * Inspired by omp.cpp's performance testing approach
 */
void performance_analysis_threads(float **f, int H, int W, float **g, int kH, int kW) {
    int max_threads = get_omp_thread_count();
    printf("\n=== Thread Performance Analysis (1-%d threads) ===\n", max_threads);
    printf("Matrix size: %dx%d, Kernel size: %dx%d\n", H, W, kH, kW);
    printf("Testing thread counts from 1 to %d\n\n", max_threads);
    
    struct timespec start, end;
    double best_time = 1e9;
    int best_threads = 1;
    
    for (int threads = 1; threads <= max_threads; threads++) {
        // Configure threads
        configure_omp_settings(threads);
        
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
        
        // Calculate efficiency
        double efficiency = (best_time / runtime) * 100.0;
        if (threads == 1) {
            efficiency = 100.0; // Baseline
            best_time = runtime;
        }
        
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
 * Get the number of available OpenMP threads
 */
int get_omp_thread_count() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return 1;
    #endif
}

/**
 * Calculate time difference in seconds
 */
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}