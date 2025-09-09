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
    int pad_bottom = kH - 1 - pad_top;
    int pad_left = (kW - 1) / 2;
    int pad_right = kW - 1 - pad_left;
    
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
 * Parallel implementation of 2D convolution using OpenMP
 * 
 * Parallelization strategy:
 * - Parallelized over output rows using #pragma omp parallel for
 * - Each thread processes a contiguous block of rows to maintain cache locality
 * - Static scheduling is used for predictable load balancing
 * - Shared variables: f, g, output, H, W, kH, kW (read-only)
 * - Private variables: i, j, ki, kj, sum, pad_h, pad_w, input_i, input_j
 */
void conv2d_parallel(float **f, int H, int W, float **g, int kH, int kW, float **output) {
    // Use precise padding calculation for both odd and even kernels
    int pad_top = (kH - 1) / 2;
    int pad_bottom = kH - 1 - pad_top;
    int pad_left = (kW - 1) / 2;
    int pad_right = kW - 1 - pad_left;
    
    // Parallelize over output rows
    #pragma omp parallel for schedule(static) \
        shared(f, g, output, H, W, kH, kW, pad_top, pad_left) \
        // private(i, j, ki, kj, sum, input_i, input_j)
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
 * Uses thread-safe random number generation
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
 * Calculate time difference in seconds
 */
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}