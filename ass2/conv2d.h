#ifndef CONV2D_H
#define CONV2D_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>

/**
 * Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
 */

// Function prototypes for convolution operations
// Serial (single-threaded) implementations
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **output);
void conv2d_serial_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output);

// Parallel (multi-threaded) implementations
void conv2d_omp_blocked(float **f, int H, int W, float **g, int kH, int kW, float **output);
void conv2d_omp_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output);

// MPI implementations
void conv2d_mpi_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm);
void conv2d_stride(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm);

// Utility functions for memory management
float** allocate_2d_array(int rows, int cols);
void free_2d_array(float **array, int rows);

// I/O functions
int read_array_from_file(const char *filename, float ***array, int *rows, int *cols);
int write_array_to_file(const char *filename, float **array, int rows, int cols);

// Random array generation
void generate_random_array(float **array, int rows, int cols);

// Performance analysis utilities
void performance_analysis_threads(float **f, int H, int W, float **g, int kH, int kW);

// Timing utilities
double get_time_diff(struct timespec start, struct timespec end);

// Performance statistics structure
typedef struct {
    double total_time;
    double computation_time;
    double communication_time;
    double broadcast_time;
    double memory_copy_time;
    long long output_elements;
    long long bytes_communicated;
    int num_communications;
} PerfStats;

// MPI implementations with performance statistics
void conv2d_mpi_stride_stats(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm, PerfStats *stats);
void conv2d_stride_stats(float **f, int H, int W, float **g, int kH, int kW, int sH, int sW, float **output, MPI_Comm comm, PerfStats *stats);

#endif // CONV2D_H