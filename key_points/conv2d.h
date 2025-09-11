#ifndef CONV2D_H
#define CONV2D_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

// Function prototypes for convolution operations
// Serial (single-threaded) implementations
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **output);

// Parallel (multi-threaded) implementations
void conv2d_omp_blocked(float **f, int H, int W, float **g, int kH, int kW, float **output);

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

#endif // CONV2D_H