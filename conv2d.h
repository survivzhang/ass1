#ifndef CONV2D_H
#define CONV2D_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

// Function prototypes for convolution operations
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **output);
void conv2d_parallel(float **f, int H, int W, float **g, int kH, int kW, float **output);

// Utility functions for memory management
float** allocate_2d_array(int rows, int cols);
void free_2d_array(float **array, int rows);

// I/O functions
int read_array_from_file(const char *filename, float ***array, int *rows, int *cols);
int write_array_to_file(const char *filename, float **array, int rows, int cols);
void generate_random_array(float **array, int rows, int cols);

// Timing utilities
double get_time_diff(struct timespec start, struct timespec end);

#endif // CONV2D_H