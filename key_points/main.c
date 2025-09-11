#include "conv2d.h"
#include <math.h>

void print_usage(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("2D Convolution with OpenMP parallelization\n\n");
    printf("Options:\n");
    printf("  -f FILE     Input feature map file\n");
    printf("  -g FILE     Input kernel file\n");
    printf("  -o FILE     Output file (optional)\n");
    printf("  -H HEIGHT   Generate random input with HEIGHT rows\n");
    printf("  -W WIDTH    Generate random input with WIDTH columns\n");
    printf("  -h HEIGHT   Kernel height (for random generation)\n");
    printf("  -w WIDTH    Kernel width (for random generation)\n");
    printf("  -t THREADS  Number of OpenMP threads (optional)\n");
    printf("  -s          Use serial implementation only\n");
    printf("  -p          Use parallel implementation only\n");
    printf("  -c          Compare serial and parallel implementations\n");
    printf("  -a          Analyze performance across different thread counts\n");
    printf("  --help      Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s -f f.txt -g g.txt\n", program_name);
    printf("  %s -f f.txt -g g.txt -o output.txt\n", program_name);
    printf("  %s -H 1000 -W 1000 -h 3 -w 3\n", program_name);
    printf("  %s -H 1000 -W 1000 -h 3 -w 3 -a\n", program_name);
    printf("  %s -f f0.txt -g g0.txt -a\n", program_name);
}

int main(int argc, char **argv) {
    // Command line arguments
    char *input_file = NULL;
    char *kernel_file = NULL;
    char *output_file = NULL;
    int H = 0, W = 0, kH = 0, kW = 0;
    int num_threads = 0;
    int use_serial = 0, use_parallel = 0, compare_mode = 0, analyze_mode = 0;
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:g:o:H:W:h:w:t:spca")) != -1) {
        switch (opt) {
            case 'f':
                input_file = optarg;
                break;
            case 'g':
                kernel_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'H':
                H = atoi(optarg);
                break;
            case 'W':
                W = atoi(optarg);
                break;
            case 'h':
                kH = atoi(optarg);
                break;
            case 'w':
                kW = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 's':
                use_serial = 1;
                break;
            case 'p':
                use_parallel = 1;
                break;
            case 'c':
                compare_mode = 1;
                break;
            case 'a':
                analyze_mode = 1;
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Handle help request
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Set number of threads if specified
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Variables for arrays
    float **f = NULL, **g = NULL, **output = NULL;
    int f_rows, f_cols, g_rows, g_cols;
    
    // Read or generate input arrays
    if (H > 0 && W > 0 && kH > 0 && kW > 0) {
        // Generate random arrays (not timed)
        printf("Generating random %dx%d input and %dx%d kernel\n", H, W, kH, kW);
        
        f = allocate_2d_array(H, W);
        g = allocate_2d_array(kH, kW);
        
        if (!f || !g) {
            fprintf(stderr, "Error allocating memory for arrays\n");
            return 1;
        }
        
        generate_random_array(f, H, W);
        generate_random_array(g, kH, kW);
        
        f_rows = H; f_cols = W;
        g_rows = kH; g_cols = kW;
        
        // Save generated arrays if filenames provided (not timed)
        if (input_file) {
            printf("Saving generated input to %s\n", input_file);
            write_array_to_file(input_file, f, H, W);
        }
        if (kernel_file) {
            printf("Saving generated kernel to %s\n", kernel_file);
            write_array_to_file(kernel_file, g, kH, kW);
        }
        
    } else if (input_file && kernel_file) {
        // Read from files (not timed)
        printf("Reading input from %s and kernel from %s\n", input_file, kernel_file);
        
        if (read_array_from_file(input_file, &f, &f_rows, &f_cols) != 0) {
            fprintf(stderr, "Error reading input file\n");
            return 1;
        }
        
        if (read_array_from_file(kernel_file, &g, &g_rows, &g_cols) != 0) {
            fprintf(stderr, "Error reading kernel file\n");
            free_2d_array(f, f_rows);
            return 1;
        }
        
        H = f_rows;
        W = f_cols;
        kH = g_rows;
        kW = g_cols;
        
    } else {
        fprintf(stderr, "Error: Must provide either input files (-f, -g) or generation parameters (-H, -W, -h, -w)\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Validate kernel size (should be odd for symmetric padding)
    if (kH % 2 == 0 || kW % 2 == 0) {
        printf("Warning: Kernel dimensions should be odd for symmetric padding\n");
    }
    
    // Allocate output array
    output = allocate_2d_array(H, W);
    if (!output) {
        fprintf(stderr, "Error allocating memory for output array\n");
        free_2d_array(f, f_rows);
        free_2d_array(g, g_rows);
        return 1;
    }
    
    // Timing variables
    struct timespec start, end;
    double serial_time = 0.0, parallel_time = 0.0;
    
    // Perform performance analysis if requested
    if (analyze_mode) {
        printf("Running performance analysis across 1-16 threads...\n");
        performance_analysis_threads(f, H, W, g, kH, kW);
        
        // Clean up and exit
        free_2d_array(f, f_rows);
        free_2d_array(g, g_rows);
        free_2d_array(output, H);
        return 0;
    }
    
    // Perform convolution based on mode
    if (use_serial || compare_mode) {
        printf("Running serial convolution (single-threaded)...\n");
        // Measure pure computation time only
        clock_gettime(CLOCK_MONOTONIC, &start);
        conv2d_serial(f, H, W, g, kH, kW, output);
        clock_gettime(CLOCK_MONOTONIC, &end);
        serial_time = get_time_diff(start, end);
        printf("Serial computation time: %.6f seconds\n", serial_time);
        
        if (!compare_mode && output_file) {
            printf("Writing output to %s\n", output_file);
            write_array_to_file(output_file, output, H, W);
        }
    }
    
    if (use_parallel || compare_mode || (!use_serial && !use_parallel)) {
        // Allocate separate output for parallel version in compare mode
        float **parallel_output = compare_mode ? allocate_2d_array(H, W) : output;
        
        printf("Running parallel convolution with %d threads...\n", omp_get_max_threads());
        // Measure pure computation time only
        clock_gettime(CLOCK_MONOTONIC, &start);
        conv2d_omp_blocked(f, H, W, g, kH, kW, parallel_output);
        clock_gettime(CLOCK_MONOTONIC, &end);
        parallel_time = get_time_diff(start, end);
        printf("Parallel computation time: %.6f seconds\n", parallel_time);
        
        if (!compare_mode && output_file) {
            printf("Writing output to %s\n", output_file);
            write_array_to_file(output_file, parallel_output, H, W);
        }
        
        // Verify results in compare mode
        if (compare_mode) {
            printf("Verifying results...\n");
            int correct = 1;
            float max_diff = 0.0f;
            
            for (int i = 0; i < H && correct; i++) {
                for (int j = 0; j < W && correct; j++) {
                    float diff = fabsf(output[i][j] - parallel_output[i][j]);
                    if (diff > max_diff) max_diff = diff;
                    if (diff > 1e-5f) {  // Allow small floating-point differences
                        printf("Mismatch at [%d][%d]: serial=%.6f, parallel=%.6f, diff=%.6f\n", 
                               i, j, output[i][j], parallel_output[i][j], diff);
                        correct = 0;
                    }
                }
            }
            
            if (correct) {
                printf("✓ Results match! Maximum difference: %.2e\n", max_diff);
                if (serial_time > 0) {
                    printf("Speedup: %.2fx\n", serial_time / parallel_time);
                }
                
                if (output_file) {
                    printf("Writing verified output to %s\n", output_file);
                    write_array_to_file(output_file, parallel_output, H, W);
                }
            } else {
                printf("✗ Results do not match!\n");
            }
            
            free_2d_array(parallel_output, H);
        }
    }
    
    // Print performance summary
    if (compare_mode && serial_time > 0 && parallel_time > 0) {
        printf("\nPerformance Summary:\n");
        printf("Array size: %dx%d, Kernel size: %dx%d\n", H, W, kH, kW);
        printf("Threads: %d\n", omp_get_max_threads());
        printf("Serial computation time:   %.6f seconds\n", serial_time);
        printf("Parallel computation time: %.6f seconds\n", parallel_time);
        printf("Speedup:                   %.2fx\n", serial_time / parallel_time);
        printf("Efficiency:                %.2f%%\n", 100.0 * serial_time / (parallel_time * omp_get_max_threads()));
    }
    
    // Clean up
    free_2d_array(f, f_rows);
    free_2d_array(g, g_rows);
    free_2d_array(output, H);
    
    return 0;
}