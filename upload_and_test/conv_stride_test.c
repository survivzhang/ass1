#include "conv2d.h"
#include <math.h>

/**
 * Group Member: Jiazheng Guo(24070858), Zichen Zhang(24064091)
 * Assignment 2: MPI+OpenMP 2D Convolution with Stride
 */

void print_usage(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("2D Convolution with stride, MPI and OpenMP parallelization\n\n");
    printf("Options:\n");
    printf("  -f FILE     Input feature map file\n");
    printf("  -g FILE     Input kernel file\n");
    printf("  -o FILE     Output file (optional)\n");
    printf("  -H HEIGHT   Generate random input with HEIGHT rows\n");
    printf("  -W WIDTH    Generate random input with WIDTH columns\n");
    printf("  -kH HEIGHT  Kernel height\n");
    printf("  -kW WIDTH   Kernel width\n");
    printf("  -sH STRIDE  Vertical stride (default: 1)\n");
    printf("  -sW STRIDE  Horizontal stride (default: 1)\n");
    printf("  -t THREADS  Number of OpenMP threads per MPI process (optional)\n");
    printf("  -m MODE     Mode: serial, omp, mpi, hybrid (default: hybrid)\n");
    printf("  --help      Show this help message\n\n");
    printf("Examples:\n");
    printf("  mpirun -np 4 %s -H 1000 -W 1000 -kH 3 -kW 3 -sW 2 -sH 3\n", program_name);
    printf("  mpirun -np 2 %s -f f.txt -g g.txt -sW 1 -sH 1 -o output.txt\n", program_name);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Command line arguments
    char *input_file = NULL;
    char *kernel_file = NULL;
    char *output_file = NULL;
    int H = 0, W = 0, kH = 0, kW = 0;
    int sH = 1, sW = 1;  // Default stride
    int num_threads = 0;
    char *mode = "hybrid";

    // Parse command line arguments (only rank 0 prints errors)
    int opt;
    while ((opt = getopt(argc, argv, "f:g:o:H:W:k:s:t:m:")) != -1) {
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
            case 'k':
                if (optind < argc && argv[optind][0] == 'H') {
                    kH = atoi(argv[optind]);
                    optind++;
                } else if (optind < argc && argv[optind][0] == 'W') {
                    kW = atoi(argv[optind]);
                    optind++;
                }
                break;
            case 's':
                if (optind < argc && argv[optind][0] == 'H') {
                    sH = atoi(argv[optind]);
                    optind++;
                } else if (optind < argc && argv[optind][0] == 'W') {
                    sW = atoi(argv[optind]);
                    optind++;
                }
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'm':
                mode = optarg;
                break;
            default:
                if (rank == 0) {
                    print_usage(argv[0]);
                }
                MPI_Finalize();
                return 1;
        }
    }

    // Manual parsing for -kH, -kW, -sH, -sW
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-kH") == 0) {
            kH = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-kW") == 0) {
            kW = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-sH") == 0) {
            sH = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-sW") == 0) {
            sW = atoi(argv[i + 1]);
        }
    }

    // Handle help
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    // Set number of threads if specified
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Variables for arrays
    float **f = NULL, **g = NULL, **output = NULL;
    int f_rows, f_cols, g_rows, g_cols;

    // Only rank 0 reads/generates data
    if (rank == 0) {
        if (H > 0 && W > 0 && kH > 0 && kW > 0) {
            printf("Generating random %dx%d input and %dx%d kernel with stride %dx%d\n",
                   H, W, kH, kW, sH, sW);

            f = allocate_2d_array(H, W);
            g = allocate_2d_array(kH, kW);

            if (!f || !g) {
                fprintf(stderr, "Error allocating memory\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            generate_random_array(f, H, W);
            generate_random_array(g, kH, kW);

            f_rows = H; f_cols = W;
            g_rows = kH; g_cols = kW;

            if (input_file) write_array_to_file(input_file, f, H, W);
            if (kernel_file) write_array_to_file(kernel_file, g, kH, kW);

        } else if (input_file && kernel_file) {
            printf("Reading input from %s and kernel from %s\n", input_file, kernel_file);

            if (read_array_from_file(input_file, &f, &f_rows, &f_cols) != 0 ||
                read_array_from_file(kernel_file, &g, &g_rows, &g_cols) != 0) {
                fprintf(stderr, "Error reading files\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            H = f_rows; W = f_cols;
            kH = g_rows; kW = g_cols;
        } else {
            fprintf(stderr, "Error: Must provide input files or generation parameters\n");
            print_usage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast dimensions to all processes
    int dims[6] = {H, W, kH, kW, sH, sW};
    MPI_Bcast(dims, 6, MPI_INT, 0, MPI_COMM_WORLD);
    H = dims[0]; W = dims[1]; kH = dims[2]; kW = dims[3]; sH = dims[4]; sW = dims[5];

    // Allocate arrays on all processes
    if (rank != 0) {
        f = allocate_2d_array(H, W);
        g = allocate_2d_array(kH, kW);
    }

    // Broadcast input data
    for (int i = 0; i < H; i++) {
        MPI_Bcast(f[i], W, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    for (int i = 0; i < kH; i++) {
        MPI_Bcast(g[i], kW, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Calculate output size
    int out_H = (H + sH - 1) / sH;
    int out_W = (W + sW - 1) / sW;
    output = allocate_2d_array(out_H, out_W);

    // Timing
    struct timespec start, end;
    double elapsed;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Running %s mode: MPI processes=%d, OpenMP threads=%d\n",
               mode, size, omp_get_max_threads());
        printf("Output size: %dx%d\n", out_H, out_W);
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    if (strcmp(mode, "serial") == 0 && rank == 0) {
        conv2d_serial_stride(f, H, W, g, kH, kW, sH, sW, output);
    } else if (strcmp(mode, "omp") == 0 && rank == 0) {
        conv2d_omp_stride(f, H, W, g, kH, kW, sH, sW, output);
    } else if (strcmp(mode, "mpi") == 0) {
        conv2d_mpi_stride(f, H, W, g, kH, kW, sH, sW, output, MPI_COMM_WORLD);
    } else {
        // hybrid (default)
        conv2d_stride(f, H, W, g, kH, kW, sH, sW, output, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = get_time_diff(start, end);

    if (rank == 0) {
        printf("Computation time: %.6f seconds\n", elapsed);

        if (output_file) {
            printf("Writing output to %s\n", output_file);
            write_array_to_file(output_file, output, out_H, out_W);
        }
    }

    // Cleanup
    free_2d_array(f, H);
    free_2d_array(g, kH);
    free_2d_array(output, out_H);

    MPI_Finalize();
    return 0;
}
