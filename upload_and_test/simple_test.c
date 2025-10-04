#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Simple MPI test - Hello from rank %d\n", rank);

        // Test division
        int H = 6, sH = 1;
        int out_H = (H + sH - 1) / sH;
        printf("H=%d, sH=%d, out_H=%d\n", H, sH, out_H);

        // Test with sH=0 (should crash)
        // sH = 0;
        // out_H = (H + sH - 1) / sH;  // This would cause division by zero
    }

    MPI_Finalize();
    return 0;
}
