

# Assignment 1: Parallel 2D Convolution with OpenMP

**Authors:** Jiazheng Guo(24070858),Zichen Zhang(24064091)
**Due:** Friday, 12th September 2025

## 1. Introduction

Convolution is a key operation in image processing, computer vision, and machine learning, particularly in convolutional neural networks (CNNs).In this project, we implemented both a serial and a parallel version of 2D convolution in C using OpenMP, with the aim of achieving significant performance improvements while ensuring correctness. The report analyses the implementation, parallelisation strategy, memory layout, cache considerations, and performance outcomes on the **Kaya HPC system**.

---

## 2. Implementation Details

### 2.1 Serial implementation of 2D convolution

```
// Serial implementation of 2D convolution with "same" padding
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **output) {
    // ...
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int input_i = i + ki - pad_top;
                    int input_j = j + kj - pad_left;
                    if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                        sum += f[input_i][input_j] * g[ki][kj];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}
```

The `conv2d_serial` function in the file implements a serial version of 2D convolution. The algorithm operates as follows:

1. Outer Loop: Iterate through each pixel position `(i, j)` in the output matrix `output` using two nested loops.
2. Inner Loop: For each output pixel, iterate through each element `(ki, kj)` in the convolution kernel g using another set of nested loops.
3. Computation: Within the inner loop, the pixel position `(input_i, input_j)` in the input image f corresponding to kernel element `g[ki][kj]`is calculated via `(i + ki - pad_top, j + kj - pad_left)`.
4. Boundary Handling: `If (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W)`, perform “same” padding. If the calculated input pixel position lies within the original image boundaries, multiply its value by the corresponding convolution kernel value and accumulate it into `sum`; if outside the boundaries, perform no operation, which is equivalent to padding with zeros.
5. Result: After accumulation completes, assign the value of `sum` to `output[i][j]`, completing the calculation for one output pixel.

### 2.2 OpenMP parallel implementation of 2D convolution

This function uses the `#pragma omp parallel for` directive to parallelize the outermost output row loop.

```
void conv2d_omp_parallel(float **f, int H, int W, float **g, int kH, int kW, float **output) {
    // ...
    #pragma omp parallel for schedule(dynamic, 1) \
        shared(f, g, output, H, W, kH, kW, pad_top, pad_left)
    for (int i = 0; i < H; i++) {
        //...
    }
```

1. Parallelization Strategy: Multiple threads jointly process different rows of the output image. The OpenMP runtime assigns iterations of the outer for loop to different threads.

2. Scheduling Strategy: `schedule(dynamic, 1)` indicates dynamic scheduling, where each loop iteration (i.e., one row) is assigned to an available thread. This strategy facilitates better load balancing, especially when handling convolutional kernels of varying sizes, as computational intensity may differ per row.

3. Variable Management:

​        • `shared(...)`: Variables like `f`, `g`, and `output` are shared across all threads. `f` and `g` are read-only, while output is written to by each thread for its assigned row.

​        • `Private Variables`: Loop variables `i`, `j`, `ki`, `kj`, and local accumulators like `sum` are private to each thread, preventing data races.

### 2.3 OpenMP blocked parallel implementation of 2D convolution

This function also employs the `#pragma omp parallel for` directive, but its primary distinction lies in utilizing a block scheduling strategy to optimize cache utilization and reduce thread synchronization overhead.

    void conv2d_omp_blocked(float **f, int H, int W, float **g, int kH, int kW, float **output) {
        // ...
        int block_size = (H > 100) ? 16 : 8;
        #pragma omp parallel for schedule(dynamic, block_size) \
            shared(f, g, output, H, W, kH, kW, pad_top, pad_left)
        for (int i = 0; i < H; i++) {
           // ...
        }

1. Parallelization Strategy: Identical to `conv2d_omp_parallel`, it still parallels the output rows.

2. Block Scheduling: The key difference lies in `schedule(dynamic, block_size)`. It bundles loop iterations (rows) into blocks of size `block_size`, then dynamically assigns these blocks to individual threads.

3. Optimization Goals:

​        • Reduce Scheduling Overhead: Assigning blocks instead of individual rows per scheduling round reduces thread management and synchronization frequency, proving particularly effective for large matrices.

​        • Improve cache locality: When a thread processes contiguous blocks of rows, its access patterns to the input image `f` and output image `output` become more localized. While processing a row, the corresponding region of the input image is likely loaded into the cache. When processing the next row within the same block, this data remains in the cache, thereby increasing cache hit rates.

## 3. Parallelisation Strategy

We provide two distinct parallelization implementations, which are not redundant but rather optimized for different scenarios.

#### 3.1 Design Rationale

1. `conv2d_omp_parallel` (Basic Parallelization): As the primary parallelization approach, it offers a simple, easy-to-understand parallel model. It demonstrates that the outermost row loop can be parallelized without data races, since each thread only writes its own row's data and performs read-only accesses to the shared input data `f` and `g`.
2. `conv2d_omp_blocked` (Advanced Parallelism): This represents a further optimization beyond the basic parallelization. It recognizes that in parallel computing, reducing overhead and optimizing cache utilization are equally important alongside parallelization itself. By introducing block scheduling, this implementation demonstrates how sacrificing some load balancing flexibility can yield higher overall performance.

#### 3.2 Comparison with the general parallel scheme

1. Parallelism within loops: Another parallelization approach involves parallelizing the innermost or intermediate loops (e.g., the convolution kernel loops `ki` or `kj`).

2. Advantages of this parallelization algorithm: Parallelizing the outermost row loop is a superior choice because it handles the coarsest-grained task (a complete output row). Each task is sufficiently large to amortize thread creation and management overhead, enabling efficient coarse-grained parallelism. Parallelizing inner loops would result in overly fine-grained tasks, leading to frequent synchronization and data races that could degrade performance. Thus, the code's choice to parallelize the outer loop aligns with parallel programming best practices.

---

## 4. Memory Layout and Cache Considerations

### 4.1 Memory Layout analysis

```
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
```

1. Implementation: The `allocate_2d_array` function uses `malloc` to allocate memory for each row separately, managing these row pointers with a pointer array `float** array`.

• `float **array = (float**)malloc(rows * sizeof(float*));`

• `array[i] = (float*)malloc(cols * sizeof(float));`

2. Design Rationale:

​        . Row-wise contiguity: Although the entire matrix may not be contiguous in memory, the `allocate_2d_array` function ensures that data within each row is stored contiguously.

​        . Flexible access pattern: This layout enables access via the intuitive syntax `array[row][col]`, aligning well with C's row-major memory access conventions.

3. Comparison with Single-Block Memory Layout:

​       • Single-Block Memory Layout (`float* array`): Some implementations allocate a contiguous memory block for the entire matrix, accessed via `array[i * cols + j]`. This approach theoretically offers optimal spatial locality since the entire matrix is contiguous.

​       • Advantages of this code: While conv2d.c sacrifices overall contiguity, it preserves row-major contiguity, which suffices for the row-major access pattern in the code (`for (int i = 0; ...)`). More importantly, it avoids the multiplication operation (`i * cols`) for each access, potentially yielding a slight performance advantage on certain compilers or hardware.

### 4.2 Cache Considerrations

```
for (int j = 0; j < W; j++) { // This is the inner-most loop for the output matrix
    // ...
    sum += f[input_i][input_j] * g[ki][kj]; // Accesses inside the inner loops
    // ...
}
```

1. Optimization Principle:

​       • In the `conv2d_serial` function, the inner loop iterates over the column indices `j` of the output matrix. This means that during loop execution, accesses to `output[i][j]` are contiguous (`j` increments). Since each row is stored contiguously in memory, this access pattern exhibits strong spatial locality, enabling efficient utilization of cache lines.

​       • Although the convolution operation exhibits complex access patterns when accessing the input image `f`, placing column accesses to the `output` matrix within the inner loop ensures that at least the write operations to the output matrix are cache-friendly.

​      • For `conv2d_omp_blocked`, the block scheduling strategy further enhances this locality. Each thread processes a contiguous block of `output` rows, making accesses to `f` and `g` more localized to some extent.

2. Comparison with conventional parallel scheduling:

​     • Conventional dynamic scheduling (`schedule(dynamic, 1)`): `conv2d_omp_parallel` employs this approach. It assigns tasks row-by-row, enabling highly granular load balancing. However, this strategy may cause threads to frequently request new tasks from the operating system scheduler, thereby increasing scheduling overhead. Additionally, different rows may be processed by different threads, potentially causing input data access patterns to jump between threads and reducing cache locality.

​    • Advantages of this code: The block scheduling strategy in `conv2d_omp_blocked` strikes a balance between these two approaches. By processing contiguous blocks of rows, a thread's accesses to the input image are more likely to concentrate within a specific memory region while handling its assigned tasks, thereby improving cache hit rates. Simultaneously, the larger block size reduces scheduling overhead, enhancing parallel efficiency—particularly when both the matrix and convolution kernel are large.

---

## 5. Performance Analysis

Performance was measured on **Kaya HPC** using different input sizes and thread counts.

### 5.1 Metrics collected:

* **Runtime (seconds)** for serial vs. parallel versions.
* **T\_serial**: Serial Time, **T_parallel**: Parallel Time, **N_threads**: Max number of threads
* **Speedup (S = T\_serial / T\_parallel).**
* **Parallel efficiency (E = S / N_threads).**
* **Scalability trends** with input size and kernel size.

### 5.2 Figures and tables:

#### 5.2.1 Table of running times

| Array  Size (H x W) | Threads | Serial Time (s) | Parallel Time (s) |
| ------------------- | ------- | --------------- | ----------------- |
| 512 x 512           | 1       | 0.007988        | 0.007981          |
| 512 x 512           | 2       | 0.007979        | 0.004161          |
| 512 x 512           | 4       | 0.007969        | 0.002206          |
| 512 x 512           | 8       | 0.007965        | 0.001215          |
| 512 x 512           | 16      | 0.007986        | 0.000879          |
| 1024 x 1024         | 1       | 0.032041        | 0.031686          |
| 1024 x 1024         | 2       | 0.031841        | 0.016522          |
| 1024 x 1024         | 4       | 0.032043        | 0.008337          |
| 1024 x 1024         | 8       | 0.031921        | 0.004344          |
| 1024 x 1024         | 16      | 0.031807        | 0.002426          |
| 2048 x 2048         | 1       | 0.13046         | 0.129714          |
| 2048 x 2048         | 2       | 0.130355        | 0.065527          |
| 2048 x 2048         | 4       | 0.130635        | 0.033267          |
| 2048 x 2048         | 8       | 0.130189        | 0.016769          |
| 2048 x 2048         | 16      | 0.130199        | 0.008872          |

This data table records the runtime of serial and parallel computations under different thread counts (1, 2, 4, 8, 16) and matrix sizes (512 x 512, 1024 x 1024, 2048 x 2048). The following conclusions can be drawn:

1. For the same matrix size, increasing the number of threads from 1 to 16 significantly reduces parallel processing time, demonstrating the effectiveness of parallelization.
2. As the matrix size increases from 512 x 512 to 2048 x 2048, both serial and parallel execution times increase substantially. This aligns with expectations, as the data volume processed grows quadratically.

#### 5.2.2  Speedup graph

<img src="C:\Users\13203\AppData\Roaming\Typora\typora-user-images\image-20250907153804943.png" alt="image-20250907153804943" style="zoom:80%;" />

The figure above illustrates the performance improvement of parallel implementation over serial implementation using a 2048 x 2048 input matrix size as an example. The chart displays two lines: one represents the ideal speedup ratio (a straight line originating from the origin), while the other shows the actual measured speedup ratio. From this, we observe:

• Linear acceleration: The measured acceleration ratio curve closely follows the ideal curve, particularly with fewer threads (1, 2, 4). This indicates near-perfect parallelization at these test points, where each additional thread delivers nearly proportional performance gains.

• Near-perfect speedup: At 8 threads, the actual speedup reaches 7.76x. This indicates the program effectively utilizes multi-core resources with minimal parallelization overhead.

• Emergence of performance bottlenecks: At 16 threads, the speedup is 14.68x. While still very high, it begins to fall slightly below the ideal 16x acceleration. This typically occurs because parallelization overhead (such as thread creation, synchronization, or scheduling) starts to become significant, or due to cache contention.

![image-20250908160228115](C:\Users\13203\AppData\Roaming\Typora\typora-user-images\image-20250908160228115.png)

The figure above illustrates the variation in the speedup across different matrix sizes and thread counts. Red indicates the trend of speedups, while blue represents the trend of thread counts. The horizontal axis denotes matrix sizes, and the vertical axis shows the corresponding speedup value. The graph reveals that as the number of parallel threads increases, the speedup for large matrices approaches the ideal speedup. This demonstrates that parallelization delivers more significant performance gains when handling large-scale computational tasks.

#### 5.2.3 Efficiency graph

#### ![image-20250908160137981](C:\Users\13203\AppData\Roaming\Typora\typora-user-images\image-20250908160137981.png)

The figure above illustrates efficiency changes across different matrix sizes and thread counts. Red indicates efficiency trends, blue represents thread count trends, the horizontal axis denotes matrix sizes, and the vertical axis shows corresponding efficiency data. It reveals that:

• Efficiency decreases with increasing thread count: For all matrix sizes, efficiency declines as the number of threads increases. Under ideal conditions, efficiency should remain around 100%. However, in practice, inter-thread overhead reduces efficiency. For example, at 16 threads, efficiency drops below 60%. This indicates that more threads are not always better; beyond a certain threshold, excessive coordination overhead diminishes each thread's contribution.

• Larger matrices exhibit higher efficiency: For the same number of threads, processing large matrices (e.g., 2048 x 2048) demonstrates significantly higher efficiency than smaller matrices (e.g., 512 x 512). This validates the principle that “parallelization is more suitable for large-scale computational problems.” When computational volume is sufficiently large, the proportion of overhead time relative to total runtime becomes relatively small, resulting in higher overall efficiency.

---

## 6. Testing and Validation



---

## 7. Conclusion

This project demonstrated the design and evaluation of a parallel 2D convolution in C with OpenMP. By parallelising the outer loops over output pixels and optimising memory layout, substantial speedups were achieved on multi-core systems. Cache efficiency and scheduling strategy played important roles in scalability. The results confirm that OpenMP parallelisation provides clear benefits for computationally intensive tasks like convolution, making it well-suited for large-scale problems encountered in image processing and deep learning.

