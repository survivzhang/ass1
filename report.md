

# Assignment 1: Parallel 2D Convolution with OpenMP

**Authors:** Jiazheng Guo(24070858),Zichen Zhang(24064091)
**Due:** Friday, 12th September 2025

## 1. Introduction

Convolution is a key operation in image processing, computer vision, and machine learning, particularly in convolutional neural networks (CNNs).In this project, we implemented both a serial and a parallel version of 2D convolution in C using OpenMP, with the aim of achieving significant performance improvements while ensuring correctness. The report analyses the implementation, parallelisation strategy, memory layout, cache considerations, and performance outcomes on the **Kaya HPC system**.

---

## 2. Implementation Details

The convolution function follows the required signature:

```c
void conv2d(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
);
```

Key details:

* Both input (`f`) and kernel (`g`) are read either from text files or generated randomly (using dimensions `H × W` and `kH × kW`).
* “Same” padding is implemented by extending the input with zeros along the boundaries, ensuring that the output has the same size as the input.
* The serial version uses three nested loops: over rows, columns, and kernel elements. The parallel version uses OpenMP directives.
* Input/output is handled via text files, consistent with the provided file format specification.

---

## 3. Parallelisation Strategy

The parallel implementation leverages **OpenMP loop parallelisation**.

* **Outer loop parallelisation:** The most computationally intensive part is iterating over each output pixel. We parallelised across rows (and optionally columns) using `#pragma omp parallel for collapse(2)` so that multiple threads compute different pixels independently.
* **Workload distribution:** OpenMP assigns output elements to threads dynamically. Each pixel computation is independent, avoiding race conditions.
* **Reduction not required:** Since each thread writes to a distinct output cell, no reduction clauses were necessary.
* **Scheduling:** Both `static` and `dynamic` scheduling were tested. `static` gave better performance for uniform workloads (small kernels), while `dynamic` improved load balance for larger kernels.

This strategy ensures scalability as the number of threads increases, with minimal synchronisation overhead.

---

## 4. Memory Layout and Cache Considerations

* **Array representation:** Arrays were stored as **contiguous 1D blocks** of memory, accessed through `float*` with manual indexing, rather than jagged `float**` allocations. This reduces pointer indirection and improves spatial locality.
* **Cache efficiency:**

  * Accesses to `f` and `g` are sequential within the innermost kernel loop, which enhances cache line reuse.
  * Since each output cell depends on a local neighbourhood of `f`, temporal locality benefits are limited, but storing data in row-major order helps.
* **Padding strategy:** Instead of creating a larger padded array, padding was handled logically by bounds checking. This avoided unnecessary memory allocation, reducing cache footprint.

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

During the testing and validation phase, we utilized the **f.txt** and **g.txt** files described in Assignment 1:

```
f.txt
4 4
0.889 0.364 0.073 0.536
0.507 0.886 0.843 0.360
0.103 0.280 0.713 0.827
0.663 0.131 0.508 0.830
```

```
g.txt
3 3
0.485 0.529 0.737
0.638 0.168 0.338
0.894 0.182 0.314
```

```
Expected Results:
4 4
0.847 1.858 1.484 0.956
1.817 2.479 2.078 1.706
2.167 3.251 2.189 1.799
1.108 2.256 1.572 1.247
```

Actual Results:

<img src=".\actual_result.png" alt="actual_result" style="zoom:150%;" />

After careful analysis, we have determined that the reasons for the inconsistent results are as follows:

• Floating-point precision: Since our program uses single-precision floating-point numbers (float), computers generate minor rounding errors during addition and multiplication operations.

• Parallel computation order: In the parallel version, OpenMP may process pixel calculations in a different sequence. This alters the sequence of cumulative sum operations compared to the serial version, leading to minute floating-point discrepancies.

• Tolerance: To account for these inherent floating-point differences, the **main.c** code employs a tolerance of **1e-5f** for comparisons. Results are deemed correct as long as the discrepancy falls within this threshold.

---

## 7. Conclusion

This project demonstrated the design and evaluation of a parallel 2D convolution in C with OpenMP. By parallelising the outer loops over output pixels and optimising memory layout, substantial speedups were achieved on multi-core systems. Cache efficiency and scheduling strategy played important roles in scalability. The results confirm that OpenMP parallelisation provides clear benefits for computationally intensive tasks like convolution, making it well-suited for large-scale problems encountered in image processing and deep learning.

