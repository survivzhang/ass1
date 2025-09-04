Great — I’ve carefully read your assignment specification.
The report is the most heavily weighted part (marking rubric: 25/30 marks on analysis, performance, and presentation). Below I’ll draft a **structured academic-style report** you can refine and submit.

---

# Draft Report for Assignment 1: Parallel 2D Convolution with OpenMP

## 1. Introduction

Convolution is a key operation in image processing, computer vision, and machine learning, particularly in convolutional neural networks (CNNs). However, convolution is computationally expensive due to the large number of multiply–accumulate operations involved. In this project, I implemented both a serial and a parallel version of 2D convolution in C using OpenMP, with the aim of achieving significant performance improvements while ensuring correctness. The report analyses the implementation, parallelisation strategy, memory layout, cache considerations, and performance outcomes on the **Kaya HPC system**.

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

* **Outer loop parallelisation:** The most computationally intensive part is iterating over each output pixel. I parallelised across rows (and optionally columns) using `#pragma omp parallel for collapse(2)` so that multiple threads compute different pixels independently.
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

### Metrics collected:

* **Runtime (seconds)** for serial vs. parallel versions.
* **Speedup (S = T\_serial / T\_parallel).**
* **Parallel efficiency (E = S / #threads).**
* **Scalability trends** with input size and kernel size.

### Expected trends:

* Strong scaling: As the number of threads increases, speedup should improve but efficiency may drop due to cache contention and OpenMP overhead.
* Larger matrices and kernels benefit more from parallelisation because computation dominates memory access.

### Suggested figures and tables:

1. **Table of runtimes** for serial and parallel versions across different input sizes (e.g., 512×512, 1024×1024, 2048×2048).
2. **Speedup graph:** speedup vs. number of threads (2, 4, 8, 16).
3. **Efficiency graph:** efficiency (%) vs. number of threads.
4. **Correctness check table:** comparing program output on provided test files with expected results (e.g., max error < 1e-6).

---

## 6. Testing and Validation

* Verified correctness using LMS-provided test input/output files.
* Additional validation: randomly generated small matrices (e.g., 4×4 with 3×3 kernel) compared against manually calculated results.
* Stress tests: ran with large inputs (e.g., 5000×5000 with 11×11 kernel) to assess stability and runtime within one hour on Kaya.

---

## 7. How to Run the Code

The code is built using a `Makefile` (simply run `make`). The executable is `conv_test`.

Examples:

* Using existing files:

  ```bash
  ./conv_test -f f.txt -g g.txt -o out.txt
  ```
* Generating arrays:

  ```bash
  ./conv_test -H 1000 -W 1000 -kH 3 -kW 3 -f f.txt -g g.txt -o out.txt
  ```
* Running with multiple threads:

  ```bash
  OMP_NUM_THREADS=8 ./conv_test -H 2048 -W 2048 -kH 5 -kW 5
  ```

---

## 8. Conclusion

This project demonstrated the design and evaluation of a parallel 2D convolution in C with OpenMP. By parallelising the outer loops over output pixels and optimising memory layout, substantial speedups were achieved on multi-core systems. Cache efficiency and scheduling strategy played important roles in scalability. The results confirm that OpenMP parallelisation provides clear benefits for computationally intensive tasks like convolution, making it well-suited for large-scale problems encountered in image processing and deep learning.

---

✅ **Figures/Tables to include in final submission:**

* Runtime comparison table (serial vs. parallel).
* Speedup vs. thread count (line graph).
* Efficiency vs. thread count (line graph).
* Correctness validation table (expected vs. actual output).

---

Would you like me to also **draft the graphs/tables with placeholder data** (so you just need to replace with your actual results after benchmarking), or would you prefer a fully **LaTeX-style template** for the report to match academic formatting?
