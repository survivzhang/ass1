import matplotlib.pyplot as plt

# Data for 2048x2048 matrix from your results.txt
threads = [1, 2, 4, 8, 16]

# Extract data for 2048x2048 matrix
# From the provided results.txt, we can extract the values for each thread count.
speedups = [1.01, 1.99, 3.93, 7.76, 14.68]
efficiencies = [100.58, 99.47, 98.17, 97.05, 91.72]

# Create Speedup graph
plt.figure(figsize=(8, 6))
plt.plot(threads, speedups, 'o-', label='Measured Speedup')
plt.plot(threads, threads, '--', label='Ideal Speedup')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('Speedup vs. Number of Threads (2048x2048 Matrix)')
plt.legend()
plt.grid(True)
plt.xticks(threads)
plt.savefig('speedup_graph.png')
plt.show()

# Create Efficiency graph
plt.figure(figsize=(8, 6))
plt.plot(threads, efficiencies, 'o-')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency (%)')
plt.title('Parallel Efficiency vs. Number of Threads (2048x2048 Matrix)')
plt.grid(True)
plt.xticks(threads)
plt.ylim(0, 110) # Set a reasonable y-axis limit
plt.savefig('efficiency_graph.png')
plt.show()