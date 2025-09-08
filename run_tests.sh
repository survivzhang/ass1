#!/bin/bash

# Define array sizes and number of threads to test
matrix_sizes=("512" "1024" "2048")
kernel_size="3"
threads=(1 2 4 8 16)

output_file="results_summary.csv"

# Clear previous results and write CSV header
echo "Array_Size,Threads,Serial_Time(s),Parallel_Time(s),Speedup,Efficiency(%)" > $output_file

echo "Running performance tests and logging to $output_file..."

for size in "${matrix_sizes[@]}"
do
    H=$size
    W=$size
    kH=$kernel_size
    kW=$kernel_size

    for t in "${threads[@]}"
    do
        echo "Testing size ${H}x${W} with ${t} threads..."
        
        # Set the number of threads for OpenMP
        export OMP_NUM_THREADS=$t
        
        # Run the program and capture the entire output to a variable
        full_output=$(./conv_test -c -H $H -W $W -h $kH -w $kW 2>&1)
        
        # Extract values using a more robust approach
        # We look for the exact line and use awk to get the value
        serial_time=$(echo "$full_output" | grep "Serial time:" | awk '{print $3}')
        parallel_time=$(echo "$full_output" | grep "Parallel time:" | awk '{print $3}')
        speedup=$(echo "$full_output" | grep "Speedup:" | awk '{print $2}' | sed 's/x//')
        efficiency=$(echo "$full_output" | grep "Efficiency:" | awk '{print $2}' | sed 's/%//')
        
        # Check if the parallel time was captured
        if [ -z "$parallel_time" ]; then
            # If "Parallel time" is not found, something went wrong.
            # We will log N/A to avoid script failure.
            echo "Warning: Parallel time not found for ${H}x${W} with ${t} threads. Logging N/A."
            parallel_time="N/A"
            speedup="N/A"
            efficiency="N/A"
        fi

        # Write the data as a single line to the CSV file
        echo "${H}x${W},${t},${serial_time},${parallel_time},${speedup},${efficiency}" >> $output_file
    done
done

echo "Tests complete. Data is saved in $output_file"