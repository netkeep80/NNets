#!/bin/bash
#
# test_multithreading_speedup.sh
#
# Experiment script to verify that multithreading actually utilizes multiple CPU cores
# and provides performance benefits.
#
# Usage: ./test_multithreading_speedup.sh [build_dir] [config_file]
#
# This script:
# 1. Runs the benchmark in single-threaded mode
# 2. Runs the benchmark in multi-threaded mode with different thread counts
# 3. Compares the results and calculates speedup
#

set -e

# Default paths
BUILD_DIR="${1:-../build}"
CONFIG_FILE="${2:-../configs/default.json}"
EXECUTABLE="${BUILD_DIR}/NNets"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first: cd build && cmake .. && make"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "=============================================="
echo "Multithreading Speedup Experiment"
echo "=============================================="
echo "Executable: $EXECUTABLE"
echo "Config: $CONFIG_FILE"
echo "Available CPU cores: $(nproc)"
echo ""

# Function to extract training time from benchmark output
extract_training_time() {
    grep "Training time:" | sed 's/.*Training time: \([0-9]*\) ms/\1/'
}

# Function to extract thread count from benchmark output
extract_thread_count() {
    grep "Threads:" | sed 's/.*Threads: \([0-9]*\).*/\1/'
}

echo "Running single-threaded benchmark..."
SINGLE_OUTPUT=$("$EXECUTABLE" -c "$CONFIG_FILE" -b --single-thread 2>&1)
SINGLE_TIME=$(echo "$SINGLE_OUTPUT" | extract_training_time)
echo "  Single-threaded training time: ${SINGLE_TIME} ms"
echo ""

echo "Running multi-threaded benchmarks..."

for THREADS in 2 4 $(nproc); do
    MULTI_OUTPUT=$("$EXECUTABLE" -c "$CONFIG_FILE" -b -j "$THREADS" 2>&1)
    MULTI_TIME=$(echo "$MULTI_OUTPUT" | extract_training_time)
    ACTUAL_THREADS=$(echo "$MULTI_OUTPUT" | extract_thread_count)

    if [ -n "$MULTI_TIME" ] && [ "$MULTI_TIME" -gt 0 ]; then
        SPEEDUP=$(echo "scale=2; $SINGLE_TIME / $MULTI_TIME" | bc)
        echo "  ${ACTUAL_THREADS} threads: ${MULTI_TIME} ms (speedup: ${SPEEDUP}x)"
    else
        echo "  ${THREADS} threads: Failed to get timing"
    fi
done

echo ""
echo "=============================================="
echo "Experiment completed"
echo "=============================================="
