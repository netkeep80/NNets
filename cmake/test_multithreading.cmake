# CMake script to test multithreading performance
# This script compares single-threaded vs multi-threaded training speed

# Check required variables
if(NOT DEFINED NNETS_EXE)
    message(FATAL_ERROR "NNETS_EXE not defined")
endif()

if(NOT DEFINED CONFIG_DIR)
    message(FATAL_ERROR "CONFIG_DIR not defined")
endif()

if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "WORK_DIR not defined")
endif()

set(CONFIG_FILE "${CONFIG_DIR}/simple.json")

message(STATUS "=== Testing Multithreading Performance ===")
message(STATUS "Executable: ${NNETS_EXE}")
message(STATUS "Config: ${CONFIG_FILE}")

# Step 1: Run single-threaded benchmark
message(STATUS "Step 1: Running single-threaded benchmark...")
execute_process(
    COMMAND "${NNETS_EXE}" -c "${CONFIG_FILE}" --single-thread -b
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE SINGLE_RESULT
    OUTPUT_VARIABLE SINGLE_OUTPUT
    ERROR_VARIABLE SINGLE_ERROR
    TIMEOUT 180
)

if(NOT SINGLE_RESULT EQUAL 0)
    message(FATAL_ERROR "Single-threaded benchmark failed with code ${SINGLE_RESULT}:\nOutput: ${SINGLE_OUTPUT}\nError: ${SINGLE_ERROR}")
endif()

# Extract training time from single-threaded output
string(REGEX MATCH "Training time: ([0-9]+) ms" SINGLE_TIME_MATCH "${SINGLE_OUTPUT}")
string(REGEX REPLACE "Training time: ([0-9]+) ms" "\\1" SINGLE_TIME "${SINGLE_TIME_MATCH}")
message(STATUS "Single-threaded training time: ${SINGLE_TIME} ms")

# Step 2: Run multi-threaded benchmark
message(STATUS "Step 2: Running multi-threaded benchmark...")
execute_process(
    COMMAND "${NNETS_EXE}" -c "${CONFIG_FILE}" -b
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE MULTI_RESULT
    OUTPUT_VARIABLE MULTI_OUTPUT
    ERROR_VARIABLE MULTI_ERROR
    TIMEOUT 180
)

if(NOT MULTI_RESULT EQUAL 0)
    message(FATAL_ERROR "Multi-threaded benchmark failed with code ${MULTI_RESULT}:\nOutput: ${MULTI_OUTPUT}\nError: ${MULTI_ERROR}")
endif()

# Extract training time from multi-threaded output
string(REGEX MATCH "Training time: ([0-9]+) ms" MULTI_TIME_MATCH "${MULTI_OUTPUT}")
string(REGEX REPLACE "Training time: ([0-9]+) ms" "\\1" MULTI_TIME "${MULTI_TIME_MATCH}")
message(STATUS "Multi-threaded training time: ${MULTI_TIME} ms")

# Extract number of threads
string(REGEX MATCH "Threads: ([0-9]+)" THREADS_MATCH "${MULTI_OUTPUT}")
string(REGEX REPLACE "Threads: ([0-9]+)" "\\1" NUM_THREADS "${THREADS_MATCH}")
message(STATUS "Number of threads used: ${NUM_THREADS}")

# Calculate speedup
if(MULTI_TIME GREATER 0)
    math(EXPR SPEEDUP_X100 "${SINGLE_TIME} * 100 / ${MULTI_TIME}")
    math(EXPR SPEEDUP_INT "${SPEEDUP_X100} / 100")
    math(EXPR SPEEDUP_FRAC "${SPEEDUP_X100} % 100")
    if(SPEEDUP_FRAC LESS 10)
        set(SPEEDUP_FRAC "0${SPEEDUP_FRAC}")
    endif()
    message(STATUS "Speedup: ${SPEEDUP_INT}.${SPEEDUP_FRAC}x")
else()
    message(STATUS "Speedup: Unable to calculate (multi-threaded time is 0)")
endif()

# Verify that both benchmarks completed successfully
string(FIND "${SINGLE_OUTPUT}" "End Benchmark" SINGLE_END_FOUND)
if(SINGLE_END_FOUND EQUAL -1)
    message(FATAL_ERROR "Single-threaded benchmark did not complete properly")
endif()

string(FIND "${MULTI_OUTPUT}" "End Benchmark" MULTI_END_FOUND)
if(MULTI_END_FOUND EQUAL -1)
    message(FATAL_ERROR "Multi-threaded benchmark did not complete properly")
endif()

message(STATUS "=== Multithreading Test PASSED ===")
message(STATUS "Summary:")
message(STATUS "  Single-threaded: ${SINGLE_TIME} ms")
message(STATUS "  Multi-threaded (${NUM_THREADS} threads): ${MULTI_TIME} ms")
if(SPEEDUP_INT)
    message(STATUS "  Speedup: ${SPEEDUP_INT}.${SPEEDUP_FRAC}x")
endif()
