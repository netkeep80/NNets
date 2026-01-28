# test_simd_benchmark.cmake
# Тест для проверки работоспособности SIMD оптимизаций

message(STATUS "=== SIMD Benchmark Test ===")

# Run benchmark with SIMD enabled (default)
message(STATUS "Testing with SIMD enabled...")
execute_process(
    COMMAND ${NNETS_EXE} -c ${CONFIG_DIR}/simple.json -b --single-thread
    WORKING_DIRECTORY ${WORK_DIR}
    RESULT_VARIABLE SIMD_RESULT
    OUTPUT_VARIABLE SIMD_OUTPUT
    ERROR_VARIABLE SIMD_ERROR
    TIMEOUT 120
)

if(NOT SIMD_RESULT EQUAL 0)
    message(FATAL_ERROR "SIMD enabled test failed: ${SIMD_ERROR}")
endif()

# Check that SIMD info is displayed
if(NOT SIMD_OUTPUT MATCHES "SIMD:")
    message(FATAL_ERROR "SIMD info not displayed in output")
endif()

message(STATUS "SIMD enabled test passed")

# Run benchmark with SIMD disabled
message(STATUS "Testing with SIMD disabled...")
execute_process(
    COMMAND ${NNETS_EXE} -c ${CONFIG_DIR}/simple.json -b --single-thread --no-simd
    WORKING_DIRECTORY ${WORK_DIR}
    RESULT_VARIABLE NO_SIMD_RESULT
    OUTPUT_VARIABLE NO_SIMD_OUTPUT
    ERROR_VARIABLE NO_SIMD_ERROR
    TIMEOUT 120
)

if(NOT NO_SIMD_RESULT EQUAL 0)
    message(FATAL_ERROR "SIMD disabled test failed: ${NO_SIMD_ERROR}")
endif()

# Verify SIMD was disabled
if(NOT NO_SIMD_OUTPUT MATCHES "disabled via --no-simd")
    message(FATAL_ERROR "--no-simd flag did not work properly")
endif()

message(STATUS "SIMD disabled test passed")
message(STATUS "=== SIMD Benchmark Test Complete ===")
