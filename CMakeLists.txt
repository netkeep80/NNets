cmake_minimum_required(VERSION 3.14)
project(NNets VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard (C++17 required for parallel algorithms)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the executable
add_executable(NNets main.cpp)

# Add include directory for nlohmann/json (header-only library)
target_include_directories(NNets PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Find threading library
find_package(Threads REQUIRED)
target_link_libraries(NNets PRIVATE Threads::Threads)

# Find TBB for parallel execution policies (optional but recommended)
find_package(TBB QUIET)
if(TBB_FOUND)
    target_link_libraries(NNets PRIVATE TBB::tbb)
    target_compile_definitions(NNets PRIVATE HAS_TBB=1)
    message(STATUS "TBB found - parallel execution policies enabled")
else()
    message(STATUS "TBB not found - using thread-based parallelism")
endif()

# Platform-specific settings
if(MSVC)
    # MSVC-specific settings (matching Visual Studio project)
    target_compile_definitions(NNets PRIVATE
        WIN32
        _CONSOLE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
    )

    # SIMD optimization option for MSVC
    option(ENABLE_SIMD_AVX2 "Enable AVX2 SIMD optimizations for MSVC" ON)

    target_compile_options(NNets PRIVATE
        /W3                    # Warning level 3
        $<$<CONFIG:Debug>:/Od>      # Disable optimization for Debug
        $<$<CONFIG:Debug>:/RTC1>    # Runtime checks
        $<$<CONFIG:Release>:/O2>    # Optimize for speed
        $<$<CONFIG:Release>:/Oi>    # Enable intrinsic functions
        $<$<CONFIG:Release>:/Oy>    # Enable frame pointer omission
    )

    # Add SIMD flags for MSVC
    if(ENABLE_SIMD_AVX2)
        target_compile_options(NNets PRIVATE /arch:AVX2)
        message(STATUS "SIMD: AVX2 enabled for MSVC")
    endif()

    # Set runtime library
    set_property(TARGET NNets PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    # GCC/Clang settings
    # Define MSVC-specific macros for cross-platform compatibility
    target_compile_definitions(NNets PRIVATE
        __fastcall=           # Remove __fastcall calling convention
        strcpy_s=strcpy       # Use standard strcpy instead of MSVC's strcpy_s
    )

    # SIMD optimization option (default: native architecture detection)
    option(ENABLE_SIMD_NATIVE "Enable SIMD optimizations for native architecture" ON)
    option(ENABLE_AVX "Enable AVX SIMD optimizations explicitly" OFF)
    option(ENABLE_SSE "Enable SSE SIMD optimizations explicitly" OFF)

    target_compile_options(NNets PRIVATE
        -Wall
        -Wextra
        -Wno-write-strings    # Suppress string literal to char* conversion warnings
        -Wno-deprecated       # Suppress deprecated header warnings (strstream)
        -fpermissive          # Allow some non-standard code constructs
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Release>:-O3>
    )

    # Add SIMD flags based on options
    if(ENABLE_SIMD_NATIVE)
        # Use -march=native for automatic detection of CPU SIMD capabilities
        target_compile_options(NNets PRIVATE -march=native)
        message(STATUS "SIMD: Using native architecture detection (-march=native)")
    elseif(ENABLE_AVX)
        target_compile_options(NNets PRIVATE -mavx -mavx2)
        message(STATUS "SIMD: AVX/AVX2 explicitly enabled")
    elseif(ENABLE_SSE)
        target_compile_options(NNets PRIVATE -msse -msse2 -msse4.1)
        message(STATUS "SIMD: SSE/SSE2/SSE4.1 explicitly enabled")
    else()
        message(STATUS "SIMD: No SIMD optimizations (scalar operations)")
    endif()
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)

# Copy config files to build directory
file(GLOB CONFIG_FILES "${CMAKE_SOURCE_DIR}/configs/*.json")
foreach(CONFIG_FILE ${CONFIG_FILES})
    get_filename_component(CONFIG_NAME ${CONFIG_FILE} NAME)
    configure_file(${CONFIG_FILE} ${CMAKE_BINARY_DIR}/configs/${CONFIG_NAME} COPYONLY)
endforeach()

# Installation rules
install(TARGETS NNets
    RUNTIME DESTINATION bin
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/configs/
    DESTINATION share/NNets/configs
    FILES_MATCHING PATTERN "*.json"
)

# ============================================================================
# Testing configuration with CTest
# ============================================================================
enable_testing()

# Helper to get the correct executable path
if(MSVC)
    set(NNETS_EXE "$<TARGET_FILE:NNets>")
else()
    set(NNETS_EXE "$<TARGET_FILE:NNets>")
endif()

# Test 1: Simple classification test (yes/no)
# Verifies basic neural network training and classification works
add_test(
    NAME test_simple_classification
    COMMAND NNets -c ${CMAKE_SOURCE_DIR}/configs/simple.json -t
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_simple_classification PROPERTIES
    TIMEOUT 120
    LABELS "classification;quick"
)

# Test 2: Default configuration test (4 classes)
# Tests with the original hardcoded configuration
add_test(
    NAME test_default_classification
    COMMAND NNets -c ${CMAKE_SOURCE_DIR}/configs/default.json -t
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_default_classification PROPERTIES
    TIMEOUT 300
    LABELS "classification;standard"
)

# Test 3: Benchmark test for training speed
# Measures and reports training performance metrics
add_test(
    NAME test_benchmark_training_speed
    COMMAND NNets -c ${CMAKE_SOURCE_DIR}/configs/benchmark.json -b
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_benchmark_training_speed PROPERTIES
    TIMEOUT 300
    LABELS "benchmark;performance"
)

# Test 4: Model save and load test
# Trains a model, saves it, then loads and verifies inference
add_test(
    NAME test_model_save_load
    COMMAND ${CMAKE_COMMAND}
        -DNNETS_EXE=$<TARGET_FILE:NNets>
        -DCONFIG_DIR=${CMAKE_SOURCE_DIR}/configs
        -DWORK_DIR=${CMAKE_BINARY_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/test_save_load.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_model_save_load PROPERTIES
    TIMEOUT 180
    LABELS "save_load;inference"
)

# Test 5: Extended classification test (more classes)
add_test(
    NAME test_extended_classification
    COMMAND NNets -c ${CMAKE_SOURCE_DIR}/configs/extended.json -t
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_extended_classification PROPERTIES
    TIMEOUT 300
    LABELS "classification;extended"
)

# Test 6: Multithreading benchmark comparison
# Compares single-threaded vs multi-threaded training performance
add_test(
    NAME test_multithreading_benchmark
    COMMAND ${CMAKE_COMMAND}
        -DNNETS_EXE=$<TARGET_FILE:NNets>
        -DCONFIG_DIR=${CMAKE_SOURCE_DIR}/configs
        -DWORK_DIR=${CMAKE_BINARY_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/test_multithreading.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_multithreading_benchmark PROPERTIES
    TIMEOUT 300
    LABELS "benchmark;multithreading;performance"
)

# Test 7: Verification mode test
# Tests --verify flag for checking model accuracy
add_test(
    NAME test_verify_mode
    COMMAND ${CMAKE_COMMAND}
        -DNNETS_EXE=$<TARGET_FILE:NNets>
        -DCONFIG_DIR=${CMAKE_SOURCE_DIR}/configs
        -DWORK_DIR=${CMAKE_BINARY_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/test_verify.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_verify_mode PROPERTIES
    TIMEOUT 180
    LABELS "verification;inference"
)

# Test 8: Retraining mode test
# Tests -r flag for retraining existing models with new classes
add_test(
    NAME test_retraining_mode
    COMMAND ${CMAKE_COMMAND}
        -DNNETS_EXE=$<TARGET_FILE:NNets>
        -DCONFIG_DIR=${CMAKE_SOURCE_DIR}/configs
        -DWORK_DIR=${CMAKE_BINARY_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/test_retraining.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_retraining_mode PROPERTIES
    TIMEOUT 600
    LABELS "retraining;training"
)

# Test 9: SIMD optimization test
# Tests SIMD vector operations and --no-simd flag
add_test(
    NAME test_simd_optimization
    COMMAND ${CMAKE_COMMAND}
        -DNNETS_EXE=$<TARGET_FILE:NNets>
        -DCONFIG_DIR=${CMAKE_SOURCE_DIR}/configs
        -DWORK_DIR=${CMAKE_BINARY_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/test_simd_benchmark.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_simd_optimization PROPERTIES
    TIMEOUT 120
    LABELS "simd;performance"
)
