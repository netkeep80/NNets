cmake_minimum_required(VERSION 3.14)
project(NNets VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the executable
add_executable(NNets main.cpp)

# Platform-specific settings
if(MSVC)
    # MSVC-specific settings (matching Visual Studio project)
    target_compile_definitions(NNets PRIVATE
        WIN32
        _CONSOLE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
    )

    target_compile_options(NNets PRIVATE
        /W3                    # Warning level 3
        $<$<CONFIG:Debug>:/Od>      # Disable optimization for Debug
        $<$<CONFIG:Debug>:/RTC1>    # Runtime checks
        $<$<CONFIG:Release>:/O2>    # Optimize for speed
        $<$<CONFIG:Release>:/Oi>    # Enable intrinsic functions
        $<$<CONFIG:Release>:/Oy>    # Enable frame pointer omission
    )

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
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)

# Installation rules
install(TARGETS NNets
    RUNTIME DESTINATION bin
)
