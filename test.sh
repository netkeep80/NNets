#!/bin/bash
# Cross-platform test script for NNets
# This script builds the project and runs all tests through cmake/ctest

set -e  # Exit on error

# Colors for output (if terminal supports them)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "NNets Test Suite"
echo "================================================"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir build
fi

cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "${YELLOW}Building...${NC}"
cmake --build . --config Release

# Run tests
echo ""
echo "================================================"
echo "Running Tests"
echo "================================================"

# Run ctest with verbose output
if ctest -C Release --output-on-failure; then
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}All tests PASSED!${NC}"
    echo -e "${GREEN}================================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}Some tests FAILED!${NC}"
    echo -e "${RED}================================================${NC}"
    exit 1
fi
