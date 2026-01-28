#!/bin/bash
#
# test_simd_speedup.sh - Тест ускорения векторных операций с SIMD
#
# Сравнивает производительность обучения нейросети с включённым и
# выключенным SIMD для измерения реального ускорения.
#
# Использование:
#   ./experiments/test_simd_speedup.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
# NNets executable can be in different locations depending on CMake generator
NNETS_EXE="$BUILD_DIR/NNets"
[ -f "$BUILD_DIR/bin/NNets" ] && NNETS_EXE="$BUILD_DIR/bin/NNets"
[ -f "$BUILD_DIR/bin/Release/NNets" ] && NNETS_EXE="$BUILD_DIR/bin/Release/NNets"
CONFIG_FILE="$PROJECT_DIR/configs/benchmark.json"

echo "=== SIMD Speedup Test ==="
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Build directory: $BUILD_DIR"

# Check if executable exists
if [ ! -f "$NNETS_EXE" ]; then
    echo "Building project..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd "$PROJECT_DIR"
fi

# Verify executable exists after build
if [ ! -f "$NNETS_EXE" ]; then
    echo "Error: Could not find NNets executable at $NNETS_EXE"
    exit 1
fi

echo ""
echo "Testing with config: $CONFIG_FILE"
echo ""

# Run benchmark with SIMD enabled (default)
echo "=== Test 1: SIMD Enabled (default) ==="
"$NNETS_EXE" -c "$CONFIG_FILE" -b --single-thread 2>&1 | tee /tmp/simd_enabled.log
SIMD_TIME=$(grep "Training time:" /tmp/simd_enabled.log | awk '{print $3}')
echo ""

# Run benchmark with SIMD disabled
echo "=== Test 2: SIMD Disabled (--no-simd) ==="
"$NNETS_EXE" -c "$CONFIG_FILE" -b --single-thread --no-simd 2>&1 | tee /tmp/simd_disabled.log
NO_SIMD_TIME=$(grep "Training time:" /tmp/simd_disabled.log | awk '{print $3}')
echo ""

# Calculate speedup
echo "=== SIMD Speedup Summary ==="
echo "SIMD Enabled:  $SIMD_TIME ms"
echo "SIMD Disabled: $NO_SIMD_TIME ms"

if [ -n "$SIMD_TIME" ] && [ -n "$NO_SIMD_TIME" ] && [ "$SIMD_TIME" -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $NO_SIMD_TIME / $SIMD_TIME" | bc)
    echo "Speedup:       ${SPEEDUP}x"
else
    echo "Could not calculate speedup (missing timing data)"
fi

echo ""
echo "=== End SIMD Speedup Test ==="

# Cleanup
rm -f /tmp/simd_enabled.log /tmp/simd_disabled.log
