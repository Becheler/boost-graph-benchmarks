#!/bin/bash
# Louvain Benchmark - Self-contained runner
# Creates a venv, installs deps, optionally builds BGL, runs benchmarks.
#
# Usage:
#   ./run_benchmark.sh          # full benchmark (default)
#   ./run_benchmark.sh --quick  # fast smoke-test (fewer trials & sizes)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Collect flags to forward to benchmark.py
BENCH_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        --quick) BENCH_FLAGS="$BENCH_FLAGS --quick" ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "=== Louvain Benchmark Suite ==="
echo "Working directory: $SCRIPT_DIR"

# --- Virtual environment ---
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
echo "Installing dependencies..."
./venv/bin/pip install -q -r ../requirements-common.txt -r requirements.txt

# --- Prepare gen-louvain source (download + patch, no compilation) ---
echo "Preparing gen-louvain source..."
bash vendor/gen-louvain/build.sh || \
    echo "  WARNING: gen-louvain source prep failed. Benchmarks will skip gen-louvain."

# --- Build all binaries via CMake (BGL + gen-louvain) ---
if [ ! -d build ] || [ ! -f build/bgl_louvain_vecS_vecS ]; then
    echo "Building all binaries with CMake..."
    mkdir -p build
    (cd build && cmake ../src -DCMAKE_BUILD_TYPE=Release -Wno-dev && cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)) && \
        echo "  Build complete" || \
        echo "  WARNING: CMake build failed. Benchmarks may skip BGL / gen-louvain."
fi

# --- Run benchmarks ---
echo ""
echo "Running benchmark suite..."
./venv/bin/python3 scripts/benchmark.py $BENCH_FLAGS

echo ""
echo "Generating visualizations..."
./venv/bin/python3 scripts/visualize.py

echo ""
echo "=== Louvain benchmark complete ==="
echo "Results in: $SCRIPT_DIR/results/"
