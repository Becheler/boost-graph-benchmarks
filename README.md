# boost-graph-benchmarks

Benchmarking Boost Graph Library algorithms for correctness and performance against reference implementations.

## Structure

```
boost-graph-benchmarks/
├── requirements-common.txt     # Shared Python deps
└── louvain/                    # Louvain community detection benchmark
    ├── run_benchmark.sh        # Self-contained entry point (creates venv, builds, runs)
    ├── requirements.txt        # Louvain-specific Python deps
    ├── src/                    # C++ sources + CMakeLists.txt
    ├── scripts/                # Python benchmark & visualisation
    ├── vendor/                 # Third-party implementations (gen-louvain)
    ├── results/                # CSV (gitignored) + PNG plots (committed)
    ├── build/                  # CMake build output (gitignored)
    └── venv/                   # Python virtualenv (gitignored)
```

## Quick Start

```bash
cd louvain && ./run_benchmark.sh
```

This builds all C++ binaries, runs correctness + runtime benchmarks, and generates plots.

See [louvain/README.md](louvain/README.md) for full details.

## Adding a New Benchmark

1. Create a new directory: `mkdir my_algorithm`
2. Add `run_benchmark.sh`, `requirements.txt`, and your benchmark scripts
3. Follow the pattern in `louvain/` for structure

## Requirements

- Python 3.8+
- C++ compiler with C++17 support
- CMake 3.15+
- Boost Graph Library headers

Python dependencies are installed automatically into per-benchmark virtual environments.
