# boost-graph-benchmarks

[![Louvain Benchmark](https://github.com/Becheler/boost-graph-benchmarks/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Becheler/boost-graph-benchmarks/actions/workflows/benchmark.yml)

Benchmarks for Boost Graph Library algorithms against reference implementations.

## Quick Start

```bash
cd louvain && ./run_benchmark.sh          # full suite
cd louvain && ./run_benchmark.sh --quick  # smoke-test
```

See [louvain/README.md](louvain/README.md) for details and results.

## Structure

```
boost-graph-benchmarks/
├── requirements-common.txt     # Shared Python deps
└── louvain/                    # Louvain community detection
    ├── run_benchmark.sh        # Entry point (venv, build, run, plot)
    ├── src/                    # C++ sources + CMake
    ├── scripts/                # benchmark.py + visualize.py
    ├── vendor/                 # gen-louvain (auto-downloaded)
    └── results/                # PNGs (committed), CSVs (gitignored)
```

## Requirements

- Python 3.8+, CMake 3.15+, C++17 compiler, Boost headers
- Python deps installed automatically into per-benchmark venvs
