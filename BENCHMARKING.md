# DiskANN Rust Benchmarking Guide

## Overview

This document describes the benchmarking strategy for evaluating DiskANN Rust performance from small-scale (100 vectors) to large-scale (1M vectors) deployments.

## Benchmark Categories

### 1. Micro-benchmarks

**Purpose**: Measure individual component performance.

**Components**:
- Distance functions (L2, Cosine, Inner Product)
- Graph operations (insert, prune, search)
- I/O operations (read, write, cache)
- Data type conversions

**Tool**: Criterion.rs

### 2. Index Build Benchmarks

**Metrics**:
- Vectors indexed per second
- Memory usage during build
- Graph quality (average degree, connectivity)
- Time breakdown by phase

**Scale Progression**:
```
100 → 1K → 10K → 100K → 1M vectors
```

### 3. Search Benchmarks

**Metrics**:
- Queries per second (QPS)
- Latency percentiles (p50, p90, p99)
- Recall@K
- Memory bandwidth utilization

**Parameters to vary**:
- K (number of neighbors): 1, 10, 100
- Search list size (L): 50, 100, 200
- Batch size: 1, 10, 100, 1000

### 4. Concurrent Operations

**Scenarios**:
- Multi-threaded search
- Search during index updates
- Parallel index building

**Metrics**:
- Throughput scaling with threads
- Contention overhead
- Memory usage per thread

## Benchmark Datasets

### Small Scale (< 10K vectors)

1. **Random-100**
   - 100 random vectors, 128D
   - Purpose: Smoke test, development

2. **SIFT-1K**
   - 1,000 SIFT descriptors, 128D
   - Purpose: Real-world features, quick validation

3. **Clustered-10K**
   - 10,000 vectors in 10 clusters, 256D
   - Purpose: Test clustering behavior

### Medium Scale (10K - 100K vectors)

1. **SIFT-100K**
   - 100,000 SIFT descriptors, 128D
   - Purpose: Standard benchmark

2. **GloVe-100K**
   - 100,000 word embeddings, 300D
   - Purpose: Higher dimensional data

3. **Deep-100K**
   - 100,000 deep features, 768D
   - Purpose: Modern embedding dimensions

### Large Scale (100K - 1M vectors)

1. **SIFT-1M**
   - 1,000,000 SIFT descriptors, 128D
   - Purpose: Million-scale benchmark

2. **GIST-1M**
   - 1,000,000 GIST descriptors, 960D
   - Purpose: High-dimensional benchmark

3. **Generated-1M**
   - 1,000,000 synthetic vectors, configurable dimension
   - Purpose: Stress testing, scaling studies

## Benchmark Implementation

### Setup Script
```bash
#!/bin/bash
# scripts/download_datasets.sh

# Create dataset directory
mkdir -p datasets/{small,medium,large}

# Download SIFT datasets
wget -P datasets/small/ ftp://ftp.irisa.fr/local/texmex/corpus/sift/sift_base.fvecs
wget -P datasets/medium/ ftp://ftp.irisa.fr/local/texmex/corpus/sift/sift_learn.fvecs
wget -P datasets/large/ ftp://ftp.irisa.fr/local/texmex/corpus/sift/sift_1M.fvecs

# Generate synthetic datasets
cargo run --bin generate_dataset -- \
  --output datasets/small/random_100.bin \
  --count 100 --dim 128 --seed 42
```

### Benchmark Runner
```rust
// benches/scaling_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_build_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_scaling");
    
    for &num_vectors in &[100, 1_000, 10_000, 100_000] {
        let vectors = load_dataset(num_vectors);
        
        group.throughput(Throughput::Elements(num_vectors as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    build_index(black_box(vectors))
                });
            },
        );
    }
}
```

### Search Benchmark Matrix
```rust
fn benchmark_search_matrix(c: &mut Criterion) {
    let index = build_large_index(1_000_000);
    let queries = generate_queries(1000);
    
    for &k in &[1, 10, 100] {
        for &search_l in &[50, 100, 200] {
            let params = SearchParams { k, search_l };
            
            c.bench_function(
                &format!("search_k{}_l{}", k, search_l),
                |b| {
                    b.iter(|| {
                        index.search(&queries[0], params)
                    });
                },
            );
        }
    }
}
```

## Performance Targets

Based on C++ DiskANN and optimized implementations:

### Build Performance
| Dataset Size | Target Build Rate | Memory Usage |
|-------------|-------------------|--------------|
| 1K          | > 10K vec/sec    | < 10 MB      |
| 10K         | > 5K vec/sec     | < 100 MB     |
| 100K        | > 2K vec/sec     | < 1 GB       |
| 1M          | > 1K vec/sec     | < 10 GB      |

### Search Performance  
| Dataset Size | Target QPS | Recall@10 | Latency P99 |
|-------------|------------|-----------|-------------|
| 1K          | > 100K     | > 99%     | < 0.1 ms    |
| 10K         | > 50K      | > 95%     | < 0.5 ms    |
| 100K        | > 10K      | > 95%     | < 2 ms      |
| 1M          | > 5K       | > 90%     | < 5 ms      |

### SIMD Performance
| Operation | Scalar | SIMD Target | Speedup |
|-----------|--------|-------------|---------|
| L2 Distance (768D) | 1.0x | 0.25x | 4x |
| Dot Product (768D) | 1.0x | 0.25x | 4x |
| Batch Distance (100x768D) | 1.0x | 0.2x | 5x |

## Running Benchmarks

### Quick Benchmark (< 5 min)
```bash
cargo bench --bench quick -- --sample-size 10
```

### Standard Benchmark (~ 30 min)
```bash
cargo bench
```

### Full Scaling Study (~ 2 hours)
```bash
cargo bench --bench scaling -- --sample-size 50
```

### Platform Comparison
```bash
# ARM64 with NEON
cargo bench --features neon

# x86-64 with AVX2  
cargo bench --features avx2

# Portable SIMD only
cargo bench --no-default-features --features simd
```

## Analysis and Reporting

### Benchmark Output
```
build_scaling/100         time:   [95.23 µs 96.45 µs 97.89 µs]
                          thrpt:  [1.0219 Melem/s 1.0371 Melem/s 1.0503 Melem/s]

build_scaling/1000        time:   [1.245 ms 1.257 ms 1.271 ms]
                          thrpt:  [786.91 Kelem/s 795.66 Kelem/s 803.21 Kelem/s]
```

### Comparison Script
```python
#!/usr/bin/env python3
# scripts/compare_benchmarks.py

import json
import sys

def compare_results(baseline, current):
    """Compare benchmark results and flag regressions"""
    for benchmark in baseline:
        if benchmark in current:
            baseline_time = baseline[benchmark]['mean']
            current_time = current[benchmark]['mean']
            
            regression = (current_time - baseline_time) / baseline_time
            if regression > 0.05:  # 5% regression threshold
                print(f"⚠️  {benchmark}: {regression*100:.1f}% slower")
```

### Visualization
Generate performance plots:
```bash
cargo bench -- --save-baseline before
# Make changes
cargo bench -- --save-baseline after
cargo benchcmp before after --threshold 5
```

## Continuous Benchmarking

### GitHub Actions Workflow
```yaml
name: Benchmarks
on:
  pull_request:
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: boa-dev/criterion-compare-action@v3
        with:
          branchName: ${{ github.base_ref }}
```

### Performance Tracking
Store results in `benchmarks/results/`:
```
benchmarks/results/
├── 2024-01-15_commit_abc123.json
├── 2024-01-16_commit_def456.json
└── baseline.json
```

## Memory Profiling

### Heap Profiling
```bash
cargo build --release
valgrind --tool=massif --massif-out-file=massif.out \
  ./target/release/benchmark_build
ms_print massif.out > memory_profile.txt
```

### Cache Analysis
```bash
cargo build --release
perf stat -e cache-misses,cache-references \
  ./target/release/benchmark_search
```

## Best Practices

1. **Warm-up**: Always include warm-up iterations
2. **Isolation**: Run benchmarks on isolated systems
3. **Reproducibility**: Fix random seeds
4. **Statistics**: Report percentiles, not just mean
5. **Context**: Record system info (CPU, RAM, OS)
6. **Incremental**: Benchmark during development, not just at end