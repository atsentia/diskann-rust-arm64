# Testing Strategy for DiskANN Rust

This document outlines the comprehensive testing strategy for the DiskANN Rust implementation. 

**Status**: Phase 1-3 Complete with comprehensive test coverage for all core functionality.

## Test Categories

### 1. Unit Tests

Located in each module's source file using `#[cfg(test)]` blocks.

#### Distance Functions (`src/distance/`)
- **Correctness**: Verify distance calculations match expected values
- **SIMD Accuracy**: Compare SIMD results with scalar implementation
- **Edge Cases**: Zero vectors, unit vectors, orthogonal vectors
- **Performance**: Ensure SIMD provides expected speedup

```bash
cargo test distance
```

#### Graph Operations (`src/graph/`)
- **Vamana Construction**: Verify graph connectivity and degree bounds
- **RobustPrune**: Test edge selection algorithm
- **Search Quality**: Ensure recall meets expectations
- **Thread Safety**: Concurrent read/write operations

```bash
cargo test graph
```

#### Dynamic Operations (`src/index/dynamic.rs`)
- **Insert**: Verify correct graph updates
- **Delete**: Test lazy deletion marking
- **Consolidation**: Ensure proper index rebuilding
- **Fragmentation**: Test automatic consolidation triggers

```bash
cargo test dynamic
```

### 2. Integration Tests

Located in `tests/` directory.

#### End-to-End Index Operations
```rust
// tests/index_operations.rs
#[test]
fn test_build_search_cycle() {
    // Build index from vectors
    // Search and verify results
    // Add new vectors
    // Delete vectors
    // Consolidate and re-verify
}
```

#### File Format Compatibility
```rust
// tests/format_compatibility.rs
#[test]
fn test_fvecs_roundtrip() {
    // Write vectors to fvecs
    // Read back and compare
}
```

### 3. Property-Based Tests

Using `proptest` for invariant testing.

```rust
// tests/properties.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_distance_symmetry(a: Vec<f32>, b: Vec<f32>) {
        let d1 = distance(&a, &b);
        let d2 = distance(&b, &a);
        assert!((d1 - d2).abs() < 1e-6);
    }
    
    #[test]
    fn test_graph_degree_bounds(vectors: Vec<Vec<f32>>) {
        let graph = build_graph(vectors);
        for node in graph.nodes() {
            assert!(node.degree() <= MAX_DEGREE);
        }
    }
}
```

### 4. Benchmarks

Located in `benches/` directory using Criterion.

#### Distance Benchmarks
```rust
// benches/distance.rs
fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");
    
    group.bench_function("scalar", |b| {
        b.iter(|| scalar_l2(&vec1, &vec2))
    });
    
    group.bench_function("simd", |b| {
        b.iter(|| simd_l2(&vec1, &vec2))
    });
    
    #[cfg(target_arch = "aarch64")]
    group.bench_function("neon", |b| {
        b.iter(|| neon_l2(&vec1, &vec2))
    });
}
```

#### Search Benchmarks
```rust
// benches/search.rs
fn bench_search_qps(c: &mut Criterion) {
    let index = build_test_index(10_000);
    
    c.bench_function("search_k10", |b| {
        b.iter(|| index.search(&query, 10))
    });
}
```

### 5. Stress Tests

Test system behavior under load.

```rust
// tests/stress.rs
#[test]
fn test_concurrent_operations() {
    let index = Arc::new(DynamicIndex::new(...));
    let mut handles = vec![];
    
    // Spawn insert threads
    for i in 0..10 {
        let idx = index.clone();
        handles.push(thread::spawn(move || {
            for j in 0..1000 {
                idx.insert(random_vector(), vec![]);
            }
        }));
    }
    
    // Spawn search threads
    for i in 0..10 {
        let idx = index.clone();
        handles.push(thread::spawn(move || {
            for j in 0..1000 {
                idx.search(&random_vector(), 10);
            }
        }));
    }
    
    // Wait and verify
    for h in handles {
        h.join().unwrap();
    }
}
```

## Running Tests

### All Tests
```bash
cargo test
```

### Specific Module
```bash
cargo test distance
cargo test graph
cargo test dynamic
```

### With Output
```bash
cargo test -- --nocapture
```

### Release Mode (Performance)
```bash
cargo test --release
```

### Specific Features
```bash
# Test without NEON
cargo test --no-default-features

# Test with all features
cargo test --all-features
```

### Benchmarks
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench distance

# Save baseline
cargo bench -- --save-baseline main

# Compare with baseline
cargo bench -- --baseline main
```

## Continuous Integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Unit Tests**: On multiple platforms (Linux, macOS, Windows)
2. **Feature Matrix**: Test different feature combinations
3. **Benchmarks**: Track performance regressions
4. **Clippy**: Lint for common mistakes
5. **Format**: Ensure consistent code style

## Test Data

### Synthetic Data
- Random vectors with known properties
- Clustered data for testing graph quality
- Edge cases (duplicates, zero vectors)

### Real Datasets
- SIFT1M: 1 million 128-dimensional SIFT descriptors
- GIST1M: 1 million 960-dimensional GIST descriptors
- Deep1B: Subset for large-scale testing

### Data Generation
```rust
// tests/common/mod.rs
pub fn generate_clustered_data(
    num_clusters: usize,
    points_per_cluster: usize,
    dimension: usize,
) -> Vec<Vec<f32>> {
    // Generate clustered test data
}

pub fn generate_random_data(
    num_vectors: usize,
    dimension: usize,
) -> Vec<Vec<f32>> {
    // Generate random test data
}
```

## Coverage

Generate coverage reports:

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage
cargo tarpaulin --out Html

# With specific features
cargo tarpaulin --features neon --out Html
```

## Performance Regression Detection

1. **Baseline**: Establish performance baselines for each release
2. **CI Integration**: Automatically run benchmarks on PRs
3. **Alerts**: Flag regressions > 5%
4. **Tracking**: Graph performance over time

## Platform-Specific Testing

### ARM64 (Apple Silicon, AWS Graviton)
```bash
# Verify NEON usage
cargo test --features neon
cargo bench --features neon
```

### x86-64 (Intel/AMD)
```bash
# Test AVX2
cargo test --features avx2
cargo bench --features avx2
```

### WebAssembly
```bash
# Install wasm-pack
cargo install wasm-pack

# Run WASM tests
wasm-pack test --node
```

## Debug Tools

### Memory Leaks
```bash
# Using valgrind (Linux)
valgrind --leak-check=full cargo test

# Using sanitizers
RUSTFLAGS="-Z sanitizer=address" cargo test --target x86_64-unknown-linux-gnu
```

### Thread Safety
```bash
RUSTFLAGS="-Z sanitizer=thread" cargo test --target x86_64-unknown-linux-gnu
```

### Performance Profiling
```bash
# Using perf (Linux)
cargo build --release
perf record --call-graph=dwarf ./target/release/bench
perf report

# Using Instruments (macOS)
cargo instruments -t "Time Profiler" --bench distance
```