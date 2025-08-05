# DiskANN Rust Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for the DiskANN Rust implementation, ensuring correctness, performance, and compatibility with the C++ implementation.

## Test Categories

### 1. Unit Tests

Located in each module's source file (`#[cfg(test)] mod tests`).

**Purpose**: Test individual components in isolation.

**Coverage**:
- Distance calculations (scalar, SIMD, platform-specific)
- Graph operations (insert, prune, search)
- Label filtering
- Data type conversions
- File I/O operations
- Memory alignment

**Example**:
```rust
cargo test --lib
cargo test --lib --release  # Performance-sensitive tests
```

### 2. Integration Tests

Located in `tests/` directory.

**Purpose**: Test component interactions and end-to-end workflows.

**Coverage**:
- Index build and search pipeline
- Cross-component data flow
- File format compatibility
- Multi-threaded operations
- Memory management

**Example**:
```rust
cargo test --test integration_tests
```

### 3. Smoke Tests

Quick sanity checks that run in < 1 minute.

**Coverage**:
- Basic index creation (100 vectors)
- Simple search operations
- All distance metrics
- All data types
- Basic label filtering

**Script**: `scripts/smoke_test.sh`
```bash
#!/bin/bash
# Run smoke tests
cargo test --test smoke_tests -- --test-threads=1
```

### 4. Correctness Tests

Verify algorithmic correctness against known results.

**Coverage**:
- Exact recall on small datasets
- Distance calculation accuracy
- Graph connectivity properties
- Label filtering accuracy
- Comparison with C++ implementation

**Dataset**: 
- SIFT-1K subset with ground truth
- Synthetic clustered data
- Edge cases (single vector, duplicates)

### 5. Stress Tests

Push the system to its limits.

**Coverage**:
- Memory pressure (large indices)
- Concurrent operations
- Long-running operations
- Resource cleanup
- Error recovery

**Example**:
```rust
cargo test --test stress_tests --release -- --ignored
```

### 6. Property-Based Tests

Using `proptest` for invariant testing.

**Coverage**:
- Distance metric properties (triangle inequality)
- Graph properties (connectivity, degree bounds)
- Index consistency
- Serialization round-trips

### 7. Platform-Specific Tests

Test SIMD implementations across platforms.

**Coverage**:
- ARM64 NEON correctness
- x86-64 AVX2/AVX-512 correctness
- Fallback implementations
- Performance characteristics

**Conditional compilation**:
```rust
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[test]
fn test_neon_specific() { ... }
```

## Test Data

### Synthetic Data
- Random vectors (uniform, gaussian)
- Clustered data (known structure)
- Adversarial cases (all same, orthogonal)
- Edge cases (zero vectors, extreme values)

### Real Datasets
- **Small** (< 10K vectors):
  - SIFT-1K
  - MNIST subset
  - GloVe-25 subset
  
- **Medium** (10K-100K vectors):
  - SIFT-10K
  - Fashion-MNIST
  - GloVe-100K
  
- **Large** (100K-1M vectors):
  - SIFT-1M
  - GIST-1M
  - Deep-1M

### Ground Truth
- Exact k-NN computed via brute force
- Stored in `tests/data/ground_truth/`
- Format: `<dataset>_<k>nn.bin`

## Test Execution

### Local Development
```bash
# Fast unit tests
cargo test --lib

# All tests
cargo test --all

# Specific module
cargo test distance::

# With output
cargo test -- --nocapture

# Release mode (important for performance tests)
cargo test --release
```

### CI Pipeline
1. **Quick Tests** (on every commit)
   - Clippy lints
   - Format check
   - Unit tests
   - Smoke tests

2. **Full Tests** (on PR)
   - All unit tests
   - Integration tests
   - Platform matrix (Linux, macOS, Windows)
   - Multiple Rust versions

3. **Nightly Tests**
   - Stress tests
   - Large dataset tests
   - Memory leak detection
   - Performance regression

## Test Metrics

### Code Coverage
- Target: 90% line coverage
- Tool: `cargo-tarpaulin`
- Exclude: Binary tools, examples

### Performance Regression
- Track key metrics:
  - Distance calculations/sec
  - Index build time
  - Search QPS
  - Memory usage
- Alert on >5% regression

## Cross-Implementation Validation

### C++ Compatibility Tests
Compare results with C++ DiskANN:

1. **Binary Format**: Ensure indices are interchangeable
2. **Search Results**: Same k-NN results (accounting for ties)
3. **Performance**: Within 10% of C++ implementation

### Test Harness
```rust
// tests/cross_validation.rs
#[test]
fn validate_against_cpp() {
    let cpp_index = load_cpp_index("data/cpp_index.bin");
    let rust_index = load_rust_index("data/rust_index.bin");
    
    for query in test_queries {
        let cpp_results = cpp_search(&cpp_index, query);
        let rust_results = rust_search(&rust_index, query);
        
        assert_recall(cpp_results, rust_results, min_recall: 0.99);
    }
}
```

## Debugging Failed Tests

### Tools
- `RUST_BACKTRACE=1` for stack traces
- `RUST_LOG=debug` for detailed logging
- `cargo test -- --test-threads=1` for sequential execution
- Memory sanitizers: `RUSTFLAGS="-Z sanitizer=address"`

### Common Issues
1. **Platform differences**: Use epsilon comparisons for floats
2. **Concurrency**: Use deterministic thread pools for tests
3. **Memory alignment**: Verify SIMD alignment requirements
4. **File paths**: Use `tempfile` crate for test files

## Test Documentation

Each test should include:
- Purpose comment
- Expected behavior
- Any prerequisites
- Performance expectations (if applicable)

Example:
```rust
/// Test that L2 distance with NEON is within epsilon of scalar version
/// and at least 2x faster for aligned 768-dimensional vectors
#[test]
fn test_neon_l2_distance_accuracy_and_performance() {
    // test implementation
}
```