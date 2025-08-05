# CLAUDE.md

This file provides guidance to Claude Code when working with the DiskANN Rust implementation.

## Project Overview

This is a pure Rust implementation of Microsoft's DiskANN algorithm, with first-class support for ARM64 NEON optimizations. The project aims to provide a memory-safe, high-performance alternative to the C++ implementation while maintaining API compatibility with the existing Rust wrapper.

## Key Design Principles

1. **Pure Rust**: Avoid unsafe code except for SIMD intrinsics
2. **Correctness First**: Ensure algorithmic correctness before optimization
3. **Platform Agnostic**: Use portable SIMD libraries (like `wide`) as the default
4. **Modular Design**: Each component should be independently testable
5. **Documentation**: Every public API must be documented with examples

## Architecture

### Distance Module (`src/distance/`)
- **simd.rs**: Portable SIMD using the `wide` crate (default)
- **scalar.rs**: Scalar fallback implementation
- **neon.rs**: ARM64 NEON specific optimizations (optional)
- Factory pattern selects best implementation at runtime

### Graph Module (`src/graph/`)
- **vamana.rs**: Core Vamana graph algorithm
- **search.rs**: Optimized search with bit vector visited tracking
- **prune.rs**: RobustPrune edge selection algorithm

### Index Module (`src/index/`)
- **memory.rs**: In-memory index implementation
- **disk.rs**: Disk-based index (TODO)
- **builder.rs**: Fluent API for index construction

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Property Tests**: Use `proptest` for invariant testing
3. **Benchmarks**: Compare against C++ implementation
4. **Integration Tests**: End-to-end index build and search

## Performance Targets

Based on C++ ARM64 NEON results:
- L2 Distance: 3.73x speedup over scalar
- Graph Search: 320K QPS
- Index Build: 2,457 points/sec

## Common Tasks

### Running Tests
```bash
cargo test
cargo test --release  # For performance tests
```

### Running Benchmarks
```bash
cargo bench
```

### Building with Specific Features
```bash
# ARM64 NEON only
cargo build --release --features neon

# With Python bindings
cargo build --release --features python
```

## Current Status

- ✅ Core distance functions (scalar, SIMD)
- ✅ Vamana graph construction
- ✅ Optimized search algorithms  
- ✅ In-memory index
- 🚧 Disk-based index
- 🚧 Product Quantization
- 🚧 Python bindings

## Next Steps

1. Complete disk-based index implementation
2. Port Product Quantization with SIMD
3. Add Python bindings for compatibility
4. Comprehensive benchmarking suite
5. Integration with existing DiskANN Rust wrapper