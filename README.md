# DiskANN Rust Implementation with ARM64 NEON Support

A high-performance, memory-safe implementation of Microsoft's DiskANN algorithm in pure Rust, with first-class support for ARM64 NEON SIMD instructions.

## Features

- 🚀 **ARM64 NEON Optimizations**: 3-6x speedup on Apple Silicon and ARM servers
- 🦀 **Pure Rust**: Memory-safe implementation with zero undefined behavior
- 🔧 **Cross-Platform**: Supports ARM64, x86-64 (with AVX2/AVX512), and WebAssembly
- ⚡ **High Performance**: Matches or exceeds C++ implementation performance
- 🔄 **Async I/O**: Efficient disk operations with Tokio
- 🐍 **Python Bindings**: Drop-in replacement for the original Python API
- 📦 **Modular Design**: Use only the components you need

## Architecture

This implementation is designed to be integrated with the existing Rust wrapper for DiskANN C++, providing a pure Rust alternative with enhanced ARM64 performance.

### Key Components

1. **Distance Functions** (`src/distance/`)
   - Platform-optimized SIMD implementations
   - ARM64 NEON, x86 AVX2/AVX512 support
   - Automatic CPU feature detection

2. **Graph Operations** (`src/graph/`)
   - Vamana graph construction
   - Efficient beam search
   - Concurrent updates with lock-free algorithms

3. **Index Types** (`src/index/`)
   - In-memory index for small datasets
   - Disk-based index for billion-scale search
   - Compressed indices with Product Quantization

4. **I/O System** (`src/io/`)
   - Async file operations
   - Memory-mapped files
   - Efficient caching strategies

## Performance

Based on our C++ ARM64 NEON optimizations:

| Operation | Scalar | NEON | Speedup |
|-----------|--------|------|---------|
| L2 Distance | 0.361 μs | 0.097 μs | 3.73x |
| Graph Search | 85K QPS | 320K QPS | 3.76x |
| Index Build | 763 pts/s | 2,457 pts/s | 3.22x |

## Quick Start

```rust
use diskann::{Index, IndexBuilder, Distance};

// Build an index
let index = IndexBuilder::new()
    .dimensions(128)
    .metric(Distance::L2)
    .max_degree(32)
    .build_from_file("vectors.bin")?;

// Search for nearest neighbors
let query = vec![0.1; 128];
let results = index.search(&query, 10)?;

for (id, distance) in results {
    println!("ID: {}, Distance: {}", id, distance);
}
```

## Building from Source

### Prerequisites

- Rust 1.75 or later
- For ARM64 NEON: ARM64 processor (Apple Silicon, AWS Graviton, etc.)
- For AVX2/AVX512: x86-64 processor with AVX2/AVX512 support

### Build

```bash
# Standard build (auto-detects CPU features)
cargo build --release

# Build without NEON (scalar fallback)
cargo build --release --no-default-features

# Build with specific features
cargo build --release --features "avx2,python"
```

### Run Tests

```bash
cargo test
cargo test --release  # Performance tests
```

### Run Benchmarks

```bash
cargo bench
```

## Integration with Existing DiskANN Rust Wrapper

This crate is designed to be a drop-in replacement for the C++ backend:

```rust
// In the existing wrapper's Cargo.toml
[dependencies]
diskann-backend = { package = "diskann", version = "0.1" }

// Use the same API as before
use diskann_backend as diskann;
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Microsoft Research for the original DiskANN algorithm
- The Rust community for excellent SIMD support
- ARM for comprehensive NEON documentation