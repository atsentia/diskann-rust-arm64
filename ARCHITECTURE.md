# DiskANN Rust Architecture

This document describes the architecture of the pure Rust DiskANN implementation.

## Project Structure

```
diskann/
├── src/
│   ├── distance/        # SIMD-optimized distance calculations
│   ├── graph/          # Vamana graph algorithm
│   ├── index/          # Index implementations (memory, dynamic, disk)
│   ├── labels/         # Label and filter support
│   ├── types/          # Multi-type vector support
│   ├── formats/        # File I/O for various formats
│   ├── io/             # Async I/O and caching
│   ├── pq/             # Product Quantization (in progress)
│   └── utils/          # Utilities and helpers
├── benches/            # Criterion benchmarks
├── tests/              # Integration tests
└── examples/           # Usage examples
```

## Core Components

### 1. Distance Functions (`src/distance/`)
The foundation of similarity search, optimized for different platforms.

#### SIMD Strategy

This implementation uses a multi-tier approach for SIMD optimizations:

### 1. Portable SIMD (Default) - `src/distance/simd.rs`
- Uses the `wide` crate which provides portable SIMD types
- `f32x8` compiles to:
  - **ARM64**: NEON instructions (128-bit registers, processes 4 at a time, uses 2 registers)
  - **x86-64 with AVX**: AVX instructions (256-bit registers, processes 8 at a time)  
  - **x86-64 without AVX**: SSE instructions (128-bit registers, processes 4 at a time)
  - **Other platforms**: Scalar fallback
- This is the default because it provides good performance across all platforms

### 2. Platform-Specific SIMD (Optional)
- **ARM64 NEON** (`src/distance/neon.rs`): Direct NEON intrinsics for maximum ARM64 performance
  - Uses `std::arch::aarch64::*` intrinsics
  - Requires `target_feature = "neon"`
  - Can achieve 3-5x speedup over scalar
  
- **x86-64 AVX2** (`src/distance/avx2.rs`): Direct AVX2 intrinsics (not yet implemented)
  - Would use `std::arch::x86_64::*` intrinsics
  - Requires `target_feature = "avx2"`
  
- **x86-64 AVX-512** (`src/distance/avx512.rs`): Direct AVX-512 intrinsics (not yet implemented)
  - Would use `std::arch::x86_64::*` intrinsics
  - Requires `target_feature = "avx512f"`

### 3. Scalar Fallback - `src/distance/scalar.rs`
- Pure Rust implementation with no SIMD
- Used when no SIMD support is available
- Serves as reference implementation for correctness

## Runtime Selection

The `create_distance_function()` factory selects the best implementation at runtime:

```rust
1. Check if platform-specific SIMD is available (NEON on ARM64, AVX on x86)
2. If not, use portable SIMD (which still compiles to platform SIMD)
3. If SIMD is completely disabled, fall back to scalar
```

## Performance Characteristics

### ARM64 (Apple Silicon, AWS Graviton)
- Portable SIMD: Uses NEON automatically via `wide` crate
- Native NEON: Slightly better performance with hand-tuned intrinsics
- Expected speedup: 3-5x over scalar

### x86-64 (Intel/AMD)
- Portable SIMD: Uses AVX2/AVX/SSE based on CPU capabilities
- Native AVX2/AVX-512: Could provide 10-20% better performance (not yet implemented)
- Expected speedup: 4-8x over scalar

### WebAssembly
- Portable SIMD: Uses WASM SIMD when available
- Scalar fallback: When WASM SIMD not supported
- Expected speedup: 2-4x over scalar (when SIMD available)

## Memory Layout

All implementations maintain the same memory layout:
- Vectors are stored as contiguous `Vec<f32>`
- Alignment is handled by the allocator
- No special alignment requirements for portable SIMD

## Correctness

All SIMD implementations are tested against the scalar implementation:
- Exact match for integer operations
- Floating-point operations tested with epsilon tolerance
- Edge cases: empty vectors, single element, non-SIMD-aligned sizes

### 2. Graph Structure (`src/graph/`)

The core Vamana graph algorithm for approximate nearest neighbor search.

#### VamanaGraph
- **Adjacency List**: Each vertex stores its neighbors
- **Bidirectional Edges**: All edges are bidirectional for robustness
- **Entry Point**: Medoid-based entry point selection
- **Thread Safety**: Uses `Arc<RwLock<>>` for concurrent access

#### Key Algorithms
- **RobustPrune**: Diverse neighbor selection for better recall
- **Greedy Search**: Beam search with early termination
- **Dynamic Updates**: Support for insert/delete operations

### 3. Index Types (`src/index/`)

#### MemoryIndex
- In-memory storage of vectors and graph
- Optimized for small to medium datasets (<10M vectors)
- Fast search with no I/O overhead

#### DynamicIndex
- Supports insert, delete, and consolidate operations
- Lazy deletion with periodic consolidation
- Thread-safe with fine-grained locking
- Automatic fragmentation management

#### StreamingIndex
- Async operations with background worker
- Non-blocking insert/delete operations
- Immediate search capability

### 4. Type System (`src/types/`)

Supports multiple vector types with transparent conversion:
- **float32**: Standard 32-bit floating point
- **float16**: Half precision for memory efficiency
- **int8**: Quantized vectors for compression
- **uint8**: Unsigned quantized vectors

### 5. Label System (`src/labels/`)

Efficient filtered search with label support:
- **LabelSet**: Compact representation of labels per vector
- **LabelIndex**: Inverted index for fast label queries
- **Universal Label**: Special label (0) that matches all queries

### 6. I/O System (`src/io/`)

#### Memory-Mapped Files
- Zero-copy access to large datasets
- Efficient random access patterns
- Platform-optimized caching

#### Async I/O
- Tokio-based async file operations
- Prefetching for sequential access
- LRU cache for repeated reads

### 7. File Formats (`src/formats/`)

Support for common vector file formats:
- **fvecs**: Float vectors (LittleEndian)
- **bvecs**: Byte vectors (int8/uint8)
- **ivecs**: Integer vectors
- **binary**: Custom binary format with metadata

## Design Principles

### 1. Safety First
- No unsafe code except for SIMD intrinsics
- All unsafe blocks are documented and tested
- Memory safety guaranteed by Rust's type system

### 2. Performance
- Zero-cost abstractions where possible
- SIMD optimization for all distance calculations
- Cache-friendly data structures
- Minimal allocations in hot paths

### 3. Modularity
- Each component is independently testable
- Clear interfaces between modules
- Feature flags for optional components

### 4. Compatibility
- File format compatibility with C++ DiskANN
- API compatibility with existing wrappers
- Cross-platform support

## Concurrency Model

### Read-Heavy Optimization
- Multiple readers, single writer pattern
- RwLock for graph structure
- Lock-free reads where possible

### Dynamic Operations
- Fine-grained locking for updates
- Lazy deletion to minimize contention
- Background consolidation thread

## Memory Management

### Vector Storage
- Contiguous storage for cache efficiency
- Optional memory mapping for large datasets
- Aligned allocation for SIMD operations

### Graph Storage
- Adjacency list with fixed maximum degree
- Compact representation (4 bytes per edge)
- Optional compression for large graphs

## Future Enhancements

### Product Quantization
- 8x-32x compression ratios
- SIMD-optimized distance calculations
- Hierarchical quantization for better recall

### Disk-Based Index
- Streaming graph construction
- Compressed graph format
- SSD-optimized layout

### GPU Acceleration
- CUDA/Metal compute shaders
- Batch distance calculations
- Parallel graph construction