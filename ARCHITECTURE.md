# DiskANN Rust Architecture

## SIMD Strategy

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