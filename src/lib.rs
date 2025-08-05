//! DiskANN: A high-performance vector search library with ARM64 NEON optimizations
//!
//! This crate provides a pure Rust implementation of Microsoft's DiskANN algorithm,
//! optimized for ARM64 processors with NEON SIMD instructions.

#![warn(missing_docs)]

/// Distance calculation functions with SIMD optimizations
pub mod distance;

/// Graph data structures and algorithms
pub mod graph;

/// Index implementations (memory and disk-based)
pub mod index;

/// I/O utilities and async operations
pub mod io;

/// Product quantization for compression
pub mod pq;

/// General utilities and helpers
pub mod utils;

/// Label and filter support
pub mod labels;

/// Data type support (int8, uint8, float16, float32)
pub mod types;

/// File format support (fvecs, bvecs, ivecs, binary)
pub mod formats;

/// Search algorithms (range search, filtered search)
pub mod search;

/// Command-line interface modules
pub mod cli;

/// CLI configuration structure
#[derive(Debug)]
pub struct Cli {
    pub verbose: bool,
    pub no_progress: bool,
}

// Re-export commonly used types
pub use distance::{Distance, DistanceFunction, create_distance_function};
pub use index::{Index, IndexBuilder};

/// Result type for DiskANN operations
pub type Result<T> = anyhow::Result<T>;

/// Error types for DiskANN operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid dimension for vectors
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    /// I/O operation failed
    #[error("I/O error: {0}")]
    Io(String),
    
    /// Index operation failed
    #[error("Index error: {0}")]
    Index(String),
    
    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Invalid state for operation
    #[error("Invalid state: {0}")]
    InvalidState(String),
    
    /// Invalid file format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}


/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if ARM64 NEON is available at runtime
#[cfg(target_arch = "aarch64")]
pub fn has_neon_support() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

/// Check if ARM64 NEON is available at runtime
#[cfg(not(target_arch = "aarch64"))]
pub fn has_neon_support() -> bool {
    false
}

/// Check if x86-64 AVX2 is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn has_avx2_support() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Check if x86-64 AVX2 is available at runtime
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2_support() -> bool {
    false
}

/// Check if x86-64 AVX512 is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn has_avx512_support() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// Check if x86-64 AVX512 is available at runtime
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512_support() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection() {
        // Just ensure these functions compile and run
        let _ = has_neon_support();
        let _ = has_avx2_support();
        let _ = has_avx512_support();
    }
}