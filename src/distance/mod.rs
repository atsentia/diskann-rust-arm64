//! Distance calculation module with SIMD optimizations
//!
//! This module provides high-performance distance calculations using
//! platform-specific SIMD instructions (ARM64 NEON, x86 AVX2/AVX512).

use crate::Result;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub mod neon;

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod avx512;

/// Portable SIMD implementations using pure Rust
pub mod simd;

/// Scalar (non-SIMD) implementations as fallback
pub mod scalar;

/// Distance metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    /// L2 (Euclidean) distance
    L2,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Inner product distance (negative dot product)
    InnerProduct,
}

/// Trait for distance calculation functions
pub trait DistanceFunction: Send + Sync {
    /// Calculate distance between two vectors
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32>;
    
    /// Calculate squared distance (useful for L2 to avoid sqrt)
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let d = self.distance(a, b)?;
        Ok(d * d)
    }
    
    /// Batch distance calculation from query to multiple points
    fn batch_distance(&self, query: &[f32], points: &[f32], distances: &mut [f32]) -> Result<()>;
    
    /// Get the metric type
    fn metric(&self) -> Distance;
}

/// Factory function to create the best distance function for the current platform
pub fn create_distance_function(metric: Distance, dimension: usize) -> Box<dyn DistanceFunction> {
    // Check CPU features at runtime and select the best implementation
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if crate::has_neon_support() {
            return Box::new(neon::NeonDistance::new(metric, dimension));
        }
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return Box::new(avx512::Avx512Distance::new(metric, dimension));
        }
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if crate::has_avx2_support() {
            return Box::new(avx2::Avx2Distance::new(metric, dimension));
        }
    }
    
    // Use portable SIMD implementation as default before scalar
    Box::new(simd::SimdDistance::new(metric, dimension))
}

/// Alignment requirements for SIMD operations
pub const SIMD_ALIGN: usize = 32;

/// Ensure a slice is properly aligned for SIMD operations
#[inline]
pub fn is_aligned(ptr: *const f32) -> bool {
    ptr as usize % SIMD_ALIGN == 0
}

/// Allocate aligned memory for vectors
pub fn aligned_vec(size: usize) -> Vec<f32> {
    // Use bytemuck for aligned allocation
    let mut vec = vec![0.0f32; size + (SIMD_ALIGN / 4)];
    let offset = vec.as_ptr() as usize % SIMD_ALIGN;
    if offset != 0 {
        let shift = (SIMD_ALIGN - offset) / 4;
        vec.drain(0..shift);
    }
    vec.truncate(size);
    vec
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aligned_vec() {
        let vec = aligned_vec(128);
        assert_eq!(vec.len(), 128);
        assert!(is_aligned(vec.as_ptr()));
    }
    
    #[test]
    fn test_distance_factory() {
        let dist = create_distance_function(Distance::L2, 128);
        assert_eq!(dist.metric(), Distance::L2);
    }
}