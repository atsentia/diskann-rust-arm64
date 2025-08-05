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

#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
pub mod sse42;

#[cfg(all(target_arch = "x86_64", feature = "fma4"))]
pub mod amd_fma4;

#[cfg(all(target_arch = "aarch64", feature = "sve"))]
pub mod arm_sve;

#[cfg(all(target_arch = "x86_64", feature = "amx"))]
pub mod intel_amx;

/// Portable SIMD implementations using pure Rust
pub mod simd;

/// Scalar (non-SIMD) implementations as fallback
pub mod scalar;

/// Distance metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    
    /// Calculate distance from query to a specific vector ID (for graph search)
    fn distance_to_query(&self, query: &[f32], target_id: usize) -> Result<f32> {
        // Default implementation - subclasses can override for specialized behavior
        Err(anyhow::anyhow!("distance_to_query not implemented for this distance function"))
    }
    
    /// Get the metric type
    fn metric(&self) -> Distance;
}

/// Factory function to create the best distance function for the current platform
pub fn create_distance_function(metric: Distance, dimension: usize) -> Box<dyn DistanceFunction> {
    // Check CPU features at runtime and select the best implementation
    // Priority order: Most advanced SIMD → Older SIMD → Portable SIMD
    
    // ARM64 processors
    #[cfg(all(target_arch = "aarch64", feature = "sve"))]
    {
        // SVE intrinsics not yet stable, so this will always be false for now
        if false { // arm_sve::ArmSveDistance::has_sve_support() when available
            log::debug!("Using ARM64 SVE SIMD optimizations for {} distance (dim={})", metric_name(metric), dimension);
            return Box::new(arm_sve::ArmSveDistance::new(metric, dimension));
        }
    }
    
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if crate::has_neon_support() {
            log::debug!("Using ARM64 NEON SIMD optimizations for {} distance (dim={})", metric_name(metric), dimension);
            return Box::new(neon::NeonDistance::new(metric, dimension));
        }
    }
    
    // x86-64 processors (newest to oldest)
    #[cfg(all(target_arch = "x86_64", feature = "amx"))]
    {
        // AMX intrinsics not yet stable, so this will always be false for now
        if dimension >= 512 && false { // intel_amx::IntelAmxDistance::has_amx_support() when available
            log::debug!("Using Intel AMX SIMD optimizations for {} distance (dim={})", metric_name(metric), dimension);
            return Box::new(intel_amx::IntelAmxDistance::new(metric, dimension));
        }
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if is_x86_feature_detected!("avx512f") {
            log::debug!("Using x86-64 AVX-512 SIMD optimizations for {} distance (dim={})", metric_name(metric), dimension);
            return Box::new(avx512::Avx512Distance::new(metric, dimension));
        }
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if crate::has_avx2_support() {
            log::debug!("Using x86-64 AVX2 SIMD optimizations for {} distance (dim={})", metric_name(metric), dimension);
            return Box::new(avx2::Avx2Distance::new(metric, dimension));
        }
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "fma4"))]
    {
        if is_x86_feature_detected!("fma4") {
            log::debug!("Using AMD FMA4 SIMD optimizations for {} distance (dim={})", metric_name(metric), dimension);
            return Box::new(amd_fma4::AmdFma4Distance::new(metric, dimension));
        }
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
    {
        if is_x86_feature_detected!("sse4.2") {
            log::debug!("Using x86-64 SSE4.2 SIMD optimizations for {} distance (dim={})", metric_name(metric), dimension);
            return Box::new(sse42::Sse42Distance::new(metric, dimension));
        }
    }
    
    // Use portable SIMD implementation as default before scalar
    log::debug!("Using portable SIMD implementation for {} distance (dim={})", metric_name(metric), dimension);
    Box::new(simd::SimdDistance::new(metric, dimension))
}

/// Get human-readable name for distance metric
fn metric_name(metric: Distance) -> &'static str {
    match metric {
        Distance::L2 => "L2",
        Distance::Cosine => "Cosine", 
        Distance::InnerProduct => "InnerProduct",
    }
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