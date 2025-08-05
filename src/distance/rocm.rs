//! AMD ROCm GPU acceleration for distance calculations
//!
//! This module provides ROCm-accelerated distance functions for AMD GPUs.
//! Note: This is currently a stub implementation. Real ROCm support requires
//! stable rocm-rs or hip-sys crates which are still in early development.

use crate::{Distance, DistanceFunction, Result, Error};

/// AMD ROCm distance calculator (stub implementation)
pub struct RocmDistance {
    metric: Distance,
    dimension: usize,
}

impl RocmDistance {
    /// Create a new ROCm distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Result<Self> {
        // Stub implementation - real ROCm support requires stable crates
        Err(anyhow::anyhow!(
            "ROCm support is not yet available. Waiting for stable rocm-rs or hip-sys crates. \
             Please use WebGPU, CUDA, or CPU SIMD implementations instead."
        ).into())
    }

    /// Check if ROCm is available
    pub fn is_available() -> bool {
        // Always return false for stub implementation
        false
    }

    /// Get ROCm device information
    pub fn get_device_info(&self) -> String {
        "ROCm support not yet implemented".to_string()
    }
}

impl DistanceFunction for RocmDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        // Fallback to scalar implementation
        crate::distance::scalar::scalar_distance(a, b, self.metric)
    }
    
    fn batch_distance(&self, query: &[f32], points: &[f32], distances: &mut [f32]) -> Result<()> {
        let dimension = self.dimension;
        let num_points = points.len() / dimension;
        
        // Fallback to scalar implementation
        for i in 0..num_points {
            let start_idx = i * dimension;
            let end_idx = start_idx + dimension;
            distances[i] = self.distance(query, &points[start_idx..end_idx])?;
        }
        
        Ok(())
    }
    
    fn metric(&self) -> Distance {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rocm_not_available() {
        assert!(!RocmDistance::is_available());
        
        let result = RocmDistance::new(Distance::L2, 128);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ROCm support is not yet available"));
    }
}