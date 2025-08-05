//! Intel AMX optimized distance calculations  
//!
//! This module provides SIMD-accelerated distance functions specifically
//! optimized for Intel processors with Advanced Matrix Extensions (AMX).

use crate::{Distance, DistanceFunction, Result, Error};

// Note: Intel AMX intrinsics are not yet available in stable Rust
// This is a forward-looking implementation for when they become available
// AMX is designed for large matrix operations, most beneficial for high dimensions

/// Intel AMX-optimized distance calculator
pub struct IntelAmxDistance {
    metric: Distance,
    dimension: usize,
    tile_size: usize, // AMX tiles are typically 16x64 bytes
}

impl IntelAmxDistance {
    /// Create a new Intel AMX distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Self {
        // AMX is most beneficial for larger dimensions (512+)
        // For smaller dimensions, other SIMD approaches are better
        let tile_size = if dimension >= 512 { 1024 } else { 64 };
        
        Self { 
            metric, 
            dimension,
            tile_size,
        }
    }
    
    /// Check if Intel AMX is available
    #[cfg(target_arch = "x86_64")]
    fn has_amx_support() -> bool {
        // Placeholder: In real implementation, this would check:
        // is_x86_feature_detected!("amx-tile") && is_x86_feature_detected!("amx-bf16")
        // For now, always return false since AMX intrinsics aren't stable
        false
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn has_amx_support() -> bool {
        false
    }
    
    /// Check if dimension is suitable for AMX optimization
    fn is_amx_beneficial(&self) -> bool {
        // AMX excels with large vectors due to its matrix-oriented design
        // For smaller vectors, the overhead isn't worth it
        self.dimension >= 512
    }
    
    /// Intel AMX-optimized L2 distance calculation (placeholder)
    #[cfg(target_arch = "x86_64")]
    fn l2_distance_amx(&self, a: &[f32], b: &[f32]) -> f32 {
        // This would use Intel AMX intrinsics when available:
        // - _tile_loadd() to load data into tiles
        // - _tile_dpbf16ps() for BF16 matrix multiply-add operations  
        // - _tile_stored() to store results
        // - Custom reduction across tiles
        
        // AMX strategy for L2 distance:
        // 1. Split vectors into matrix tiles
        // 2. Compute (a-b) using tile operations
        // 3. Compute (a-b)^2 using tile multiply
        // 4. Sum across all tiles
        
        // For now, fall back to optimized scalar implementation
        Self::l2_distance_scalar_optimized(a, b)
    }
    
    /// Intel AMX-optimized dot product (placeholder)
    #[cfg(target_arch = "x86_64")]
    fn dot_product_amx(&self, a: &[f32], b: &[f32]) -> f32 {
        // This would use AMX matrix multiply capabilities:
        // - Reshape vectors as 1xN and Nx1 matrices
        // - Use _tile_dpbf16ps() for efficient dot product
        // - Single tile operation for the entire computation
        
        // For now, fall back to optimized scalar implementation
        Self::dot_product_scalar_optimized(a, b)
    }
    
    /// Intel AMX-optimized cosine distance (placeholder)
    #[cfg(target_arch = "x86_64")]
    fn cosine_distance_amx(&self, a: &[f32], b: &[f32]) -> f32 {
        // This would compute dot product and norms using three
        // separate AMX tile operations, then combine results
        
        // For now, fall back to optimized scalar implementation
        Self::cosine_distance_scalar_optimized(a, b)
    }
    
    /// Optimized scalar fallback for L2 distance (loop unrolled)
    fn l2_distance_scalar_optimized(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let len = a.len();
        let mut i = 0;
        
        // Unroll loop by 8 for better performance
        while i + 8 <= len {
            let diff0 = a[i] - b[i];
            let diff1 = a[i + 1] - b[i + 1];
            let diff2 = a[i + 2] - b[i + 2];
            let diff3 = a[i + 3] - b[i + 3];
            let diff4 = a[i + 4] - b[i + 4];
            let diff5 = a[i + 5] - b[i + 5];
            let diff6 = a[i + 6] - b[i + 6];
            let diff7 = a[i + 7] - b[i + 7];
            
            sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3
                 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
            
            i += 8;
        }
        
        // Handle remaining elements
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        
        sum.sqrt()
    }
    
    /// Optimized scalar fallback for dot product
    fn dot_product_scalar_optimized(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let len = a.len();
        let mut i = 0;
        
        // Unroll loop by 8 for better performance
        while i + 8 <= len {
            sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3]
                 + a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5] + a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
            i += 8;
        }
        
        // Handle remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }
    
    /// Optimized scalar fallback for cosine distance
    fn cosine_distance_scalar_optimized(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        let len = a.len();
        let mut i = 0;
        
        // Unroll loop by 4 for better performance (3 operations per iteration)
        while i + 4 <= len {
            let a0 = a[i];
            let b0 = b[i];
            let a1 = a[i + 1];
            let b1 = b[i + 1];
            let a2 = a[i + 2];
            let b2 = b[i + 2];
            let a3 = a[i + 3];
            let b3 = b[i + 3];
            
            dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
            norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
            
            i += 4;
        }
        
        // Handle remaining elements
        while i < len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
            i += 1;
        }
        
        let norm_product = (norm_a * norm_b).sqrt();
        if norm_product > 0.0 {
            1.0 - (dot / norm_product)
        } else {
            0.0
        }
    }
}

impl DistanceFunction for IntelAmxDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(Error::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            }.into());
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if Self::has_amx_support() && self.is_amx_beneficial() {
                let result = match self.metric {
                    Distance::L2 => self.l2_distance_amx(a, b),
                    Distance::Cosine => self.cosine_distance_amx(a, b),
                    Distance::InnerProduct => -self.dot_product_amx(a, b), // Negative for max-heap
                };
                return Ok(result);
            }
        }
        
        // Fallback for non-x86_64 or non-AMX systems, but use optimized scalar
        let result = match self.metric {
            Distance::L2 => Self::l2_distance_scalar_optimized(a, b),
            Distance::Cosine => Self::cosine_distance_scalar_optimized(a, b),
            Distance::InnerProduct => -Self::dot_product_scalar_optimized(a, b),
        };
        Ok(result)
    }
    
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(Error::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            }.into());
        }
        
        match self.metric {
            Distance::L2 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if Self::has_amx_support() && self.is_amx_beneficial() {
                        // AMX-optimized squared distance would avoid sqrt
                        let mut sum = 0.0f32;
                        for i in 0..a.len() {
                            let diff = a[i] - b[i];
                            sum += diff * diff;
                        }
                        return Ok(sum);
                    }
                }
                
                // Use optimized scalar fallback without sqrt
                let mut sum = 0.0f32;
                let len = a.len();
                let mut i = 0;
                
                while i + 8 <= len {
                    let diff0 = a[i] - b[i];
                    let diff1 = a[i + 1] - b[i + 1];
                    let diff2 = a[i + 2] - b[i + 2];
                    let diff3 = a[i + 3] - b[i + 3];
                    let diff4 = a[i + 4] - b[i + 4];
                    let diff5 = a[i + 5] - b[i + 5];
                    let diff6 = a[i + 6] - b[i + 6];
                    let diff7 = a[i + 7] - b[i + 7];
                    
                    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3
                         + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
                    
                    i += 8;
                }
                
                while i < len {
                    let diff = a[i] - b[i];
                    sum += diff * diff;
                    i += 1;
                }
                
                Ok(sum)
            },
            _ => {
                let d = self.distance(a, b)?;
                Ok(d * d)
            }
        }
    }
    
    fn batch_distance(&self, query: &[f32], points: &[f32], distances: &mut [f32]) -> Result<()> {
        let dimension = self.dimension;
        let num_points = points.len() / dimension;
        
        if distances.len() != num_points {
            return Err(anyhow::anyhow!("Distances array length mismatch").into());
        }
        
        // AMX would excel at batch processing with matrix operations
        // treating the query as a 1xD matrix and points as an NxD matrix
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

// Future implementation notes for when Intel AMX intrinsics become available:
//
// Key AMX advantages:
// 1. Massive compute throughput for large matrices (8KB tiles)
// 2. BF16 and INT8 support for reduced precision with maintained accuracy
// 3. Extremely efficient for batch operations
// 4. Best-in-class performance for dimensions > 1024
//
// Example AMX intrinsics that would be used:
// - _tile_loadd() - Load data into 8KB tiles
// - _tile_dpbf16ps() - BF16 matrix multiply-add
// - _tile_dpbusd() - INT8 matrix multiply-add  
// - _tile_stored() - Store tile data back to memory
// - _tile_zero() - Zero tile contents
//
// Performance expectations:
// - 10-50x speedup for very large dimensions (2048+)
// - Most beneficial with batch processing
// - Requires careful memory layout optimization
// - Best with reduced precision (BF16/INT8)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_intel_amx_fallback() {
        // Since AMX intrinsics aren't available yet, test fallback behavior
        let vectors = generate_random_vectors(2, 1024); // Large dimension for AMX
        let amx_dist = IntelAmxDistance::new(Distance::L2, 1024);
        
        let distance = amx_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }
    
    #[test]
    fn test_intel_amx_small_dimension() {
        // Test that small dimensions work (though not AMX-optimized)
        let vectors = generate_random_vectors(2, 64);
        let amx_dist = IntelAmxDistance::new(Distance::L2, 64);
        
        let distance = amx_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }
    
    #[test]
    fn test_intel_amx_beneficial_check() {
        let amx_small = IntelAmxDistance::new(Distance::L2, 128);
        let amx_large = IntelAmxDistance::new(Distance::L2, 1024);
        
        assert!(!amx_small.is_amx_beneficial()); // Too small for AMX
        assert!(amx_large.is_amx_beneficial());  // Large enough for AMX
    }
    
    #[test]
    fn test_intel_amx_optimized_scalar() {
        // Test the optimized scalar implementations
        let vectors = generate_random_vectors(2, 512);
        let amx_dist = IntelAmxDistance::new(Distance::Cosine, 512);
        
        let distance = amx_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance <= 2.0); // Cosine distance is bounded by [0, 2]
    }
}