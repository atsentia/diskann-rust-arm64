//! Pure Rust SIMD implementations for distance calculations
//!
//! This module provides portable SIMD implementations that work across platforms.

use crate::{Distance, DistanceFunction, Result, Error};
use wide::{f32x4, f32x8};

/// SIMD-optimized distance calculator using the `wide` crate
pub struct SimdDistance {
    metric: Distance,
    dimension: usize,
}

impl SimdDistance {
    /// Create a new SIMD distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Self {
        Self { metric, dimension }
    }
    
    /// SIMD L2 distance using f32x8 (256-bit vectors)
    #[inline]
    fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum8 = f32x8::splat(0.0);
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= len {
            let a8 = f32x8::from(&a[i..i + 8]);
            let b8 = f32x8::from(&b[i..i + 8]);
            let diff = a8 - b8;
            sum8 = diff.mul_add(diff, sum8);
            i += 8;
        }
        
        // Sum the 8 lanes
        let sum8_array: [f32; 8] = sum8.into();
        let mut sum = sum8_array.iter().sum::<f32>();
        
        // Process remaining elements with f32x4
        if i + 4 <= len {
            let a4 = f32x4::from(&a[i..i + 4]);
            let b4 = f32x4::from(&b[i..i + 4]);
            let diff = a4 - b4;
            let sum4 = diff * diff;
            let sum4_array: [f32; 4] = sum4.into();
            sum += sum4_array.iter().sum::<f32>();
            i += 4;
        }
        
        // Handle remaining scalar elements
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        
        sum.sqrt()
    }
    
    /// SIMD squared L2 distance (no sqrt)
    #[inline]
    fn l2_distance_squared_simd(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum8 = f32x8::splat(0.0);
        let mut i = 0;
        
        // Process 16 elements at a time (2 x f32x8)
        while i + 16 <= len {
            // First 8
            let a8_1 = f32x8::from(&a[i..i + 8]);
            let b8_1 = f32x8::from(&b[i..i + 8]);
            let diff1 = a8_1 - b8_1;
            sum8 = diff1.mul_add(diff1, sum8);
            
            // Next 8
            let a8_2 = f32x8::from(&a[i + 8..i + 16]);
            let b8_2 = f32x8::from(&b[i + 8..i + 16]);
            let diff2 = a8_2 - b8_2;
            sum8 = diff2.mul_add(diff2, sum8);
            
            i += 16;
        }
        
        // Process remaining 8 elements
        while i + 8 <= len {
            let a8 = f32x8::from(&a[i..i + 8]);
            let b8 = f32x8::from(&b[i..i + 8]);
            let diff = a8 - b8;
            sum8 = diff.mul_add(diff, sum8);
            i += 8;
        }
        
        // Sum the lanes
        let sum8_array: [f32; 8] = sum8.into();
        let mut sum = sum8_array.iter().sum::<f32>();
        
        // Handle remaining elements
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        
        sum
    }
    
    /// SIMD dot product
    #[inline]
    fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum8 = f32x8::splat(0.0);
        let mut i = 0;
        
        // Process 16 elements at a time for better throughput
        while i + 16 <= len {
            let a8_1 = f32x8::from(&a[i..i + 8]);
            let b8_1 = f32x8::from(&b[i..i + 8]);
            sum8 = a8_1.mul_add(b8_1, sum8);
            
            let a8_2 = f32x8::from(&a[i + 8..i + 16]);
            let b8_2 = f32x8::from(&b[i + 8..i + 16]);
            sum8 = a8_2.mul_add(b8_2, sum8);
            
            i += 16;
        }
        
        // Process remaining 8
        while i + 8 <= len {
            let a8 = f32x8::from(&a[i..i + 8]);
            let b8 = f32x8::from(&b[i..i + 8]);
            sum8 = a8.mul_add(b8, sum8);
            i += 8;
        }
        
        // Sum the lanes
        let sum8_array: [f32; 8] = sum8.into();
        let mut sum = sum8_array.iter().sum::<f32>();
        
        // Handle remaining scalar elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }
    
    /// SIMD vector norm
    #[inline]
    fn vector_norm_simd(vec: &[f32]) -> f32 {
        let len = vec.len();
        let mut sum8 = f32x8::splat(0.0);
        let mut i = 0;
        
        while i + 8 <= len {
            let v8 = f32x8::from(&vec[i..i + 8]);
            sum8 = v8.mul_add(v8, sum8);
            i += 8;
        }
        
        let sum8_array: [f32; 8] = sum8.into();
        let mut sum = sum8_array.iter().sum::<f32>();
        
        while i < len {
            sum += vec[i] * vec[i];
            i += 1;
        }
        
        sum.sqrt()
    }
    
    /// SIMD cosine distance
    fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product_simd(a, b);
        let norm_a = Self::vector_norm_simd(a);
        let norm_b = Self::vector_norm_simd(b);
        
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0 // Maximum distance for zero vectors
        } else {
            1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }
    
    /// SIMD batch L2 distance
    fn batch_l2_distance_simd(
        query: &[f32],
        points: &[f32],
        distances: &mut [f32],
        dim: usize,
    ) -> Result<()> {
        let num_points = points.len() / dim;
        if distances.len() < num_points {
            return Err(Error::InvalidParameter(
                "Distances buffer too small".to_string()
            ).into());
        }
        
        // Process multiple points in parallel when possible
        for i in 0..num_points {
            let point_offset = i * dim;
            let point = &points[point_offset..point_offset + dim];
            distances[i] = Self::l2_distance_simd(query, point);
        }
        
        Ok(())
    }
}

impl DistanceFunction for SimdDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: a.len(),
            }.into());
        }
        
        match self.metric {
            Distance::L2 => Ok(Self::l2_distance_simd(a, b)),
            Distance::Cosine => Ok(Self::cosine_distance_simd(a, b)),
            Distance::InnerProduct => Ok(-Self::dot_product_simd(a, b)),
        }
    }
    
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: a.len(),
            }.into());
        }
        
        match self.metric {
            Distance::L2 => Ok(Self::l2_distance_squared_simd(a, b)),
            _ => {
                let d = self.distance(a, b)?;
                Ok(d * d)
            }
        }
    }
    
    fn batch_distance(&self, query: &[f32], points: &[f32], distances: &mut [f32]) -> Result<()> {
        if query.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            }.into());
        }
        
        match self.metric {
            Distance::L2 => Self::batch_l2_distance_simd(query, points, distances, self.dimension),
            _ => {
                // Fallback to individual calculations for other metrics
                let num_points = points.len() / self.dimension;
                for i in 0..num_points {
                    let point_offset = i * self.dimension;
                    let point = &points[point_offset..point_offset + self.dimension];
                    distances[i] = self.distance(query, point)?;
                }
                Ok(())
            }
        }
    }
    
    fn metric(&self) -> Distance {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simd_l2_distance() {
        let calc = SimdDistance::new(Distance::L2, 4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let dist = calc.distance(&a, &b).unwrap();
        assert_relative_eq!(dist, 8.0, epsilon = 1e-5);
        
        let dist_sq = calc.distance_squared(&a, &b).unwrap();
        assert_relative_eq!(dist_sq, 64.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_simd_cosine_distance() {
        let calc = SimdDistance::new(Distance::Cosine, 3);
        
        // Test orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = calc.distance(&a, &b).unwrap();
        assert_relative_eq!(dist, 1.0, epsilon = 1e-5);
        
        // Test identical vectors
        let c = vec![1.0, 2.0, 3.0];
        let d = vec![1.0, 2.0, 3.0];
        let dist2 = calc.distance(&c, &d).unwrap();
        assert_relative_eq!(dist2, 0.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_simd_inner_product() {
        let calc = SimdDistance::new(Distance::InnerProduct, 3);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        // Inner product distance is negative dot product
        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let dist = calc.distance(&a, &b).unwrap();
        assert_relative_eq!(dist, -32.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_simd_batch_distance() {
        let calc = SimdDistance::new(Distance::L2, 4);
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let points = vec![
            5.0, 6.0, 7.0, 8.0,  // First point
            1.0, 2.0, 3.0, 4.0,  // Second point (same as query)
            0.0, 0.0, 0.0, 0.0,  // Third point (origin)
        ];
        let mut distances = vec![0.0; 3];
        
        calc.batch_distance(&query, &points, &mut distances).unwrap();
        
        assert_relative_eq!(distances[0], 8.0, epsilon = 1e-5);
        assert_relative_eq!(distances[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(distances[2], (1.0 + 4.0 + 9.0 + 16.0f32).sqrt(), epsilon = 1e-5);
    }
    
    #[test]
    fn test_large_vectors() {
        // Test with various sizes to ensure SIMD paths work correctly
        for size in [7, 8, 15, 16, 31, 32, 127, 128, 256, 512] {
            let calc = SimdDistance::new(Distance::L2, size);
            let a = vec![1.0; size];
            let b = vec![2.0; size];
            
            let dist = calc.distance(&a, &b).unwrap();
            let expected = (size as f32).sqrt(); // sqrt(n * 1^2)
            assert_relative_eq!(dist, expected, epsilon = 1e-5);
        }
    }
}