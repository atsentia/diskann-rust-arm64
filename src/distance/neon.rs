//! ARM64 NEON optimized distance calculations
//!
//! This module provides SIMD-accelerated distance functions specifically
//! optimized for ARM64 processors with NEON support.

use crate::{Distance, DistanceFunction, Result, Error};
use std::arch::aarch64::*;

/// NEON-optimized distance calculator
pub struct NeonDistance {
    metric: Distance,
    dimension: usize,
}

impl NeonDistance {
    /// Create a new NEON distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Self {
        Self { metric, dimension }
    }
    
    /// NEON-optimized L2 distance calculation
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn l2_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = vdupq_n_f32(0.0);
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= len {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff, diff);
            i += 4;
        }
        
        // Reduce the vector to a single sum
        let sum = vaddvq_f32(sum_vec);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0f32;
        while i < len {
            let diff = a[i] - b[i];
            remainder_sum += diff * diff;
            i += 1;
        }
        
        (sum + remainder_sum).sqrt()
    }
    
    /// NEON-optimized squared L2 distance (avoids sqrt)
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn l2_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = vdupq_n_f32(0.0);
        let mut i = 0;
        
        // Process 8 elements at a time for better throughput
        while i + 8 <= len {
            // First 4 elements
            let a_vec1 = vld1q_f32(a.as_ptr().add(i));
            let b_vec1 = vld1q_f32(b.as_ptr().add(i));
            let diff1 = vsubq_f32(a_vec1, b_vec1);
            sum_vec = vfmaq_f32(sum_vec, diff1, diff1);
            
            // Next 4 elements
            let a_vec2 = vld1q_f32(a.as_ptr().add(i + 4));
            let b_vec2 = vld1q_f32(b.as_ptr().add(i + 4));
            let diff2 = vsubq_f32(a_vec2, b_vec2);
            sum_vec = vfmaq_f32(sum_vec, diff2, diff2);
            
            i += 8;
        }
        
        // Process remaining 4 elements if any
        while i + 4 <= len {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff, diff);
            i += 4;
        }
        
        // Reduce the vector to a single sum
        let sum = vaddvq_f32(sum_vec);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0f32;
        while i < len {
            let diff = a[i] - b[i];
            remainder_sum += diff * diff;
            i += 1;
        }
        
        sum + remainder_sum
    }
    
    /// NEON-optimized dot product
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = vdupq_n_f32(0.0);
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= len {
            let a_vec1 = vld1q_f32(a.as_ptr().add(i));
            let b_vec1 = vld1q_f32(b.as_ptr().add(i));
            sum_vec = vfmaq_f32(sum_vec, a_vec1, b_vec1);
            
            let a_vec2 = vld1q_f32(a.as_ptr().add(i + 4));
            let b_vec2 = vld1q_f32(b.as_ptr().add(i + 4));
            sum_vec = vfmaq_f32(sum_vec, a_vec2, b_vec2);
            
            i += 8;
        }
        
        // Process remaining 4 elements
        while i + 4 <= len {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
            i += 4;
        }
        
        // Reduce the vector
        let sum = vaddvq_f32(sum_vec);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0f32;
        while i < len {
            remainder_sum += a[i] * b[i];
            i += 1;
        }
        
        sum + remainder_sum
    }
    
    /// NEON-optimized vector norm
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn vector_norm_neon(vec: &[f32]) -> f32 {
        let len = vec.len();
        let mut sum_vec = vdupq_n_f32(0.0);
        let mut i = 0;
        
        while i + 4 <= len {
            let v = vld1q_f32(vec.as_ptr().add(i));
            sum_vec = vfmaq_f32(sum_vec, v, v);
            i += 4;
        }
        
        let sum = vaddvq_f32(sum_vec);
        
        let mut remainder_sum = 0.0f32;
        while i < len {
            remainder_sum += vec[i] * vec[i];
            i += 1;
        }
        
        (sum + remainder_sum).sqrt()
    }
    
    /// NEON-optimized cosine distance
    #[target_feature(enable = "neon")]
    unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product_neon(a, b);
        let norm_a = Self::vector_norm_neon(a);
        let norm_b = Self::vector_norm_neon(b);
        
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0 // Maximum distance for zero vectors
        } else {
            1.0 - (dot / (norm_a * norm_b))
        }
    }
    
    /// NEON-optimized batch L2 distance
    #[target_feature(enable = "neon")]
    unsafe fn batch_l2_distance_neon(
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
        
        for i in 0..num_points {
            let point_offset = i * dim;
            let point = &points[point_offset..point_offset + dim];
            distances[i] = Self::l2_distance_neon(query, point);
        }
        
        Ok(())
    }
}

impl DistanceFunction for NeonDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: a.len(),
            }.into());
        }
        
        // Safety: We've verified the dimensions match
        unsafe {
            match self.metric {
                Distance::L2 => Ok(Self::l2_distance_neon(a, b)),
                Distance::Cosine => Ok(Self::cosine_distance_neon(a, b)),
                Distance::InnerProduct => Ok(-Self::dot_product_neon(a, b)),
            }
        }
    }
    
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: a.len(),
            }.into());
        }
        
        unsafe {
            match self.metric {
                Distance::L2 => Ok(Self::l2_distance_squared_neon(a, b)),
                _ => {
                    let d = self.distance(a, b)?;
                    Ok(d * d)
                }
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
        
        unsafe {
            match self.metric {
                Distance::L2 => Self::batch_l2_distance_neon(query, points, distances, self.dimension),
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
    fn test_l2_distance() {
        if !crate::has_neon_support() {
            eprintln!("Skipping NEON tests - no NEON support detected");
            return;
        }
        
        let calc = NeonDistance::new(Distance::L2, 4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let dist = calc.distance(&a, &b).unwrap();
        assert_relative_eq!(dist, 8.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_cosine_distance() {
        if !crate::has_neon_support() {
            eprintln!("Skipping NEON tests - no NEON support detected");
            return;
        }
        
        let calc = NeonDistance::new(Distance::Cosine, 3);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let dist = calc.distance(&a, &b).unwrap();
        assert_relative_eq!(dist, 1.0, epsilon = 1e-5); // Orthogonal vectors
    }
    
    #[test]
    fn test_batch_distance() {
        if !crate::has_neon_support() {
            eprintln!("Skipping NEON tests - no NEON support detected");
            return;
        }
        
        let calc = NeonDistance::new(Distance::L2, 4);
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let points = vec![
            5.0, 6.0, 7.0, 8.0,  // First point
            1.0, 2.0, 3.0, 4.0,  // Second point (same as query)
        ];
        let mut distances = vec![0.0; 2];
        
        calc.batch_distance(&query, &points, &mut distances).unwrap();
        
        assert_relative_eq!(distances[0], 8.0, epsilon = 1e-5);
        assert_relative_eq!(distances[1], 0.0, epsilon = 1e-5);
    }
}