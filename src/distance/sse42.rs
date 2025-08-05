//! x86-64 SSE4.2 optimized distance calculations
//!
//! This module provides SIMD-accelerated distance functions for older
//! x86-64 processors with SSE4.2 support (2008+).

use crate::{Distance, DistanceFunction, Result, Error};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE4.2-optimized distance calculator  
pub struct Sse42Distance {
    metric: Distance,
    dimension: usize,
}

impl Sse42Distance {
    /// Create a new SSE4.2 distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Self {
        Self { metric, dimension }
    }
    
    /// SSE4.2-optimized L2 distance calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn l2_distance_sse42(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = _mm_setzero_ps();
        let mut i = 0;
        
        // Process 4 elements at a time (128-bit / 32-bit = 4)
        while i + 4 <= len {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
            let diff = _mm_sub_ps(a_vec, b_vec);
            let squared = _mm_mul_ps(diff, diff);
            sum_vec = _mm_add_ps(sum_vec, squared);
            i += 4;
        }
        
        // Horizontal sum of vector elements
        let sum_high = _mm_unpackhi_ps(sum_vec, sum_vec);
        let sum_low = _mm_unpacklo_ps(sum_vec, sum_vec);
        let sum_combined = _mm_add_ps(sum_high, sum_low);
        let sum_final = _mm_hadd_ps(sum_combined, sum_combined);
        let mut sum = _mm_cvtss_f32(sum_final);
        
        // Handle remaining elements
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        
        sum.sqrt()
    }
    
    /// SSE4.2-optimized squared L2 distance (avoids sqrt)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn l2_distance_squared_sse42(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = _mm_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time for better throughput (2x4)
        while i + 8 <= len {
            // First 4 elements
            let a_vec1 = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec1 = _mm_loadu_ps(b.as_ptr().add(i));
            let diff1 = _mm_sub_ps(a_vec1, b_vec1);
            let squared1 = _mm_mul_ps(diff1, diff1);
            sum_vec = _mm_add_ps(sum_vec, squared1);
            
            // Second 4 elements
            let a_vec2 = _mm_loadu_ps(a.as_ptr().add(i + 4));
            let b_vec2 = _mm_loadu_ps(b.as_ptr().add(i + 4));
            let diff2 = _mm_sub_ps(a_vec2, b_vec2);
            let squared2 = _mm_mul_ps(diff2, diff2);
            sum_vec = _mm_add_ps(sum_vec, squared2);
            
            i += 8;
        }
        
        // Process remaining 4 elements
        if i + 4 <= len {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
            let diff = _mm_sub_ps(a_vec, b_vec);
            let squared = _mm_mul_ps(diff, diff);
            sum_vec = _mm_add_ps(sum_vec, squared);
            i += 4;
        }
        
        // Horizontal sum of vector elements
        let sum_high = _mm_unpackhi_ps(sum_vec, sum_vec);
        let sum_low = _mm_unpacklo_ps(sum_vec, sum_vec);
        let sum_combined = _mm_add_ps(sum_high, sum_low);
        let sum_final = _mm_hadd_ps(sum_combined, sum_combined);
        let mut sum = _mm_cvtss_f32(sum_final);
        
        // Handle remaining elements
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        
        sum
    }
    
    /// SSE4.2-optimized dot product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn dot_product_sse42(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = _mm_setzero_ps();
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= len {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
            let product = _mm_mul_ps(a_vec, b_vec);
            sum_vec = _mm_add_ps(sum_vec, product);
            i += 4;
        }
        
        // Horizontal sum of vector elements
        let sum_high = _mm_unpackhi_ps(sum_vec, sum_vec);
        let sum_low = _mm_unpacklo_ps(sum_vec, sum_vec);
        let sum_combined = _mm_add_ps(sum_high, sum_low);
        let sum_final = _mm_hadd_ps(sum_combined, sum_combined);
        let mut sum = _mm_cvtss_f32(sum_final);
        
        // Handle remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }
    
    /// SSE4.2-optimized cosine distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn cosine_distance_sse42(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut dot_sum = _mm_setzero_ps();
        let mut norm_a_sum = _mm_setzero_ps();
        let mut norm_b_sum = _mm_setzero_ps();
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= len {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
            
            let dot_product = _mm_mul_ps(a_vec, b_vec);
            let norm_a_sq = _mm_mul_ps(a_vec, a_vec);
            let norm_b_sq = _mm_mul_ps(b_vec, b_vec);
            
            dot_sum = _mm_add_ps(dot_sum, dot_product);
            norm_a_sum = _mm_add_ps(norm_a_sum, norm_a_sq);
            norm_b_sum = _mm_add_ps(norm_b_sum, norm_b_sq);
            
            i += 4;
        }
        
        // Horizontal sum for all three vectors
        let dot_high = _mm_unpackhi_ps(dot_sum, dot_sum);
        let dot_low = _mm_unpacklo_ps(dot_sum, dot_sum);
        let dot_combined = _mm_add_ps(dot_high, dot_low);
        let dot_final = _mm_hadd_ps(dot_combined, dot_combined);
        let mut dot = _mm_cvtss_f32(dot_final);
        
        let norm_a_high = _mm_unpackhi_ps(norm_a_sum, norm_a_sum);
        let norm_a_low = _mm_unpacklo_ps(norm_a_sum, norm_a_sum);
        let norm_a_combined = _mm_add_ps(norm_a_high, norm_a_low);
        let norm_a_final = _mm_hadd_ps(norm_a_combined, norm_a_combined);
        let mut norm_a = _mm_cvtss_f32(norm_a_final);
        
        let norm_b_high = _mm_unpackhi_ps(norm_b_sum, norm_b_sum);
        let norm_b_low = _mm_unpacklo_ps(norm_b_sum, norm_b_sum);
        let norm_b_combined = _mm_add_ps(norm_b_high, norm_b_low);
        let norm_b_final = _mm_hadd_ps(norm_b_combined, norm_b_combined);
        let mut norm_b = _mm_cvtss_f32(norm_b_final);
        
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

impl DistanceFunction for Sse42Distance {
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(Error::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            }.into());
        }
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let result = match self.metric {
                Distance::L2 => Self::l2_distance_sse42(a, b),
                Distance::Cosine => Self::cosine_distance_sse42(a, b),
                Distance::InnerProduct => -Self::dot_product_sse42(a, b), // Negative for max-heap
            };
            Ok(result)
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for non-x86_64 architectures
            crate::distance::scalar::scalar_distance(a, b, self.metric)
        }
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
                unsafe {
                    Ok(Self::l2_distance_squared_sse42(a, b))
                }
                
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let d = self.distance(a, b)?;
                    Ok(d * d)
                }
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

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_sse42_l2_distance() {
        if !is_x86_feature_detected!("sse4.2") {
            return; // Skip test if SSE4.2 not available
        }
        
        let vectors = generate_random_vectors(2, 128);
        let sse42_dist = Sse42Distance::new(Distance::L2, 128);
        
        let distance = sse42_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }
    
    #[test]
    fn test_sse42_cosine_distance() {  
        if !is_x86_feature_detected!("sse4.2") {
            return; // Skip test if SSE4.2 not available
        }
        
        let vectors = generate_random_vectors(2, 64);
        let sse42_dist = Sse42Distance::new(Distance::Cosine, 64);
        
        let distance = sse42_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance <= 2.0); // Cosine distance is bounded by [0, 2]
    }
    
    #[test]
    fn test_sse42_vs_scalar_accuracy() {
        if !is_x86_feature_detected!("sse4.2") {
            return; // Skip test if SSE4.2 not available
        }
        
        let vectors = generate_random_vectors(2, 100);
        let sse42_dist = Sse42Distance::new(Distance::L2, 100);
        
        let sse42_result = sse42_dist.distance(&vectors[0], &vectors[1]).unwrap();
        let scalar_result = crate::distance::scalar::scalar_distance(&vectors[0], &vectors[1], Distance::L2).unwrap();
        
        // Allow small numerical difference due to floating point precision
        let diff = (sse42_result - scalar_result).abs();
        assert!(diff < 1e-5, "SSE4.2 result {} differs too much from scalar result {}", sse42_result, scalar_result);
    }
}