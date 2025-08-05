//! x86-64 AVX2 optimized distance calculations
//!
//! This module provides SIMD-accelerated distance functions specifically
//! optimized for x86-64 processors with AVX2 support.

use crate::{Distance, DistanceFunction, Result, Error};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-optimized distance calculator
pub struct Avx2Distance {
    metric: Distance,
    dimension: usize,
}

impl Avx2Distance {
    /// Create a new AVX2 distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Self {
        Self { metric, dimension }
    }
    
    /// AVX2-optimized L2 distance calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time (256-bit / 32-bit = 8)
        while i + 8 <= len {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
            i += 8;
        }
        
        // Reduce the vector to a single sum
        let mut sum_array = [0.0f32; 8];
        _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
        let mut sum = sum_array.iter().sum::<f32>();
        
        // Handle remaining elements
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        
        sum.sqrt()
    }
    
    /// AVX2-optimized squared L2 distance (avoids sqrt)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn l2_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 16 elements at a time for better throughput (2x8)
        while i + 16 <= len {
            // First 8 elements
            let a_vec1 = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec1 = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff1 = _mm256_sub_ps(a_vec1, b_vec1);
            sum_vec = _mm256_fmadd_ps(diff1, diff1, sum_vec);
            
            // Second 8 elements
            let a_vec2 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let b_vec2 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            let diff2 = _mm256_sub_ps(a_vec2, b_vec2);
            sum_vec = _mm256_fmadd_ps(diff2, diff2, sum_vec);
            
            i += 16;
        }
        
        // Process remaining 8 elements
        if i + 8 <= len {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
            i += 8;
        }
        
        // Reduce the vector to a single sum
        let mut sum_array = [0.0f32; 8];
        _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
        let mut sum = sum_array.iter().sum::<f32>();
        
        // Handle remaining elements
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        
        sum
    }
    
    /// AVX2-optimized dot product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut sum_vec = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= len {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            i += 8;
        }
        
        // Reduce the vector to a single sum
        let mut sum_array = [0.0f32; 8];
        _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
        let mut sum = sum_array.iter().sum::<f32>();
        
        // Handle remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }
    
    /// AVX2-optimized cosine distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= len {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            
            dot_sum = _mm256_fmadd_ps(a_vec, b_vec, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(a_vec, a_vec, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(b_vec, b_vec, norm_b_sum);
            
            i += 8;
        }
        
        // Reduce vectors to single sums
        let mut dot_array = [0.0f32; 8];
        let mut norm_a_array = [0.0f32; 8];
        let mut norm_b_array = [0.0f32; 8];
        
        _mm256_storeu_ps(dot_array.as_mut_ptr(), dot_sum);
        _mm256_storeu_ps(norm_a_array.as_mut_ptr(), norm_a_sum);
        _mm256_storeu_ps(norm_b_array.as_mut_ptr(), norm_b_sum);
        
        let mut dot = dot_array.iter().sum::<f32>();
        let mut norm_a = norm_a_array.iter().sum::<f32>();
        let mut norm_b = norm_b_array.iter().sum::<f32>();
        
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

impl DistanceFunction for Avx2Distance {
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
                Distance::L2 => Self::l2_distance_avx2(a, b),
                Distance::Cosine => Self::cosine_distance_avx2(a, b),
                Distance::InnerProduct => -Self::dot_product_avx2(a, b), // Negative for max-heap
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
                    Ok(Self::l2_distance_squared_avx2(a, b))
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
    fn test_avx2_l2_distance() {
        if !crate::has_avx2_support() {
            return; // Skip test if AVX2 not available
        }
        
        let vectors = generate_random_vectors(2, 128);
        let avx2_dist = Avx2Distance::new(Distance::L2, 128);
        
        let distance = avx2_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }
    
    #[test]
    fn test_avx2_cosine_distance() {
        if !crate::has_avx2_support() {
            return; // Skip test if AVX2 not available
        }
        
        let vectors = generate_random_vectors(2, 64);
        let avx2_dist = Avx2Distance::new(Distance::Cosine, 64);
        
        let distance = avx2_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance <= 2.0); // Cosine distance is bounded by [0, 2]
    }
    
    #[test]
    fn test_avx2_vs_scalar_accuracy() {
        if !crate::has_avx2_support() {
            return; // Skip test if AVX2 not available
        }
        
        let vectors = generate_random_vectors(2, 100);
        let avx2_dist = Avx2Distance::new(Distance::L2, 100);
        
        let avx2_result = avx2_dist.distance(&vectors[0], &vectors[1]).unwrap();
        let scalar_result = crate::distance::scalar::scalar_distance(&vectors[0], &vectors[1], Distance::L2).unwrap();
        
        // Allow small numerical difference due to floating point precision
        let diff = (avx2_result - scalar_result).abs();
        assert!(diff < 1e-5, "AVX2 result {} differs too much from scalar result {}", avx2_result, scalar_result);
    }
}