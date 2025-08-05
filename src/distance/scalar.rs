//! Scalar (non-SIMD) distance calculations
//!
//! This module provides fallback implementations for platforms without SIMD support
//! or when SIMD features are disabled.

use crate::{Distance, DistanceFunction, Result, Error};

/// Scalar distance calculator (no SIMD)
pub struct ScalarDistance {
    metric: Distance,
    dimension: usize,
}

impl ScalarDistance {
    /// Create a new scalar distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Self {
        Self { metric, dimension }
    }
    
    /// Scalar L2 distance calculation
    #[inline]
    fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
    
    /// Scalar squared L2 distance (avoids sqrt)
    #[inline]
    fn l2_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }
    
    /// Scalar dot product
    #[inline]
    fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
    
    /// Scalar vector norm
    #[inline]
    fn vector_norm_scalar(vec: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for &v in vec {
            sum += v * v;
        }
        sum.sqrt()
    }
    
    /// Scalar cosine distance
    fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product_scalar(a, b);
        let norm_a = Self::vector_norm_scalar(a);
        let norm_b = Self::vector_norm_scalar(b);
        
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0 // Maximum distance for zero vectors
        } else {
            1.0 - (dot / (norm_a * norm_b))
        }
    }
}

impl DistanceFunction for ScalarDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: a.len(),
            }.into());
        }
        
        match self.metric {
            Distance::L2 => Ok(Self::l2_distance_scalar(a, b)),
            Distance::Cosine => Ok(Self::cosine_distance_scalar(a, b)),
            Distance::InnerProduct => Ok(-Self::dot_product_scalar(a, b)),
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
            Distance::L2 => Ok(Self::l2_distance_squared_scalar(a, b)),
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
        
        let num_points = points.len() / self.dimension;
        if distances.len() < num_points {
            return Err(Error::InvalidParameter(
                "Distances buffer too small".to_string()
            ).into());
        }
        
        for i in 0..num_points {
            let point_offset = i * self.dimension;
            let point = &points[point_offset..point_offset + self.dimension];
            distances[i] = self.distance(query, point)?;
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
    use approx::assert_relative_eq;
    
    #[test]
    fn test_l2_distance() {
        let calc = ScalarDistance::new(Distance::L2, 4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let dist = calc.distance(&a, &b).unwrap();
        assert_relative_eq!(dist, 8.0, epsilon = 1e-5);
        
        let dist_sq = calc.distance_squared(&a, &b).unwrap();
        assert_relative_eq!(dist_sq, 64.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_cosine_distance() {
        let calc = ScalarDistance::new(Distance::Cosine, 3);
        
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
    fn test_inner_product() {
        let calc = ScalarDistance::new(Distance::InnerProduct, 3);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        // Inner product distance is negative dot product
        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let dist = calc.distance(&a, &b).unwrap();
        assert_relative_eq!(dist, -32.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_batch_distance() {
        let calc = ScalarDistance::new(Distance::L2, 4);
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
    fn test_dimension_mismatch() {
        let calc = ScalarDistance::new(Distance::L2, 4);
        let a = vec![1.0, 2.0, 3.0];  // Wrong dimension
        let b = vec![4.0, 5.0, 6.0, 7.0];
        
        assert!(calc.distance(&a, &b).is_err());
    }
}