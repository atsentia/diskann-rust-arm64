//! Test Data Generation Module
//! 
//! This module provides utilities for generating standardized test datasets
//! for comprehensive parity testing.

use super::*;

/// Generate various types of test datasets for comprehensive testing
pub struct TestDataGenerator {
    seed: u64,
}

impl TestDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
    
    /// Generate uniform random vectors
    pub fn generate_uniform(&self, count: usize, dimension: usize) -> Vec<Vec<f32>> {
        utils::generate_random_vectors(count, dimension, self.seed)
    }
    
    /// Generate clustered data with specified number of clusters
    pub fn generate_clustered(&self, num_clusters: usize, points_per_cluster: usize, dimension: usize) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut data = Vec::new();
        
        for _ in 0..num_clusters {
            // Generate cluster center
            let center: Vec<f32> = (0..dimension).map(|_| rng.gen::<f32>() * 100.0).collect();
            
            // Generate points around center
            for _ in 0..points_per_cluster {
                let point: Vec<f32> = center.iter()
                    .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0) // Small variance around center
                    .collect();
                data.push(point);
            }
        }
        
        data
    }
    
    /// Generate sparse vectors (mostly zeros)
    pub fn generate_sparse(&self, count: usize, dimension: usize, sparsity: f32) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        
        (0..count).map(|_| {
            let mut vector = vec![0.0; dimension];
            let non_zero_count = (dimension as f32 * sparsity) as usize;
            
            for _ in 0..non_zero_count {
                let idx = rng.gen_range(0..dimension);
                vector[idx] = rng.gen::<f32>();
            }
            
            vector
        }).collect()
    }
    
    /// Generate vectors with known ground truth relationships
    pub fn generate_with_ground_truth(&self, count: usize, dimension: usize) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let data = self.generate_uniform(count, dimension);
        
        // For each vector, compute true nearest neighbors (brute force)
        let ground_truth: Vec<Vec<usize>> = data.iter().enumerate().map(|(i, query)| {
            let mut distances: Vec<(usize, f32)> = data.iter().enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, vec)| {
                    let dist = l2_distance(query, vec);
                    (j, dist)
                })
                .collect();
            
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.into_iter().take(10).map(|(idx, _)| idx).collect()
        }).collect();
        
        (data, ground_truth)
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}