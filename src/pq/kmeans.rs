//! K-means clustering implementation for Product Quantization
//!
//! This module provides efficient K-means clustering with SIMD optimizations
//! for generating PQ codebooks.

use crate::{Result, DistanceFunction};
use crate::distance::create_distance_function;
use crate::utils::aligned::AlignedVec;
use rand::prelude::*;
use std::f32;

/// K-means clustering parameters
#[derive(Debug, Clone)]
pub struct KMeansParams {
    /// Number of clusters (centroids)
    pub k: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance (change in centroids)
    pub tolerance: f32,
    /// Random seed for reproducible results
    pub seed: Option<u64>,
    /// Use K-means++ initialization
    pub use_plus_plus_init: bool,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            k: 256,
            max_iterations: 100,
            tolerance: 1e-4,
            seed: None,
            use_plus_plus_init: true,
        }
    }
}

/// K-means clustering result
#[derive(Debug)]
pub struct KMeansResult {
    /// Cluster centroids [k x dimension]
    pub centroids: Vec<Vec<f32>>,
    /// Assignment of each point to cluster
    pub assignments: Vec<usize>,
    /// Final inertia (sum of squared distances to centroids)
    pub inertia: f32,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// SIMD-optimized K-means clustering
pub struct KMeans {
    params: KMeansParams,
    distance_fn: Box<dyn DistanceFunction>,
    rng: StdRng,
}

impl KMeans {
    /// Create a new K-means clusterer
    pub fn new(params: KMeansParams, dimension: usize) -> Self {
        let seed = params.seed.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);
        
        Self {
            params,
            distance_fn: create_distance_function(crate::Distance::L2, dimension),
            rng,
        }
    }
    
    /// Fit K-means to the data
    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<KMeansResult> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot cluster empty dataset"));
        }
        
        if data.len() < self.params.k {
            return Err(anyhow::anyhow!(
                "Number of data points ({}) must be >= k ({})",
                data.len(),
                self.params.k
            ));
        }
        
        let dimension = data[0].len();
        let n_points = data.len();
        
        // Validate dimensions
        for (i, point) in data.iter().enumerate() {
            if point.len() != dimension {
                return Err(anyhow::anyhow!(
                    "Dimension mismatch at point {}: expected {}, got {}",
                    i, dimension, point.len()
                ));
            }
        }
        
        // Initialize centroids
        let mut centroids = if self.params.use_plus_plus_init {
            self.init_plus_plus(data)?
        } else {
            self.init_random(data)?
        };
        
        let mut assignments = vec![0; n_points];
        let mut prev_inertia = f32::INFINITY;
        let mut converged = false;
        
        for iteration in 0..self.params.max_iterations {
            // Assignment step: assign each point to nearest centroid
            let mut inertia = 0.0;
            let mut changed = false;
            
            for (point_idx, point) in data.iter().enumerate() {
                let (closest_centroid, distance) = self.find_closest_centroid(point, &centroids)?;
                
                if assignments[point_idx] != closest_centroid {
                    assignments[point_idx] = closest_centroid;
                    changed = true;
                }
                
                inertia += distance * distance;
            }
            
            // Check for convergence
            if !changed || (prev_inertia - inertia).abs() < self.params.tolerance {
                converged = true;
                
                return Ok(KMeansResult {
                    centroids,
                    assignments,
                    inertia,
                    iterations: iteration + 1,
                    converged,
                });
            }
            
            // Update step: recalculate centroids
            centroids = self.update_centroids(data, &assignments)?;
            prev_inertia = inertia;
        }
        
        // Final inertia calculation
        let mut final_inertia = 0.0;
        for (point_idx, point) in data.iter().enumerate() {
            let centroid_idx = assignments[point_idx];
            let distance = self.distance_fn.distance(point, &centroids[centroid_idx])?;
            final_inertia += distance * distance;
        }
        
        Ok(KMeansResult {
            centroids,
            assignments,
            inertia: final_inertia,
            iterations: self.params.max_iterations,
            converged,
        })
    }
    
    /// Initialize centroids using K-means++ algorithm
    fn init_plus_plus(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut centroids = Vec::with_capacity(self.params.k);
        let n_points = data.len();
        
        // Choose first centroid randomly
        let first_idx = self.rng.gen_range(0..n_points);
        centroids.push(data[first_idx].clone());
        
        // Choose remaining centroids with probability proportional to squared distance
        for _ in 1..self.params.k {
            let mut distances = vec![f32::INFINITY; n_points];
            
            // Calculate minimum squared distance to existing centroids
            for (point_idx, point) in data.iter().enumerate() {
                for centroid in &centroids {
                    let dist = self.distance_fn.distance(point, centroid)?;
                    distances[point_idx] = distances[point_idx].min(dist * dist);
                }
            }
            
            // Create probability distribution
            let total_weight: f32 = distances.iter().sum();
            if total_weight == 0.0 {
                // All points are already centroids, choose randomly
                let remaining: Vec<usize> = (0..n_points)
                    .filter(|&i| !centroids.iter().any(|c| c == &data[i]))
                    .collect();
                
                if !remaining.is_empty() {
                    let idx = remaining[self.rng.gen_range(0..remaining.len())];
                    centroids.push(data[idx].clone());
                }
                continue;
            }
            
            // Sample proportional to squared distance
            let mut cumulative_prob = 0.0;
            let target = self.rng.gen::<f32>() * total_weight;
            
            for (point_idx, &weight) in distances.iter().enumerate() {
                cumulative_prob += weight;
                if cumulative_prob >= target {
                    centroids.push(data[point_idx].clone());
                    break;
                }
            }
        }
        
        Ok(centroids)
    }
    
    /// Initialize centroids randomly from data points
    fn init_random(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.shuffle(&mut self.rng);
        
        let centroids = indices
            .into_iter()
            .take(self.params.k)
            .map(|i| data[i].clone())
            .collect();
        
        Ok(centroids)
    }
    
    /// Find the closest centroid to a point
    fn find_closest_centroid(&self, point: &[f32], centroids: &[Vec<f32>]) -> Result<(usize, f32)> {
        let mut min_distance = f32::INFINITY;
        let mut closest_idx = 0;
        
        for (idx, centroid) in centroids.iter().enumerate() {
            let distance = self.distance_fn.distance(point, centroid)?;
            if distance < min_distance {
                min_distance = distance;
                closest_idx = idx;
            }
        }
        
        Ok((closest_idx, min_distance))
    }
    
    /// Update centroids as the mean of assigned points
    fn update_centroids(&mut self, data: &[Vec<f32>], assignments: &[usize]) -> Result<Vec<Vec<f32>>> {
        let dimension = data[0].len();
        let mut centroids = vec![vec![0.0; dimension]; self.params.k];
        let mut counts = vec![0; self.params.k];
        
        // Sum up points for each cluster
        for (point_idx, &cluster_idx) in assignments.iter().enumerate() {
            counts[cluster_idx] += 1;
            for (dim, &value) in data[point_idx].iter().enumerate() {
                centroids[cluster_idx][dim] += value;
            }
        }
        
        // Compute means, handle empty clusters
        for (cluster_idx, count) in counts.iter().enumerate() {
            if *count == 0 {
                // Handle empty cluster by reinitializing randomly
                let random_point_idx = self.rng.gen_range(0..data.len());
                centroids[cluster_idx] = data[random_point_idx].clone();
            } else {
                let count_f32 = *count as f32;
                for dim in 0..dimension {
                    centroids[cluster_idx][dim] /= count_f32;
                }
            }
        }
        
        Ok(centroids)
    }
    
    /// Predict cluster assignments for new data
    pub fn predict(&self, data: &[Vec<f32>], centroids: &[Vec<f32>]) -> Result<Vec<usize>> {
        let mut assignments = Vec::with_capacity(data.len());
        
        for point in data {
            let (closest_idx, _) = self.find_closest_centroid(point, centroids)?;
            assignments.push(closest_idx);
        }
        
        Ok(assignments)
    }
    
    /// Calculate inertia (within-cluster sum of squares)
    pub fn calculate_inertia(&self, data: &[Vec<f32>], centroids: &[Vec<f32>], assignments: &[usize]) -> Result<f32> {
        let mut inertia = 0.0;
        
        for (point_idx, point) in data.iter().enumerate() {
            let centroid_idx = assignments[point_idx];
            let distance = self.distance_fn.distance(point, &centroids[centroid_idx])?;
            inertia += distance * distance;
        }
        
        Ok(inertia)
    }
}

/// Elbow method for determining optimal k
pub fn find_optimal_k(
    data: &[Vec<f32>],
    k_range: std::ops::Range<usize>,
    dimension: usize,
    trials: usize,
) -> Result<Vec<(usize, f32)>> {
    let mut results = Vec::new();
    
    for k in k_range {
        let mut best_inertia = f32::INFINITY;
        
        // Run multiple trials to get more stable results
        for trial in 0..trials {
            let params = KMeansParams {
                k,
                seed: Some(trial as u64),
                ..Default::default()
            };
            
            let mut kmeans = KMeans::new(params, dimension);
            match kmeans.fit(data) {
                Ok(result) => {
                    if result.inertia < best_inertia {
                        best_inertia = result.inertia;
                    }
                }
                Err(_) => continue, // Skip failed runs
            }
        }
        
        if best_inertia != f32::INFINITY {
            results.push((k, best_inertia));
        }
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_kmeans_basic() {
        // Create simple 2D data with clear clusters
        let data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.1],  // Cluster 1
            vec![5.0, 5.0], vec![5.1, 5.1], vec![5.0, 5.1],  // Cluster 2
        ];
        
        let params = KMeansParams {
            k: 2,
            max_iterations: 50,
            seed: Some(42),
            ..Default::default()
        };
        
        let mut kmeans = KMeans::new(params, 2);
        let result = kmeans.fit(&data).unwrap();
        
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), 6);
        assert!(result.converged);
        assert!(result.inertia > 0.0);
    }
    
    #[test]
    fn test_kmeans_plus_plus_init() {
        let data = generate_random_vectors(100, 16);
        
        let params = KMeansParams {
            k: 8,
            use_plus_plus_init: true,
            seed: Some(42),
            ..Default::default()
        };
        
        let mut kmeans = KMeans::new(params, 16);
        let result = kmeans.fit(&data).unwrap();
        
        assert_eq!(result.centroids.len(), 8);
        assert_eq!(result.assignments.len(), 100);
        
        // All assignments should be valid
        for &assignment in &result.assignments {
            assert!(assignment < 8);
        }
    }
    
    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        let params = KMeansParams::default();
        let mut kmeans = KMeans::new(params, 16);
        
        assert!(kmeans.fit(&data).is_err());
    }
    
    #[test]
    fn test_insufficient_data() {
        let data = vec![vec![1.0, 2.0]]; // Only 1 point for k=2
        let params = KMeansParams {
            k: 2,
            ..Default::default()
        };
        let mut kmeans = KMeans::new(params, 2);
        
        assert!(kmeans.fit(&data).is_err());
    }
    
    #[test]
    fn test_predict() {
        let train_data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1],
            vec![5.0, 5.0], vec![5.1, 5.1],
        ];
        
        let test_data = vec![
            vec![0.05, 0.05],  // Should be cluster 0
            vec![5.05, 5.05],  // Should be cluster 1
        ];
        
        let params = KMeansParams {
            k: 2,
            seed: Some(42),
            ..Default::default()
        };
        
        let mut kmeans = KMeans::new(params, 2);
        let result = kmeans.fit(&train_data).unwrap();
        
        let predictions = kmeans.predict(&test_data, &result.centroids).unwrap();
        assert_eq!(predictions.len(), 2);
    }
    
    #[test]
    fn test_find_optimal_k() {
        let data = generate_random_vectors(50, 8);
        let results = find_optimal_k(&data, 2..6, 8, 3).unwrap();
        
        assert!(!results.is_empty());
        
        // Inertia should generally decrease with more clusters
        for i in 1..results.len() {
            assert!(results[i].1 <= results[i-1].1);
        }
    }
}