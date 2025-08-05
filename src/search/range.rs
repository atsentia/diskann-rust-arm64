//! Range search implementation for finding all neighbors within a distance threshold
//!
//! This module implements range search functionality for finding all vectors
//! within a specified distance from a query point.

use crate::{Result, Distance, DistanceFunction};
use crate::distance::create_distance_function;
use crate::graph::VamanaGraph;
use std::collections::BinaryHeap;
use hashbrown::HashSet;
use std::cmp::Ordering;

/// Neighbor with distance information for range search
#[derive(Debug, Clone, Copy)]
pub struct RangeNeighbor {
    pub id: usize,
    pub distance: f32,
}

impl Eq for RangeNeighbor {}

impl PartialEq for RangeNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for RangeNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // For max-heap (farthest first)
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for RangeNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Range search parameters
#[derive(Debug, Clone)]
pub struct RangeSearchParams {
    /// Maximum distance threshold
    pub radius: f32,
    /// Maximum number of results to return (0 = unlimited)
    pub max_results: usize,
    /// Search list size for traversal
    pub search_list_size: usize,
}

impl Default for RangeSearchParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            max_results: 0, // Unlimited
            search_list_size: 100,
        }
    }
}

/// Range search implementation
pub struct RangeSearcher {
    distance_fn: Box<dyn DistanceFunction>,
}

impl RangeSearcher {
    /// Create a new range searcher
    pub fn new(metric: Distance, dimension: usize) -> Self {
        Self {
            distance_fn: create_distance_function(metric, dimension),
        }
    }
    
    /// Perform range search on a graph
    pub fn search(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Vec<f32>],
        params: &RangeSearchParams,
    ) -> Result<Vec<RangeNeighbor>> {
        // Use graph traversal to find candidates
        let candidates = self.graph_range_search(graph, query, vectors, params)?;
        
        // Filter by distance and sort
        let mut results: Vec<RangeNeighbor> = candidates
            .into_iter()
            .filter(|neighbor| neighbor.distance <= params.radius)
            .collect();
        
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        // Limit results if specified
        if params.max_results > 0 && results.len() > params.max_results {
            results.truncate(params.max_results);
        }
        
        Ok(results)
    }
    
    /// Range search using dynamic index with Option<Vec<f32>>
    pub fn search_dynamic(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Option<Vec<f32>>],
        params: &RangeSearchParams,
    ) -> Result<Vec<RangeNeighbor>> {
        let candidates = self.graph_range_search_dynamic(graph, query, vectors, params)?;
        
        let mut results: Vec<RangeNeighbor> = candidates
            .into_iter()
            .filter(|neighbor| neighbor.distance <= params.radius)
            .collect();
        
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        if params.max_results > 0 && results.len() > params.max_results {
            results.truncate(params.max_results);
        }
        
        Ok(results)
    }
    
    /// Brute force range search (for small datasets or ground truth)
    pub fn brute_force_search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        params: &RangeSearchParams,
    ) -> Result<Vec<RangeNeighbor>> {
        let mut results = Vec::new();
        
        for (id, vector) in vectors.iter().enumerate() {
            let distance = self.distance_fn.distance(query, vector)?;
            if distance <= params.radius {
                results.push(RangeNeighbor { id, distance });
            }
        }
        
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        if params.max_results > 0 && results.len() > params.max_results {
            results.truncate(params.max_results);
        }
        
        Ok(results)
    }
    
    /// Graph-based range search implementation
    fn graph_range_search(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Vec<f32>],
        params: &RangeSearchParams,
    ) -> Result<Vec<RangeNeighbor>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = Vec::new();
        
        // Start from entry point
        let entry_point = graph.get_entry_point();
        let entry_dist = self.distance_fn.distance(query, &vectors[entry_point])?;
        
        candidates.push(RangeNeighbor {
            id: entry_point,
            distance: entry_dist,
        });
        visited.insert(entry_point);
        
        if entry_dist <= params.radius {
            results.push(RangeNeighbor {
                id: entry_point,
                distance: entry_dist,
            });
        }
        
        // Expand search
        let mut expansion_count = 0;
        while !candidates.is_empty() && expansion_count < params.search_list_size {
            let current = candidates.pop().unwrap();
            expansion_count += 1;
            
            // Get neighbors from graph
            let neighbors = graph.get_neighbors(current.id);
            
            for neighbor_id in neighbors {
                if !visited.contains(&neighbor_id) && neighbor_id < vectors.len() {
                    visited.insert(neighbor_id);
                    
                    let distance = self.distance_fn.distance(query, &vectors[neighbor_id])?;
                    
                    // Add to results if within radius
                    if distance <= params.radius {
                        results.push(RangeNeighbor {
                            id: neighbor_id,
                            distance,
                        });
                    }
                    
                    // Add to candidates for further expansion (even if outside radius)
                    if distance <= params.radius * 2.0 { // Expand slightly beyond radius
                        candidates.push(RangeNeighbor {
                            id: neighbor_id,
                            distance,
                        });
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Graph-based range search for dynamic index
    fn graph_range_search_dynamic(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Option<Vec<f32>>],
        params: &RangeSearchParams,
    ) -> Result<Vec<RangeNeighbor>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = Vec::new();
        
        // Find valid entry point
        let entry_point = graph.find_valid_entry_point(vectors);
        let entry_vec = match vectors[entry_point].as_ref() {
            Some(vec) => vec,
            None => return Ok(Vec::new()),
        };
        
        let entry_dist = self.distance_fn.distance(query, entry_vec)?;
        
        candidates.push(RangeNeighbor {
            id: entry_point,
            distance: entry_dist,
        });
        visited.insert(entry_point);
        
        if entry_dist <= params.radius {
            results.push(RangeNeighbor {
                id: entry_point,
                distance: entry_dist,
            });
        }
        
        // Expand search
        let mut expansion_count = 0;
        while !candidates.is_empty() && expansion_count < params.search_list_size {
            let current = candidates.pop().unwrap();
            expansion_count += 1;
            
            let neighbors = graph.get_neighbors(current.id);
            
            for neighbor_id in neighbors {
                if !visited.contains(&neighbor_id) && neighbor_id < vectors.len() {
                    if let Some(neighbor_vec) = vectors[neighbor_id].as_ref() {
                        visited.insert(neighbor_id);
                        
                        let distance = self.distance_fn.distance(query, neighbor_vec)?;
                        
                        if distance <= params.radius {
                            results.push(RangeNeighbor {
                                id: neighbor_id,
                                distance,
                            });
                        }
                        
                        if distance <= params.radius * 2.0 {
                            candidates.push(RangeNeighbor {
                                id: neighbor_id,
                                distance,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_range_search_brute_force() {
        let vectors = generate_random_vectors(100, 16);
        let searcher = RangeSearcher::new(Distance::L2, 16);
        
        let params = RangeSearchParams {
            radius: 2.0,
            max_results: 10,
            search_list_size: 50,
        };
        
        let results = searcher.brute_force_search(&vectors[0], &vectors, &params).unwrap();
        
        // Should find at least the query vector itself (distance 0)
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0);
        assert_eq!(results[0].distance, 0.0);
        
        // All results should be within radius
        for result in &results {
            assert!(result.distance <= params.radius);
        }
        
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i-1].distance);
        }
    }
    
    #[test]
    fn test_range_search_params() {
        let params = RangeSearchParams::default();
        assert_eq!(params.radius, 1.0);
        assert_eq!(params.max_results, 0); // Unlimited
        assert_eq!(params.search_list_size, 100);
    }
}