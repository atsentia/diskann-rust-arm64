//! Filtered search implementation with label constraints
//!
//! This module provides search functionality that filters results based on labels,
//! enabling category-specific or subset-based nearest neighbor search.

use crate::{Result, Distance, DistanceFunction};
use crate::distance::create_distance_function;
use crate::graph::VamanaGraph;
use crate::labels::{LabelIndex, LabelFilter};
use std::collections::BinaryHeap;
use hashbrown::HashSet;
use std::cmp::Ordering;

/// Search result with labels
#[derive(Debug, Clone)]
pub struct FilteredNeighbor {
    pub id: usize,
    pub distance: f32,
    pub labels: Vec<u32>,
}

impl Eq for FilteredNeighbor {}

impl PartialEq for FilteredNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for FilteredNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap (closest first)
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for FilteredNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Filtered search parameters
#[derive(Debug, Clone)]
pub struct FilteredSearchParams {
    /// Number of results to return
    pub k: usize,
    /// Search list size for graph traversal
    pub search_list_size: usize,
    /// Label filter to apply
    pub filter: LabelFilter,
    /// Whether to include vector labels in results
    pub include_labels: bool,
}

impl Default for FilteredSearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            search_list_size: 100,
            filter: LabelFilter::Any,
            include_labels: true,
        }
    }
}

/// Filtered search implementation
pub struct FilteredSearcher {
    distance_fn: Box<dyn DistanceFunction>,
}

impl FilteredSearcher {
    /// Create a new filtered searcher
    pub fn new(metric: Distance, dimension: usize) -> Self {
        Self {
            distance_fn: create_distance_function(metric, dimension),
        }
    }
    
    /// Perform filtered search using graph and label index
    pub fn search(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Vec<f32>],
        label_index: &LabelIndex,
        params: &FilteredSearchParams,
    ) -> Result<Vec<FilteredNeighbor>> {
        // Get candidate set from label index
        let label_candidates = label_index.get_candidates(&params.filter)?;
        
        if label_candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        // Use graph traversal with label filtering
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        
        // Start from best entry point among filtered candidates
        let entry_point = self.find_best_entry_point(
            graph,
            query,
            vectors,
            &label_candidates,
        )?;
        
        let entry_dist = self.distance_fn.distance(query, &vectors[entry_point])?;
        
        candidates.push(FilteredNeighbor {
            id: entry_point,
            distance: entry_dist,
            labels: label_index.get_labels(entry_point).unwrap_or_default(),
        });
        
        results.push(FilteredNeighbor {
            id: entry_point,
            distance: entry_dist,
            labels: if params.include_labels {
                label_index.get_labels(entry_point).unwrap_or_default()
            } else {
                Vec::new()
            },
        });
        
        visited.insert(entry_point);
        
        // Expand search
        let mut expansion_count = 0;
        while !candidates.is_empty() && expansion_count < params.search_list_size {
            let current = candidates.pop().unwrap();
            expansion_count += 1;
            
            // Early termination condition
            if results.len() >= params.k && current.distance > results.peek().unwrap().distance {
                break;
            }
            
            // Explore neighbors
            let neighbors = graph.get_neighbors(current.id);
            
            for &neighbor_id in &neighbors {
                if !visited.contains(&neighbor_id) && neighbor_id < vectors.len() {
                    visited.insert(neighbor_id);
                    
                    // Check if neighbor matches filter
                    if label_candidates.contains(&neighbor_id) {
                        let distance = self.distance_fn.distance(query, &vectors[neighbor_id])?;
                        
                        let neighbor = FilteredNeighbor {
                            id: neighbor_id,
                            distance,
                            labels: if params.include_labels {
                                label_index.get_labels(neighbor_id).unwrap_or_default()
                            } else {
                                Vec::new()
                            },
                        };
                        
                        // Add to results
                        if results.len() < params.k {
                            results.push(neighbor.clone());
                        } else if distance < results.peek().unwrap().distance {
                            results.pop();
                            results.push(neighbor.clone());
                        }
                        
                        // Add to candidates for further exploration
                        candidates.push(neighbor);
                    }
                }
            }
        }
        
        // Convert to sorted vector
        let mut final_results: Vec<FilteredNeighbor> = results.into_iter().collect();
        final_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        final_results.truncate(params.k);
        
        Ok(final_results)
    }
    
    /// Perform filtered search on dynamic index
    pub fn search_dynamic(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Option<Vec<f32>>],
        label_index: &LabelIndex,
        params: &FilteredSearchParams,
    ) -> Result<Vec<FilteredNeighbor>> {
        let label_candidates = label_index.get_candidates(&params.filter)?;
        
        if label_candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        // Filter candidates to only valid (non-deleted) vectors
        let valid_candidates: HashSet<usize> = label_candidates
            .into_iter()
            .filter(|&id| id < vectors.len() && vectors[id].is_some())
            .collect();
        
        if valid_candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        
        // Find best entry point among valid candidates
        let entry_point = self.find_best_entry_point_dynamic(
            graph,
            query,
            vectors,
            &valid_candidates,
        )?;
        
        let entry_vec = vectors[entry_point].as_ref().unwrap();
        let entry_dist = self.distance_fn.distance(query, entry_vec)?;
        
        candidates.push(FilteredNeighbor {
            id: entry_point,
            distance: entry_dist,
            labels: label_index.get_labels(entry_point).unwrap_or_default(),
        });
        
        results.push(FilteredNeighbor {
            id: entry_point,
            distance: entry_dist,
            labels: if params.include_labels {
                label_index.get_labels(entry_point).unwrap_or_default()
            } else {
                Vec::new()
            },
        });
        
        visited.insert(entry_point);
        
        // Expand search
        let mut expansion_count = 0;
        while !candidates.is_empty() && expansion_count < params.search_list_size {
            let current = candidates.pop().unwrap();
            expansion_count += 1;
            
            if results.len() >= params.k && current.distance > results.peek().unwrap().distance {
                break;
            }
            
            let neighbors = graph.get_neighbors(current.id);
            
            for &neighbor_id in &neighbors {
                if !visited.contains(&neighbor_id) && valid_candidates.contains(&neighbor_id) {
                    if let Some(neighbor_vec) = vectors[neighbor_id].as_ref() {
                        visited.insert(neighbor_id);
                        
                        let distance = self.distance_fn.distance(query, neighbor_vec)?;
                        
                        let neighbor = FilteredNeighbor {
                            id: neighbor_id,
                            distance,
                            labels: if params.include_labels {
                                label_index.get_labels(neighbor_id).unwrap_or_default()
                            } else {
                                Vec::new()
                            },
                        };
                        
                        if results.len() < params.k {
                            results.push(neighbor.clone());
                        } else if distance < results.peek().unwrap().distance {
                            results.pop();
                            results.push(neighbor.clone());
                        }
                        
                        candidates.push(neighbor);
                    }
                }
            }
        }
        
        let mut final_results: Vec<FilteredNeighbor> = results.into_iter().collect();
        final_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        final_results.truncate(params.k);
        
        Ok(final_results)
    }
    
    /// Find the best entry point among filtered candidates
    fn find_best_entry_point(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Vec<f32>],
        candidates: &HashSet<usize>,
    ) -> Result<usize> {
        let graph_entry = graph.get_entry_point();
        
        // If graph entry point is in candidates, use it
        if candidates.contains(&graph_entry) {
            return Ok(graph_entry);
        }
        
        // Otherwise, find the closest candidate to the query
        let mut best_id = *candidates.iter().next().unwrap();
        let mut best_distance = self.distance_fn.distance(query, &vectors[best_id])?;
        
        for &candidate_id in candidates {
            if candidate_id < vectors.len() {
                let distance = self.distance_fn.distance(query, &vectors[candidate_id])?;
                if distance < best_distance {
                    best_distance = distance;
                    best_id = candidate_id;
                }
            }
        }
        
        Ok(best_id)
    }
    
    /// Find the best entry point for dynamic index
    fn find_best_entry_point_dynamic(
        &self,
        graph: &VamanaGraph,
        query: &[f32],
        vectors: &[Option<Vec<f32>>],
        candidates: &HashSet<usize>,
    ) -> Result<usize> {
        let graph_entry = graph.find_valid_entry_point(vectors);
        
        if candidates.contains(&graph_entry) {
            return Ok(graph_entry);
        }
        
        let mut best_id = *candidates.iter().next().unwrap();
        let mut best_distance = {
            let vec = vectors[best_id].as_ref().unwrap();
            self.distance_fn.distance(query, vec)?
        };
        
        for &candidate_id in candidates {
            if let Some(vec) = vectors[candidate_id].as_ref() {
                let distance = self.distance_fn.distance(query, vec)?;
                if distance < best_distance {
                    best_distance = distance;
                    best_id = candidate_id;
                }
            }
        }
        
        Ok(best_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    use crate::labels::LabelFilter;
    
    #[test]
    fn test_filtered_search_params() {
        let params = FilteredSearchParams::default();
        assert_eq!(params.k, 10);
        assert_eq!(params.search_list_size, 100);
        assert!(params.include_labels);
        assert!(matches!(params.filter, LabelFilter::Any));
    }
    
    #[test]
    fn test_filtered_neighbor_ordering() {
        let n1 = FilteredNeighbor {
            id: 0,
            distance: 1.0,
            labels: vec![1],
        };
        
        let n2 = FilteredNeighbor {
            id: 1,
            distance: 2.0,
            labels: vec![2],
        };
        
        // n1 should be "greater" (closer to top of min-heap)
        assert!(n1 > n2);
    }
}