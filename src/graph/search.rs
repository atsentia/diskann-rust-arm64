//! Optimized search algorithms for DiskANN
//!
//! This module provides efficient search implementations with various optimizations.

use crate::{Result, Error};
use std::collections::{BinaryHeap, HashSet};

/// Search parameters for tuning performance vs accuracy
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Search list size (L parameter)
    pub search_list_size: usize,
    /// Number of results to return
    pub k: usize,
    /// Alpha parameter for early termination
    pub alpha: f32,
    /// Whether to use bit vector for visited tracking
    pub use_bitvector: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            search_list_size: 100,
            k: 10,
            alpha: 1.2,
            use_bitvector: true,
        }
    }
}

/// Visited set abstraction for efficient tracking
pub trait VisitedSet {
    fn insert(&mut self, id: usize) -> bool;
    fn contains(&self, id: usize) -> bool;
    fn clear(&mut self);
}

/// HashSet-based visited tracking
pub struct HashSetVisited {
    set: HashSet<usize>,
}

impl HashSetVisited {
    pub fn new() -> Self {
        Self {
            set: HashSet::new(),
        }
    }
}

impl VisitedSet for HashSetVisited {
    fn insert(&mut self, id: usize) -> bool {
        self.set.insert(id)
    }
    
    fn contains(&self, id: usize) -> bool {
        self.set.contains(&id)
    }
    
    fn clear(&mut self) {
        self.set.clear();
    }
}

/// Bit vector-based visited tracking for better cache locality
pub struct BitVectorVisited {
    bits: Vec<u64>,
    size: usize,
}

impl BitVectorVisited {
    pub fn new(size: usize) -> Self {
        let num_words = (size + 63) / 64;
        Self {
            bits: vec![0; num_words],
            size,
        }
    }
    
    #[inline]
    fn bit_index(id: usize) -> (usize, u64) {
        let word_idx = id / 64;
        let bit_idx = id % 64;
        (word_idx, 1u64 << bit_idx)
    }
}

impl VisitedSet for BitVectorVisited {
    fn insert(&mut self, id: usize) -> bool {
        if id >= self.size {
            return false;
        }
        
        let (word_idx, bit_mask) = Self::bit_index(id);
        let was_set = self.bits[word_idx] & bit_mask != 0;
        self.bits[word_idx] |= bit_mask;
        !was_set
    }
    
    fn contains(&self, id: usize) -> bool {
        if id >= self.size {
            return false;
        }
        
        let (word_idx, bit_mask) = Self::bit_index(id);
        self.bits[word_idx] & bit_mask != 0
    }
    
    fn clear(&mut self) {
        self.bits.fill(0);
    }
}

/// Search scratch space for reuse across searches
pub struct SearchScratch {
    pub candidates: BinaryHeap<super::vamana::Neighbor>,
    pub w: BinaryHeap<super::vamana::Neighbor>,
    pub visited_hash: HashSetVisited,
    pub visited_bits: BitVectorVisited,
}

impl SearchScratch {
    pub fn new(num_vertices: usize) -> Self {
        Self {
            candidates: BinaryHeap::with_capacity(200),
            w: BinaryHeap::with_capacity(200),
            visited_hash: HashSetVisited::new(),
            visited_bits: BitVectorVisited::new(num_vertices),
        }
    }
    
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.w.clear();
        self.visited_hash.clear();
        self.visited_bits.clear();
    }
}

/// Optimized beam search implementation
pub fn beam_search<F>(
    entry_point: usize,
    query: &[f32],
    params: &SearchParams,
    scratch: &mut SearchScratch,
    graph: &[Vec<usize>],
    vectors: &[Vec<f32>],
    mut distance_fn: F,
) -> Result<Vec<(usize, f32)>>
where
    F: FnMut(&[f32], &[f32]) -> Result<f32>,
{
    scratch.clear();
    
    // Choose visited set type
    let visited: &mut dyn VisitedSet = if params.use_bitvector {
        &mut scratch.visited_bits
    } else {
        &mut scratch.visited_hash
    };
    
    // Initialize with entry point
    let entry_dist = distance_fn(query, &vectors[entry_point])?;
    scratch.candidates.push(super::vamana::Neighbor {
        id: entry_point,
        distance: entry_dist,
    });
    scratch.w.push(super::vamana::Neighbor {
        id: entry_point,
        distance: entry_dist,
    });
    visited.insert(entry_point);
    
    // Main search loop
    while let Some(current) = scratch.candidates.pop() {
        // Early termination check
        if let Some(best) = scratch.w.peek() {
            if current.distance > best.distance * params.alpha {
                break;
            }
        }
        
        // Explore neighbors
        for &neighbor_id in &graph[current.id] {
            if visited.insert(neighbor_id) {
                let dist = distance_fn(query, &vectors[neighbor_id])?;
                
                // Check if this neighbor should be added
                let should_add = scratch.w.len() < params.search_list_size || 
                    dist < scratch.w.peek().unwrap().distance;
                
                if should_add {
                    let neighbor = super::vamana::Neighbor {
                        id: neighbor_id,
                        distance: dist,
                    };
                    
                    scratch.candidates.push(neighbor);
                    scratch.w.push(neighbor);
                    
                    // Maintain search list size
                    if scratch.w.len() > params.search_list_size {
                        scratch.w.pop();
                    }
                }
            }
        }
    }
    
    // Extract top-k results
    let mut results = Vec::with_capacity(params.k);
    while results.len() < params.k && !scratch.w.is_empty() {
        let neighbor = scratch.w.pop().unwrap();
        results.push((neighbor.id, neighbor.distance));
    }
    
    // Results are in reverse order (furthest first), so reverse
    results.reverse();
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitvector_visited() {
        let mut visited = BitVectorVisited::new(1000);
        
        assert!(!visited.contains(42));
        assert!(visited.insert(42));
        assert!(visited.contains(42));
        assert!(!visited.insert(42)); // Already inserted
        
        visited.clear();
        assert!(!visited.contains(42));
    }
    
    #[test]
    fn test_search_params() {
        let params = SearchParams::default();
        assert_eq!(params.search_list_size, 100);
        assert_eq!(params.k, 10);
        assert_eq!(params.alpha, 1.2);
        assert!(params.use_bitvector);
    }
}