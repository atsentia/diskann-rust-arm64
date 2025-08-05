//! In-memory index implementation
//!
//! This module provides an in-memory index using the Vamana graph algorithm.

use crate::{Distance, Index, Result, Error};
use crate::graph::{VamanaGraph, SearchParams, SearchScratch, beam_search};
use crate::distance::create_distance_function;
use parking_lot::RwLock;
use std::sync::Arc;

/// In-memory index implementation
pub struct MemoryIndex {
    /// The underlying graph
    graph: VamanaGraph,
    /// Stored vectors
    vectors: Arc<Vec<Vec<f32>>>,
    /// Search scratch spaces (one per thread)
    scratch_pool: RwLock<Vec<SearchScratch>>,
    /// Index configuration
    dimension: usize,
    metric: Distance,
    search_params: SearchParams,
}

impl MemoryIndex {
    /// Build a new in-memory index
    pub fn build(
        vectors: Vec<Vec<f32>>,
        dimension: usize,
        metric: Distance,
        max_degree: usize,
        search_list_size: usize,
        alpha: f32,
    ) -> Result<Self> {
        // Validate vectors
        if vectors.is_empty() {
            return Err(Error::InvalidParameter("No vectors provided".to_string()).into());
        }
        
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != dimension {
                return Err(Error::DimensionMismatch {
                    expected: dimension,
                    actual: vec.len(),
                }.into());
            }
        }
        
        // Build graph
        let num_vertices = vectors.len();
        let mut graph = VamanaGraph::new(
            num_vertices,
            dimension,
            metric,
            max_degree,
            search_list_size,
            alpha,
        );
        
        graph.build(&vectors)?;
        
        // Create search parameters
        let search_params = SearchParams {
            search_list_size,
            k: 10, // Default, will be overridden in search
            alpha,
            use_bitvector: num_vertices > 10000, // Use bit vector for large graphs
        };
        
        Ok(Self {
            graph,
            vectors: Arc::new(vectors),
            scratch_pool: RwLock::new(Vec::new()),
            dimension,
            metric,
            search_params,
        })
    }
    
    /// Get or create a search scratch space
    fn get_scratch(&self) -> SearchScratch {
        let mut pool = self.scratch_pool.write();
        pool.pop().unwrap_or_else(|| SearchScratch::new(self.vectors.len()))
    }
    
    /// Return a scratch space to the pool
    fn return_scratch(&self, scratch: SearchScratch) {
        let mut pool = self.scratch_pool.write();
        if pool.len() < 64 { // Limit pool size
            pool.push(scratch);
        }
    }
    
    /// Add a new vector to the index
    pub fn add(&mut self, vector: Vec<f32>) -> Result<usize> {
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            }.into());
        }
        
        // This would require making vectors mutable and rebuilding parts of the graph
        // For now, return an error
        Err(Error::Index("Dynamic insertion not yet implemented".to_string()).into())
    }
    
    /// Remove a vector from the index
    pub fn remove(&mut self, id: usize) -> Result<()> {
        if id >= self.vectors.len() {
            return Err(Error::InvalidParameter("Invalid vector ID".to_string()).into());
        }
        
        // This would require marking vectors as deleted and handling in search
        // For now, return an error
        Err(Error::Index("Dynamic deletion not yet implemented".to_string()).into())
    }
    
    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let graph_stats = self.graph.stats();
        
        IndexStats {
            num_vectors: self.vectors.len(),
            dimension: self.dimension,
            metric: self.metric,
            memory_usage_bytes: self.estimate_memory_usage(),
            graph_degree_avg: graph_stats.avg_degree,
            graph_degree_max: graph_stats.max_degree,
        }
    }
    
    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self) -> usize {
        let vector_memory = self.vectors.len() * self.dimension * 4; // f32 = 4 bytes
        let graph_memory = self.graph.stats().num_edges * 8; // Assuming 8 bytes per edge
        let overhead = 1024 * 1024; // 1MB overhead estimate
        
        vector_memory + graph_memory + overhead
    }
}

impl Index for MemoryIndex {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            }.into());
        }
        
        // Update search params with requested k
        let mut params = self.search_params.clone();
        params.k = k;
        params.search_list_size = params.search_list_size.max(k);
        
        // Use the lower-level search for better performance
        self.graph.search(query, k, &self.vectors)
    }
    
    fn size(&self) -> usize {
        self.vectors.len()
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn metric(&self) -> Distance {
        self.metric
    }
}

/// Index statistics
#[derive(Debug)]
pub struct IndexStats {
    pub num_vectors: usize,
    pub dimension: usize,
    pub metric: Distance,
    pub memory_usage_bytes: usize,
    pub graph_degree_avg: f32,
    pub graph_degree_max: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_memory_index_build() {
        let vectors = generate_random_vectors(100, 16);
        let index = MemoryIndex::build(
            vectors.clone(),
            16,
            Distance::L2,
            32,
            50,
            1.2,
        ).unwrap();
        
        assert_eq!(index.size(), 100);
        assert_eq!(index.dimension(), 16);
        assert_eq!(index.metric(), Distance::L2);
    }
    
    #[test]
    fn test_memory_index_search() {
        let vectors = generate_random_vectors(100, 16);
        let index = MemoryIndex::build(
            vectors.clone(),
            16,
            Distance::L2,
            32,
            50,
            1.2,
        ).unwrap();
        
        // Search with first vector
        let results = index.search(&vectors[0], 5).unwrap();
        
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 0); // Should find itself
        assert_relative_eq!(results[0].1, 0.0, epsilon = 1e-6);
        
        // Distances should be non-decreasing
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i-1].1);
        }
    }
    
    #[test]
    fn test_memory_index_stats() {
        let vectors = generate_random_vectors(50, 8);
        let index = MemoryIndex::build(
            vectors,
            8,
            Distance::L2,
            16,
            32,
            1.2,
        ).unwrap();
        
        let stats = index.stats();
        assert_eq!(stats.num_vectors, 50);
        assert_eq!(stats.dimension, 8);
        assert!(stats.graph_degree_avg > 0.0);
        assert!(stats.graph_degree_avg <= 16.0);
        assert!(stats.memory_usage_bytes > 0);
    }
    
    #[test]
    fn test_dimension_validation() {
        let vectors = generate_random_vectors(10, 8);
        let index = MemoryIndex::build(
            vectors,
            8,
            Distance::L2,
            16,
            32,
            1.2,
        ).unwrap();
        
        // Wrong dimension query
        let bad_query = vec![1.0; 16];
        assert!(index.search(&bad_query, 5).is_err());
    }
}