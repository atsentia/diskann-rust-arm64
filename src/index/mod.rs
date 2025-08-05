//! Index implementations for DiskANN
//!
//! This module provides both in-memory and disk-based index implementations.

pub mod memory;
pub mod disk;
pub mod builder;
pub mod dynamic;

// Re-export key types from sub-modules
pub use disk::{PQFlashIndex, PQFlashConfig, QueryStats, DiskIndexStats};
pub use memory::{MemoryIndex, IndexStats};
pub use dynamic::DynamicIndex;

use crate::{Distance, Result};

/// Trait for all index implementations
pub trait Index: Send + Sync {
    /// Search for k nearest neighbors
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>>;
    
    /// Get the number of indexed vectors
    fn size(&self) -> usize;
    
    /// Get the dimension of vectors
    fn dimension(&self) -> usize;
    
    /// Get the distance metric
    fn metric(&self) -> Distance;
    
    /// Range search - find all neighbors within radius
    fn range_search(&self, query: &[f32], radius: f32, search_list_size: usize) -> Result<Vec<(usize, f32)>> {
        // Default implementation using regular search with filtering
        let initial_k = search_list_size.min(self.size());
        let mut results = self.search(query, initial_k)?;
        results.retain(|(_, dist)| *dist <= radius);
        Ok(results)
    }
    
    /// Save index to disk
    fn save(&self, _path: &str) -> Result<()> {
        Err(anyhow::anyhow!("Save not implemented for this index type"))
    }
    
    /// Get index statistics
    fn stats(&self) -> IndexStats {
        IndexStats {
            num_vectors: self.size(),
            dimension: self.dimension(),
            metric: self.metric(),
            memory_usage_bytes: 0, // Default, override in implementations
            graph_degree_avg: 0.0,
            graph_degree_max: 0,
        }
    }
    
    /// Get memory usage in bytes
    fn memory_usage_bytes(&self) -> usize {
        self.stats().memory_usage_bytes
    }
}

/// Builder for creating indices
pub struct IndexBuilder {
    dimension: Option<usize>,
    metric: Distance,
    max_degree: usize,
    search_list_size: usize,
    alpha: f32,
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self {
            dimension: None,
            metric: Distance::L2,
            max_degree: 64,
            search_list_size: 100,
            alpha: 1.2,
        }
    }
}

impl IndexBuilder {
    /// Create a new index builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the vector dimension
    pub fn dimensions(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }
    
    /// Set the distance metric
    pub fn metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }
    
    /// Set the maximum degree (R parameter)
    pub fn max_degree(mut self, degree: usize) -> Self {
        self.max_degree = degree;
        self
    }
    
    /// Set the search list size (L parameter)
    pub fn search_list_size(mut self, size: usize) -> Self {
        self.search_list_size = size;
        self
    }
    
    /// Set the alpha parameter for graph construction
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
    
    /// Build an in-memory index from vectors
    pub fn build_from_vectors(self, vectors: Vec<Vec<f32>>) -> Result<Box<dyn Index>> {
        let dimension = self.dimension.unwrap_or_else(|| {
            vectors.first().map(|v| v.len()).unwrap_or(0)
        });
        
        Ok(Box::new(memory::MemoryIndex::build(
            vectors,
            dimension,
            self.metric,
            self.max_degree,
            self.search_list_size,
            self.alpha,
        )?))
    }
    
    /// Build a concrete MemoryIndex from vectors (for serialization)
    pub fn build_memory_index(self, vectors: Vec<Vec<f32>>) -> Result<memory::MemoryIndex> {
        let dimension = self.dimension.unwrap_or_else(|| {
            vectors.first().map(|v| v.len()).unwrap_or(0)
        });
        
        memory::MemoryIndex::build(
            vectors,
            dimension,
            self.metric,
            self.max_degree,
            self.search_list_size,
            self.alpha,
        )
    }
    
    /// Build an index from a file
    pub fn build_from_file(self, path: &str) -> Result<Box<dyn Index>> {
        // TODO: Implement file loading
        unimplemented!("File loading not yet implemented")
    }
}