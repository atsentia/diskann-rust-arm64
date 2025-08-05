//! In-memory index implementation
//!
//! This module provides an in-memory index using the Vamana graph algorithm.

use crate::{Distance, Index, Result};
use crate::graph::VamanaGraph;
use crate::graph::vamana::SerializableVamanaGraph;
use std::sync::Arc;
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};

/// Serializable wrapper for MemoryIndex
#[derive(Serialize, Deserialize)]
struct SerializableMemoryIndex {
    graph: SerializableVamanaGraph,
    vectors: Vec<Vec<f32>>,
    dimension: usize,
    metric: Distance,
}

/// In-memory index implementation
pub struct MemoryIndex {
    /// The underlying graph
    graph: VamanaGraph,
    /// Stored vectors
    vectors: Arc<Vec<Vec<f32>>>,
    /// Index configuration
    dimension: usize,
    metric: Distance,
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
            return Err(anyhow::anyhow!("No vectors provided"));
        }
        
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != dimension {
                return Err(anyhow::anyhow!("Dimension mismatch: expected {}, got {}", dimension, vec.len()));
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
        
        Ok(Self {
            graph,
            vectors: Arc::new(vectors),
            dimension,
            metric,
        })
    }
    
    
    /// Add a new vector to the index
    pub fn add(&mut self, vector: Vec<f32>) -> Result<usize> {
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!("Dimension mismatch: expected {}, got {}", self.dimension, vector.len()));
        }
        
        // This would require making vectors mutable and rebuilding parts of the graph
        // For now, return an error
Err(anyhow::anyhow!("Dynamic insertion not yet implemented"))
    }
    
    /// Remove a vector from the index
    pub fn remove(&mut self, id: usize) -> Result<()> {
        if id >= self.vectors.len() {
            return Err(anyhow::anyhow!("Invalid vector ID"));
        }
        
        // This would require marking vectors as deleted and handling in search
        // For now, return an error
Err(anyhow::anyhow!("Dynamic deletion not yet implemented"))
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
    
    /// Save index to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // Manually extract graph data for serialization
        // Access the private fields of VamanaGraph through serialization
        let serialized_graph = bincode::serialize(&self.graph)?;
        let deserialized: SerializableVamanaGraph = bincode::deserialize(&serialized_graph)?;
        
        let serializable_graph = deserialized;
        
        let serializable = SerializableMemoryIndex {
            graph: serializable_graph,
            vectors: (*self.vectors).clone(),
            dimension: self.dimension,
            metric: self.metric,
        };
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &serializable)?;
        Ok(())
    }
    
    /// Load index from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let serializable: SerializableMemoryIndex = bincode::deserialize_from(reader)?;
        
        // Reconstruct VamanaGraph from serializable data
        let graph = VamanaGraph::from_serializable(serializable.graph);
        
        Ok(MemoryIndex {
            graph,
            vectors: Arc::new(serializable.vectors),
            dimension: serializable.dimension,
            metric: serializable.metric,
        })
    }
}

impl Index for MemoryIndex {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow::anyhow!("Dimension mismatch: expected {}, got {}", self.dimension, query.len()));
        }
        
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
    
    fn save(&self, path: &str) -> Result<()> {
        self.save(path)
    }
    
    fn stats(&self) -> IndexStats {
        self.stats()
    }
    
    fn memory_usage_bytes(&self) -> usize {
        self.stats().memory_usage_bytes
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