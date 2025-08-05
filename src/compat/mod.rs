//! Microsoft DiskANN C++ API Compatibility Layer
//!
//! This module provides a compatibility layer that matches the C++ DiskANN API,
//! making it easy to migrate from the C++ implementation to Rust.

pub mod cpp_api;
pub mod c_api;

use crate::{
    Result, Error, Distance,
    index::{Index, MemoryIndex, PQFlashIndex, PQFlashConfig}
};
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;

/// DiskANN Index compatible with C++ API
pub struct DiskANNIndex {
    inner: DiskANNIndexInner,
}

enum DiskANNIndexInner {
    Memory(Arc<RwLock<Box<dyn Index>>>),
    Disk(Arc<RwLock<PQFlashIndex>>),
}

/// Index build parameters matching C++ API
#[derive(Debug, Clone)]
pub struct BuildParams {
    pub num_threads: u32,
    pub max_degree: u32,
    pub search_list_size: u32,
    pub max_occlusion_size: u32,
    pub alpha: f32,
    pub saturate_graph: bool,
    pub use_pq_build: bool,
    pub num_pq_chunks: u32,
    pub use_opq: bool,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            num_threads: 0, // 0 means use all available
            max_degree: 64,
            search_list_size: 100,
            max_occlusion_size: 750,
            alpha: 1.2,
            saturate_graph: false,
            use_pq_build: false,
            num_pq_chunks: 0,
            use_opq: false,
        }
    }
}

/// Search parameters matching C++ API
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub search_list_size: u32,
    pub beamwidth: u32,
    pub reorder_data: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            search_list_size: 100,
            beamwidth: 2,
            reorder_data: true,
        }
    }
}

impl DiskANNIndex {
    /// Build index from data file (C++ compatible)
    pub fn build(
        data_file: &str,
        index_prefix: &str,
        params: &BuildParams,
        metric: Distance,
    ) -> Result<Self> {
        // Load vectors from file
        let (vectors, dimension) = if data_file.ends_with(".fvecs") {
            crate::formats::read_fvecs(data_file)?
        } else if data_file.ends_with(".bin") {
            let dimension = Self::infer_dimension(data_file)?;
            let vectors = crate::formats::read_binary_vectors(data_file, dimension)?;
            (vectors, dimension)
        } else {
            return Err(Error::InvalidParameter(
                "Unsupported file format. Use .fvecs or .bin".to_string()
            ).into());
        };

        // Set thread count
        if params.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(params.num_threads as usize)
                .build_global()
                .ok();
        }

        if params.use_pq_build {
            // Build PQ index
            let pq_config = PQFlashConfig {
                max_degree: params.max_degree as usize,
                search_list_size: params.search_list_size as usize,
                alpha: params.alpha,
                pq_params: crate::index::disk::PQParams {
                    num_chunks: params.num_pq_chunks as usize,
                    bits_per_chunk: 8,
                },
                num_threads: params.num_threads as usize,
                use_reorder_data: true,
                beam_width: 4,
            };
            
            let index_path = format!("{}.pq", index_prefix);
            let mut index = PQFlashIndex::new(dimension, metric, pq_config);
            index.build_and_save(&vectors, &index_path)?;
            index.load(&index_path)?;
            
            Ok(Self {
                inner: DiskANNIndexInner::Disk(Arc::new(RwLock::new(index))),
            })
        } else {
            // Build in-memory index
            let index = crate::IndexBuilder::new()
                .dimensions(dimension)
                .metric(metric)
                .max_degree(params.max_degree as usize)
                .search_list_size(params.search_list_size as usize)
                .alpha(params.alpha)
                .build_from_vectors(vectors)?;
            
            // Save to disk
            index.save(&format!("{}.bin", index_prefix))?;
            
            Ok(Self {
                inner: DiskANNIndexInner::Memory(Arc::new(RwLock::new(index))),
            })
        }
    }

    /// Load index from disk (C++ compatible)
    pub fn load(
        index_prefix: &str,
        num_points: usize,
        dimension: usize,
        metric: Distance,
    ) -> Result<Self> {
        let bin_path = format!("{}.bin", index_prefix);
        let pq_path = format!("{}.pq", index_prefix);
        
        if Path::new(&pq_path).exists() {
            // Load PQ index - create new index and load data
            let mut index = PQFlashIndex::new(128, Distance::L2, PQFlashConfig::default()); // Default values
            index.load(&pq_path)?;
            Ok(Self {
                inner: DiskANNIndexInner::Disk(Arc::new(RwLock::new(index))),
            })
        } else if Path::new(&bin_path).exists() {
            // Load in-memory index
            let index = MemoryIndex::load(&bin_path)?;
            Ok(Self {
                inner: DiskANNIndexInner::Memory(Arc::new(RwLock::new(Box::new(index)))),
            })
        } else {
            Err(Error::InvalidParameter(
                format!("Index files not found with prefix: {}", index_prefix)
            ).into())
        }
    }

    /// Search for k nearest neighbors (C++ compatible)
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        neighbors: &mut [u32],
        distances: &mut [f32],
    ) -> Result<u32> {
        match &self.inner {
            DiskANNIndexInner::Memory(index) => {
                let index = index.read();
                let results = index.search(query, k)?;
                
                for (i, (id, dist)) in results.iter().enumerate() {
                    if i < neighbors.len() && i < distances.len() {
                        neighbors[i] = *id as u32;
                        distances[i] = *dist;
                    }
                }
                
                Ok(results.len() as u32)
            }
            DiskANNIndexInner::Disk(index) => {
                let index = index.read();
                let search_list_size = k * 10; // Default search parameter
                let (results, _stats) = index.search(query, k, search_list_size)?;
                
                for (i, (id, dist)) in results.iter().enumerate() {
                    if i < neighbors.len() && i < distances.len() {
                        neighbors[i] = *id;
                        distances[i] = *dist;
                    }
                }
                
                Ok(results.len() as u32)
            }
        }
    }

    /// Batch search (C++ compatible)
    pub fn batch_search(
        &self,
        queries: &[f32],
        num_queries: usize,
        dimension: usize,
        k: usize,
        params: &SearchParams,
        neighbors: &mut [u32],
        distances: &mut [f32],
    ) -> Result<()> {
        let neighbors_per_query = k;
        
        for i in 0..num_queries {
            let query_start = i * dimension;
            let query_end = query_start + dimension;
            let query = &queries[query_start..query_end];
            
            let result_start = i * neighbors_per_query;
            let neighbor_slice = &mut neighbors[result_start..result_start + k];
            let distance_slice = &mut distances[result_start..result_start + k];
            
            self.search(query, k, params, neighbor_slice, distance_slice)?;
        }
        
        Ok(())
    }

    /// Get index stats (C++ compatible)
    pub fn get_stats(&self) -> IndexStats {
        match &self.inner {
            DiskANNIndexInner::Memory(index) => {
                let index = index.read();
                let stats = index.stats();
                IndexStats {
                    num_points: stats.num_vectors,
                    dimension: index.dimension(),
                    graph_degree: stats.graph_degree_avg,
                    max_degree: stats.graph_degree_max as u32,
                    indexing_time: 0.0, // Not tracked
                    memory_usage: index.memory_usage_bytes() as u64,
                }
            }
            DiskANNIndexInner::Disk(index) => {
                let index = index.read();
                IndexStats {
                    num_points: index.size(),
                    dimension: index.dimension(),
                    graph_degree: 0.0, // Not available for disk index
                    max_degree: 0,
                    indexing_time: 0.0,
                    memory_usage: index.memory_usage_bytes() as u64,
                }
            }
        }
    }

    /// Helper to infer dimension from binary file
    fn infer_dimension(file_path: &str) -> Result<usize> {
        use std::fs;
        let metadata = fs::metadata(file_path)?;
        let file_size = metadata.len() as usize;
        
        // Try common dimensions
        for dim in [128, 256, 512, 768, 960, 1024, 1536, 2048] {
            if file_size % (dim * 4) == 0 {
                return Ok(dim);
            }
        }
        
        Err(Error::InvalidParameter(
            "Cannot infer dimension from file size. Please specify dimension.".to_string()
        ).into())
    }
}

/// Index statistics (C++ compatible)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_points: usize,
    pub dimension: usize,
    pub graph_degree: f32,
    pub max_degree: u32,
    pub indexing_time: f64,
    pub memory_usage: u64,
}

/// Thread-safe wrapper for concurrent operations
pub struct ConcurrentIndex {
    inner: Arc<DiskANNIndex>,
}

impl ConcurrentIndex {
    pub fn new(index: DiskANNIndex) -> Self {
        Self {
            inner: Arc::new(index),
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let mut neighbors = vec![0u32; k];
        let mut distances = vec![0.0f32; k];
        
        let count = self.inner.search(
            query,
            k,
            params,
            &mut neighbors,
            &mut distances,
        )?;
        
        neighbors.truncate(count as usize);
        distances.truncate(count as usize);
        
        Ok((neighbors, distances))
    }
}

// Re-export commonly used types
pub use crate::Distance as Metric;
pub use crate::Distance::L2 as METRIC_L2;
pub use crate::Distance::InnerProduct as METRIC_IP;
pub use crate::Distance::Cosine as METRIC_COSINE;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_cpp_api_compatibility() {
        // Generate test data
        let vectors = crate::utils::generate_random_vectors(1000, 128);
        let dir = tempdir().unwrap();
        let data_path = dir.path().join("test.fvecs");
        let index_prefix = dir.path().join("test_index").to_str().unwrap().to_string();
        
        // Save vectors
        crate::formats::write_fvecs(&data_path, &vectors).unwrap();
        
        // Build index using C++ compatible API
        let params = BuildParams::default();
        let index = DiskANNIndex::build(
            data_path.to_str().unwrap(),
            &index_prefix,
            &params,
            METRIC_L2,
        ).unwrap();
        
        // Search
        let query = &vectors[0];
        let mut neighbors = vec![0u32; 10];
        let mut distances = vec![0.0f32; 10];
        let search_params = SearchParams::default();
        
        let count = index.search(
            query,
            10,
            &search_params,
            &mut neighbors,
            &mut distances,
        ).unwrap();
        
        assert_eq!(count, 10);
        assert_eq!(neighbors[0], 0); // Should find itself
        assert!(distances[0] < 0.01); // Distance to itself should be ~0
    }
}