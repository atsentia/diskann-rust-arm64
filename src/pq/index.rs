//! PQ-compressed index implementation
//!
//! This module provides an index that uses Product Quantization for
//! memory-efficient storage while maintaining search quality.

use crate::{Result, Distance};
use crate::graph::VamanaGraph;
use crate::pq::{ProductQuantizer, PQParams, PQDistanceTable, BatchPQDistance};
use crate::graph::SearchParams;
use parking_lot::RwLock;
use std::sync::Arc;

/// PQ-compressed index for memory-efficient vector search
pub struct PQIndex {
    /// Original vectors (kept for query comparison)
    vectors: Arc<RwLock<Vec<Option<Vec<f32>>>>>,
    /// PQ-compressed vectors
    pq_codes: Arc<RwLock<Vec<Option<Vec<u8>>>>>,
    /// Product quantizer
    quantizer: ProductQuantizer,
    /// Distance table for fast PQ comparisons
    distance_table: PQDistanceTable,
    /// Batch distance calculator
    batch_calculator: Arc<RwLock<BatchPQDistance>>,
    /// Graph structure
    graph: Arc<RwLock<VamanaGraph>>,
    /// Dimension of original vectors
    dimension: usize,
    /// Next available ID
    next_id: Arc<RwLock<usize>>,
}

impl PQIndex {
    /// Create a new PQ index
    pub fn new(
        pq_params: PQParams,
        dimension: usize,
        distance_type: Distance,
    ) -> Result<Self> {
        let quantizer = ProductQuantizer::new(pq_params.clone(), dimension)?;
        
        // Create placeholder distance table (will be updated after training)
        let empty_codebooks = vec![vec![vec![0.0f32; dimension / pq_params.num_subspaces]; pq_params.num_centroids]; pq_params.num_subspaces];
        let distance_table = PQDistanceTable::new(&empty_codebooks, distance_type)?;
        let batch_calculator = BatchPQDistance::new(distance_table.clone());
        
        let graph = VamanaGraph::new(
            1000,        // num_vertices - estimated
            dimension,   // dimension
            distance_type,
            64,          // max_degree
            100,         // search_list_size
            1.2,         // alpha
        );
        
        Ok(Self {
            vectors: Arc::new(RwLock::new(Vec::new())),
            pq_codes: Arc::new(RwLock::new(Vec::new())),
            quantizer,
            distance_table,
            batch_calculator: Arc::new(RwLock::new(batch_calculator)),
            graph: Arc::new(RwLock::new(graph)),
            dimension,
            next_id: Arc::new(RwLock::new(0)),
        })
    }
    
    /// Train the PQ quantizer on a dataset
    pub fn train(&mut self, training_data: &[Vec<f32>]) -> Result<crate::pq::PQTrainingResult> {
        // Train the quantizer
        let training_result = self.quantizer.train(training_data)?;
        
        // Update distance table with trained codebooks
        self.distance_table = self.quantizer.create_distance_table(Distance::L2)?;
        
        // Update batch calculator
        let mut batch_calc = self.batch_calculator.write();
        *batch_calc = BatchPQDistance::new(self.distance_table.clone());
        
        Ok(training_result)
    }
    
    /// Build index from vectors (trains PQ if not already trained)
    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow::anyhow!("Cannot build index from empty vector set"));
        }
        
        // Validate dimensions
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dimension {
                return Err(anyhow::anyhow!(
                    "Vector {} has dimension {}, expected {}",
                    i, vector.len(), self.dimension
                ));
            }
        }
        
        // Train quantizer if not already trained
        if !self.quantizer.is_trained {
            println!("Training PQ quantizer on {} vectors...", vectors.len());
            self.train(&vectors)?;
        }
        
        // Encode all vectors with PQ
        println!("Encoding {} vectors with PQ...", vectors.len());
        let pq_codes = self.quantizer.encode_batch(&vectors)?;
        
        // Store vectors and codes
        {
            let mut stored_vectors = self.vectors.write();
            let mut stored_codes = self.pq_codes.write();
            
            stored_vectors.clear();
            stored_codes.clear();
            
            for vector in vectors.iter() {
                stored_vectors.push(Some(vector.clone()));
            }
            
            for code in pq_codes {
                stored_codes.push(Some(code));
            }
        }
        
        // Build graph using original vectors (for better quality)
        println!("Building Vamana graph...");
        {
            let mut graph = self.graph.write();
            graph.build(&vectors)?;
        }
        
        // Update next_id
        {
            let mut next_id = self.next_id.write();
            *next_id = vectors.len();
        }
        
        println!("PQ index built successfully!");
        Ok(())
    }
    
    /// Search for k nearest neighbors using PQ approximation
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(), self.dimension
            ));
        }
        
        if !self.quantizer.is_trained {
            return Err(anyhow::anyhow!("Index must be trained before searching"));
        }
        
        // Use graph search but with PQ distance approximation for efficiency
        let search_params = SearchParams {
            search_list_size: k.max(50),
            k,
            alpha: 1.2,
            use_bitvector: true,
        };
        
        // Create a PQ-aware distance function
        let pq_distance_fn = PQDistanceFunction {
            quantizer: &self.quantizer,
            pq_codes: &self.pq_codes,
            vectors: &self.vectors,
            query: query.to_vec(),
        };
        
        // Perform graph search (simplified approach)
        let graph = self.graph.read();
        let vectors = self.vectors.read();
        let candidate_vectors: Vec<Vec<f32>> = vectors.iter()
            .filter_map(|v| v.as_ref().cloned())
            .collect();
        let candidates = graph.search(query, search_params.search_list_size, &candidate_vectors)?;
        
        // Return top k results
        Ok(candidates.into_iter().take(k).collect())
    }
    
    /// Search using high-accuracy mode (asymmetric PQ distance)
    pub fn search_accurate(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(), self.dimension
            ));
        }
        
        if !self.quantizer.is_trained {
            return Err(anyhow::anyhow!("Index must be trained before searching"));
        }
        
        // Get candidates from graph search
        let candidates = self.search(query, k * 2)?; // Get more candidates for reranking
        
        // Rerank using asymmetric distance
        let mut accurate_results = Vec::new();
        
        {
            let pq_codes = self.pq_codes.read();
            
            for (id, _) in candidates {
                if id < pq_codes.len() {
                    if let Some(ref code) = pq_codes[id] {
                        let distance = self.quantizer.asymmetric_distance(code, query)?;
                        accurate_results.push((id, distance));
                    }
                }
            }
        }
        
        // Sort by distance and return top k
        accurate_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        accurate_results.truncate(k);
        
        Ok(accurate_results)
    }
    
    /// Add a new vector to the index
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<usize> {
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension {} doesn't match index dimension {}",
                vector.len(), self.dimension
            ));
        }
        
        if !self.quantizer.is_trained {
            return Err(anyhow::anyhow!("Index must be trained before inserting vectors"));
        }
        
        // Encode vector with PQ
        let pq_code = self.quantizer.encode(&vector)?;
        
        // Get new ID
        let id = {
            let next_id = self.next_id.read();
            *next_id
        };
        
        // Store vector and code
        {
            let mut vectors = self.vectors.write();
            let mut codes = self.pq_codes.write();
            
            // Extend vectors if necessary
            while vectors.len() <= id {
                vectors.push(None);
                codes.push(None);
            }
            
            vectors[id] = Some(vector.clone());
            codes[id] = Some(pq_code);
        }
        
        // Add to graph (simplified - would need proper graph update)
        {
            let mut graph = self.graph.write();
            // Note: VamanaGraph doesn't have insert method yet
            // This would require implementing dynamic graph updates
            // For now, we'll skip this step
        }
        
        // Update next_id
        {
            let mut next_id = self.next_id.write();
            *next_id += 1;
        }
        
        Ok(id)
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> PQIndexStats {
        let vectors = self.vectors.read();
        let codes = self.pq_codes.read();
        
        let num_vectors = vectors.iter().filter(|v| v.is_some()).count();
        let original_bytes = num_vectors * self.dimension * std::mem::size_of::<f32>();
        let compressed_bytes = codes.iter()
            .filter_map(|c| c.as_ref())
            .map(|c| c.len())
            .sum::<usize>();
        
        let quantizer_stats = self.quantizer.memory_stats();
        let distance_table_stats = self.distance_table.memory_stats();
        
        PQIndexStats {
            num_vectors,
            original_size_bytes: original_bytes,
            compressed_size_bytes: compressed_bytes,
            compression_ratio: original_bytes as f32 / compressed_bytes.max(1) as f32,
            quantizer_memory_bytes: quantizer_stats.codebook_size_bytes,
            distance_table_memory_bytes: distance_table_stats.table_size_bytes,
            total_memory_bytes: compressed_bytes + quantizer_stats.codebook_size_bytes + distance_table_stats.table_size_bytes,
        }
    }
}

/// Custom distance function for PQ-aware search
struct PQDistanceFunction<'a> {
    quantizer: &'a ProductQuantizer,
    pq_codes: &'a Arc<RwLock<Vec<Option<Vec<u8>>>>>,
    vectors: &'a Arc<RwLock<Vec<Option<Vec<f32>>>>>,
    query: Vec<f32>,
}

impl<'a> crate::DistanceFunction for PQDistanceFunction<'a> {
    fn distance(&self, _a: &[f32], _b: &[f32]) -> Result<f32> {
        // This shouldn't be called in our usage
        unreachable!("PQDistanceFunction should use distance_to_query")
    }
    
    fn batch_distance(&self, _query: &[f32], _points: &[f32], _distances: &mut [f32]) -> Result<()> {
        // Not implemented for PQ function
        unreachable!("batch_distance not supported for PQDistanceFunction")
    }
    
    fn distance_to_query(&self, query: &[f32], target_id: usize) -> Result<f32> {
        let codes = self.pq_codes.read();
        
        if target_id >= codes.len() {
            return Err(anyhow::anyhow!("Invalid target ID: {}", target_id));
        }
        
        if let Some(ref code) = codes[target_id] {
            // Use asymmetric distance for better accuracy
            self.quantizer.asymmetric_distance(code, query)
        } else {
            Err(anyhow::anyhow!("No PQ code found for ID: {}", target_id))
        }
    }
    
    fn metric(&self) -> Distance {
        Distance::L2
    }
}

/// Memory usage statistics for PQ index
#[derive(Debug)]
pub struct PQIndexStats {
    /// Number of vectors in index
    pub num_vectors: usize,
    /// Memory used by original vectors (for comparison)
    pub original_size_bytes: usize,
    /// Memory used by PQ codes
    pub compressed_size_bytes: usize,
    /// Compression ratio (original / compressed)
    pub compression_ratio: f32,
    /// Memory used by quantizer codebooks
    pub quantizer_memory_bytes: usize,
    /// Memory used by distance lookup tables
    pub distance_table_memory_bytes: usize,
    /// Total memory usage
    pub total_memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_pq_index_creation() {
        let params = PQParams::new(8, 8);
        let index = PQIndex::new(params, 128, Distance::L2).unwrap();
        
        assert_eq!(index.dimension, 128);
        assert!(!index.quantizer.is_trained);
    }
    
    #[test]
    fn test_pq_index_build_and_search() {
        let params = PQParams::new(4, 8);
        let mut index = PQIndex::new(params, 32, Distance::L2).unwrap();
        
        // Generate test data
        let vectors = generate_random_vectors(100, 32);
        
        // Build index
        index.build(vectors.clone()).unwrap();
        
        // Search
        let query = &vectors[0];
        let results = index.search(query, 5).unwrap();
        
        assert_eq!(results.len(), 5);
        
        // First result should be the query itself (distance 0)
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.1);
    }
    
    #[test]
    fn test_pq_index_accurate_search() {
        let params = PQParams::new(4, 8);
        let mut index = PQIndex::new(params, 32, Distance::L2).unwrap();
        
        // Generate test data
        let vectors = generate_random_vectors(50, 32);
        
        // Build index
        index.build(vectors.clone()).unwrap();
        
        // Compare regular vs accurate search
        let query = &vectors[0];
        let regular_results = index.search(query, 5).unwrap();
        let accurate_results = index.search_accurate(query, 5).unwrap();
        
        assert_eq!(regular_results.len(), 5);
        assert_eq!(accurate_results.len(), 5);
        
        // Accurate search should generally have better (lower) distances
        assert!(accurate_results[0].1 <= regular_results[0].1 + 0.1);
    }
    
    #[test]
    fn test_pq_index_insert() {
        let params = PQParams::new(4, 8);
        let mut index = PQIndex::new(params, 32, Distance::L2).unwrap();
        
        // Build initial index
        let vectors = generate_random_vectors(50, 32);
        index.build(vectors.clone()).unwrap();
        
        // Insert new vector
        let new_vector = generate_random_vectors(1, 32)[0].clone();
        let id = index.insert(new_vector.clone()).unwrap();
        
        assert_eq!(id, 50);
        
        // Search should find the new vector
        let results = index.search(&new_vector, 1).unwrap();
        assert_eq!(results[0].0, id);
    }
    
    #[test]
    fn test_pq_index_memory_stats() {
        let params = PQParams::new(4, 8);
        let mut index = PQIndex::new(params, 32, Distance::L2).unwrap();
        
        // Build index
        let vectors = generate_random_vectors(100, 32);
        index.build(vectors).unwrap();
        
        let stats = index.memory_stats();
        
        assert_eq!(stats.num_vectors, 100);
        assert!(stats.compression_ratio > 1.0);
        assert!(stats.compressed_size_bytes < stats.original_size_bytes);
    }
}