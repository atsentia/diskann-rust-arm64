//! PQ distance calculations for compressed vectors
//!
//! This module implements fast approximate distance calculations using
//! precomputed lookup tables for product quantized vectors.

use crate::{Result, Distance};
use crate::distance::simd;
use hashbrown::HashMap;

/// Precomputed distance lookup table for PQ codes
/// 
/// For each subspace and each pair of centroid codes, stores the distance
/// between centroids. This enables O(M) distance calculations instead of O(d).
#[derive(Debug, Clone)]
pub struct PQDistanceTable {
    /// Distance lookup tables [M x K x K] where M = num_subspaces, K = num_centroids
    pub tables: Vec<Vec<Vec<f32>>>,
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Number of centroids per subspace
    pub num_centroids: usize,
    /// Distance metric used
    pub distance_type: Distance,
}

impl PQDistanceTable {
    /// Create a new distance table from codebooks
    pub fn new(
        codebooks: &[Vec<Vec<f32>>],
        distance_type: Distance,
    ) -> Result<Self> {
        if codebooks.is_empty() {
            return Err(anyhow::anyhow!("Cannot create distance table from empty codebooks"));
        }
        
        let num_subspaces = codebooks.len();
        let num_centroids = codebooks[0].len();
        
        // Validate codebook structure
        for (i, codebook) in codebooks.iter().enumerate() {
            if codebook.len() != num_centroids {
                return Err(anyhow::anyhow!(
                    "Codebook {} has {} centroids, expected {}",
                    i, codebook.len(), num_centroids
                ));
            }
        }
        
        let mut tables = Vec::with_capacity(num_subspaces);
        
        // Build distance table for each subspace
        for codebook in codebooks {
            let mut subspace_table = vec![vec![0.0f32; num_centroids]; num_centroids];
            
            // Compute all pairwise distances between centroids
            for i in 0..num_centroids {
                for j in 0..num_centroids {
                    let distance = match distance_type {
                        Distance::L2 => simd::l2_distance(&codebook[i], &codebook[j])?,
                        Distance::Cosine => simd::cosine_distance(&codebook[i], &codebook[j])?,
                        Distance::InnerProduct => simd::inner_product_distance(&codebook[i], &codebook[j])?,
                    };
                    subspace_table[i][j] = distance;
                }
            }
            
            tables.push(subspace_table);
        }
        
        Ok(Self {
            tables,
            num_subspaces,
            num_centroids,
            distance_type,
        })
    }
    
    /// Calculate approximate distance between two PQ codes
    pub fn distance(&self, code1: &[u8], code2: &[u8]) -> Result<f32> {
        if code1.len() != self.num_subspaces || code2.len() != self.num_subspaces {
            return Err(anyhow::anyhow!(
                "Code length mismatch: expected {}, got {} and {}",
                self.num_subspaces, code1.len(), code2.len()
            ));
        }
        
        let mut total_distance = 0.0f32;
        
        // Sum distances across all subspaces
        for (subspace_idx, (&c1, &c2)) in code1.iter().zip(code2.iter()).enumerate() {
            if c1 as usize >= self.num_centroids || c2 as usize >= self.num_centroids {
                return Err(anyhow::anyhow!(
                    "Invalid code in subspace {}: codes {} and {} (max {})",
                    subspace_idx, c1, c2, self.num_centroids - 1
                ));
            }
            
            let subspace_distance = self.tables[subspace_idx][c1 as usize][c2 as usize];
            
            match self.distance_type {
                Distance::L2 => {
                    // For L2, sum squared distances
                    total_distance += subspace_distance * subspace_distance;
                }
                Distance::Cosine | Distance::InnerProduct => {
                    // For other metrics, sum directly
                    total_distance += subspace_distance;
                }
            }
        }
        
        // For L2, return square root of sum
        if self.distance_type == Distance::L2 {
            total_distance = total_distance.sqrt();
        }
        
        Ok(total_distance)
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> PQDistanceTableStats {
        let table_entries = self.num_subspaces * self.num_centroids * self.num_centroids;
        let table_bytes = table_entries * std::mem::size_of::<f32>();
        
        PQDistanceTableStats {
            table_size_bytes: table_bytes,
            num_entries: table_entries,
            lookup_time_complexity: format!("O({})", self.num_subspaces),
        }
    }
}

/// Memory and performance statistics for PQ distance tables
#[derive(Debug)]
pub struct PQDistanceTableStats {
    /// Total memory used by lookup tables
    pub table_size_bytes: usize,
    /// Total number of precomputed distances
    pub num_entries: usize,
    /// Time complexity description
    pub lookup_time_complexity: String,
}

/// Batch distance calculator for PQ codes
pub struct BatchPQDistance {
    table: PQDistanceTable,
    /// Cache for repeated queries
    query_cache: HashMap<Vec<u8>, Vec<f32>>,
}

impl BatchPQDistance {
    /// Create a new batch distance calculator
    pub fn new(table: PQDistanceTable) -> Self {
        Self {
            table,
            query_cache: HashMap::new(),
        }
    }
    
    /// Calculate distances from one query code to many database codes
    pub fn distances_to_many(&mut self, query_code: &[u8], db_codes: &[Vec<u8>]) -> Result<Vec<f32>> {
        let mut distances = Vec::with_capacity(db_codes.len());
        
        for db_code in db_codes {
            let distance = self.table.distance(query_code, db_code)?;
            distances.push(distance);
        }
        
        Ok(distances)
    }
    
    /// Clear the query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let entries = self.query_cache.len();
        let memory_bytes = entries * (
            self.table.num_subspaces + // query code size
            self.query_cache.values().next().map_or(0, |v| v.len() * std::mem::size_of::<f32>()) // distance values
        );
        (entries, memory_bytes)
    }
}

/// Convenience function for computing PQ distance
pub fn pq_distance(
    code1: &[u8],
    code2: &[u8],
    distance_table: &PQDistanceTable,
) -> Result<f32> {
    distance_table.distance(code1, code2)
}

/// Create distance table from codebooks
pub fn create_distance_table(
    codebooks: &[Vec<Vec<f32>>],
    distance_type: Distance,
) -> Result<PQDistanceTable> {
    PQDistanceTable::new(codebooks, distance_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pq::{ProductQuantizer, PQParams};  
    use crate::utils::generate_random_vectors;
    
    fn create_test_quantizer() -> ProductQuantizer {
        let params = PQParams::new(4, 8); // 4 subspaces, 8 bits (256 centroids)
        let mut pq = ProductQuantizer::new(params, 32).unwrap();
        
        // Train on random data
        let training_data = generate_random_vectors(100, 32);
        pq.train(&training_data).unwrap();
        
        pq
    }
    
    #[test]
    fn test_distance_table_creation() {
        let pq = create_test_quantizer();
        let table = PQDistanceTable::new(&pq.codebooks, Distance::L2).unwrap();
        
        assert_eq!(table.num_subspaces, 4);
        assert_eq!(table.num_centroids, 256);
        assert_eq!(table.tables.len(), 4);
        
        for subspace_table in &table.tables {
            assert_eq!(subspace_table.len(), 256);
            for row in subspace_table {
                assert_eq!(row.len(), 256);
            }
        }
    }
    
    #[test]
    fn test_pq_distance_calculation() {
        let pq = create_test_quantizer();
        let table = PQDistanceTable::new(&pq.codebooks, Distance::L2).unwrap();
        
        let code1 = vec![10u8, 20, 30, 40];
        let code2 = vec![11u8, 21, 31, 41];
        
        let distance = table.distance(&code1, &code2).unwrap();
        assert!(distance >= 0.0);
        
        // Distance to self should be 0
        let self_distance = table.distance(&code1, &code1).unwrap();
        assert!(self_distance < 1e-6);
    }
    
    #[test] 
    fn test_batch_distance_calculation() {
        let pq = create_test_quantizer();
        let table = PQDistanceTable::new(&pq.codebooks, Distance::L2).unwrap();
        let mut batch_calc = BatchPQDistance::new(table);
        
        let query_code = vec![10u8, 20, 30, 40];
        let db_codes = vec![
            vec![11u8, 21, 31, 41],
            vec![12u8, 22, 32, 42],
            vec![13u8, 23, 33, 43],
        ];
        
        let distances = batch_calc.distances_to_many(&query_code, &db_codes).unwrap();
        assert_eq!(distances.len(), 3);
        
        for distance in distances {
            assert!(distance >= 0.0);
        }
    }
    
    #[test]
    fn test_distance_consistency() {
        let pq = create_test_quantizer();
        let table = PQDistanceTable::new(&pq.codebooks, Distance::L2).unwrap();
        
        // Test that distance is symmetric for L2
        let code1 = vec![1u8, 2, 3, 4];
        let code2 = vec![5u8, 6, 7, 8];
        
        let d1 = table.distance(&code1, &code2).unwrap();
        let d2 = table.distance(&code2, &code1).unwrap();
        
        assert!((d1 - d2).abs() < 1e-6);
    }
    
    #[test]
    fn test_memory_stats() {
        let pq = create_test_quantizer();
        let table = PQDistanceTable::new(&pq.codebooks, Distance::L2).unwrap();
        
        let stats = table.memory_stats();
        
        // Should have 4 subspaces * 256 * 256 entries * 4 bytes
        assert_eq!(stats.num_entries, 4 * 256 * 256);
        assert_eq!(stats.table_size_bytes, 4 * 256 * 256 * 4);
        assert_eq!(stats.lookup_time_complexity, "O(4)");
    }
}