//! Codebook management for product quantization
//!
//! This module handles codebook storage, serialization, and lookups.

use crate::{Result, Distance};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Codebook for product quantization
/// 
/// Stores centroids for each subspace and provides methods for
/// encoding/decoding and serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// Centroids for each subspace [M x K x d/M]
    /// M = num_subspaces, K = num_centroids per subspace, d/M = subspace dimension
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Number of centroids per subspace
    pub num_centroids: usize,
    /// Dimension of each subspace
    pub subspace_dimension: usize,
    /// Original vector dimension
    pub dimension: usize,
    /// Distance metric used for training
    pub distance_type: Distance,
    /// Number of bits per subquantizer
    pub bits_per_subquantizer: usize,
}

impl Codebook {
    /// Create a new codebook from centroids
    pub fn new(
        centroids: Vec<Vec<Vec<f32>>>,
        distance_type: Distance,
        bits_per_subquantizer: usize,
    ) -> Result<Self> {
        if centroids.is_empty() {
            return Err(anyhow::anyhow!("Cannot create codebook with empty centroids"));
        }
        
        let num_subspaces = centroids.len();
        let num_centroids = centroids[0].len();
        let subspace_dimension = if num_centroids > 0 {
            centroids[0][0].len()
        } else {
            return Err(anyhow::anyhow!("Cannot create codebook with empty centroids"));
        };
        
        // Validate structure
        for (i, subspace) in centroids.iter().enumerate() {
            if subspace.len() != num_centroids {
                return Err(anyhow::anyhow!(
                    "Subspace {} has {} centroids, expected {}",
                    i, subspace.len(), num_centroids
                ));
            }
            
            for (j, centroid) in subspace.iter().enumerate() {
                if centroid.len() != subspace_dimension {
                    return Err(anyhow::anyhow!(
                        "Centroid [{}, {}] has dimension {}, expected {}",
                        i, j, centroid.len(), subspace_dimension
                    ));
                }
            }
        }
        
        // Validate that num_centroids matches bits
        let expected_centroids = 1 << bits_per_subquantizer;
        if num_centroids != expected_centroids {
            return Err(anyhow::anyhow!(
                "Number of centroids ({}) doesn't match 2^bits (2^{} = {})",
                num_centroids, bits_per_subquantizer, expected_centroids
            ));
        }
        
        let dimension = num_subspaces * subspace_dimension;
        
        Ok(Self {
            centroids,
            num_subspaces,
            num_centroids,
            subspace_dimension,
            dimension,
            distance_type,
            bits_per_subquantizer,
        })
    }
    
    /// Get centroids for a specific subspace
    pub fn get_subspace_centroids(&self, subspace_idx: usize) -> Result<&Vec<Vec<f32>>> {
        if subspace_idx >= self.num_subspaces {
            return Err(anyhow::anyhow!(
                "Subspace index {} out of range (max {})",
                subspace_idx, self.num_subspaces - 1
            ));
        }
        
        Ok(&self.centroids[subspace_idx])
    }
    
    /// Get a specific centroid
    pub fn get_centroid(&self, subspace_idx: usize, centroid_idx: usize) -> Result<&Vec<f32>> {
        if subspace_idx >= self.num_subspaces {
            return Err(anyhow::anyhow!(
                "Subspace index {} out of range (max {})",
                subspace_idx, self.num_subspaces - 1
            ));
        }
        
        if centroid_idx >= self.num_centroids {
            return Err(anyhow::anyhow!(
                "Centroid index {} out of range (max {})",
                centroid_idx, self.num_centroids - 1
            ));
        }
        
        Ok(&self.centroids[subspace_idx][centroid_idx])
    }
    
    /// Find closest centroid in a subspace
    pub fn find_closest_centroid(&self, subspace_idx: usize, subvector: &[f32]) -> Result<(usize, f32)> {
        if subspace_idx >= self.num_subspaces {
            return Err(anyhow::anyhow!(
                "Subspace index {} out of range (max {})",
                subspace_idx, self.num_subspaces - 1
            ));
        }
        
        if subvector.len() != self.subspace_dimension {
            return Err(anyhow::anyhow!(
                "Subvector dimension {} doesn't match subspace dimension {}",
                subvector.len(), self.subspace_dimension
            ));
        }
        
        let subspace_centroids = &self.centroids[subspace_idx];
        let mut min_distance = f32::INFINITY;
        let mut best_idx = 0;
        
        for (idx, centroid) in subspace_centroids.iter().enumerate() {
            let mut distance = 0.0f32;
            for (a, b) in subvector.iter().zip(centroid.iter()) {
                let diff = a - b;
                distance += diff * diff;
            }
            distance = distance.sqrt(); // L2 distance
            
            if distance < min_distance {
                min_distance = distance;
                best_idx = idx;
            }
        }
        
        Ok((best_idx, min_distance))
    }
    
    /// Reconstruct a vector from PQ codes
    pub fn reconstruct(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.num_subspaces {
            return Err(anyhow::anyhow!(
                "Code length {} doesn't match number of subspaces {}",
                codes.len(), self.num_subspaces
            ));
        }
        
        let mut reconstructed = vec![0.0f32; self.dimension];
        
        for (subspace_idx, &code) in codes.iter().enumerate() {
            if code as usize >= self.num_centroids {
                return Err(anyhow::anyhow!(
                    "Invalid code {} for subspace {} (max {})",
                    code, subspace_idx, self.num_centroids - 1
                ));
            }
            
            let centroid = &self.centroids[subspace_idx][code as usize];
            let start_dim = subspace_idx * self.subspace_dimension;
            
            for (i, &value) in centroid.iter().enumerate() {
                reconstructed[start_dim + i] = value;
            }
        }
        
        Ok(reconstructed)
    }
    
    /// Save codebook to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }
    
    /// Load codebook from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let codebook = bincode::deserialize_from(reader)?;
        Ok(codebook)
    }
    
    /// Save codebook in JSON format (human readable)
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
    
    /// Load codebook from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let codebook = serde_json::from_reader(reader)?;
        Ok(codebook)
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> CodebookStats {
        let centroid_count = self.num_subspaces * self.num_centroids;
        let total_floats = centroid_count * self.subspace_dimension;
        let memory_bytes = total_floats * std::mem::size_of::<f32>();
        
        CodebookStats {
            memory_bytes,
            num_subspaces: self.num_subspaces,
            num_centroids: self.num_centroids,
            subspace_dimension: self.subspace_dimension,
            total_centroids: centroid_count,
            compression_ratio: (self.dimension * std::mem::size_of::<f32>()) as f32 
                             / self.num_subspaces as f32,
        }
    }
    
    /// Validate codebook integrity
    pub fn validate(&self) -> Result<()> {
        // Check basic structure
        if self.centroids.len() != self.num_subspaces {
            return Err(anyhow::anyhow!(
                "Centroids length {} doesn't match num_subspaces {}",
                self.centroids.len(), self.num_subspaces
            ));
        }
        
        // Check each subspace
        for (i, subspace) in self.centroids.iter().enumerate() {
            if subspace.len() != self.num_centroids {
                return Err(anyhow::anyhow!(
                    "Subspace {} has {} centroids, expected {}",
                    i, subspace.len(), self.num_centroids
                ));
            }
            
            // Check each centroid
            for (j, centroid) in subspace.iter().enumerate() {
                if centroid.len() != self.subspace_dimension {
                    return Err(anyhow::anyhow!(
                        "Centroid [{}, {}] has dimension {}, expected {}",
                        i, j, centroid.len(), self.subspace_dimension
                    ));
                }
                
                // Check for NaN or infinite values
                for (k, &value) in centroid.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(anyhow::anyhow!(
                            "Invalid value {} at centroid [{}, {}][{}]",
                            value, i, j, k
                        ));
                    }
                }
            }
        }
        
        // Check dimension consistency
        if self.dimension != self.num_subspaces * self.subspace_dimension {
            return Err(anyhow::anyhow!(
                "Dimension {} doesn't match num_subspaces * subspace_dimension ({} * {} = {})",
                self.dimension, self.num_subspaces, self.subspace_dimension,
                self.num_subspaces * self.subspace_dimension
            ));
        }
        
        Ok(())
    }
}

/// Codebook memory and structure statistics
#[derive(Debug)]
pub struct CodebookStats {
    /// Total memory used by codebook in bytes
    pub memory_bytes: usize,
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Number of centroids per subspace
    pub num_centroids: usize,
    /// Dimension of each subspace
    pub subspace_dimension: usize,
    /// Total number of centroids across all subspaces
    pub total_centroids: usize,
    /// Compression ratio (original bytes / compressed bytes)
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pq::{ProductQuantizer, PQParams};
    use crate::utils::generate_random_vectors;
    use tempfile::NamedTempFile;
    
    fn create_test_codebook() -> Codebook {
        let params = PQParams::new(4, 8); // 4 subspaces, 8 bits (256 centroids)
        let mut pq = ProductQuantizer::new(params, 32).unwrap();
        
        // Train on random data
        let training_data = generate_random_vectors(100, 32);
        pq.train(&training_data).unwrap();
        
        Codebook::new(pq.codebooks, Distance::L2, 8).unwrap()
    }
    
    #[test]
    fn test_codebook_creation() {
        let codebook = create_test_codebook();
        
        assert_eq!(codebook.num_subspaces, 4);
        assert_eq!(codebook.num_centroids, 256);
        assert_eq!(codebook.subspace_dimension, 8);
        assert_eq!(codebook.dimension, 32);
        assert_eq!(codebook.bits_per_subquantizer, 8);
        
        codebook.validate().unwrap();
    }
    
    #[test]
    fn test_centroid_access() {
        let codebook = create_test_codebook();
        
        // Test getting subspace centroids
        let subspace_0 = codebook.get_subspace_centroids(0).unwrap();
        assert_eq!(subspace_0.len(), 256);
        
        // Test getting specific centroid
        let centroid = codebook.get_centroid(0, 0).unwrap();
        assert_eq!(centroid.len(), 8);
        
        // Test out of bounds
        assert!(codebook.get_subspace_centroids(10).is_err());
        assert!(codebook.get_centroid(0, 300).is_err());
    }
    
    #[test]
    fn test_closest_centroid() {
        let codebook = create_test_codebook();
        
        let test_subvector = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let (idx, distance) = codebook.find_closest_centroid(0, &test_subvector).unwrap();
        
        assert!(idx < 256);
        assert!(distance >= 0.0);
        
        // Test wrong dimension
        let wrong_dim = vec![0.1, 0.2];
        assert!(codebook.find_closest_centroid(0, &wrong_dim).is_err());
    }
    
    #[test]
    fn test_reconstruction() {
        let codebook = create_test_codebook();
        
        let codes = vec![10u8, 20, 30, 40];
        let reconstructed = codebook.reconstruct(&codes).unwrap();
        
        assert_eq!(reconstructed.len(), 32);
        
        // Test wrong code length
        let wrong_codes = vec![10u8, 20];
        assert!(codebook.reconstruct(&wrong_codes).is_err());
        
        // Test invalid code
        let invalid_codes = vec![255u8, 255, 255, 255];
        assert!(codebook.reconstruct(&invalid_codes).is_err());
    }
    
    #[test]
    fn test_serialization() {
        let codebook = create_test_codebook();
        
        // Test binary serialization
        let temp_file = NamedTempFile::new().unwrap();
        codebook.save(temp_file.path()).unwrap();
        
        let loaded_codebook = Codebook::load(temp_file.path()).unwrap();
        assert_eq!(codebook.num_subspaces, loaded_codebook.num_subspaces);
        assert_eq!(codebook.num_centroids, loaded_codebook.num_centroids);
        assert_eq!(codebook.dimension, loaded_codebook.dimension);
        
        // Test JSON serialization
        let temp_json = NamedTempFile::with_suffix(".json").unwrap();
        codebook.save_json(temp_json.path()).unwrap();
        
        let loaded_json = Codebook::load_json(temp_json.path()).unwrap();
        assert_eq!(codebook.num_subspaces, loaded_json.num_subspaces);
        assert_eq!(codebook.num_centroids, loaded_json.num_centroids);
    }
    
    #[test]
    fn test_memory_stats() {
        let codebook = create_test_codebook();
        let stats = codebook.memory_stats();
        
        assert_eq!(stats.num_subspaces, 4);
        assert_eq!(stats.num_centroids, 256);
        assert_eq!(stats.total_centroids, 4 * 256);
        assert_eq!(stats.memory_bytes, 4 * 256 * 8 * 4); // subspaces * centroids * dim * sizeof(f32)
    }
}