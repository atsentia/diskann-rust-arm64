//! Product quantizer implementation
//!
//! This module implements the core product quantization algorithm for
//! memory-efficient vector storage and search.

use crate::{Result, Distance};
use crate::pq::kmeans::{KMeans, KMeansParams};
use crate::utils::aligned::AlignedVec;

/// Product Quantization parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PQParams {
    /// Number of subspaces (M)
    pub num_subspaces: usize,
    /// Number of centroids per subspace (K = 2^nbits)
    pub num_centroids: usize,
    /// Number of bits per subquantizer (log2(num_centroids))
    pub bits_per_subquantizer: usize,
    /// Random seed for reproducible training
    pub seed: Option<u64>,
    /// K-means parameters for training
    pub kmeans_params: KMeansParams,
}

impl PQParams {
    /// Create new PQ parameters
    pub fn new(num_subspaces: usize, bits_per_subquantizer: usize) -> Self {
        let num_centroids = 1 << bits_per_subquantizer; // 2^bits
        
        Self {
            num_subspaces,
            num_centroids,
            bits_per_subquantizer,
            seed: None,
            kmeans_params: KMeansParams {
                k: num_centroids,
                max_iterations: 100,
                tolerance: 1e-4,
                seed: None,
                use_plus_plus_init: true,
            },
        }
    }
    
    /// Common configuration: 8 subspaces, 8 bits each (256 centroids per subspace)
    pub fn default_8x8() -> Self {
        Self::new(8, 8)
    }
    
    /// Common configuration: 16 subspaces, 8 bits each
    pub fn default_16x8() -> Self {
        Self::new(16, 8)
    }
    
    /// Validate parameters against vector dimension
    pub fn validate(&self, dimension: usize) -> Result<()> {
        if dimension % self.num_subspaces != 0 {
            return Err(anyhow::anyhow!(
                "Vector dimension ({}) must be divisible by number of subspaces ({})",
                dimension, self.num_subspaces
            ));
        }
        
        if self.bits_per_subquantizer == 0 || self.bits_per_subquantizer > 16 {
            return Err(anyhow::anyhow!(
                "Bits per subquantizer must be between 1 and 16, got {}",
                self.bits_per_subquantizer
            ));
        }
        
        if self.num_centroids != (1 << self.bits_per_subquantizer) {
            return Err(anyhow::anyhow!(
                "Number of centroids ({}) must equal 2^bits_per_subquantizer (2^{} = {})",
                self.num_centroids, self.bits_per_subquantizer, 1 << self.bits_per_subquantizer
            ));
        }
        
        Ok(())
    }
}

/// Product Quantizer for vector compression
pub struct ProductQuantizer {
    /// PQ configuration parameters
    pub params: PQParams,
    /// Vector dimension
    pub dimension: usize,
    /// Subspace dimension (dimension / num_subspaces)
    pub subspace_dim: usize,
    /// Codebooks for each subspace [M x K x d/M]
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// Whether the quantizer has been trained
    pub is_trained: bool,
}

impl ProductQuantizer {
    /// Create a new untrained product quantizer
    pub fn new(params: PQParams, dimension: usize) -> Result<Self> {
        params.validate(dimension)?;
        
        let subspace_dim = dimension / params.num_subspaces;
        
        let num_subspaces = params.num_subspaces;
        
        Ok(Self {
            params,
            dimension,
            subspace_dim,
            codebooks: vec![Vec::new(); num_subspaces],
            is_trained: false,
        })
    }
    
    /// Train the product quantizer on a dataset
    pub fn train(&mut self, data: &[Vec<f32>]) -> Result<PQTrainingResult> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot train on empty dataset"));
        }
        
        // Validate data dimensions
        for (i, vector) in data.iter().enumerate() {
            if vector.len() != self.dimension {
                return Err(anyhow::anyhow!(
                    "Vector {} has dimension {}, expected {}",
                    i, vector.len(), self.dimension
                ));
            }
        }
        
        let training_start = std::time::Instant::now();
        let mut training_stats = PQTrainingResult {
            subspace_inertias: Vec::with_capacity(self.params.num_subspaces),
            total_training_time: std::time::Duration::new(0, 0),
            convergence_info: Vec::new(),
        };
        
        // Train each subspace independently
        for subspace_idx in 0..self.params.num_subspaces {
            let start_dim = subspace_idx * self.subspace_dim;
            let end_dim = start_dim + self.subspace_dim;
            
            // Extract subspace data
            let subspace_data: Vec<Vec<f32>> = data
                .iter()
                .map(|vector| vector[start_dim..end_dim].to_vec())
                .collect();
            
            // Train K-means for this subspace
            let mut kmeans_params = self.params.kmeans_params.clone();
            if let Some(seed) = self.params.seed {
                kmeans_params.seed = Some(seed + subspace_idx as u64);
            }
            
            let mut kmeans = KMeans::new(kmeans_params, self.subspace_dim);
            let result = kmeans.fit(&subspace_data)?;
            
            // Store codebook
            self.codebooks[subspace_idx] = result.centroids;
            
            // Record training statistics  
            training_stats.subspace_inertias.push(result.inertia);
            training_stats.convergence_info.push((
                result.iterations,
                result.converged,
                result.inertia,
            ));
        }
        
        self.is_trained = true;
        training_stats.total_training_time = training_start.elapsed();
        
        Ok(training_stats)
    }
    
    /// Encode a single vector using the trained quantizer
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(anyhow::anyhow!("Quantizer must be trained before encoding"));
        }
        
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension {} doesn't match quantizer dimension {}",
                vector.len(), self.dimension
            ));
        }
        
        let mut codes = Vec::with_capacity(self.params.num_subspaces);
        
        for subspace_idx in 0..self.params.num_subspaces {
            let start_dim = subspace_idx * self.subspace_dim;
            let end_dim = start_dim + self.subspace_dim;
            let subvector = &vector[start_dim..end_dim];
            
            // Find closest centroid in this subspace
            let code = self.encode_subspace(subvector, subspace_idx)?;
            codes.push(code);
        }
        
        Ok(codes)
    }
    
    /// Encode multiple vectors efficiently
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        if !self.is_trained {
            return Err(anyhow::anyhow!("Quantizer must be trained before encoding"));
        }
        
        let mut encoded = Vec::with_capacity(vectors.len());
        
        for vector in vectors {
            encoded.push(self.encode(vector)?);
        }
        
        Ok(encoded)
    }
    
    /// Decode a PQ code back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if !self.is_trained {
            return Err(anyhow::anyhow!("Quantizer must be trained before decoding"));
        }
        
        if codes.len() != self.params.num_subspaces {
            return Err(anyhow::anyhow!(
                "Code length {} doesn't match number of subspaces {}",
                codes.len(), self.params.num_subspaces
            ));
        }
        
        let mut decoded = vec![0.0f32; self.dimension];
        
        for (subspace_idx, &code) in codes.iter().enumerate() {
            if code as usize >= self.params.num_centroids {
                return Err(anyhow::anyhow!(
                    "Invalid code {} for subspace {} (max {})",
                    code, subspace_idx, self.params.num_centroids - 1
                ));
            }
            
            let start_dim = subspace_idx * self.subspace_dim;
            let centroid = &self.codebooks[subspace_idx][code as usize];
            
            for (i, &value) in centroid.iter().enumerate() {
                decoded[start_dim + i] = value;
            }
        }
        
        Ok(decoded)
    }
    
    /// Calculate reconstruction error for a set of vectors
    pub fn reconstruction_error(&self, vectors: &[Vec<f32>]) -> Result<f32> {
        if vectors.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_error = 0.0;
        
        for vector in vectors {
            let encoded = self.encode(vector)?;
            let decoded = self.decode(&encoded)?;
            
            // Calculate L2 error
            let mut error = 0.0;
            for (original, reconstructed) in vector.iter().zip(decoded.iter()) {
                let diff = original - reconstructed;
                error += diff * diff;
            }
            total_error += error.sqrt();
        }
        
        Ok(total_error / vectors.len() as f32)
    }
    
    /// Calculate asymmetric distance between PQ code and full vector
    /// 
    /// This is more accurate than symmetric PQ distance when the query
    /// is a full vector and the database contains PQ codes.
    pub fn asymmetric_distance(&self, pq_code: &[u8], full_vector: &[f32]) -> Result<f32> {
        if !self.is_trained {
            return Err(anyhow::anyhow!("Quantizer must be trained before computing asymmetric distance"));
        }
        
        if pq_code.len() != self.params.num_subspaces {
            return Err(anyhow::anyhow!(
                "PQ code length {} doesn't match number of subspaces {}",
                pq_code.len(), self.params.num_subspaces
            ));
        }
        
        if full_vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension {} doesn't match quantizer dimension {}",
                full_vector.len(), self.dimension
            ));
        }
        
        let mut total_distance = 0.0f32;
        
        // Calculate distance for each subspace
        for (subspace_idx, &code) in pq_code.iter().enumerate() {
            if code as usize >= self.params.num_centroids {
                return Err(anyhow::anyhow!(
                    "Invalid code {} in subspace {} (max {})",
                    code, subspace_idx, self.params.num_centroids - 1
                ));
            }
            
            let start_dim = subspace_idx * self.subspace_dim;
            let end_dim = start_dim + self.subspace_dim;
            let subvector = &full_vector[start_dim..end_dim];
            
            // Get centroid for this code
            let centroid = &self.codebooks[subspace_idx][code as usize];
            
            // Calculate L2 distance in this subspace
            let mut subspace_distance = 0.0f32;
            for (a, b) in subvector.iter().zip(centroid.iter()) {
                let diff = a - b;
                subspace_distance += diff * diff;
            }
            
            total_distance += subspace_distance;
        }
        
        Ok(total_distance.sqrt())
    }
    
    /// Create distance table for fast PQ-to-PQ distance calculations
    pub fn create_distance_table(&self, distance_type: Distance) -> Result<crate::pq::PQDistanceTable> {
        if !self.is_trained {
            return Err(anyhow::anyhow!("Quantizer must be trained before creating distance table"));
        }
        
        crate::pq::distance::PQDistanceTable::new(&self.codebooks, distance_type)
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> PQMemoryStats {
        let codebook_size = self.params.num_subspaces * self.params.num_centroids * self.subspace_dim;
        let codebook_bytes = codebook_size * std::mem::size_of::<f32>();
        
        PQMemoryStats {
            codebook_size_bytes: codebook_bytes,
            bytes_per_vector: self.params.num_subspaces,
            compression_ratio: (self.dimension * std::mem::size_of::<f32>()) as f32 
                             / self.params.num_subspaces as f32,
        }
    }
    
    /// Find closest centroid in a specific subspace
    fn encode_subspace(&self, subvector: &[f32], subspace_idx: usize) -> Result<u8> {
        let codebook = &self.codebooks[subspace_idx];
        let mut min_distance = f32::INFINITY;
        let mut best_code = 0u8;
        
        for (code, centroid) in codebook.iter().enumerate() {
            let mut distance = 0.0f32;
            for (a, b) in subvector.iter().zip(centroid.iter()) {
                let diff = a - b;
                distance += diff * diff;
            }
            
            if distance < min_distance {
                min_distance = distance;
                best_code = code as u8;
            }
        }
        
        Ok(best_code)
    }
}

/// Product quantization training statistics
#[derive(Debug)]
pub struct PQTrainingResult {
    /// Inertia (within-cluster sum of squares) for each subspace
    pub subspace_inertias: Vec<f32>,
    /// Total time spent training
    pub total_training_time: std::time::Duration,
    /// Convergence information (iterations, converged, final_inertia) per subspace
    pub convergence_info: Vec<(usize, bool, f32)>,
}

impl PQTrainingResult {
    /// Get average reconstruction error across all subspaces
    pub fn average_inertia(&self) -> f32 {
        if self.subspace_inertias.is_empty() {
            0.0
        } else {
            self.subspace_inertias.iter().sum::<f32>() / self.subspace_inertias.len() as f32
        }
    }
    
    /// Check if all subspaces converged
    pub fn all_converged(&self) -> bool {
        self.convergence_info.iter().all(|(_, converged, _)| *converged)
    }
}

/// Memory usage statistics for PQ
#[derive(Debug)]
pub struct PQMemoryStats {
    /// Total bytes used by codebooks
    pub codebook_size_bytes: usize,
    /// Bytes per encoded vector
    pub bytes_per_vector: usize,
    /// Compression ratio (original_size / compressed_size)
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_pq_params_validation() {
        // Valid parameters
        let params = PQParams::new(8, 8);
        assert!(params.validate(128).is_ok());
        
        // Invalid: dimension not divisible by subspaces
        assert!(params.validate(100).is_err());
        
        // Invalid: too many bits
        let invalid_params = PQParams::new(8, 17);
        assert!(invalid_params.validate(128).is_err());
    }
    
    #[test]
    fn test_pq_basic_training() {
        let data = generate_random_vectors(100, 64);
        let params = PQParams::new(8, 8);
        let mut pq = ProductQuantizer::new(params, 64).unwrap();
        
        let result = pq.train(&data).unwrap();
        
        assert!(pq.is_trained);
        assert_eq!(result.subspace_inertias.len(), 8);
        assert_eq!(pq.codebooks.len(), 8);
        
        // Each codebook should have 256 centroids (2^8)
        for codebook in &pq.codebooks {
            assert_eq!(codebook.len(), 256);
            // Each centroid should have dimension 64/8 = 8
            for centroid in codebook {
                assert_eq!(centroid.len(), 8);
            }
        }
    }
    
    #[test]
    fn test_pq_encode_decode() {
        let data = generate_random_vectors(50, 32);
        let params = PQParams::new(4, 8);
        let mut pq = ProductQuantizer::new(params, 32).unwrap();
        
        pq.train(&data).unwrap();
        
        // Test single vector encoding/decoding
        let test_vector = &data[0];
        let encoded = pq.encode(test_vector).unwrap();
        let decoded = pq.decode(&encoded).unwrap();
        
        assert_eq!(encoded.len(), 4);  // 4 subspaces
        assert_eq!(decoded.len(), 32); // Original dimension
        
        // Decoded vector should be approximately similar to original
        let mut mse = 0.0;
        for (orig, dec) in test_vector.iter().zip(decoded.iter()) {
            let diff = orig - dec;
            mse += diff * diff;
        }
        mse /= test_vector.len() as f32;
        
        // MSE should be reasonable (not perfect due to quantization)
        assert!(mse < 10.0); // This is a loose bound for random data
    }
    
    #[test]
    fn test_pq_batch_encoding() {
        let data = generate_random_vectors(20, 16);
        let params = PQParams::new(2, 8);
        let mut pq = ProductQuantizer::new(params, 16).unwrap();
        
        pq.train(&data).unwrap();
        
        let encoded_batch = pq.encode_batch(&data).unwrap();
        assert_eq!(encoded_batch.len(), 20);
        
        for encoded in &encoded_batch {
            assert_eq!(encoded.len(), 2); // 2 subspaces
        }
    }
    
    #[test]
    fn test_pq_error_handling() {
        let params = PQParams::new(4, 8);
        let mut pq = ProductQuantizer::new(params, 16).unwrap();
        
        // Should fail when not trained
        let test_vector = vec![1.0; 16];
        assert!(pq.encode(&test_vector).is_err());
        assert!(pq.decode(&[1, 2, 3, 4]).is_err());
        
        // Train with valid data
        let data = generate_random_vectors(10, 16);
        pq.train(&data).unwrap();
        
        // Should fail with wrong dimensions
        let wrong_dim_vector = vec![1.0; 8];
        assert!(pq.encode(&wrong_dim_vector).is_err());
        
        // Should fail with wrong code length
        assert!(pq.decode(&[1, 2]).is_err());
    }
    
    #[test]
    fn test_memory_stats() {
        let params = PQParams::new(8, 8);
        let pq = ProductQuantizer::new(params, 128).unwrap();
        
        let stats = pq.memory_stats();
        
        assert_eq!(stats.bytes_per_vector, 8);
        assert_eq!(stats.compression_ratio, (128 * 4) as f32 / 8.0); // 64x compression
        
        // Codebook size: 8 subspaces * 256 centroids * 16 dims * 4 bytes
        assert_eq!(stats.codebook_size_bytes, 8 * 256 * 16 * 4);
    }
}