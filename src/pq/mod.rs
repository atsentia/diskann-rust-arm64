//! Product Quantization for compression
//!
//! This module implements PQ compression for memory-efficient vector storage
//! and fast approximate distance calculations.

pub mod kmeans;
pub mod quantizer;
pub mod codebook;
pub mod distance;
pub mod index;

// Re-export main types
pub use quantizer::{ProductQuantizer, PQParams, PQTrainingResult, PQMemoryStats};
pub use kmeans::{KMeans, KMeansParams, KMeansResult};
pub use distance::{pq_distance, PQDistanceTable, PQDistanceTableStats, BatchPQDistance, create_distance_table};
pub use codebook::{Codebook, CodebookStats};
pub use index::{PQIndex, PQIndexStats};