//! Microsoft DiskANN Rust API compatibility layer
//!
//! This module provides an API compatible with Microsoft's Rust implementation
//! of DiskANN, allowing easy migration from the Microsoft version to our pure
//! Rust implementation.
//!
//! The API is designed to match the structure and naming conventions of the
//! Microsoft DiskANN Rust implementation while internally using our optimized
//! pure Rust implementation.

pub mod common;
pub mod index;
pub mod model;
pub mod utils;

// Internal vector module for Microsoft compatibility
pub mod vector;

// Re-export key types for convenience
pub use common::{ANNResult, ANNError};
pub use index::{create_inmem_index, create_disk_index, ANNInmemIndex, ANNDiskIndex};
pub use model::{IndexConfiguration, IndexWriteParameters, IndexWriteParametersBuilder, DiskIndexBuildParameters};

// Re-export vector types that Microsoft DiskANN uses
pub use vector::{Metric, FullPrecisionDistance, Half};