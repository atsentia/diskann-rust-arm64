//! NVIDIA CUDA GPU acceleration for distance calculations
//!
//! This module provides CUDA-accelerated distance functions for NVIDIA GPUs
//! using the cudarc crate for safe Rust bindings.

use crate::{Distance, DistanceFunction, Result, Error};

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use cudarc::driver::*;
    use std::sync::Arc;

    /// NVIDIA CUDA distance calculator
    pub struct CudaDistance {
        metric: Distance,
        dimension: usize,
        device: Arc<CudaDevice>,
        l2_kernel: CudaFunction,
        cosine_kernel: CudaFunction,
        dot_kernel: CudaFunction,
    }

    impl CudaDistance {
        /// Create a new CUDA distance calculator
        pub fn new(metric: Distance, dimension: usize) -> Result<Self> {
            // Initialize CUDA device
            let device = CudaDevice::new(0)
                .map_err(|e| anyhow::anyhow!("Failed to initialize CUDA device: {}", e))?;

            // Load CUDA kernels (PTX code embedded at compile time)
            let ptx = include_str!("shaders/distance_kernels.ptx");
            device.load_ptx(ptx.into(), "distance_kernels", &["l2_distance", "cosine_distance", "dot_product"])
                .map_err(|e| anyhow::anyhow!("Failed to load CUDA kernels: {}", e))?;

            let l2_kernel = device.get_func("distance_kernels", "l2_distance")
                .map_err(|e| anyhow::anyhow!("Failed to get L2 kernel: {}", e))?;
            
            let cosine_kernel = device.get_func("distance_kernels", "cosine_distance")
                .map_err(|e| anyhow::anyhow!("Failed to get cosine kernel: {}", e))?;
            
            let dot_kernel = device.get_func("distance_kernels", "dot_product")
                .map_err(|e| anyhow::anyhow!("Failed to get dot product kernel: {}", e))?;

            Ok(Self {
                metric,
                dimension,
                device: Arc::new(device),
                l2_kernel,
                cosine_kernel,
                dot_kernel,
            })
        }

        /// Check if CUDA is available
        pub fn is_available() -> bool {
            CudaDevice::new(0).is_ok()
        }

        /// Batch distance calculation on CUDA
        pub fn batch_distance_cuda(&self, query: &[f32], points: &[f32]) -> Result<Vec<f32>> {
            let num_points = points.len() / self.dimension;
            
            // Allocate GPU memory
            let query_gpu = self.device.htod_copy(query.to_vec())
                .map_err(|e| anyhow::anyhow!("Failed to copy query to GPU: {}", e))?;
            
            let points_gpu = self.device.htod_copy(points.to_vec())
                .map_err(|e| anyhow::anyhow!("Failed to copy points to GPU: {}", e))?;
            
            let mut results_gpu = self.device.alloc_zeros::<f32>(num_points)
                .map_err(|e| anyhow::anyhow!("Failed to allocate results on GPU: {}", e))?;

            // Launch appropriate kernel
            let kernel = match self.metric {
                Distance::L2 => &self.l2_kernel,
                Distance::Cosine => &self.cosine_kernel,
                Distance::InnerProduct => &self.dot_kernel,
            };

            let block_size = 256;
            let grid_size = (num_points + block_size - 1) / block_size;

            unsafe {
                kernel.launch(LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                }, (
                    &query_gpu,
                    &points_gpu,
                    &mut results_gpu,
                    self.dimension as u32,
                    num_points as u32,
                )).map_err(|e| anyhow::anyhow!("Failed to launch CUDA kernel: {}", e))?;
            }

            // Copy results back to CPU
            let results = self.device.dtoh_sync_copy(&results_gpu)
                .map_err(|e| anyhow::anyhow!("Failed to copy results from GPU: {}", e))?;

            Ok(results)
        }

        /// Get CUDA device information
        pub fn get_device_info(&self) -> String {
            match self.device.compute_cap() {
                Ok((major, minor)) => format!("NVIDIA CUDA Device (Compute Capability {}.{})", major, minor),
                Err(_) => "NVIDIA CUDA Device (Unknown Capability)".to_string(),
            }
        }
    }

    impl DistanceFunction for CudaDistance {
        fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
            if a.len() != b.len() {
                return Err(Error::DimensionMismatch {
                    expected: a.len(),
                    actual: b.len(),
                }.into());
            }
            
            // For single vector pairs, CPU is more efficient due to GPU overhead
            crate::distance::scalar::scalar_distance(a, b, self.metric)
        }
        
        fn batch_distance(&self, query: &[f32], points: &[f32], distances: &mut [f32]) -> Result<()> {
            let dimension = self.dimension;
            let num_points = points.len() / dimension;
            
            if distances.len() != num_points {
                return Err(anyhow::anyhow!("Distances array length mismatch").into());
            }
            
            // Use CUDA for large batches (>128 vectors for efficiency)
            if num_points >= 128 {
                log::debug!("Using NVIDIA CUDA GPU for batch distance calculation");
                let results = self.batch_distance_cuda(query, points)?;
                distances.copy_from_slice(&results);
            } else {
                // Small batches on CPU
                for i in 0..num_points {
                    let start_idx = i * dimension;
                    let end_idx = start_idx + dimension;
                    distances[i] = self.distance(query, &points[start_idx..end_idx])?;
                }
            }
            
            Ok(())
        }
        
        fn metric(&self) -> Distance {
            self.metric
        }
    }
}

// Fallback implementation when CUDA feature is disabled
#[cfg(not(feature = "cuda"))]
mod cuda_impl {
    use super::*;

    pub struct CudaDistance {
        metric: Distance,
        dimension: usize,
    }

    impl CudaDistance {
        pub fn new(metric: Distance, dimension: usize) -> Result<Self> {
            Err(anyhow::anyhow!("CUDA support not enabled. Use --features cuda").into())
        }

        pub fn is_available() -> bool {
            false
        }

        pub fn get_device_info(&self) -> String {
            "CUDA support not compiled".to_string()
        }
    }

    impl DistanceFunction for CudaDistance {
        fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
            crate::distance::scalar::scalar_distance(a, b, self.metric)
        }
        
        fn batch_distance(&self, query: &[f32], points: &[f32], distances: &mut [f32]) -> Result<()> {
            let dimension = self.dimension;
            let num_points = points.len() / dimension;
            
            for i in 0..num_points {
                let start_idx = i * dimension;
                let end_idx = start_idx + dimension;
                distances[i] = self.distance(query, &points[start_idx..end_idx])?;
            }
            
            Ok(())
        }
        
        fn metric(&self) -> Distance {
            self.metric
        }
    }
}

pub use cuda_impl::CudaDistance;

#[cfg(feature = "cuda")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_cuda_availability() {
        let available = CudaDistance::is_available();
        println!("CUDA available: {}", available);
        
        if available {
            let cuda_dist = CudaDistance::new(Distance::L2, 128).unwrap();
            println!("Device info: {}", cuda_dist.get_device_info());
        }
    }
    
    #[test]
    fn test_cuda_batch_distance() {
        if !CudaDistance::is_available() {
            return; // Skip if CUDA not available
        }
        
        let vectors = generate_random_vectors(200, 128); // Large batch for GPU
        let cuda_dist = CudaDistance::new(Distance::L2, 128).unwrap();
        
        let query = &vectors[0];
        let points: Vec<f32> = vectors[1..].iter().flatten().copied().collect();
        let mut distances = vec![0.0; 199];
        
        cuda_dist.batch_distance(query, &points, &mut distances).unwrap();
        
        // Verify results are reasonable
        for &dist in &distances {
            assert!(dist >= 0.0);
            assert!(dist.is_finite());
        }
    }
}