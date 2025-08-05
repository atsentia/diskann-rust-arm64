//! AMD ROCm GPU acceleration for distance calculations
//!
//! This module provides ROCm-accelerated distance functions for AMD GPUs
//! using the hip-rs crate for HIP (Heterogeneous Interface for Portability).

use crate::{Distance, DistanceFunction, Result, Error};

#[cfg(feature = "rocm")]
mod rocm_impl {
    use super::*;
    use hip_rs::{Device, Context, Module, Function, DevicePtr, MemoryAdvice};
    use std::sync::Arc;

    /// AMD ROCm distance calculator
    pub struct RocmDistance {
        metric: Distance,
        dimension: usize,
        device: Device,
        context: Context,
        l2_kernel: Function,
        cosine_kernel: Function,
        dot_kernel: Function,
    }

    impl RocmDistance {
        /// Create a new ROCm distance calculator
        pub fn new(metric: Distance, dimension: usize) -> Result<Self> {
            // Initialize ROCm device
            let device = Device::get(0)
                .map_err(|e| anyhow::anyhow!("Failed to get ROCm device: {}", e))?;
            
            let context = Context::create_for_device(&device)
                .map_err(|e| anyhow::anyhow!("Failed to create ROCm context: {}", e))?;

            // Load HIP kernels (compiled HSACO binary embedded at compile time)
            let hsaco = include_bytes!("shaders/distance_kernels.hsaco");
            let module = Module::load_from_bytes(&context, hsaco)
                .map_err(|e| anyhow::anyhow!("Failed to load ROCm kernels: {}", e))?;

            let l2_kernel = module.get_function("l2_distance")
                .map_err(|e| anyhow::anyhow!("Failed to get L2 kernel: {}", e))?;
            
            let cosine_kernel = module.get_function("cosine_distance")
                .map_err(|e| anyhow::anyhow!("Failed to get cosine kernel: {}", e))?;
            
            let dot_kernel = module.get_function("dot_product")
                .map_err(|e| anyhow::anyhow!("Failed to get dot product kernel: {}", e))?;

            Ok(Self {
                metric,
                dimension,
                device,
                context,
                l2_kernel,
                cosine_kernel,
                dot_kernel,
            })
        }

        /// Check if ROCm is available
        pub fn is_available() -> bool {
            Device::get_count().unwrap_or(0) > 0
        }

        /// Batch distance calculation on ROCm
        pub fn batch_distance_rocm(&self, query: &[f32], points: &[f32]) -> Result<Vec<f32>> {
            let num_points = points.len() / self.dimension;
            
            // Allocate GPU memory
            let query_gpu = DevicePtr::allocate(&self.context, query.len())
                .map_err(|e| anyhow::anyhow!("Failed to allocate query memory: {}", e))?;
            
            let points_gpu = DevicePtr::allocate(&self.context, points.len())
                .map_err(|e| anyhow::anyhow!("Failed to allocate points memory: {}", e))?;
            
            let results_gpu = DevicePtr::allocate(&self.context, num_points)
                .map_err(|e| anyhow::anyhow!("Failed to allocate results memory: {}", e))?;

            // Copy data to GPU
            query_gpu.copy_from_slice(query)
                .map_err(|e| anyhow::anyhow!("Failed to copy query to GPU: {}", e))?;
            
            points_gpu.copy_from_slice(points)
                .map_err(|e| anyhow::anyhow!("Failed to copy points to GPU: {}", e))?;

            // Launch appropriate kernel
            let kernel = match self.metric {
                Distance::L2 => &self.l2_kernel,
                Distance::Cosine => &self.cosine_kernel,
                Distance::InnerProduct => &self.dot_kernel,
            };

            // ROCm/HIP optimal block size for AMD GPUs (typically 64 work-items per workgroup)
            let block_size = 64;
            let grid_size = (num_points + block_size - 1) / block_size;

            kernel.launch(
                (grid_size, 1, 1),
                (block_size, 1, 1),
                0, // shared memory
                &[
                    &query_gpu as &dyn hip_rs::KernelParam,
                    &points_gpu as &dyn hip_rs::KernelParam,
                    &results_gpu as &dyn hip_rs::KernelParam,
                    &(self.dimension as u32) as &dyn hip_rs::KernelParam,
                    &(num_points as u32) as &dyn hip_rs::KernelParam,
                ]
            ).map_err(|e| anyhow::anyhow!("Failed to launch ROCm kernel: {}", e))?;

            // Synchronize and copy results back
            self.context.synchronize()
                .map_err(|e| anyhow::anyhow!("Failed to synchronize: {}", e))?;
            
            let mut results = vec![0.0f32; num_points];
            results_gpu.copy_to_slice(&mut results)
                .map_err(|e| anyhow::anyhow!("Failed to copy results from GPU: {}", e))?;

            Ok(results)
        }

        /// Get ROCm device information
        pub fn get_device_info(&self) -> String {
            match self.device.name() {
                Ok(name) => format!("AMD ROCm Device: {}", name),
                Err(_) => "AMD ROCm Device (Unknown)".to_string(),
            }
        }

        /// Check if we should use ROCm for this workload
        fn should_use_gpu(&self, batch_size: usize) -> bool {
            // ROCm overhead is higher than CUDA, need larger batches
            batch_size >= 256
        }
    }

    impl DistanceFunction for RocmDistance {
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
            
            // Use ROCm for very large batches (ROCm has higher overhead than CUDA)
            if self.should_use_gpu(num_points) {
                log::debug!("Using AMD ROCm GPU for batch distance calculation");
                let results = self.batch_distance_rocm(query, points)?;
                distances.copy_from_slice(&results);
            } else {
                // Smaller batches on CPU
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

// Fallback implementation when ROCm feature is disabled
#[cfg(not(feature = "rocm"))]
mod rocm_impl {
    use super::*;

    pub struct RocmDistance {
        metric: Distance,
        dimension: usize,
    }

    impl RocmDistance {
        pub fn new(metric: Distance, dimension: usize) -> Result<Self> {
            Err(anyhow::anyhow!("ROCm support not enabled. Use --features rocm").into())
        }

        pub fn is_available() -> bool {
            false
        }

        pub fn get_device_info(&self) -> String {
            "ROCm support not compiled".to_string()
        }
    }

    impl DistanceFunction for RocmDistance {
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

pub use rocm_impl::RocmDistance;

#[cfg(feature = "rocm")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_rocm_availability() {
        let available = RocmDistance::is_available();
        println!("ROCm available: {}", available);
        
        if available {
            let rocm_dist = RocmDistance::new(Distance::L2, 128).unwrap();
            println!("Device info: {}", rocm_dist.get_device_info());
        }
    }
    
    #[test]
    fn test_rocm_batch_distance() {
        if !RocmDistance::is_available() {
            return; // Skip if ROCm not available
        }
        
        let vectors = generate_random_vectors(300, 128); // Large batch for GPU
        let rocm_dist = RocmDistance::new(Distance::L2, 128).unwrap();
        
        let query = &vectors[0];
        let points: Vec<f32> = vectors[1..].iter().flatten().copied().collect();
        let mut distances = vec![0.0; 299];
        
        rocm_dist.batch_distance(query, &points, &mut distances).unwrap();
        
        // Verify results are reasonable
        for &dist in &distances {
            assert!(dist >= 0.0);
            assert!(dist.is_finite());
        }
    }
}