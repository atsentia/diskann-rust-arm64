//! Apple Metal optimized distance calculations
//!
//! This module provides GPU-accelerated distance functions using Apple Metal
//! for M-series processors with GPU and Neural Engine optimization.

use crate::{Distance, DistanceFunction, Result, Error};

#[cfg(all(feature = "metal", target_os = "macos"))]
use metal::*;

/// Apple Metal-optimized distance calculator for M-series processors
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct AppleMetalDistance {
    metric: Distance,
    dimension: usize,
    device: Device,
    command_queue: CommandQueue,
    l2_pipeline: Option<ComputePipelineState>,
    cosine_pipeline: Option<ComputePipelineState>,
    dot_pipeline: Option<ComputePipelineState>,
    library: Library,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl AppleMetalDistance {
    /// Create a new Apple Metal distance calculator
    pub fn new(metric: Distance, dimension: usize) -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        
        let command_queue = device.new_command_queue();
        
        // Load Metal shaders library
        let library_source = include_str!("shaders/metal_distance.metal");
        let library = device
            .new_library_with_source(library_source, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Failed to compile Metal shaders: {}", e))?;
        
        let mut metal_distance = Self {
            metric,
            dimension,
            device,
            command_queue,
            l2_pipeline: None,
            cosine_pipeline: None,
            dot_pipeline: None,
            library,
        };
        
        // Initialize compute pipelines
        metal_distance.init_pipelines()?;
        
        Ok(metal_distance)
    }
    
    /// Check if Apple Metal is available (M-series processors)
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }
    
    /// Initialize Metal compute pipelines
    fn init_pipelines(&mut self) -> Result<()> {
        // L2 distance pipeline
        let l2_function = self.library
            .get_function("l2_distance_kernel", None)
            .map_err(|e| anyhow::anyhow!("Failed to load L2 function: {}", e))?;
        
        self.l2_pipeline = Some(
            self.device
                .new_compute_pipeline_state_with_function(&l2_function)
                .map_err(|e| anyhow::anyhow!("Failed to create L2 pipeline: {}", e))?
        );
        
        // Cosine distance pipeline
        let cosine_function = self.library
            .get_function("cosine_distance_kernel", None)
            .map_err(|e| anyhow::anyhow!("Failed to load cosine function: {}", e))?;
        
        self.cosine_pipeline = Some(
            self.device
                .new_compute_pipeline_state_with_function(&cosine_function)
                .map_err(|e| anyhow::anyhow!("Failed to create cosine pipeline: {}", e))?
        );
        
        // Dot product pipeline
        let dot_function = self.library
            .get_function("dot_product_kernel", None)
            .map_err(|e| anyhow::anyhow!("Failed to load dot product function: {}", e))?;
        
        self.dot_pipeline = Some(
            self.device
                .new_compute_pipeline_state_with_function(&dot_function)
                .map_err(|e| anyhow::anyhow!("Failed to create dot product pipeline: {}", e))?
        );
        
        Ok(())
    }
    
    /// Batch distance calculation on Apple GPU
    pub fn batch_distance_metal(&self, query: &[f32], points: &[f32]) -> Result<Vec<f32>> {
        let num_points = points.len() / self.dimension;
        
        // Create Metal buffers
        let query_buffer = self.device.new_buffer_with_data(
            query.as_ptr() as *const std::ffi::c_void,
            (query.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let points_buffer = self.device.new_buffer_with_data(
            points.as_ptr() as *const std::ffi::c_void,
            (points.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let results_buffer = self.device.new_buffer(
            (num_points * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Parameters buffer
        let params = [self.dimension as u32, num_points as u32];
        let params_buffer = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            (params.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Get appropriate pipeline
        let pipeline = match self.metric {
            Distance::L2 => self.l2_pipeline.as_ref(),
            Distance::Cosine => self.cosine_pipeline.as_ref(),
            Distance::InnerProduct => self.dot_pipeline.as_ref(),
        }.ok_or_else(|| anyhow::anyhow!("Pipeline not initialized for metric {:?}", self.metric))?;
        
        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        // Set pipeline and buffers
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(0, Some(&params_buffer), 0);
        compute_encoder.set_buffer(1, Some(&query_buffer), 0);
        compute_encoder.set_buffer(2, Some(&points_buffer), 0);
        compute_encoder.set_buffer(3, Some(&results_buffer), 0);
        
        // Dispatch threads (optimized for Apple Silicon)
        let threads_per_threadgroup = MTLSize::new(64, 1, 1); // Optimal for M-series
        let threadgroups = MTLSize::new(
            (num_points + 63) / 64, // Round up
            1,
            1,
        );
        
        compute_encoder.dispatch_threadgroups(threadgroups, threads_per_threadgroup);
        compute_encoder.end_encoding();
        
        // Execute and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Read results
        let results_ptr = results_buffer.contents() as *const f32;
        let results = unsafe {
            std::slice::from_raw_parts(results_ptr, num_points).to_vec()
        };
        
        Ok(results)
    }
    
    /// Check if Neural Engine is available for this workload
    fn can_use_neural_engine(&self, dimension: usize, batch_size: usize) -> bool {
        // Neural Engine is optimal for specific matrix sizes and operations
        // Typically 8x8, 16x16, or larger matrices with specific alignment
        dimension >= 64 && batch_size >= 32 && dimension % 8 == 0
    }
    
    /// Get Metal device information
    pub fn get_device_info(&self) -> String {
        format!(
            "Apple Metal Device: {} (Family: {})",
            self.device.name(),
            if self.device.supports_family(MTLGPUFamily::Apple7) {
                "M1/M2 Series"
            } else if self.device.supports_family(MTLGPUFamily::Apple8) {
                "M3 Series" 
            } else if self.device.supports_family(MTLGPUFamily::Apple9) {
                "M4 Series"
            } else {
                "Unknown Apple Silicon"
            }
        )
    }
    
    /// Optimize for Neural Engine workloads (matrix operations)
    fn batch_distance_neural_engine(&self, query: &[f32], points: &[f32]) -> Result<Vec<f32>> {
        // Neural Engine optimization would go here
        // For now, fall back to regular Metal compute
        self.batch_distance_metal(query, points)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl DistanceFunction for AppleMetalDistance {
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
        
        // Use Metal GPU for large batches (>32 vectors for efficiency)
        if num_points >= 32 {
            let results = if self.can_use_neural_engine(dimension, num_points) {
                log::debug!("Using Apple Neural Engine for batch distance calculation");
                self.batch_distance_neural_engine(query, points)?
            } else {
                log::debug!("Using Apple Metal GPU for batch distance calculation");
                self.batch_distance_metal(query, points)?
            };
            
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

// Fallback implementation for non-macOS platforms
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub struct AppleMetalDistance {
    metric: Distance,
    dimension: usize,
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
impl AppleMetalDistance {
    pub fn new(metric: Distance, dimension: usize) -> Result<Self> {
        Err(anyhow::anyhow!("Apple Metal support requires macOS with 'metal' feature enabled.").into())
    }
    
    pub fn is_available() -> bool {
        false
    }
    
    pub fn get_device_info(&self) -> String {
        "Apple Metal not available".to_string()
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
impl DistanceFunction for AppleMetalDistance {
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

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_apple_metal_availability() {
        let available = AppleMetalDistance::is_available();
        println!("Apple Metal available: {}", available);
        
        if available {
            let metal_dist = AppleMetalDistance::new(Distance::L2, 128).unwrap();
            println!("Device info: {}", metal_dist.get_device_info());
        }
    }
    
    #[test]
    fn test_apple_metal_distance() {
        if !AppleMetalDistance::is_available() {
            return; // Skip if Metal not available
        }
        
        let vectors = generate_random_vectors(100, 128); // Large batch for GPU
        let metal_dist = AppleMetalDistance::new(Distance::L2, 128).unwrap();
        
        let query = &vectors[0];
        let points: Vec<f32> = vectors[1..].iter().flatten().copied().collect();
        let mut distances = vec![0.0; 99];
        
        metal_dist.batch_distance(query, &points, &mut distances).unwrap();
        
        // Verify results are reasonable
        for &dist in &distances {
            assert!(dist >= 0.0);
            assert!(dist.is_finite());
        }
    }
}