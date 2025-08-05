//! WebGPU optimized distance calculations
//!
//! This module provides GPU-accelerated distance functions using WebGPU
//! for cross-platform compatibility (NVIDIA, AMD, Intel, Apple, mobile).

use crate::{Distance, DistanceFunction, Result, Error};

#[cfg(feature = "webgpu")]
use wgpu::*;
#[cfg(feature = "webgpu")]
use bytemuck::{Pod, Zeroable};

/// WebGPU-optimized distance calculator
#[cfg(feature = "webgpu")]
pub struct WebGpuDistance {
    metric: Distance,
    dimension: usize,
    device: Device,
    queue: Queue,
    l2_pipeline: Option<ComputePipeline>,
    cosine_pipeline: Option<ComputePipeline>,
    dot_pipeline: Option<ComputePipeline>,
}

#[cfg(feature = "webgpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    dimension: u32,
    num_vectors: u32,
    _padding: [u32; 2],
}

#[cfg(feature = "webgpu")]
impl WebGpuDistance {
    /// Create a new WebGPU distance calculator
    pub async fn new(metric: Distance, dimension: usize) -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable GPU adapter"))?;
        
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("DiskANN GPU Device"),
                    features: Features::empty(),
                    limits: Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create GPU device: {}", e))?;
        
        let mut gpu_distance = Self {
            metric,
            dimension,
            device,
            queue,
            l2_pipeline: None,
            cosine_pipeline: None,
            dot_pipeline: None,
        };
        
        // Initialize compute pipelines based on metric
        gpu_distance.init_pipelines().await?;
        
        Ok(gpu_distance)
    }
    
    /// Check if WebGPU is available
    pub async fn is_available() -> bool {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_some()
    }
    
    /// Initialize compute pipelines
    async fn init_pipelines(&mut self) -> Result<()> {
        // L2 distance compute shader
        let l2_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("L2 Distance Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/l2_distance.wgsl").into()),
        });
        
        self.l2_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("L2 Distance Pipeline"),
            layout: None,
            module: &l2_shader,
            entry_point: "l2_distance_main",
        }));
        
        // Cosine distance compute shader
        let cosine_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Cosine Distance Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/cosine_distance.wgsl").into()),
        });
        
        self.cosine_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Cosine Distance Pipeline"),
            layout: None,
            module: &cosine_shader,
            entry_point: "cosine_distance_main",
        }));
        
        // Dot product compute shader
        let dot_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Dot Product Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/dot_product.wgsl").into()),
        });
        
        self.dot_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Dot Product Pipeline"),
            layout: None,
            module: &dot_shader,
            entry_point: "dot_product_main",
        }));
        
        Ok(())
    }
    
    /// Batch distance calculation on GPU
    pub async fn batch_distance_gpu(&self, query: &[f32], points: &[f32]) -> Result<Vec<f32>> {
        let num_points = points.len() / self.dimension;
        let workgroup_size = 64u32; // Optimal for most GPUs
        let num_workgroups = (num_points as u32 + workgroup_size - 1) / workgroup_size;
        
        // Create buffers
        let query_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Query Buffer"),
            contents: bytemuck::cast_slice(query),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        let points_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Points Buffer"),
            contents: bytemuck::cast_slice(points),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        let results_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Results Buffer"),
            size: (num_points * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let params = GpuParams {
            dimension: self.dimension as u32,
            num_vectors: num_points as u32,
            _padding: [0; 2],
        };
        
        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create bind group
        let bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Distance Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Distance Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: query_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: points_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Dispatch compute shader
        let pipeline = match self.metric {
            Distance::L2 => self.l2_pipeline.as_ref(),
            Distance::Cosine => self.cosine_pipeline.as_ref(), 
            Distance::InnerProduct => self.dot_pipeline.as_ref(),
        }.ok_or_else(|| anyhow::anyhow!("Pipeline not initialized for metric {:?}", self.metric))?;
        
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Distance Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Distance Compute Pass"),
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Copy results back to CPU
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (num_points * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (num_points * std::mem::size_of::<f32>()) as u64,
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Map and read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
        
        self.device.poll(Maintain::Wait);
        receiver.receive().await.unwrap().map_err(|e| anyhow::anyhow!("Failed to map buffer: {}", e))?;
        
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
    
    /// Get GPU information
    pub fn get_gpu_info(&self) -> String {
        format!("WebGPU Device: {}", self.device.features().contains(Features::empty()))
    }
}

#[cfg(feature = "webgpu")]
impl DistanceFunction for WebGpuDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(Error::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            }.into());
        }
        
        // For single vector pairs, fallback to CPU (GPU has overhead)
        crate::distance::scalar::scalar_distance(a, b, self.metric)
    }
    
    fn batch_distance(&self, query: &[f32], points: &[f32], distances: &mut [f32]) -> Result<()> {
        let dimension = self.dimension;
        let num_points = points.len() / dimension;
        
        if distances.len() != num_points {
            return Err(anyhow::anyhow!("Distances array length mismatch").into());
        }
        
        // Use GPU for batch operations (>100 vectors for efficiency)
        if num_points >= 100 {
            // This would need async context, so for now fallback to CPU
            // In real implementation, we'd use a different API design
            for i in 0..num_points {
                let start_idx = i * dimension;
                let end_idx = start_idx + dimension;
                distances[i] = self.distance(query, &points[start_idx..end_idx])?;
            }
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

// Fallback implementation when WebGPU is not available
#[cfg(not(feature = "webgpu"))]
pub struct WebGpuDistance {
    metric: Distance,
    dimension: usize,
}

#[cfg(not(feature = "webgpu"))]
impl WebGpuDistance {
    pub async fn new(metric: Distance, dimension: usize) -> Result<Self> {
        Err(anyhow::anyhow!("WebGPU support not compiled in. Enable 'webgpu' feature.").into())
    }
    
    pub async fn is_available() -> bool {
        false
    }
}

#[cfg(not(feature = "webgpu"))]
impl DistanceFunction for WebGpuDistance {
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

#[cfg(all(test, feature = "webgpu"))]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[tokio::test]
    async fn test_webgpu_availability() {
        let available = WebGpuDistance::is_available().await;
        println!("WebGPU available: {}", available);
    }
    
    #[tokio::test] 
    async fn test_webgpu_distance() {
        if !WebGpuDistance::is_available().await {
            return; // Skip if WebGPU not available
        }
        
        let vectors = generate_random_vectors(2, 128);
        let gpu_dist = WebGpuDistance::new(Distance::L2, 128).await.unwrap();
        
        let distance = gpu_dist.distance(&vectors[0], &vectors[1]).unwrap();
        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }
}