//! Platform-Specific GPU Optimization Example
//!
//! This example shows how to use platform-specific GPU features for
//! optimal performance on different systems.

use diskann::{Distance, Result};
use std::time::Instant;

#[cfg(target_os = "macos")]
mod apple_example {
    use super::*;
    use diskann::distance::apple_metal::MetalDistance;
    
    pub fn run_apple_optimized() -> Result<()> {
        println!("Running Apple Metal optimized example\n");
        
        let dimension = 768;
        let batch_size = 10000;
        
        // Create Metal distance function
        let distance_fn = MetalDistance::new(Distance::Cosine, dimension)?;
        
        // Check if Neural Engine is available
        if distance_fn.has_neural_engine() {
            println!("✓ Neural Engine detected - will use for appropriate workloads");
        } else {
            println!("✗ No Neural Engine - using GPU compute");
        }
        
        // Generate test data
        let query = vec![0.5; dimension];
        let points: Vec<f32> = (0..batch_size * dimension)
            .map(|i| (i as f32 / 100.0).sin())
            .collect();
        let mut distances = vec![0.0; batch_size];
        
        // Benchmark
        let start = Instant::now();
        distance_fn.batch_distance(&query, &points, &mut distances)?;
        let elapsed = start.elapsed();
        
        println!("Processed {} vectors in {:.3} ms", batch_size, elapsed.as_millis());
        println!("Throughput: {:.0} vectors/second", 
                 batch_size as f64 / elapsed.as_secs_f64());
        
        // Show some results
        println!("\nFirst 5 distances:");
        for (i, &dist) in distances.iter().take(5).enumerate() {
            println!("  Vector {}: {:.6}", i, dist);
        }
        
        Ok(())
    }
}

#[cfg(target_os = "windows")]
mod windows_example {
    use super::*;
    
    #[cfg(feature = "directml")]
    use diskann::distance::qualcomm_directml::QualcommDistance;
    
    #[cfg(feature = "cuda")]
    use diskann::distance::cuda::CudaDistance;
    
    pub fn run_windows_optimized() -> Result<()> {
        println!("Running Windows optimized example\n");
        
        let dimension = 1024;
        let batch_size = 50000;
        
        // Try Qualcomm Snapdragon X first (for ARM64 Windows devices)
        #[cfg(all(target_arch = "aarch64", feature = "directml"))]
        {
            match QualcommDistance::new(Distance::L2, dimension) {
                Ok(distance_fn) => {
                    println!("✓ Using Qualcomm Snapdragon X NPU");
                    return benchmark_gpu(distance_fn, dimension, batch_size);
                }
                Err(_) => println!("✗ Qualcomm NPU not available"),
            }
        }
        
        // Try NVIDIA CUDA
        #[cfg(feature = "cuda")]
        {
            match CudaDistance::new(Distance::L2, dimension) {
                Ok(distance_fn) => {
                    println!("✓ Using NVIDIA CUDA GPU");
                    return benchmark_gpu(distance_fn, dimension, batch_size);
                }
                Err(_) => println!("✗ CUDA not available"),
            }
        }
        
        // Fallback to CPU
        println!("Using CPU SIMD optimizations");
        let distance_fn = diskann::create_distance_function(Distance::L2, dimension);
        benchmark_gpu(distance_fn, dimension, batch_size)
    }
    
    fn benchmark_gpu(
        distance_fn: impl diskann::DistanceFunction,
        dimension: usize,
        batch_size: usize,
    ) -> Result<()> {
        // Generate test data
        let query = vec![1.0; dimension];
        let points: Vec<f32> = (0..batch_size * dimension)
            .map(|i| (i as f32).cos())
            .collect();
        let mut distances = vec![0.0; batch_size];
        
        // Warm-up
        distance_fn.batch_distance(&query, &points, &mut distances)?;
        
        // Benchmark
        let start = Instant::now();
        for _ in 0..10 {
            distance_fn.batch_distance(&query, &points, &mut distances)?;
        }
        let elapsed = start.elapsed();
        
        println!("Processed {} vectors x 10 iterations in {:.3} ms", 
                 batch_size, elapsed.as_millis());
        println!("Average throughput: {:.0} vectors/second", 
                 (batch_size * 10) as f64 / elapsed.as_secs_f64());
        
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux_example {
    use super::*;
    
    #[cfg(feature = "cuda")]
    use diskann::distance::cuda::CudaDistance;
    
    #[cfg(feature = "rocm")]
    use diskann::distance::rocm::RocmDistance;
    
    pub fn run_linux_optimized() -> Result<()> {
        println!("Running Linux optimized example\n");
        
        let dimension = 2048; // Large dimension to show GPU benefits
        let batch_size = 100000;
        
        // Try AMD ROCm first
        #[cfg(feature = "rocm")]
        {
            match RocmDistance::new(Distance::InnerProduct, dimension) {
                Ok(distance_fn) => {
                    println!("✓ Using AMD ROCm GPU");
                    return benchmark_gpu(distance_fn, dimension, batch_size);
                }
                Err(_) => println!("✗ ROCm not available"),
            }
        }
        
        // Try NVIDIA CUDA
        #[cfg(feature = "cuda")]
        {
            match CudaDistance::new(Distance::InnerProduct, dimension) {
                Ok(distance_fn) => {
                    println!("✓ Using NVIDIA CUDA GPU");
                    
                    // Get GPU info
                    if let Ok(device_name) = distance_fn.get_device_name() {
                        println!("GPU: {}", device_name);
                    }
                    
                    return benchmark_gpu(distance_fn, dimension, batch_size);
                }
                Err(_) => println!("✗ CUDA not available"),
            }
        }
        
        // Fallback
        println!("Using CPU optimizations");
        let distance_fn = diskann::create_distance_function(Distance::InnerProduct, dimension);
        benchmark_gpu(distance_fn, dimension, batch_size)
    }
    
    fn benchmark_gpu(
        distance_fn: impl diskann::DistanceFunction,
        dimension: usize,
        batch_size: usize,
    ) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Generate normalized random vectors
        let mut query = vec![0.0; dimension];
        let mut norm = 0.0;
        for i in 0..dimension {
            query[i] = rng.gen_range(-1.0..1.0);
            norm += query[i] * query[i];
        }
        norm = norm.sqrt();
        for i in 0..dimension {
            query[i] /= norm;
        }
        
        // Generate batch
        let points: Vec<f32> = (0..batch_size * dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let mut distances = vec![0.0; batch_size];
        
        // Benchmark with different batch sizes
        for size in [1000, 10000, 50000, 100000].iter() {
            if *size > batch_size {
                break;
            }
            
            let sub_points = &points[0..*size * dimension];
            let mut sub_distances = vec![0.0; *size];
            
            let start = Instant::now();
            distance_fn.batch_distance(&query, sub_points, &mut sub_distances)?;
            let elapsed = start.elapsed();
            
            println!("Batch size {}: {:.3} ms ({:.0} vectors/sec)", 
                     size, 
                     elapsed.as_millis(),
                     *size as f64 / elapsed.as_secs_f64());
        }
        
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("Platform-Specific GPU Optimization Example");
    println!("==========================================\n");
    
    #[cfg(target_os = "macos")]
    apple_example::run_apple_optimized()?;
    
    #[cfg(target_os = "windows")]
    windows_example::run_windows_optimized()?;
    
    #[cfg(target_os = "linux")]
    linux_example::run_linux_optimized()?;
    
    #[cfg(feature = "webgpu")]
    {
        println!("\nCross-platform WebGPU is also available!");
        println!("WebGPU provides consistent performance across all platforms.");
    }
    
    Ok(())
}