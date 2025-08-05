//! Multi-GPU Usage Example
//!
//! This example demonstrates how to use multiple GPUs for parallel processing
//! of different batches.

use diskann::{Distance, Result};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use rand::Rng;

#[cfg(feature = "cuda")]
use diskann::distance::cuda::CudaDistance;

#[cfg(feature = "metal")]
use diskann::distance::apple_metal::MetalDistance;

fn generate_random_data(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..num_vectors * dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

fn process_batch_on_gpu(
    gpu_id: usize,
    batch_id: usize,
    query: Arc<Vec<f32>>,
    points: Vec<f32>,
    dimension: usize,
) -> Result<(usize, Vec<f32>, f64)> {
    let start = Instant::now();
    let batch_size = points.len() / dimension;
    let mut distances = vec![0.0; batch_size];
    
    // Create GPU-specific distance function
    #[cfg(feature = "cuda")]
    {
        // For CUDA, you might specify device ID
        let distance_fn = CudaDistance::new(Distance::L2, dimension)?;
        distance_fn.batch_distance(&query, &points, &mut distances)?;
    }
    
    #[cfg(feature = "metal")]
    {
        // For Metal, it automatically uses the best available device
        let distance_fn = MetalDistance::new(Distance::L2, dimension)?;
        distance_fn.batch_distance(&query, &points, &mut distances)?;
    }
    
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        // Fallback to CPU
        let distance_fn = diskann::create_distance_function(Distance::L2, dimension);
        distance_fn.batch_distance(&query, &points, &mut distances)?;
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    Ok((batch_id, distances, elapsed))
}

fn main() -> Result<()> {
    println!("Multi-GPU Processing Example\n");
    
    // Parameters
    let dimension = 1024;
    let batch_size = 10000;
    let num_batches = 8;
    
    // Generate query
    let query = Arc::new(generate_random_data(1, dimension));
    
    // Generate batches
    let batches: Vec<Vec<f32>> = (0..num_batches)
        .map(|_| generate_random_data(batch_size, dimension))
        .collect();
    
    println!("Processing {} batches of {} vectors each", num_batches, batch_size);
    println!("Vector dimension: {}\n", dimension);
    
    // Process batches in parallel using multiple GPUs/threads
    let handles: Vec<_> = batches
        .into_iter()
        .enumerate()
        .map(|(batch_id, points)| {
            let query_clone = Arc::clone(&query);
            thread::spawn(move || {
                process_batch_on_gpu(
                    batch_id % 2, // Alternate between 2 GPUs if available
                    batch_id,
                    query_clone,
                    points,
                    dimension,
                )
            })
        })
        .collect();
    
    // Collect results
    let mut total_time = 0.0;
    let mut results = Vec::new();
    
    for handle in handles {
        let (batch_id, distances, elapsed) = handle.join().unwrap()?;
        println!("Batch {} completed in {:.3} ms", batch_id, elapsed * 1000.0);
        total_time += elapsed;
        results.push((batch_id, distances));
    }
    
    // Sort results by batch ID
    results.sort_by_key(|r| r.0);
    
    println!("\nSummary:");
    println!("Total processing time: {:.3} ms", total_time * 1000.0);
    println!("Average time per batch: {:.3} ms", total_time * 1000.0 / num_batches as f64);
    println!("Total vectors processed: {}", batch_size * num_batches);
    println!("Throughput: {:.0} vectors/second", 
             (batch_size * num_batches) as f64 / total_time);
    
    // Verify results
    let first_distances = &results[0].1;
    let mut max_diff = 0.0f32;
    
    for (_, distances) in results.iter().skip(1) {
        for (d1, d2) in first_distances.iter().zip(distances.iter()) {
            max_diff = max_diff.max((d1 - d2).abs());
        }
    }
    
    println!("\nResult verification:");
    println!("Maximum difference between batches: {:.6}", max_diff);
    println!("Results are {}", if max_diff < 1e-4 { "consistent ✓" } else { "inconsistent ✗" });
    
    Ok(())
}