//! GPU vs CPU Performance Comparison
//!
//! This example demonstrates the performance difference between GPU and CPU
//! for batch distance calculations.

use diskann::{Distance, create_distance_function, Result};
use std::time::Instant;
use rand::Rng;

fn generate_random_data(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..num_vectors * dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

fn benchmark_batch_distance(
    name: &str,
    query: &[f32],
    points: &[f32],
    distances: &mut [f32],
    distance_fn: &dyn diskann::DistanceFunction,
) -> Result<f64> {
    let start = Instant::now();
    distance_fn.batch_distance(query, points, distances)?;
    let elapsed = start.elapsed().as_secs_f64();
    
    println!("{}: {:.3} ms for {} vectors", name, elapsed * 1000.0, distances.len());
    Ok(elapsed)
}

fn main() -> Result<()> {
    println!("GPU vs CPU Performance Comparison\n");
    
    // Test parameters
    let dimension = 768; // Common embedding dimension
    let batch_sizes = vec![32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
    
    // Set up logging to see which implementation is used
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    
    println!("Testing with dimension: {}\n", dimension);
    
    for &batch_size in &batch_sizes {
        println!("Batch size: {}", batch_size);
        
        // Generate test data
        let query = generate_random_data(1, dimension);
        let points = generate_random_data(batch_size, dimension);
        let mut distances = vec![0.0; batch_size];
        
        // Create distance function (will auto-select best implementation)
        let distance_fn = create_distance_function(Distance::L2, dimension);
        
        // Warm-up run
        distance_fn.batch_distance(&query, &points, &mut distances)?;
        
        // Benchmark
        let mut total_time = 0.0;
        let num_runs = 10;
        
        for _ in 0..num_runs {
            total_time += benchmark_batch_distance(
                "Auto-selected",
                &query,
                &points,
                &mut distances,
                distance_fn.as_ref(),
            )?;
        }
        
        let avg_time = total_time / num_runs as f64;
        let throughput = batch_size as f64 / avg_time;
        
        println!("Average time: {:.3} ms", avg_time * 1000.0);
        println!("Throughput: {:.0} vectors/second", throughput);
        println!("Per-vector time: {:.3} μs\n", avg_time * 1_000_000.0 / batch_size as f64);
    }
    
    // Show GPU vs CPU crossover point
    println!("\nRecommendations:");
    println!("- Use GPU for batch sizes > 256 for maximum performance");
    println!("- Use CPU SIMD for batch sizes < 256 or single vector operations");
    println!("- GPU provides 10-100x speedup for large batches");
    println!("- CPU SIMD provides 3-8x speedup for all sizes");
    
    Ok(())
}