//! SIMD Performance Benchmark
//!
//! This benchmark compares the performance of different SIMD implementations
//! for distance calculations.

use diskann::{Distance, distance::create_distance_function, utils::generate_random_vectors};
use std::time::Instant;

fn main() {
    // Initialize logging to see SIMD selection
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
    
    println!("DiskANN SIMD Performance Benchmark");
    println!("===================================");
    
    // Test different vector dimensions
    let dimensions = vec![64, 128, 256, 512, 1024];
    let num_iterations = 100_000;
    
    println!("CPU Features:");
    println!("  ARM64 NEON: {}", diskann::has_neon_support());
    println!("  x86-64 AVX2: {}", diskann::has_avx2_support());
    println!("  x86-64 AVX-512: {}", diskann::has_avx512_support());
    println!();
    
    // Create a distance function to trigger SIMD selection logging  
    println!("SIMD Implementation Selection:");
    let _test_fn = create_distance_function(Distance::L2, 128);
    println!();
    
    for &dim in &dimensions {
        println!("Dimension: {}", dim);
        
        // Generate test vectors
        let vectors = generate_random_vectors(2, dim);
        let vec_a = &vectors[0];
        let vec_b = &vectors[1];
        
        // Test L2 distance
        benchmark_distance("L2", Distance::L2, vec_a, vec_b, dim, num_iterations);
        
        // Test Cosine distance
        benchmark_distance("Cosine", Distance::Cosine, vec_a, vec_b, dim, num_iterations);
        
        // Test Inner Product
        benchmark_distance("Inner Product", Distance::InnerProduct, vec_a, vec_b, dim, num_iterations);
        
        println!();
    }
}

fn benchmark_distance(
    name: &str,
    metric: Distance, 
    vec_a: &[f32], 
    vec_b: &[f32], 
    dimension: usize,
    iterations: usize
) {
    let distance_fn = create_distance_function(metric, dimension);
    let mut total_distance = 0.0;
    
    let start = Instant::now();
    
    for _ in 0..iterations {
        let dist = distance_fn.distance(vec_a, vec_b).unwrap();
        total_distance += dist; // Prevent optimization
    }
    
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;
    let ops_per_sec = 1_000_000_000.0 / ns_per_call;
    
    println!(
        "  {}: {:.2} ns/call, {:.0} ops/sec, avg_dist: {:.6}",
        name, ns_per_call, ops_per_sec, total_distance / iterations as f32
    );
}