//! SIMD Performance Benchmark
//!
//! This benchmark compares the performance of different SIMD implementations
//! for distance calculations on M2 ARM64.

use diskann::{Distance, distance::create_distance_function};
use std::time::Instant;
use rand::Rng;

fn main() {
    println!("DiskANN SIMD Performance Benchmark - M2 ARM64");
    println!("===============================================");
    
    // Test different vector dimensions
    let dimensions = vec![64, 128, 256, 512, 768, 1024];
    let num_iterations = 1_000_000;
    
    println!("CPU Features:");
    println!("  ARM64 NEON: {}", diskann::has_neon_support());
    println!("  x86-64 AVX2: {}", diskann::has_avx2_support());
    println!("  x86-64 AVX-512: {}", diskann::has_avx512_support());
    println!();
    
    for &dim in &dimensions {
        println!("=== Dimension: {} ===", dim);
        
        // Generate test vectors
        let vec_a = generate_random_vector(dim);
        let vec_b = generate_random_vector(dim);
        
        // Test L2 distance
        benchmark_distance("L2", Distance::L2, &vec_a, &vec_b, dim, num_iterations);
        
        // Test Cosine distance
        benchmark_distance("Cosine", Distance::Cosine, &vec_a, &vec_b, dim, num_iterations);
        
        // Test Inner Product
        benchmark_distance("Inner Product", Distance::InnerProduct, &vec_a, &vec_b, dim, num_iterations);
        
        println!();
    }
    
    println!("=== Performance Summary ===");
    println!("Expected ARM64 NEON speedup: 3-5x over scalar");
    println!("L2 distance typically fastest, followed by Inner Product, then Cosine");
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
    
    // Warmup
    for _ in 0..1000 {
        total_distance += distance_fn.distance(vec_a, vec_b).unwrap();
    }
    
    let start = Instant::now();
    
    for _ in 0..iterations {
        let dist = distance_fn.distance(vec_a, vec_b).unwrap();
        total_distance += dist; // Prevent optimization
    }
    
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;
    let ops_per_sec = 1_000_000_000.0 / ns_per_call;
    
    println!(
        "  {}: {:.2} ns/call, {:.1}M ops/sec (avg_dist: {:.6})",
        name, ns_per_call, ops_per_sec / 1_000_000.0, total_distance / (iterations + 1000) as f32
    );
}

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}