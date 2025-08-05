//! Comprehensive benchmark example using proper Rust measurement practices

use diskann::{Distance, IndexBuilder};
use diskann::utils::metrics::{Timer, Metrics, benchmark, measure_throughput};
use rand::prelude::*;
use std::hint::black_box;

fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DiskANN Rust Benchmark - Following Best Practices\n");
    
    // Test parameters
    let dimension = 768; // Common embedding dimension
    let num_vectors = 10_000;
    let num_queries = 1_000;
    let k = 10;
    
    println!("Configuration:");
    println!("  Dimension: {}", dimension);
    println!("  Vectors: {}", num_vectors);
    println!("  Queries: {}", num_queries);
    println!("  K: {}\n", k);
    
    // Generate test data
    println!("Generating test data...");
    let timer = Timer::new();
    let vectors = generate_random_vectors(num_vectors, dimension);
    let queries = generate_random_vectors(num_queries, dimension);
    println!("Data generation took: {:?}\n", timer.elapsed());
    
    // Benchmark index building
    println!("=== Index Build Benchmark ===");
    let build_stats = benchmark("Index Build", 1, 5, || {
        IndexBuilder::new()
            .dimensions(dimension)
            .metric(Distance::L2)
            .max_degree(32)
            .search_list_size(100)
            .alpha(1.2)
            .build_from_vectors(vectors.clone())
            .unwrap()
    });
    
    let points_per_sec = num_vectors as f64 / build_stats.mean.as_secs_f64();
    println!("Build rate: {:.0} points/sec\n", points_per_sec);
    
    // Build index for search benchmarks
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .max_degree(32)
        .search_list_size(100)
        .build_from_vectors(vectors.clone())?;
    
    // Benchmark search with proper timing
    println!("=== Search Benchmark ===");
    
    // Individual search timing
    let mut search_metrics = Metrics::new();
    for query in queries.iter().take(100) {
        search_metrics.measure(|| {
            black_box(index.search(query, k).unwrap())
        });
    }
    
    let search_stats = search_metrics.stats();
    search_stats.print("Individual Search");
    
    // Throughput measurement
    println!("\n=== Throughput Measurement ===");
    let mut query_idx = 0;
    let throughput = measure_throughput("Search", 1, 5, || {
        let query = &queries[query_idx % queries.len()];
        black_box(index.search(query, k).unwrap());
        query_idx += 1;
    });
    
    // Batch search simulation
    println!("\n=== Batch Search Benchmark ===");
    let batch_sizes = vec![1, 10, 100];
    
    for &batch_size in &batch_sizes {
        let batch_stats = benchmark(&format!("Batch Size {}", batch_size), 10, 100, || {
            for i in 0..batch_size {
                let query = &queries[i % queries.len()];
                black_box(index.search(query, k).unwrap());
            }
        });
        
        let qps = batch_size as f64 / batch_stats.mean.as_secs_f64();
        println!("  Effective QPS: {:.0}", qps);
    }
    
    // Memory usage estimation
    println!("\n=== Memory Usage ===");
    let vector_memory = num_vectors * dimension * 4; // f32 = 4 bytes
    let graph_memory_estimate = num_vectors * 32 * 8; // avg degree * pointer size
    let total_memory = vector_memory + graph_memory_estimate;
    
    println!("  Vectors: {:.2} MB", vector_memory as f64 / 1_048_576.0);
    println!("  Graph (estimated): {:.2} MB", graph_memory_estimate as f64 / 1_048_576.0);
    println!("  Total (estimated): {:.2} MB", total_memory as f64 / 1_048_576.0);
    println!("  Per vector: {:.2} KB", total_memory as f64 / num_vectors as f64 / 1024.0);
    
    // Distance function micro-benchmark
    println!("\n=== Distance Function Micro-benchmark ===");
    use diskann::distance::create_distance_function;
    
    let distance_fn = create_distance_function(Distance::L2, dimension);
    let v1 = &vectors[0];
    let v2 = &vectors[1];
    
    let dist_stats = benchmark("L2 Distance", 10_000, 100_000, || {
        black_box(distance_fn.distance(v1, v2).unwrap())
    });
    
    let ops_per_sec = 1.0 / dist_stats.mean.as_secs_f64();
    println!("  Operations/sec: {:.0}", ops_per_sec);
    println!("  Nanoseconds/op: {:.0}", dist_stats.mean.as_nanos());
    
    // Platform capabilities
    println!("\n=== Platform Capabilities ===");
    println!("  ARM64 NEON: {}", if diskann::has_neon_support() { "✓" } else { "✗" });
    println!("  x86-64 AVX2: {}", if diskann::has_avx2_support() { "✓" } else { "✗" });
    
    #[cfg(target_arch = "aarch64")]
    {
        // ARM64 specific info
        println!("  Target: ARM64");
        println!("  SIMD: NEON enabled");
    }
    
    #[cfg(target_arch = "x86_64")]
    {
        // x86-64 specific info
        println!("  Target: x86-64");
        if is_x86_feature_detected!("avx2") {
            println!("  SIMD: AVX2 available");
        } else if is_x86_feature_detected!("sse4.1") {
            println!("  SIMD: SSE4.1 available");
        }
    }
    
    Ok(())
}