//! Windows ML NPU Benchmark for Qualcomm Snapdragon X
//!
//! This benchmark tests the performance of DiskANN operations
//! using Windows ML with NPU acceleration on Snapdragon X processors.

use anyhow::Result;
use diskann::{Distance, create_distance_function, IndexBuilder};
use std::time::{Duration, Instant};
use console::style;

fn main() -> Result<()> {
    println!("{}", style("🚀 Windows ML NPU Benchmark for Qualcomm Snapdragon X").bold().blue());
    println!("{}", style("Testing DiskANN performance with NPU acceleration").dim());
    println!();

    // Test parameters
    let dimensions = vec![128, 256, 512, 768, 1024];
    let num_vectors = 10000;
    let num_queries = 1000;
    let k = 10;

    for dim in dimensions {
        println!("{}", style(format!("📊 Testing dimension: {}", dim)).yellow());
        
        // Generate random vectors
        let mut vectors = Vec::with_capacity(num_vectors);
        let mut queries = Vec::with_capacity(num_queries);
        
        println!("  Generating test data...");
        for _ in 0..num_vectors {
            let vec: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            vectors.push(vec);
        }
        
        for _ in 0..num_queries {
            let vec: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            queries.push(vec);
        }

        // Test 1: Distance computation performance
        println!("\n  {} Distance Computation Benchmark", style("1.").bold());
        benchmark_distance_computation(&vectors, &queries, dim)?;

        // Test 2: Index building performance
        println!("\n  {} Index Building Benchmark", style("2.").bold());
        let index = benchmark_index_building(&vectors, dim)?;

        // Test 3: Search performance
        println!("\n  {} Search Performance Benchmark", style("3.").bold());
        benchmark_search_performance(&index, &queries, k)?;

        println!("\n{}", style("─".repeat(60)).dim());
    }

    // Check NPU availability
    check_npu_status();

    Ok(())
}

fn benchmark_distance_computation(vectors: &[Vec<f32>], queries: &[Vec<f32>], dim: usize) -> Result<()> {
    let distance_fn = create_distance_function(Distance::L2, dim);
    
    // Check if using hardware acceleration
    if diskann::distance::QualcommDistance::is_available() {
        println!("    ✅ NPU acceleration available");
    } else {
        println!("    ⚠️  NPU acceleration not available, using CPU SIMD");
    }

    // Warm up
    for _ in 0..100 {
        let _ = distance_fn.distance(&vectors[0], &queries[0]);
    }

    // Benchmark single distance computation
    let mut total_time = Duration::ZERO;
    let iterations = 10000;
    
    for i in 0..iterations {
        let vec_idx = i % vectors.len();
        let query_idx = i % queries.len();
        
        let start = Instant::now();
        let _ = distance_fn.distance(&vectors[vec_idx], &queries[query_idx])?;
        total_time += start.elapsed();
    }

    let avg_time = total_time / iterations as u32;
    let ops_per_sec = 1_000_000.0 / avg_time.as_micros() as f64;

    println!("    Average time per distance: {:?}", avg_time);
    println!("    Distance computations/sec: {:.0}", ops_per_sec);

    // Benchmark batch distance computation
    let batch_size = 100;
    let start = Instant::now();
    
    for query in queries.iter().take(batch_size) {
        for vector in vectors.iter().take(batch_size) {
            let _ = distance_fn.distance(query, vector)?;
        }
    }
    
    let batch_time = start.elapsed();
    let batch_ops = (batch_size * batch_size) as f64;
    let batch_ops_per_sec = batch_ops * 1_000_000.0 / batch_time.as_micros() as f64;

    println!("    Batch computation time: {:?}", batch_time);
    println!("    Batch operations/sec: {:.0}", batch_ops_per_sec);

    Ok(())
}

fn benchmark_index_building(vectors: &[Vec<f32>], dim: usize) -> Result<Box<dyn diskann::Index>> {
    println!("    Building index with {} vectors...", vectors.len());
    
    let start = Instant::now();
    
    let index = IndexBuilder::new(dim, Distance::L2)
        .with_max_degree(64)
        .with_search_list_size(100)
        .with_alpha(1.2)
        .build_memory_index(vectors.clone())?;
    
    let build_time = start.elapsed();
    let vectors_per_sec = vectors.len() as f64 * 1000.0 / build_time.as_millis() as f64;

    println!("    Index build time: {:?}", build_time);
    println!("    Vectors indexed/sec: {:.0}", vectors_per_sec);

    Ok(index)
}

fn benchmark_search_performance(index: &dyn diskann::Index, queries: &[Vec<f32>], k: usize) -> Result<()> {
    // Warm up
    for query in queries.iter().take(10) {
        let _ = index.search(query, k)?;
    }

    // Single query latency
    let mut latencies = Vec::new();
    
    for query in queries.iter().take(100) {
        let start = Instant::now();
        let _ = index.search(query, k)?;
        latencies.push(start.elapsed());
    }

    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p90 = latencies[latencies.len() * 9 / 10];
    let p99 = latencies[latencies.len() * 99 / 100];

    println!("    Search latencies:");
    println!("      P50: {:?}", p50);
    println!("      P90: {:?}", p90);
    println!("      P99: {:?}", p99);

    // Throughput test
    let start = Instant::now();
    let mut total_queries = 0;
    
    while start.elapsed() < Duration::from_secs(1) {
        for query in queries.iter() {
            let _ = index.search(query, k)?;
            total_queries += 1;
        }
    }
    
    let qps = total_queries as f64 / start.elapsed().as_secs_f64();
    println!("    Queries per second: {:.0}", qps);

    Ok(())
}

fn check_npu_status() {
    println!("\n{}", style("🔍 NPU Status Check").bold().green());
    
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        if diskann::distance::QualcommDistance::is_available() {
            println!("✅ Qualcomm NPU detected and available");
            println!("✅ Windows ML DirectML support active");
        } else {
            println!("❌ Qualcomm NPU not detected");
            println!("⚠️  Falling back to CPU SIMD optimizations");
        }
    }
    
    #[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
    {
        println!("ℹ️  Not running on Windows ARM64 - NPU features unavailable");
    }
}