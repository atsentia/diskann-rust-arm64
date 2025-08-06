use anyhow::Result;
use diskann_rust::{
    distance::{create_distance_function, Distance},
    graph::{VamanaBuilder, VamanaOptimized},
    index::{DynamicMemoryIndex, MemoryIndex},
    utils::{
        data::DataGenerator,
        timer::Timer,
    },
};
use rand::prelude::*;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    println!("=== DiskANN Rust Comprehensive Benchmark ===");
    println!("Platform: Ampere ARM64 (Neoverse-N1)");
    println!("Cores: 8");
    println!("Date: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
    println!();

    // Test configurations
    let dimensions = vec![64, 128, 256, 512, 768, 1024];
    let dataset_sizes = vec![1000, 5000, 10000, 25000, 50000];
    let num_queries = 100;

    // Run SIMD benchmarks
    simd_benchmarks(&dimensions)?;
    
    // Run graph construction benchmarks
    graph_construction_benchmarks(&dataset_sizes)?;
    
    // Run search benchmarks
    search_benchmarks(&dataset_sizes)?;
    
    // Run memory vs optimized comparison
    implementation_comparison()?;

    Ok(())
}

fn simd_benchmarks(dimensions: &[usize]) -> Result<()> {
    println!("=== SIMD Distance Function Benchmarks ===");
    println!("Testing L2, Inner Product, and Cosine distances");
    println!();

    for &dim in dimensions {
        println!("Dimension: {}", dim);
        
        // Generate test vectors
        let mut rng = thread_rng();
        let vec1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let vec2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        // Test each distance type
        for distance_type in &[Distance::L2, Distance::InnerProduct, Distance::Cosine] {
            let distance_fn = create_distance_function(*distance_type, dim);
            
            // Warmup
            for _ in 0..1000 {
                let _ = distance_fn.distance(&vec1, &vec2);
            }
            
            // Benchmark
            let iterations = 1_000_000;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = distance_fn.distance(&vec1, &vec2);
            }
            let elapsed = start.elapsed();
            
            let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
            println!(
                "  {:?}: {:.2}M ops/sec ({:.2} ns/op)",
                distance_type,
                ops_per_sec / 1_000_000.0,
                elapsed.as_nanos() as f64 / iterations as f64
            );
        }
        println!();
    }
    
    Ok(())
}

fn graph_construction_benchmarks(dataset_sizes: &[usize]) -> Result<()> {
    println!("=== Graph Construction Benchmarks ===");
    println!("Testing Vamana graph building with different dataset sizes");
    println!();
    
    let dim = 128;
    let max_degree = 64;
    let alpha = 1.2;
    let num_threads = 8;
    
    for &size in dataset_sizes {
        println!("Dataset size: {} vectors ({}D)", size, dim);
        
        // Generate dataset
        let data = DataGenerator::new()
            .with_seed(42)
            .generate_uniform(size, dim, -1.0, 1.0);
        
        // Test standard Vamana
        {
            let mut timer = Timer::new();
            timer.start();
            
            let mut builder = VamanaBuilder::new(data.clone(), Distance::L2);
            builder
                .max_degree(max_degree)
                .alpha(alpha)
                .num_threads(num_threads)
                .build_graph();
            
            let elapsed = timer.elapsed();
            let points_per_sec = size as f64 / elapsed;
            
            println!("  Standard Vamana: {:.2}s ({:.0} points/sec)", elapsed, points_per_sec);
        }
        
        // Test optimized Vamana
        {
            let mut timer = Timer::new();
            timer.start();
            
            let mut builder = VamanaOptimized::new(data.clone(), Distance::L2);
            builder
                .with_max_degree(max_degree)
                .with_alpha(alpha)
                .with_num_threads(num_threads)
                .build();
            
            let elapsed = timer.elapsed();
            let points_per_sec = size as f64 / elapsed;
            
            println!("  Optimized Vamana: {:.2}s ({:.0} points/sec)", elapsed, points_per_sec);
        }
        
        println!();
        
        // Stop if build time exceeds 60 seconds
        if size >= 25000 {
            println!("  (Skipping larger sizes to keep benchmark under 60s)");
            break;
        }
    }
    
    Ok(())
}

fn search_benchmarks(dataset_sizes: &[usize]) -> Result<()> {
    println!("=== Search Performance Benchmarks ===");
    println!("Testing search QPS and latency");
    println!();
    
    let dim = 128;
    let num_queries = 100;
    let k = 10;
    
    for &size in dataset_sizes.iter().take(3) {  // Only test smaller sizes for search
        println!("Dataset size: {} vectors", size);
        
        // Build index
        let data = DataGenerator::new()
            .with_seed(42)
            .generate_uniform(size, dim, -1.0, 1.0);
        
        let index = MemoryIndex::build(
            data,
            Distance::L2,
            64,  // max_degree
            1.2, // alpha
            Some(8), // num_threads
        )?;
        
        // Generate queries
        let queries = DataGenerator::new()
            .with_seed(123)
            .generate_uniform(num_queries, dim, -1.0, 1.0);
        
        // Warmup
        for query in queries.iter().take(10) {
            let _ = index.search(query, k, 100);
        }
        
        // Benchmark single queries
        let mut latencies = Vec::new();
        let start = Instant::now();
        
        for query in &queries {
            let query_start = Instant::now();
            let _ = index.search(query, k, 100);
            latencies.push(query_start.elapsed());
        }
        
        let total_elapsed = start.elapsed();
        let qps = num_queries as f64 / total_elapsed.as_secs_f64();
        
        // Calculate latency percentiles
        latencies.sort();
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[latencies.len() * 99 / 100];
        
        println!("  Single queries:");
        println!("    QPS: {:.0}", qps);
        println!("    Latency P50: {:.2} µs", p50.as_micros());
        println!("    Latency P99: {:.2} µs", p99.as_micros());
        
        // Benchmark batch queries
        let batch_size = 10;
        let batch_start = Instant::now();
        
        for batch_start_idx in (0..num_queries).step_by(batch_size) {
            let batch_end = (batch_start_idx + batch_size).min(num_queries);
            for i in batch_start_idx..batch_end {
                let _ = index.search(&queries[i], k, 100);
            }
        }
        
        let batch_elapsed = batch_start.elapsed();
        let batch_qps = num_queries as f64 / batch_elapsed.as_secs_f64();
        
        println!("  Batch queries (size={}): {:.0} QPS", batch_size, batch_qps);
        println!();
    }
    
    Ok(())
}

fn implementation_comparison() -> Result<()> {
    println!("=== Implementation Comparison ===");
    println!("Comparing different Vamana implementations");
    println!();
    
    let sizes = vec![5000, 10000];
    let dim = 128;
    
    for size in sizes {
        println!("Dataset: {} vectors, {}D", size, dim);
        
        let data = DataGenerator::new()
            .with_seed(42)
            .generate_uniform(size, dim, -1.0, 1.0);
        
        // Standard Vamana
        let standard_start = Instant::now();
        let standard_graph = VamanaBuilder::new(data.clone(), Distance::L2)
            .max_degree(64)
            .alpha(1.2)
            .num_threads(8)
            .build_graph();
        let standard_time = standard_start.elapsed();
        
        // Optimized Vamana
        let optimized_start = Instant::now();
        let optimized_graph = VamanaOptimized::new(data.clone(), Distance::L2)
            .with_max_degree(64)
            .with_alpha(1.2)
            .with_num_threads(8)
            .build();
        let optimized_time = optimized_start.elapsed();
        
        // Calculate speedup
        let speedup = standard_time.as_secs_f64() / optimized_time.as_secs_f64();
        
        println!("  Standard:  {:.2}s", standard_time.as_secs_f64());
        println!("  Optimized: {:.2}s", optimized_time.as_secs_f64());
        println!("  Speedup:   {:.2}x", speedup);
        
        // Compare graph quality
        let standard_edges: usize = standard_graph.adjacency_list.iter()
            .map(|neighbors| neighbors.len())
            .sum();
        let optimized_edges: usize = optimized_graph.graph.iter()
            .map(|neighbors| neighbors.len())
            .sum();
        
        println!("  Graph edges - Standard: {}, Optimized: {}", 
                standard_edges, optimized_edges);
        println!("  Avg degree - Standard: {:.1}, Optimized: {:.1}",
                standard_edges as f64 / size as f64,
                optimized_edges as f64 / size as f64);
        println!();
    }
    
    // Test dynamic operations
    println!("=== Dynamic Index Operations ===");
    let size = 5000;
    let data = DataGenerator::new()
        .with_seed(42)
        .generate_uniform(size, dim, -1.0, 1.0);
    
    let mut dynamic_index = DynamicMemoryIndex::new(Distance::L2, dim, 64, 1.2);
    
    // Batch insert
    let insert_start = Instant::now();
    for (i, vector) in data.iter().enumerate() {
        dynamic_index.insert(i, vector.clone());
    }
    let insert_time = insert_start.elapsed();
    
    println!("Batch insert {} vectors: {:.2}s ({:.0} vectors/sec)",
            size, insert_time.as_secs_f64(), 
            size as f64 / insert_time.as_secs_f64());
    
    // Test consolidation
    let consolidate_start = Instant::now();
    dynamic_index.consolidate()?;
    let consolidate_time = consolidate_start.elapsed();
    
    println!("Consolidation time: {:.2}s", consolidate_time.as_secs_f64());
    
    // Test deletion
    let delete_ids: Vec<usize> = (0..100).collect();
    let delete_start = Instant::now();
    for id in &delete_ids {
        dynamic_index.delete(*id)?;
    }
    let delete_time = delete_start.elapsed();
    
    println!("Delete {} vectors: {:.3}s", delete_ids.len(), delete_time.as_secs_f64());
    
    Ok(())
}