//! Memory Scaling and Efficiency Benchmarks
//!
//! This example tests how DiskANN scales with different memory constraints
//! and measures the trade-offs between memory usage, performance, and accuracy.

use diskann::{Distance, IndexBuilder, PQFlashIndex, PQFlashConfig, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, ProcessExt};

#[derive(Debug, Clone)]
struct MemoryConfig {
    name: String,
    num_vectors: usize,
    dimension: usize,
    max_degree: usize,
    pq_chunks: Option<usize>,
    pq_bits: Option<u8>,
    cache_size: usize,
}

#[derive(Debug)]
struct MemoryBenchmarkResult {
    config: MemoryConfig,
    build_time: Duration,
    peak_memory_mb: f64,
    index_memory_mb: f64,
    search_memory_mb: f64,
    avg_search_time_ms: f64,
    queries_per_second: f64,
    recall_at_10: f64,
}

fn get_process_memory_mb() -> f64 {
    let mut system = System::new();
    system.refresh_processes();
    
    let pid = std::process::id();
    if let Some(process) = system.process(pid as _) {
        process.memory() as f64 / 1024.0 // KB to MB
    } else {
        0.0
    }
}

fn measure_memory_usage<F, T>(operation: F) -> Result<(T, f64)>
where
    F: FnOnce() -> Result<T>,
{
    // Force garbage collection
    for _ in 0..3 {
        let _large_vec: Vec<u8> = Vec::with_capacity(100 * 1024 * 1024);
    }
    std::thread::sleep(Duration::from_millis(100));
    
    let start_memory = get_process_memory_mb();
    let result = operation()?;
    let end_memory = get_process_memory_mb();
    
    Ok((result, end_memory - start_memory))
}

fn run_memory_benchmark(config: MemoryConfig) -> Result<MemoryBenchmarkResult> {
    println!("\n=== {} Configuration ===", config.name);
    
    // Generate test data
    let vectors = diskann::utils::generate_random_vectors(config.num_vectors, config.dimension);
    let queries = diskann::utils::generate_random_vectors(100, config.dimension);
    
    let build_start = Instant::now();
    let (index_result, build_memory) = if config.pq_chunks.is_some() {
        // Disk-based PQ index
        println!("Building PQ Flash Index...");
        let pq_config = PQFlashConfig {
            dimension: config.dimension,
            metric: Distance::L2,
            num_chunks: config.pq_chunks.unwrap(),
            bits_per_chunk: config.pq_bits.unwrap(),
            search_cache_size: config.cache_size,
            reorder_data: true,
        };
        
        measure_memory_usage(|| {
            PQFlashIndex::build_from_vectors(
                "temp_index.pq",
                vectors.clone(),
                pq_config,
            )
        })?
    } else {
        // In-memory index
        println!("Building in-memory index...");
        measure_memory_usage(|| {
            IndexBuilder::new()
                .dimensions(config.dimension)
                .metric(Distance::L2)
                .max_degree(config.max_degree)
                .search_list_size(100)
                .build_from_vectors(vectors.clone())
        })?
    };
    
    let build_time = build_start.elapsed();
    let peak_memory_mb = build_memory;
    
    println!("Build completed in {:.2}s, peak memory: {:.2} MB", 
             build_time.as_secs_f64(), peak_memory_mb);
    
    // Measure index memory footprint
    let index_memory_mb = get_process_memory_mb();
    
    // Benchmark searches
    let mut search_times = Vec::new();
    let mut search_memory_usage = Vec::new();
    let mut recalls = Vec::new();
    
    // Build ground truth with in-memory index
    let gt_index = IndexBuilder::new()
        .dimensions(config.dimension)
        .metric(Distance::L2)
        .max_degree(64)
        .search_list_size(100)
        .build_from_vectors(vectors)?;
    
    for query in &queries {
        let gt_results = gt_index.search(query, 10)?;
        
        let (results, search_mem) = measure_memory_usage(|| {
            let start = Instant::now();
            let res = if config.pq_chunks.is_some() {
                let mut idx = index_result.as_ref().unwrap();
                idx.search(query, 10)
            } else {
                index_result.as_ref().unwrap().search(query, 10)
            };
            search_times.push(start.elapsed());
            res
        })?;
        
        search_memory_usage.push(search_mem);
        
        // Calculate recall
        let mut correct = 0;
        for (id, _) in &results {
            for (gt_id, _) in &gt_results {
                if id == gt_id {
                    correct += 1;
                    break;
                }
            }
        }
        recalls.push(correct as f64 / 10.0);
    }
    
    // Clean up disk index if created
    if config.pq_chunks.is_some() {
        let _ = std::fs::remove_file("temp_index.pq");
    }
    
    // Calculate statistics
    let avg_search_time = search_times.iter().sum::<Duration>() / search_times.len() as u32;
    let avg_search_time_ms = avg_search_time.as_secs_f64() * 1000.0;
    let queries_per_second = 1000.0 / avg_search_time_ms;
    let search_memory_mb = search_memory_usage.iter().sum::<f64>() / search_memory_usage.len() as f64;
    let recall_at_10 = recalls.iter().sum::<f64>() / recalls.len() as f64;
    
    Ok(MemoryBenchmarkResult {
        config,
        build_time,
        peak_memory_mb,
        index_memory_mb,
        search_memory_mb,
        avg_search_time_ms,
        queries_per_second,
        recall_at_10,
    })
}

fn main() -> Result<()> {
    println!("Memory Scaling and Efficiency Benchmarks");
    println!("========================================");
    
    let base_vectors = 100_000;
    let dimension = 768;
    
    // Define different memory configurations
    let configs = vec![
        // Full in-memory index
        MemoryConfig {
            name: "In-Memory Full".to_string(),
            num_vectors: base_vectors,
            dimension,
            max_degree: 64,
            pq_chunks: None,
            pq_bits: None,
            cache_size: 0,
        },
        // Reduced degree in-memory
        MemoryConfig {
            name: "In-Memory Compact".to_string(),
            num_vectors: base_vectors,
            dimension,
            max_degree: 32,
            pq_chunks: None,
            pq_bits: None,
            cache_size: 0,
        },
        // PQ compression - high quality
        MemoryConfig {
            name: "PQ High Quality".to_string(),
            num_vectors: base_vectors,
            dimension,
            max_degree: 64,
            pq_chunks: Some(96),
            pq_bits: Some(8),
            cache_size: 10000,
        },
        // PQ compression - balanced
        MemoryConfig {
            name: "PQ Balanced".to_string(),
            num_vectors: base_vectors,
            dimension,
            max_degree: 64,
            pq_chunks: Some(48),
            pq_bits: Some(8),
            cache_size: 5000,
        },
        // PQ compression - aggressive
        MemoryConfig {
            name: "PQ Aggressive".to_string(),
            num_vectors: base_vectors,
            dimension,
            max_degree: 64,
            pq_chunks: Some(24),
            pq_bits: Some(4),
            cache_size: 1000,
        },
    ];
    
    // Run benchmarks
    let mut results = Vec::new();
    for config in configs {
        match run_memory_benchmark(config) {
            Ok(result) => results.push(result),
            Err(e) => eprintln!("Benchmark failed: {}", e),
        }
    }
    
    // Print results
    println!("\n=== Memory Usage Summary ===");
    println!("{:-<110}", "");
    println!("{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
             "Configuration", "Build Mem", "Index Mem", "Search Mem", "Total Mem", "Compression", "Recall");
    println!("{:-<110}", "");
    
    let baseline_memory = results[0].index_memory_mb;
    for result in &results {
        let total_memory = result.index_memory_mb + result.search_memory_mb;
        let compression = baseline_memory / result.index_memory_mb;
        
        println!("{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.1}x {:>10.2}",
                 result.config.name,
                 result.peak_memory_mb,
                 result.index_memory_mb,
                 result.search_memory_mb,
                 total_memory,
                 compression,
                 result.recall_at_10);
    }
    
    println!("\n=== Performance Summary ===");
    println!("{:-<80}", "");
    println!("{:<20} {:>15} {:>15} {:>15} {:>15}",
             "Configuration", "Build Time (s)", "Search (ms)", "QPS", "Efficiency");
    println!("{:-<80}", "");
    
    let baseline_qps = results[0].queries_per_second;
    for result in &results {
        let efficiency = (result.queries_per_second / baseline_qps) * 
                        (baseline_memory / result.index_memory_mb) * 
                        result.recall_at_10;
        
        println!("{:<20} {:>15.2} {:>15.2} {:>15.0} {:>15.2}",
                 result.config.name,
                 result.build_time.as_secs_f64(),
                 result.avg_search_time_ms,
                 result.queries_per_second,
                 efficiency);
    }
    
    // Memory scaling analysis
    println!("\n=== Memory Scaling Analysis ===");
    
    let memory_points = vec![(10_000, "10K"), (50_000, "50K"), (100_000, "100K"), (500_000, "500K")];
    
    println!("\nProjected memory usage for different dataset sizes:");
    println!("{:<10} {:>15} {:>15} {:>15}",
             "Size", "In-Memory (MB)", "PQ Balanced (MB)", "Compression");
    
    for (size, label) in memory_points {
        let in_memory = (size as f64 / base_vectors as f64) * baseline_memory;
        let pq_memory = (size as f64 / base_vectors as f64) * results[2].index_memory_mb;
        let compression = in_memory / pq_memory;
        
        println!("{:<10} {:>15.2} {:>15.2} {:>15.1}x",
                 label, in_memory, pq_memory, compression);
    }
    
    // Recommendations
    println!("\n=== Recommendations ===");
    println!("1. For datasets < 1GB: Use in-memory index for best performance");
    println!("2. For datasets 1-10GB: Use PQ Balanced for good trade-off");
    println!("3. For datasets > 10GB: Use PQ Aggressive with larger cache");
    println!("4. Memory-constrained systems: PQ provides 10-50x compression");
    println!("5. Quality requirements: Adjust PQ chunks/bits for recall vs size");
    
    Ok(())
}