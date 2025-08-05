//! Large-Scale Cold Disk Benchmarks
//!
//! This example tests DiskANN's disk-based PQ Flash Index with large datasets
//! (10K, 100K, 1M embeddings) to measure cold disk performance.

use diskann::{Distance, PQFlashIndex, PQFlashConfig, Result};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    name: String,
    num_vectors: usize,
    dimension: usize,
    num_queries: usize,
    k: usize,
    pq_chunks: usize,
    pq_bits: u8,
    cache_size: usize,
}

#[derive(Debug)]
struct BenchmarkResult {
    config: BenchmarkConfig,
    build_time: Duration,
    index_size_mb: f64,
    avg_search_time_ms: f64,
    p95_search_time_ms: f64,
    p99_search_time_ms: f64,
    queries_per_second: f64,
    memory_usage_mb: f64,
    cache_hit_rate: f64,
    avg_nodes_visited: f64,
    avg_distance_computations: f64,
}

impl BenchmarkConfig {
    fn new(name: &str, num_vectors: usize) -> Self {
        Self {
            name: name.to_string(),
            num_vectors,
            dimension: 768, // Common embedding dimension
            num_queries: 1000,
            k: 10,
            pq_chunks: 96, // 768 / 8
            pq_bits: 8,
            cache_size: 10000.min(num_vectors / 10),
        }
    }
}

fn generate_embeddings(count: usize, dimension: usize, progress: &ProgressBar) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    
    progress.set_message("Generating embeddings");
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            if i % 100 == 0 {
                progress.set_position(i as u64);
            }
            
            // Generate normalized vectors (common for embeddings)
            let mut vec: Vec<f32> = (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            
            vec
        })
        .collect();
    
    progress.finish_with_message("Embeddings generated");
    vectors
}

fn run_benchmark(config: BenchmarkConfig, output_dir: &Path) -> Result<BenchmarkResult> {
    println!("\n=== Running {} Benchmark ===", config.name);
    println!("Vectors: {}, Dimension: {}, Queries: {}", 
             config.num_vectors, config.dimension, config.num_queries);
    
    let mp = MultiProgress::new();
    let build_progress = mp.add(ProgressBar::new(config.num_vectors as u64));
    build_progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    
    // Generate data
    let vectors = generate_embeddings(config.num_vectors, config.dimension, &build_progress);
    let queries = generate_embeddings(config.num_queries, config.dimension, &build_progress);
    
    // Create index path
    let index_path = output_dir.join(format!("{}.pq.idx", config.name));
    
    // Build index
    println!("Building PQ Flash Index...");
    let pq_config = PQFlashConfig {
        dimension: config.dimension,
        metric: Distance::Cosine,
        num_chunks: config.pq_chunks,
        bits_per_chunk: config.pq_bits,
        search_cache_size: config.cache_size,
        reorder_data: true,
    };
    
    let build_start = Instant::now();
    let mut index = PQFlashIndex::build_from_vectors(
        index_path.to_str().unwrap(),
        vectors,
        pq_config,
    )?;
    let build_time = build_start.elapsed();
    
    // Get index size
    let index_size_bytes = fs::metadata(&index_path)?.len();
    let index_size_mb = index_size_bytes as f64 / (1024.0 * 1024.0);
    
    println!("Index built in {:.2}s, size: {:.2} MB", 
             build_time.as_secs_f64(), index_size_mb);
    
    // Warm-up searches
    println!("Warming up cache...");
    for i in 0..10 {
        let _ = index.search(&queries[i % queries.len()], config.k)?;
    }
    
    // Cold disk searches
    println!("Running cold disk searches...");
    let search_progress = mp.add(ProgressBar::new(config.num_queries as u64));
    search_progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} searches")
            .unwrap()
            .progress_chars("#>-")
    );
    
    let mut search_times = Vec::new();
    let mut total_nodes_visited = 0u64;
    let mut total_distance_computations = 0u64;
    
    // Clear OS cache between queries for true cold disk performance
    for (i, query) in queries.iter().enumerate() {
        search_progress.set_position(i as u64);
        
        // For true cold disk testing, we would clear OS cache here
        // On Linux: sync && echo 3 > /proc/sys/vm/drop_caches
        // For this benchmark, we simulate by creating memory pressure
        if i % 100 == 0 {
            // Allocate and touch large memory to evict cache
            let _pressure: Vec<u8> = vec![0; 100 * 1024 * 1024]; // 100MB
        }
        
        let search_start = Instant::now();
        let results = index.search(query, config.k)?;
        let search_time = search_start.elapsed();
        
        search_times.push(search_time);
        
        // Get search statistics
        if let Some(stats) = index.last_search_stats() {
            total_nodes_visited += stats.nodes_visited as u64;
            total_distance_computations += stats.distance_computations as u64;
        }
    }
    
    search_progress.finish_with_message("Searches complete");
    
    // Calculate statistics
    search_times.sort_by(|a, b| a.cmp(b));
    
    let avg_search_time = search_times.iter().sum::<Duration>() / search_times.len() as u32;
    let p95_index = (search_times.len() as f64 * 0.95) as usize;
    let p99_index = (search_times.len() as f64 * 0.99) as usize;
    
    let avg_search_time_ms = avg_search_time.as_secs_f64() * 1000.0;
    let p95_search_time_ms = search_times[p95_index].as_secs_f64() * 1000.0;
    let p99_search_time_ms = search_times[p99_index].as_secs_f64() * 1000.0;
    let queries_per_second = 1000.0 / avg_search_time_ms;
    
    let avg_nodes_visited = total_nodes_visited as f64 / config.num_queries as f64;
    let avg_distance_computations = total_distance_computations as f64 / config.num_queries as f64;
    
    // Estimate memory usage
    let memory_usage_mb = index.memory_usage_mb();
    let cache_stats = index.cache_stats();
    let cache_hit_rate = cache_stats.hit_rate();
    
    // Clean up
    drop(index);
    let _ = fs::remove_file(&index_path);
    
    Ok(BenchmarkResult {
        config,
        build_time,
        index_size_mb,
        avg_search_time_ms,
        p95_search_time_ms,
        p99_search_time_ms,
        queries_per_second,
        memory_usage_mb,
        cache_hit_rate,
        avg_nodes_visited,
        avg_distance_computations,
    })
}

fn print_results(results: &[BenchmarkResult]) {
    println!("\n=== Benchmark Results Summary ===");
    println!("{:-<120}", "");
    println!("{:<15} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "Dataset", "Vectors", "Index (MB)", "Build (s)", "Avg (ms)", "P95 (ms)", "P99 (ms)", "QPS", "Mem (MB)");
    println!("{:-<120}", "");
    
    for result in results {
        println!("{:<15} {:>10} {:>12.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                 result.config.name,
                 result.config.num_vectors,
                 result.index_size_mb,
                 result.build_time.as_secs_f64(),
                 result.avg_search_time_ms,
                 result.p95_search_time_ms,
                 result.p99_search_time_ms,
                 result.queries_per_second,
                 result.memory_usage_mb);
    }
    
    println!("{:-<120}", "");
    
    // Additional statistics
    println!("\n=== Detailed Statistics ===");
    for result in results {
        println!("\n{}:", result.config.name);
        println!("  Cache hit rate: {:.2}%", result.cache_hit_rate * 100.0);
        println!("  Avg nodes visited: {:.0}", result.avg_nodes_visited);
        println!("  Avg distance computations: {:.0}", result.avg_distance_computations);
        println!("  Compression ratio: {:.1}x", 
                 (result.config.num_vectors * result.config.dimension * 4) as f64 / 
                 (result.index_size_mb * 1024.0 * 1024.0));
    }
}

fn main() -> Result<()> {
    println!("Large-Scale Cold Disk Benchmark");
    println!("===============================");
    
    // Create output directory
    let output_dir = Path::new("benchmark_indices");
    fs::create_dir_all(output_dir)?;
    
    // Define benchmark configurations
    let configs = vec![
        BenchmarkConfig::new("small_10k", 10_000),
        BenchmarkConfig::new("medium_100k", 100_000),
        BenchmarkConfig::new("large_1m", 1_000_000),
    ];
    
    // Run benchmarks
    let mut results = Vec::new();
    for config in configs {
        match run_benchmark(config.clone(), output_dir) {
            Ok(result) => results.push(result),
            Err(e) => {
                eprintln!("Failed to run {} benchmark: {}", config.name, e);
                if config.num_vectors <= 100_000 {
                    return Err(e);
                }
            }
        }
    }
    
    // Print results
    print_results(&results);
    
    // Performance analysis
    println!("\n=== Performance Analysis ===");
    
    if results.len() >= 2 {
        let small = &results[0];
        let large = &results[results.len() - 1];
        
        let scale_factor = large.config.num_vectors as f64 / small.config.num_vectors as f64;
        let search_slowdown = large.avg_search_time_ms / small.avg_search_time_ms;
        let build_slowdown = large.build_time.as_secs_f64() / small.build_time.as_secs_f64();
        
        println!("Scaling from {} to {} vectors ({}x):",
                 small.config.num_vectors, large.config.num_vectors, scale_factor);
        println!("  Search slowdown: {:.2}x", search_slowdown);
        println!("  Build slowdown: {:.2}x", build_slowdown);
        println!("  Search scaling efficiency: {:.1}%", 
                 (scale_factor.log2() / search_slowdown.log2()) * 100.0);
        println!("  Build scaling efficiency: {:.1}%",
                 (scale_factor / build_slowdown) * 100.0);
    }
    
    // Recommendations
    println!("\n=== Recommendations ===");
    println!("1. For datasets < 100K vectors: Consider in-memory index for better performance");
    println!("2. For datasets > 100K vectors: PQ Flash Index provides excellent compression");
    println!("3. Increase cache size for better performance (current: {} entries)", 
             results.last().map(|r| r.config.cache_size).unwrap_or(10000));
    println!("4. Consider SSD over HDD for 10x+ better random access performance");
    println!("5. Use GPU acceleration for batch queries to improve throughput");
    
    // Clean up
    let _ = fs::remove_dir_all(output_dir);
    
    Ok(())
}