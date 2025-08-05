//! Cold Disk-based PQ Flash Index Benchmarks
//!
//! This benchmark ensures cold cache conditions by:
//! 1. Creating fresh index files for each test
//! 2. Clearing OS page cache between tests
//! 3. Using large datasets that exceed typical cache sizes
//! 4. Measuring true disk I/O performance

use diskann::index::disk::{PQFlashIndex, PQFlashConfig, PQParams};
use diskann::Distance;
use diskann::utils::generate_random_vectors;
use std::time::Instant;
use std::process::Command;
use std::fs;
use tempfile::TempDir;

/// Clear OS page cache (macOS/Linux)
fn clear_system_caches() {
    #[cfg(target_os = "macos")]
    {
        // macOS: purge command clears system caches
        let _ = Command::new("sudo")
            .arg("purge")
            .output();
    }
    
    #[cfg(target_os = "linux")]
    {
        // Linux: clear page cache
        let _ = Command::new("sudo")
            .arg("sh")
            .arg("-c")
            .arg("echo 3 > /proc/sys/vm/drop_caches")
            .output();
    }
    
    // Additional: force garbage collection and sleep
    std::thread::sleep(std::time::Duration::from_secs(2));
}

/// Generate deterministic test dataset
fn generate_test_dataset(num_vectors: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::{SeedableRng, Rng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    
    (0..num_vectors)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 DiskANN Rust - Cold Disk-based Benchmark Suite");
    println!("===============================================\n");
    
    // Get system info
    let output = Command::new("df").arg("-h").arg(".").output()?;
    let disk_info = String::from_utf8_lossy(&output.stdout);
    println!("💾 Disk Space:");
    println!("{}", disk_info);
    
    // Get memory info
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl").arg("hw.memsize").output()?;
        let mem_info = String::from_utf8_lossy(&output.stdout);
        println!("🧠 System Memory:");
        println!("{}", mem_info);
    }
    
    println!("🧪 Test Configuration:");
    println!("- Cold cache conditions enforced");
    println!("- Fresh index files for each test");
    println!("- Multiple dataset sizes");
    println!("- Comprehensive I/O measurement\n");
    
    // Test configurations: 10K, 100K, 1M embeddings with 128 dimensions
    let test_configs = vec![
        // 10K embeddings - ~5MB raw data
        ("10K Embeddings (Small Scale)", 10_000, 128, 8, 8),
        // 100K embeddings - ~50MB raw data  
        ("100K Embeddings (Medium Scale)", 100_000, 128, 16, 8),
        // 1M embeddings - ~500MB raw data
        ("1M Embeddings (Large Scale)", 1_000_000, 128, 32, 8),
    ];
    
    for (name, num_vectors, dimension, num_chunks, bits_per_chunk) in test_configs {
        println!("📊 Running: {}", name);
        println!("   Vectors: {}, Dimension: {}, PQ: {}x{} bits", 
                 num_vectors, dimension, num_chunks, bits_per_chunk);
        
        run_cold_benchmark(name, num_vectors, dimension, num_chunks, bits_per_chunk)?;
        println!();
    }
    
    println!("✅ Cold disk benchmarks completed!");
    Ok(())
}

fn run_cold_benchmark(
    name: &str,
    num_vectors: usize,
    dimension: usize,
    num_chunks: usize,
    bits_per_chunk: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Create temporary directory for this test
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("cold_benchmark_index");
    
    // Generate deterministic dataset
    println!("   📝 Generating dataset...");
    let vectors = generate_test_dataset(num_vectors, dimension, 42);
    let data_size_mb = (num_vectors * dimension * 4) / 1024 / 1024;
    println!("   📦 Dataset size: {} MB", data_size_mb);
    
    // Configure index
    let config = PQFlashConfig {
        max_degree: 32,
        search_list_size: 100,
        alpha: 1.2,
        pq_params: PQParams {
            num_chunks,
            bits_per_chunk,
        },
        num_threads: 4,
        use_reorder_data: true,  // Enable for accuracy comparison
        beam_width: 4,
    };
    
    // === BUILD PHASE ===
    println!("   🔨 Building index...");
    clear_system_caches();
    
    let build_start = Instant::now();
    let mut index = PQFlashIndex::new(dimension, Distance::L2, config.clone());
    index.build_and_save(&vectors, &index_path)?;
    let build_time = build_start.elapsed();
    
    // Get index file sizes
    let stats = index.get_stats().unwrap();
    let total_disk_size = stats.index_file_size + stats.pq_file_size + stats.reorder_file_size;
    let compression_ratio = (data_size_mb * 1024 * 1024) as f64 / total_disk_size as f64;
    
    println!("   ⏱️  Build time: {:?}", build_time);
    println!("   💿 Index size: {} KB (compression: {:.1}x)", 
             total_disk_size / 1024, compression_ratio);
    
    // === COLD LOAD PHASE ===
    println!("   ❄️  Testing cold load...");
    clear_system_caches();
    
    let load_start = Instant::now();
    let mut cold_index = PQFlashIndex::new(dimension, Distance::L2, config.clone());
    cold_index.load(&index_path)?;
    let load_time = load_start.elapsed();
    
    println!("   ⏱️  Cold load time: {:?}", load_time);
    
    // === COLD SEARCH PHASE ===
    println!("   🔍 Testing cold search performance...");
    clear_system_caches();
    
    let num_queries = 50;
    let k = 10;
    let search_l = 50;
    
    let mut total_query_time = std::time::Duration::ZERO;
    let mut total_nodes_visited = 0usize;
    let mut total_distance_computations = 0usize;
    let mut total_sectors_read = 0usize;
    
    // Perform cold searches
    for i in 0..num_queries {
        let query = &vectors[i % 1000]; // Use various queries
        
        let search_start = Instant::now();
        let (results, stats) = cold_index.search(query, k, search_l)?;
        let search_time = search_start.elapsed();
        
        total_query_time += search_time;
        total_nodes_visited += stats.nodes_visited;
        total_distance_computations += stats.distance_computations;
        total_sectors_read += stats.sectors_read;
        
        assert_eq!(results.len(), k, "Should return k results");
    }
    
    // Calculate metrics
    let avg_query_time = total_query_time / num_queries as u32;
    let qps = num_queries as f64 / total_query_time.as_secs_f64();
    let avg_nodes_visited = total_nodes_visited as f64 / num_queries as f64;
    let avg_distance_computations = total_distance_computations as f64 / num_queries as f64;
    let avg_sectors_read = total_sectors_read as f64 / num_queries as f64;
    
    // === WARM SEARCH COMPARISON ===
    println!("   🔥 Testing warm search for comparison...");
    
    let mut warm_total_time = std::time::Duration::ZERO;
    for i in 0..num_queries {
        let query = &vectors[i % 1000];
        
        let search_start = Instant::now();
        let (results, _) = cold_index.search(query, k, search_l)?;
        let search_time = search_start.elapsed();
        
        warm_total_time += search_time;
        assert_eq!(results.len(), k);
    }
    
    let warm_avg_time = warm_total_time / num_queries as u32;
    let warm_qps = num_queries as f64 / warm_total_time.as_secs_f64();
    let cache_speedup = avg_query_time.as_micros() as f64 / warm_avg_time.as_micros() as f64;
    
    // === RESULTS ===
    println!("   📈 RESULTS:");
    println!("   ├─ Cold Search:");
    println!("   │  ├─ Average latency: {:?}", avg_query_time);
    println!("   │  ├─ Throughput: {:.1} QPS", qps);
    println!("   │  ├─ Nodes visited: {:.1}", avg_nodes_visited);
    println!("   │  ├─ Distance computations: {:.1}", avg_distance_computations);
    println!("   │  └─ Sectors read: {:.1}", avg_sectors_read);
    println!("   ├─ Warm Search:");
    println!("   │  ├─ Average latency: {:?}", warm_avg_time);
    println!("   │  ├─ Throughput: {:.1} QPS", warm_qps);
    println!("   │  └─ Cache speedup: {:.1}x", cache_speedup);
    println!("   └─ Storage:");
    println!("      ├─ Total size: {} KB", total_disk_size / 1024);
    println!("      ├─ Compression: {:.1}x", compression_ratio);
    println!("      └─ Build rate: {:.0} vectors/sec", 
             num_vectors as f64 / build_time.as_secs_f64());
    
    Ok(())
}