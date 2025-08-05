//! Comprehensive benchmark suite for DiskANN on M2 ARM64
//! 
//! Runs individual benchmarks with max 60 second timeout each

use diskann::*;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;
use chrono::Local;
use rand::Rng;

const OUTPUT_DIR: &str = "examples/runs/ampereARM64small";

fn main() -> Result<()> {
    // Create timestamp for this run
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    println!("DiskANN Benchmark Suite - Ampere ARM64");
    println!("==================================");
    println!("Timestamp: {}", timestamp);
    println!("Results will be saved to: {}/", OUTPUT_DIR);
    println!();

    // 1. SIMD Distance Benchmarks
    run_simd_benchmark(&timestamp)?;
    
    // 2. Index Construction Benchmark
    run_index_construction_benchmark(&timestamp)?;
    
    // 3. Search Performance Benchmark
    run_search_benchmark(&timestamp)?;
    
    // 4. Disk Index Benchmark
    run_disk_index_benchmark(&timestamp)?;
    
    // 5. Product Quantization Benchmark
    run_pq_benchmark(&timestamp)?;
    
    // 6. Memory vs Disk Comparison
    run_memory_vs_disk_benchmark(&timestamp)?;
    
    println!("\nAll benchmarks complete!");
    Ok(())
}

fn run_simd_benchmark(timestamp: &str) -> Result<()> {
    println!("\n=== SIMD Distance Benchmark ===");
    let output_path = format!("{}/simd_benchmark_{}.log", OUTPUT_DIR, timestamp);
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "SIMD Distance Benchmark")?;
    writeln!(output, "=======================")?;
    writeln!(output, "Platform: M2 ARM64")?;
    writeln!(output, "Date: {}", Local::now())?;
    writeln!(output)?;
    
    // CPU features
    writeln!(output, "CPU Features:")?;
    writeln!(output, "  ARM64 NEON: {}", has_neon_support())?;
    writeln!(output, "  x86-64 AVX2: {}", has_avx2_support())?;
    writeln!(output, "  x86-64 AVX-512: {}", has_avx512_support())?;
    writeln!(output)?;
    
    let dimensions = vec![64, 128, 256, 512, 768, 1024];
    let num_iterations = 1_000_000;
    
    for dim in dimensions {
        println!("  Testing dimension {}...", dim);
        writeln!(output, "Dimension: {}", dim)?;
        
        // Generate test vectors
        let vec_a = generate_random_vector(dim);
        let vec_b = generate_random_vector(dim);
        
        // Test each distance metric
        for metric in [Distance::L2, Distance::Cosine, Distance::InnerProduct] {
            let distance_fn = distance::create_distance_function(metric, dim);
            
            // Warmup
            for _ in 0..1000 {
                let _ = distance_fn.distance(&vec_a, &vec_b);
            }
            
            // Benchmark
            let start = Instant::now();
            let mut total_dist = 0.0;
            
            for _ in 0..num_iterations {
                total_dist += distance_fn.distance(&vec_a, &vec_b)?;
            }
            
            let elapsed = start.elapsed();
            let ns_per_op = elapsed.as_nanos() as f64 / num_iterations as f64;
            let ops_per_sec = 1_000_000_000.0 / ns_per_op;
            
            writeln!(output, "  {:?}: {:.2} ns/op, {:.2}M ops/sec", 
                     metric, ns_per_op, ops_per_sec / 1_000_000.0)?;
        }
        writeln!(output)?;
    }
    
    println!("  Results saved to: {}", output_path);
    Ok(())
}

fn run_index_construction_benchmark(timestamp: &str) -> Result<()> {
    println!("\n=== Index Construction Benchmark ===");
    let output_path = format!("{}/index_construction_{}.log", OUTPUT_DIR, timestamp);
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "Index Construction Benchmark")?;
    writeln!(output, "===========================")?;
    writeln!(output, "Date: {}", Local::now())?;
    writeln!(output)?;
    
    let configs = vec![
        (1000, 128, "1K vectors, 128 dim"),
        (10000, 128, "10K vectors, 128 dim"),
        (50000, 128, "50K vectors, 128 dim"),
        (10000, 768, "10K vectors, 768 dim"),
    ];
    
    for (num_vectors, dim, desc) in configs {
        println!("  Building {}...", desc);
        writeln!(output, "{}", desc)?;
        
        let vectors = generate_random_vectors_batch(num_vectors, dim);
        
        // Test with different parameters
        let params = vec![
            (32, 50, 1.2, "R=32, L=50"),
            (64, 100, 1.2, "R=64, L=100"),
        ];
        
        for (r, l, alpha, param_desc) in params {
            let start = Instant::now();
            
            let index = IndexBuilder::new()
                .dimensions(dim)
                .metric(Distance::L2)
                .max_degree(r)
                .search_list_size(l)
                .alpha(alpha)
                .build_from_vectors(vectors.clone())?;
            
            let elapsed = start.elapsed();
            let points_per_sec = num_vectors as f64 / elapsed.as_secs_f64();
            
            writeln!(output, "  {}: {:.2}s ({:.0} points/sec)", 
                     param_desc, elapsed.as_secs_f64(), points_per_sec)?;
            
            // If taking too long, skip larger configs
            if elapsed.as_secs() > 30 {
                writeln!(output, "  (Skipping larger configurations due to time)")?;
                break;
            }
        }
        writeln!(output)?;
    }
    
    println!("  Results saved to: {}", output_path);
    Ok(())
}

fn run_search_benchmark(timestamp: &str) -> Result<()> {
    println!("\n=== Search Performance Benchmark ===");
    let output_path = format!("{}/search_performance_{}.log", OUTPUT_DIR, timestamp);
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "Search Performance Benchmark")?;
    writeln!(output, "===========================")?;
    writeln!(output, "Date: {}", Local::now())?;
    writeln!(output)?;
    
    // Build test index
    println!("  Building test index...");
    let num_vectors = 50000;
    let dim = 128;
    let vectors = generate_random_vectors_batch(num_vectors, dim);
    
    let index = IndexBuilder::new()
        .dimensions(dim)
        .metric(Distance::L2)
        .max_degree(64)
        .search_list_size(100)
        .build_from_vectors(vectors.clone())?;
    
    writeln!(output, "Index: {} vectors, {} dimensions", num_vectors, dim)?;
    writeln!(output)?;
    
    let k_values = vec![1, 10, 50, 100];
    let num_queries = 1000;
    
    for k in k_values {
        println!("  Testing k={}...", k);
        writeln!(output, "k={}", k)?;
        
        let mut query_times = Vec::new();
        
        for i in 0..num_queries {
            let query = &vectors[i % vectors.len()];
            
            let start = Instant::now();
            let _ = index.search(query, k)?;
            query_times.push(start.elapsed());
        }
        
        // Calculate statistics
        query_times.sort();
        let avg_time = query_times.iter().sum::<Duration>() / query_times.len() as u32;
        let p50 = query_times[query_times.len() / 2];
        let p95 = query_times[query_times.len() * 95 / 100];
        let p99 = query_times[query_times.len() * 99 / 100];
        
        let qps = 1_000_000.0 / avg_time.as_micros() as f64;
        
        writeln!(output, "  QPS: {:.0}", qps)?;
        writeln!(output, "  Avg latency: {:.2} μs", avg_time.as_micros() as f64)?;
        writeln!(output, "  P50 latency: {:.2} μs", p50.as_micros() as f64)?;
        writeln!(output, "  P95 latency: {:.2} μs", p95.as_micros() as f64)?;
        writeln!(output, "  P99 latency: {:.2} μs", p99.as_micros() as f64)?;
        writeln!(output)?;
    }
    
    println!("  Results saved to: {}", output_path);
    Ok(())
}

fn run_disk_index_benchmark(timestamp: &str) -> Result<()> {
    println!("\n=== Disk Index Benchmark ===");
    let output_path = format!("{}/disk_index_{}.log", OUTPUT_DIR, timestamp);
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "Disk Index Benchmark")?;
    writeln!(output, "===================")?;
    writeln!(output, "Date: {}", Local::now())?;
    writeln!(output)?;
    
    // Build test data
    println!("  Building disk index...");
    let num_vectors = 10000;
    let dim = 128;
    let vectors = generate_random_vectors_batch(num_vectors, dim);
    
    let pq_config = PQFlashConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        pq_params: index::disk::PQParams {
            num_chunks: 16,
            bits_per_chunk: 8,
        },
        num_threads: 0,
        use_reorder_data: true,
        beam_width: 4,
    };
    
    let disk_path = format!("{}/temp_disk_index", OUTPUT_DIR);
    
    // Build disk index
    let build_start = Instant::now();
    PQFlashIndex::build_from_vectors(&disk_path, vectors.clone(), pq_config)?;
    let build_time = build_start.elapsed();
    
    writeln!(output, "Build Statistics:")?;
    writeln!(output, "  Vectors: {}", num_vectors)?;
    writeln!(output, "  Dimensions: {}", dim)?;
    writeln!(output, "  Build time: {:.2}s", build_time.as_secs_f64())?;
    writeln!(output)?;
    
    // Load and test search
    println!("  Testing disk search...");
    let mut disk_index = PQFlashIndex::new(dim, Distance::L2, pq_config);
    disk_index.load(&disk_path)?;
    
    let k = 10;
    let num_queries = 100;
    let mut query_times = Vec::new();
    let mut stats_vec = Vec::new();
    
    for i in 0..num_queries {
        let query = &vectors[i % vectors.len()];
        
        let start = Instant::now();
        let (results, stats) = disk_index.search(query, k, 100)?;
        query_times.push(start.elapsed());
        stats_vec.push(stats);
    }
    
    // Calculate statistics
    let avg_time = query_times.iter().sum::<Duration>() / query_times.len() as u32;
    let avg_nodes_visited = stats_vec.iter().map(|s| s.nodes_visited).sum::<usize>() as f64 / stats_vec.len() as f64;
    let avg_distance_comps = stats_vec.iter().map(|s| s.distance_computations).sum::<usize>() as f64 / stats_vec.len() as f64;
    
    writeln!(output, "Search Statistics (k={}):", k)?;
    writeln!(output, "  Avg latency: {:.2} ms", avg_time.as_secs_f64() * 1000.0)?;
    writeln!(output, "  Avg nodes visited: {:.1}", avg_nodes_visited)?;
    writeln!(output, "  Avg distance computations: {:.1}", avg_distance_comps)?;
    writeln!(output)?;
    
    // Cleanup
    std::fs::remove_file(&disk_path).ok();
    std::fs::remove_file(format!("{}.pq_compressed.bin", disk_path)).ok();
    std::fs::remove_file(format!("{}.pq_compressed.pq_metadata.json", disk_path)).ok();
    std::fs::remove_file(format!("{}.reorder_data.bin", disk_path)).ok();
    
    println!("  Results saved to: {}", output_path);
    Ok(())
}

fn run_pq_benchmark(timestamp: &str) -> Result<()> {
    println!("\n=== Product Quantization Benchmark ===");
    let output_path = format!("{}/pq_benchmark_{}.log", OUTPUT_DIR, timestamp);
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "Product Quantization Benchmark")?;
    writeln!(output, "=============================")?;
    writeln!(output, "Date: {}", Local::now())?;
    writeln!(output)?;
    
    let num_vectors = 10000;
    let dim = 128;
    let vectors = generate_random_vectors_batch(num_vectors, dim);
    
    println!("  Training PQ codebooks...");
    
    let chunk_configs = vec![
        (8, "8 chunks (16x compression)"),
        (16, "16 chunks (8x compression)"),
        (32, "32 chunks (4x compression)"),
    ];
    
    for (num_chunks, desc) in chunk_configs {
        writeln!(output, "{}", desc)?;
        
        let pq_params = pq::PQParams::new(num_chunks, 8); // 8 bits per subquantizer
        
        // Train PQ
        let train_start = Instant::now();
        let mut pq = pq::ProductQuantizer::new(512, pq_params)?; // 512 dimensions
        let _training_result = pq.train(&vectors)?;
        let train_time = train_start.elapsed();
        
        // Encode all vectors
        let encode_start = Instant::now();
        let encoded: Result<Vec<_>, _> = vectors.iter()
            .map(|v| pq.encode(v))
            .collect();
        let encoded = encoded?;
        let encode_time = encode_start.elapsed();
        
        // Test reconstruction error on sample
        let mut total_error = 0.0;
        for i in 0..100 {
            let original = &vectors[i];
            let reconstructed = pq.decode(&encoded[i])?;
            let error: f32 = original.iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            total_error += error;
        }
        let avg_error = total_error / 100.0;
        
        writeln!(output, "  Training time: {:.2}s", train_time.as_secs_f64())?;
        writeln!(output, "  Encoding rate: {:.0} vectors/sec", num_vectors as f64 / encode_time.as_secs_f64())?;
        writeln!(output, "  Avg reconstruction error: {:.4}", avg_error)?;
        writeln!(output)?;
    }
    
    println!("  Results saved to: {}", output_path);
    Ok(())
}

fn run_memory_vs_disk_benchmark(timestamp: &str) -> Result<()> {
    println!("\n=== Memory vs Disk Comparison ===");
    let output_path = format!("{}/memory_vs_disk_{}.log", OUTPUT_DIR, timestamp);
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "Memory vs Disk Index Comparison")?;
    writeln!(output, "==============================")?;
    writeln!(output, "Date: {}", Local::now())?;
    writeln!(output)?;
    
    let num_vectors = 10000;
    let dim = 128;
    let vectors = generate_random_vectors_batch(num_vectors, dim);
    
    println!("  Building memory index...");
    
    // Build memory index
    let mem_start = Instant::now();
    let mem_index = IndexBuilder::new()
        .dimensions(dim)
        .metric(Distance::L2)
        .max_degree(64)
        .search_list_size(100)
        .build_from_vectors(vectors.clone())?;
    let mem_build_time = mem_start.elapsed();
    
    println!("  Building disk index...");
    
    // Build disk index
    let pq_config = PQFlashConfig {
        max_degree: 64,
        search_list_size: 100,
        alpha: 1.2,
        pq_params: index::disk::PQParams {
            num_chunks: 16,
            bits_per_chunk: 8,
        },
        num_threads: 0,
        use_reorder_data: true,
        beam_width: 4,
    };
    
    let disk_path = format!("{}/temp_disk_index", OUTPUT_DIR);
    let disk_start = Instant::now();
    PQFlashIndex::build_from_vectors(&disk_path, vectors.clone(), pq_config)?;
    let disk_build_time = disk_start.elapsed();
    
    let mut disk_index = PQFlashIndex::new(dim, Distance::L2, pq_config);
    disk_index.load(&disk_path)?;
    
    writeln!(output, "Build Times:")?;
    writeln!(output, "  Memory index: {:.2}s", mem_build_time.as_secs_f64())?;
    writeln!(output, "  Disk index: {:.2}s", disk_build_time.as_secs_f64())?;
    writeln!(output)?;
    
    // Compare search performance
    println!("  Comparing search performance...");
    let k = 10;
    let num_queries = 100;
    
    writeln!(output, "Search Performance (k={}, {} queries):", k, num_queries)?;
    
    // Memory index search
    let mut mem_times = Vec::new();
    for i in 0..num_queries {
        let query = &vectors[i];
        let start = Instant::now();
        let _ = mem_index.search(query, k)?;
        mem_times.push(start.elapsed());
    }
    
    let mem_avg = mem_times.iter().sum::<Duration>() / mem_times.len() as u32;
    let mem_qps = 1_000_000.0 / mem_avg.as_micros() as f64;
    
    // Disk index search
    let mut disk_times = Vec::new();
    for i in 0..num_queries {
        let query = &vectors[i];
        let start = Instant::now();
        let _ = disk_index.search(query, k, 100)?;
        disk_times.push(start.elapsed());
    }
    
    let disk_avg = disk_times.iter().sum::<Duration>() / disk_times.len() as u32;
    let disk_qps = 1_000_000.0 / disk_avg.as_micros() as f64;
    
    writeln!(output)?;
    writeln!(output, "Memory Index:")?;
    writeln!(output, "  Avg latency: {:.2} μs", mem_avg.as_micros() as f64)?;
    writeln!(output, "  QPS: {:.0}", mem_qps)?;
    
    writeln!(output)?;
    writeln!(output, "Disk Index:")?;
    writeln!(output, "  Avg latency: {:.2} μs", disk_avg.as_micros() as f64)?;
    writeln!(output, "  QPS: {:.0}", disk_qps)?;
    writeln!(output, "  Slowdown: {:.2}x", disk_avg.as_secs_f64() / mem_avg.as_secs_f64())?;
    
    // Cleanup
    std::fs::remove_file(&disk_path).ok();
    std::fs::remove_file(format!("{}.pq_compressed.bin", disk_path)).ok();
    std::fs::remove_file(format!("{}.pq_compressed.pq_metadata.json", disk_path)).ok();
    std::fs::remove_file(format!("{}.reorder_data.bin", disk_path)).ok();
    
    println!("  Results saved to: {}", output_path);
    Ok(())
}

// Helper functions
fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn generate_random_vectors_batch(num: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..num).map(|_| generate_random_vector(dim)).collect()
}