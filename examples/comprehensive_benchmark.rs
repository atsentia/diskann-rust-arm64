//! Comprehensive benchmarks for DiskANN on M2 ARM64
//! 
//! Runs various benchmarks with 60-second timeout each and logs results

use diskann::*;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use chrono::Local;

const MAX_RUNTIME: Duration = Duration::from_secs(60);
const OUTPUT_DIR: &str = "examples/runs/macM2arm64";

fn main() -> Result<()> {
    // Create output file with timestamp
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let output_path = format!("{}/benchmark_{}.log", OUTPUT_DIR, timestamp);
    let mut output = File::create(&output_path)?;
    
    println!("Running comprehensive benchmarks on M2 ARM64...");
    println!("Results will be saved to: {}", output_path);
    writeln!(output, "DiskANN Comprehensive Benchmark Results")?;
    writeln!(output, "Platform: macOS M2 ARM64")?;
    writeln!(output, "Date: {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(output, "Max runtime per test: 60 seconds\n")?;
    
    // System info
    log_system_info(&mut output)?;
    
    // Run benchmarks
    benchmark_distance_functions(&mut output)?;
    benchmark_graph_construction(&mut output)?;
    benchmark_search_performance(&mut output)?;
    benchmark_memory_vs_disk(&mut output)?;
    benchmark_pq_compression(&mut output)?;
    benchmark_concurrent_operations(&mut output)?;
    
    println!("\nBenchmark complete! Results saved to: {}", output_path);
    Ok(())
}

fn log_system_info(output: &mut File) -> Result<()> {
    writeln!(output, "=== System Information ===")?;
    
    // Get CPU info
    if let Ok(info) = std::process::Command::new("sysctl")
        .args(&["-n", "machdep.cpu.brand_string"])
        .output()
    {
        let cpu = String::from_utf8_lossy(&info.stdout);
        writeln!(output, "CPU: {}", cpu.trim())?;
    }
    
    // Get memory info
    if let Ok(info) = std::process::Command::new("sysctl")
        .args(&["-n", "hw.memsize"])
        .output()
    {
        let mem_bytes = String::from_utf8_lossy(&info.stdout)
            .trim()
            .parse::<u64>()
            .unwrap_or(0);
        writeln!(output, "Memory: {} GB", mem_bytes / (1024 * 1024 * 1024))?;
    }
    
    // Rust version
    writeln!(output, "Rust: {}", env!("CARGO_PKG_RUST_VERSION"))?;
    writeln!(output, "DiskANN version: {}\n", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

fn benchmark_distance_functions(output: &mut File) -> Result<()> {
    writeln!(output, "=== Distance Function Benchmarks ===")?;
    println!("Running distance function benchmarks...");
    
    let dimensions = vec![128, 256, 512, 768, 1024];
    let num_iterations = 1_000_000;
    
    for dim in dimensions {
        writeln!(output, "\nDimension: {}", dim)?;
        
        // Generate test vectors
        let v1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 0.1).collect();
        
        // Test different distance functions
        for metric in [Distance::L2, Distance::Cosine, Distance::InnerProduct] {
            let distance_fn = distance::create_distance_function(metric, dim);
            
            // Warm up
            for _ in 0..1000 {
                let _ = distance_fn.distance(&v1, &v2);
            }
            
            // Time limited benchmark
            let start = Instant::now();
            let stop_flag = Arc::new(AtomicBool::new(false));
            let stop_flag_clone = stop_flag.clone();
            
            thread::spawn(move || {
                thread::sleep(MAX_RUNTIME);
                stop_flag_clone.store(true, Ordering::Relaxed);
            });
            
            let mut count = 0u64;
            while !stop_flag.load(Ordering::Relaxed) && count < num_iterations as u64 {
                let _ = distance_fn.distance(&v1, &v2);
                count += 1;
            }
            
            let elapsed = start.elapsed();
            let ops_per_sec = count as f64 / elapsed.as_secs_f64();
            let ns_per_op = elapsed.as_nanos() as f64 / count as f64;
            
            writeln!(output, "  {:?}: {:.2} M ops/sec ({:.2} ns/op)", 
                     metric, ops_per_sec / 1_000_000.0, ns_per_op)?;
        }
    }
    
    writeln!(output)?;
    Ok(())
}

fn benchmark_graph_construction(output: &mut File) -> Result<()> {
    writeln!(output, "=== Graph Construction Benchmarks ===")?;
    println!("Running graph construction benchmarks...");
    
    let test_configs = vec![
        (1000, 128, "1K vectors, 128 dim"),
        (10000, 128, "10K vectors, 128 dim"),
        (1000, 768, "1K vectors, 768 dim"),
    ];
    
    for (num_vectors, dim, desc) in test_configs {
        writeln!(output, "\n{}", desc)?;
        
        // Generate random vectors
        let vectors = generate_random_vectors(num_vectors, dim);
        
        // Build with different parameters
        let params = vec![
            (32, 50, "R=32, L=50"),
            (64, 100, "R=64, L=100"),
        ];
        
        for (max_degree, search_list_size, param_desc) in params {
            let start = Instant::now();
            let timeout = start + MAX_RUNTIME;
            
            let builder = IndexBuilder::new()
                .dimensions(dim)
                .metric(Distance::L2)
                .max_degree(max_degree)
                .search_list_size(search_list_size)
                .alpha(1.2);
            
            match builder.build_with_timeout(vectors.clone(), timeout) {
                Ok(index) => {
                    let elapsed = start.elapsed();
                    let points_per_sec = num_vectors as f64 / elapsed.as_secs_f64();
                    writeln!(output, "  {}: {:.2}s ({:.0} points/sec)", 
                             param_desc, elapsed.as_secs_f64(), points_per_sec)?;
                }
                Err(_) => {
                    writeln!(output, "  {}: Timeout (>60s)", param_desc)?;
                }
            }
        }
    }
    
    writeln!(output)?;
    Ok(())
}

fn benchmark_search_performance(output: &mut File) -> Result<()> {
    writeln!(output, "=== Search Performance Benchmarks ===")?;
    println!("Running search performance benchmarks...");
    
    // Build test index
    let num_vectors = 10000;
    let dim = 128;
    let vectors = generate_random_vectors(num_vectors, dim);
    
    let index = IndexBuilder::new()
        .dimensions(dim)
        .metric(Distance::L2)
        .max_degree(64)
        .search_list_size(100)
        .build_from_vectors(vectors.clone())?;
    
    let k_values = vec![1, 10, 50, 100];
    let num_queries = 1000;
    
    for k in k_values {
        writeln!(output, "\nk={}", k)?;
        
        let start = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();
        
        thread::spawn(move || {
            thread::sleep(MAX_RUNTIME);
            stop_flag_clone.store(true, Ordering::Relaxed);
        });
        
        let mut total_queries = 0u64;
        let mut total_time = Duration::ZERO;
        
        while !stop_flag.load(Ordering::Relaxed) && total_queries < num_queries as u64 {
            let query_idx = (total_queries % vectors.len() as u64) as usize;
            let query = &vectors[query_idx];
            
            let query_start = Instant::now();
            let _ = index.search(query, k)?;
            total_time += query_start.elapsed();
            
            total_queries += 1;
        }
        
        let elapsed = start.elapsed();
        let qps = total_queries as f64 / elapsed.as_secs_f64();
        let avg_latency_us = total_time.as_micros() as f64 / total_queries as f64;
        
        writeln!(output, "  QPS: {:.0}, Avg latency: {:.2} μs", qps, avg_latency_us)?;
    }
    
    writeln!(output)?;
    Ok(())
}

fn benchmark_memory_vs_disk(output: &mut File) -> Result<()> {
    writeln!(output, "=== Memory vs Disk Index Benchmarks ===")?;
    println!("Running memory vs disk benchmarks...");
    
    let num_vectors = 10000;
    let dim = 128;
    let vectors = generate_random_vectors(num_vectors, dim);
    
    // Build memory index
    let mem_index = IndexBuilder::new()
        .dimensions(dim)
        .metric(Distance::L2)
        .max_degree(64)
        .search_list_size(100)
        .build_from_vectors(vectors.clone())?;
    
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
    PQFlashIndex::build_from_vectors(&disk_path, vectors.clone(), pq_config)?;
    let mut disk_index = PQFlashIndex::new(dim, Distance::L2, pq_config);
    disk_index.load(&disk_path)?;
    
    // Benchmark both
    let num_queries = 100;
    let k = 10;
    
    writeln!(output, "\nMemory Index:")?;
    let mut mem_times = Vec::new();
    for i in 0..num_queries {
        let query = &vectors[i % vectors.len()];
        let start = Instant::now();
        let _ = mem_index.search(query, k)?;
        mem_times.push(start.elapsed());
    }
    
    let mem_avg = mem_times.iter().sum::<Duration>() / mem_times.len() as u32;
    let mem_p99 = percentile(&mut mem_times, 0.99);
    writeln!(output, "  Avg latency: {:.2} μs", mem_avg.as_micros() as f64)?;
    writeln!(output, "  P99 latency: {:.2} μs", mem_p99.as_micros() as f64)?;
    
    writeln!(output, "\nDisk Index:")?;
    let mut disk_times = Vec::new();
    for i in 0..num_queries {
        let query = &vectors[i % vectors.len()];
        let start = Instant::now();
        let _ = disk_index.search(query, k, 100)?;
        disk_times.push(start.elapsed());
    }
    
    let disk_avg = disk_times.iter().sum::<Duration>() / disk_times.len() as u32;
    let disk_p99 = percentile(&mut disk_times, 0.99);
    writeln!(output, "  Avg latency: {:.2} μs", disk_avg.as_micros() as f64)?;
    writeln!(output, "  P99 latency: {:.2} μs", disk_p99.as_micros() as f64)?;
    
    // Cleanup
    std::fs::remove_file(&disk_path).ok();
    std::fs::remove_file(format!("{}.pq_compressed.bin", disk_path)).ok();
    std::fs::remove_file(format!("{}.pq_compressed.pq_metadata.json", disk_path)).ok();
    std::fs::remove_file(format!("{}.reorder_data.bin", disk_path)).ok();
    
    writeln!(output)?;
    Ok(())
}

fn benchmark_pq_compression(output: &mut File) -> Result<()> {
    writeln!(output, "=== Product Quantization Benchmarks ===")?;
    println!("Running PQ compression benchmarks...");
    
    let num_vectors = 10000;
    let dim = 128;
    let vectors = generate_random_vectors(num_vectors, dim);
    
    let chunk_configs = vec![
        (8, "8 chunks"),
        (16, "16 chunks"),
        (32, "32 chunks"),
    ];
    
    for (num_chunks, desc) in chunk_configs {
        writeln!(output, "\n{} (compression: {}x)", desc, dim / num_chunks)?;
        
        let pq_params = pq::PQParams::new(num_chunks, 8); // 8 bits per subquantizer
        
        // Build PQ
        let start = Instant::now();
        let mut pq = pq::ProductQuantizer::new(dimension, pq_params)?;
        let _training_result = pq.train(&vectors)?;
        let train_time = start.elapsed();
        
        // Encode vectors
        let start = Instant::now();
        let encoded: Result<Vec<_>, _> = vectors.iter()
            .map(|v| pq.encode(v))
            .collect();
        let encoded = encoded?;
        let encode_time = start.elapsed();
        
        // Test reconstruction error
        let mut total_error = 0.0;
        for (original, encoded) in vectors.iter().zip(encoded.iter()).take(100) {
            let reconstructed = pq.decode(encoded)?;
            let error: f32 = original.iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            total_error += error;
        }
        
        let avg_error = total_error / 100.0;
        let encode_rate = num_vectors as f64 / encode_time.as_secs_f64();
        
        writeln!(output, "  Training time: {:.2}s", train_time.as_secs_f64())?;
        writeln!(output, "  Encoding rate: {:.0} vectors/sec", encode_rate)?;
        writeln!(output, "  Avg reconstruction error: {:.4}", avg_error)?;
    }
    
    writeln!(output)?;
    Ok(())
}

fn benchmark_concurrent_operations(output: &mut File) -> Result<()> {
    writeln!(output, "=== Concurrent Operations Benchmarks ===")?;
    println!("Running concurrent operations benchmarks...");
    
    let num_vectors = 10000;
    let dim = 128;
    let vectors = generate_random_vectors(num_vectors, dim);
    
    // Build dynamic index
    let dynamic_index = Arc::new(parking_lot::RwLock::new(
        DynamicIndex::new(dim, Distance::L2, 64, 100, 1.2)
    ));
    
    // Initial build
    {
        let mut index = dynamic_index.write();
        for (i, vec) in vectors.iter().take(num_vectors / 2).enumerate() {
            index.insert(vec.clone(), vec![])?;
        }
    }
    
    let thread_counts = vec![1, 2, 4, 8];
    
    for num_threads in thread_counts {
        writeln!(output, "\nThreads: {}", num_threads)?;
        
        let start = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let operations = Arc::new(AtomicU64::new(0));
        
        // Spawn worker threads
        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let index_clone = dynamic_index.clone();
            let vectors_clone = vectors.clone();
            let stop_clone = stop_flag.clone();
            let ops_clone = operations.clone();
            
            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                
                while !stop_clone.load(Ordering::Relaxed) {
                    // Mix of operations: 70% search, 20% insert, 10% delete
                    let op = rng.gen_range(0..10);
                    
                    if op < 7 {
                        // Search
                        let query_idx = rng.gen_range(0..vectors_clone.len());
                        let index = index_clone.read();
                        let _ = index.search(&vectors_clone[query_idx], 10);
                    } else if op < 9 {
                        // Insert
                        let vec_idx = rng.gen_range(0..vectors_clone.len());
                        let mut index = index_clone.write();
                        let _ = index.insert(vectors_clone[vec_idx].clone(), vec![]);
                    } else {
                        // Delete
                        let id = rng.gen_range(0..num_vectors);
                        let mut index = index_clone.write();
                        let _ = index.delete(id);
                    }
                    
                    ops_clone.fetch_add(1, Ordering::Relaxed);
                }
            });
            
            handles.push(handle);
        }
        
        // Run for limited time
        thread::sleep(Duration::from_secs(10));
        stop_flag.store(true, Ordering::Relaxed);
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let elapsed = start.elapsed();
        let total_ops = operations.load(Ordering::Relaxed);
        let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
        
        writeln!(output, "  Operations/sec: {:.0}", ops_per_sec)?;
    }
    
    writeln!(output)?;
    Ok(())
}

// Helper functions
fn generate_random_vectors(num_vectors: usize, dimension: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..num_vectors)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn percentile(times: &mut Vec<Duration>, p: f64) -> Duration {
    times.sort();
    let idx = ((times.len() as f64 - 1.0) * p) as usize;
    times[idx]
}

// Extension trait for timeout support
trait IndexBuilderExt {
    fn build_with_timeout(self, vectors: Vec<Vec<f32>>, timeout: Instant) -> Result<Box<dyn Index>>;
}

impl IndexBuilderExt for IndexBuilder {
    fn build_with_timeout(self, vectors: Vec<Vec<f32>>, timeout: Instant) -> Result<Box<dyn Index>> {
        // This is a simplified version - in reality we'd need to modify the actual build process
        // to check for timeout during construction
        if Instant::now() > timeout {
            return Err(anyhow::anyhow!("Timeout"));
        }
        self.build_from_vectors(vectors)
    }
}

use std::sync::atomic::AtomicU64;
use rand::Rng;