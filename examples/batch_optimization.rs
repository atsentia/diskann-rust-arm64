//! Batch Size Optimization Example
//!
//! This example helps find the optimal batch size for GPU processing
//! based on your specific hardware and workload.

use diskann::{Distance, create_distance_function, Result};
use std::time::{Duration, Instant};
use rand::Rng;

struct BenchmarkResult {
    batch_size: usize,
    total_time: Duration,
    throughput: f64,
    latency_ms: f64,
    efficiency_score: f64,
}

fn benchmark_batch_size(
    distance_fn: &dyn diskann::DistanceFunction,
    dimension: usize,
    batch_size: usize,
    num_iterations: usize,
) -> Result<BenchmarkResult> {
    let mut rng = rand::thread_rng();
    
    // Generate test data
    let query: Vec<f32> = (0..dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    let points: Vec<f32> = (0..batch_size * dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    let mut distances = vec![0.0; batch_size];
    
    // Warm-up
    for _ in 0..5 {
        distance_fn.batch_distance(&query, &points, &mut distances)?;
    }
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..num_iterations {
        distance_fn.batch_distance(&query, &points, &mut distances)?;
    }
    let total_time = start.elapsed();
    
    let total_vectors = batch_size * num_iterations;
    let throughput = total_vectors as f64 / total_time.as_secs_f64();
    let latency_ms = total_time.as_millis() as f64 / num_iterations as f64;
    
    // Efficiency score: throughput per latency (higher is better)
    let efficiency_score = throughput / latency_ms;
    
    Ok(BenchmarkResult {
        batch_size,
        total_time,
        throughput,
        latency_ms,
        efficiency_score,
    })
}

fn find_optimal_batch_size(
    distance_fn: &dyn diskann::DistanceFunction,
    dimension: usize,
) -> Result<()> {
    println!("Finding optimal batch size for dimension {}...\n", dimension);
    
    // Test various batch sizes
    let batch_sizes = vec![
        1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    ];
    
    let mut results = Vec::new();
    
    for &batch_size in &batch_sizes {
        // Adjust iterations based on batch size for consistent test duration
        let num_iterations = (100_000 / batch_size).max(10).min(1000);
        
        print!("Testing batch size {:6}... ", batch_size);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        match benchmark_batch_size(distance_fn, dimension, batch_size, num_iterations) {
            Ok(result) => {
                println!("✓ {:.0} vectors/sec, {:.2} ms latency", 
                         result.throughput, result.latency_ms);
                results.push(result);
            }
            Err(e) => {
                println!("✗ Failed: {}", e);
            }
        }
    }
    
    if results.is_empty() {
        println!("No successful benchmarks!");
        return Ok(());
    }
    
    // Find best configurations
    let best_throughput = results.iter()
        .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
        .unwrap();
    
    let best_latency = results.iter()
        .min_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap())
        .unwrap();
    
    let best_efficiency = results.iter()
        .max_by(|a, b| a.efficiency_score.partial_cmp(&b.efficiency_score).unwrap())
        .unwrap();
    
    // Print results table
    println!("\n{:-<80}", "");
    println!("{:>10} | {:>15} | {:>15} | {:>15} | {:>15}", 
             "Batch Size", "Throughput", "Latency (ms)", "Efficiency", "Recommendation");
    println!("{:-<80}", "");
    
    for result in &results {
        let recommendation = if result.batch_size == best_throughput.batch_size {
            "Best Throughput"
        } else if result.batch_size == best_latency.batch_size {
            "Best Latency"
        } else if result.batch_size == best_efficiency.batch_size {
            "Best Overall"
        } else {
            ""
        };
        
        println!("{:>10} | {:>15.0} | {:>15.2} | {:>15.2} | {:>15}",
                 result.batch_size,
                 result.throughput,
                 result.latency_ms,
                 result.efficiency_score,
                 recommendation);
    }
    println!("{:-<80}", "");
    
    // Recommendations
    println!("\nRecommendations:");
    println!("================");
    println!("✓ For maximum throughput: batch size = {} ({:.0} vectors/sec)",
             best_throughput.batch_size, best_throughput.throughput);
    println!("✓ For minimum latency: batch size = {} ({:.2} ms)",
             best_latency.batch_size, best_latency.latency_ms);
    println!("✓ For best overall efficiency: batch size = {}",
             best_efficiency.batch_size);
    
    // Memory usage estimation
    let memory_per_batch = |size: usize| {
        let vectors_mb = (size * dimension * 4) as f64 / (1024.0 * 1024.0);
        let distances_mb = (size * 4) as f64 / (1024.0 * 1024.0);
        vectors_mb + distances_mb
    };
    
    println!("\nMemory Usage:");
    println!("=============");
    for size in [256, 1024, 4096, 16384, 65536].iter() {
        println!("Batch size {:6}: {:>8.2} MB", size, memory_per_batch(*size));
    }
    
    // GPU vs CPU recommendation
    let cpu_optimal = results.iter()
        .filter(|r| r.batch_size <= 256)
        .max_by(|a, b| a.efficiency_score.partial_cmp(&b.efficiency_score).unwrap());
    
    let gpu_optimal = results.iter()
        .filter(|r| r.batch_size > 256)
        .max_by(|a, b| a.efficiency_score.partial_cmp(&b.efficiency_score).unwrap());
    
    if let (Some(cpu), Some(gpu)) = (cpu_optimal, gpu_optimal) {
        let speedup = gpu.throughput / cpu.throughput;
        println!("\nGPU Advantage:");
        println!("==============");
        println!("CPU optimal (batch={}): {:.0} vectors/sec", cpu.batch_size, cpu.throughput);
        println!("GPU optimal (batch={}): {:.0} vectors/sec", gpu.batch_size, gpu.throughput);
        println!("GPU speedup: {:.1}x", speedup);
    }
    
    Ok(())
}

fn main() -> Result<()> {
    println!("Batch Size Optimization Tool");
    println!("============================\n");
    
    // Enable debug logging to see which implementation is selected
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    
    // Test different dimensions
    let dimensions = vec![128, 256, 512, 768, 1024, 2048];
    
    for dimension in dimensions {
        println!("\n{:=<80}\n", format!(" Dimension: {} ", dimension));
        
        let distance_fn = create_distance_function(Distance::L2, dimension);
        find_optimal_batch_size(distance_fn.as_ref(), dimension)?;
        
        println!("\nPress Enter to continue to next dimension...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
    }
    
    println!("\nGeneral Guidelines:");
    println!("===================");
    println!("1. Batch sizes 1-32: CPU SIMD is usually optimal");
    println!("2. Batch sizes 64-256: Performance crossover point");
    println!("3. Batch sizes 512+: GPU provides significant advantage");
    println!("4. Larger dimensions benefit more from GPU acceleration");
    println!("5. Consider memory constraints for very large batches");
    
    Ok(())
}