//! Performance Benchmarking Tool
//!
//! Run comprehensive performance tests on DiskANN implementation

use diskann::{Distance, IndexBuilder, DynamicIndex, Result};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use rand::Rng;

#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    operations: usize,
    duration: Duration,
    throughput: f64,
    latency_ms: f64,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("{:<30} {:>10} ops in {:>8.2} ms = {:>10.0} ops/sec (latency: {:>6.2} ms)",
                 self.name,
                 self.operations,
                 self.duration.as_millis(),
                 self.throughput,
                 self.latency_ms);
    }
}

fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn benchmark<F>(name: &str, operations: usize, mut f: F) -> BenchmarkResult
where
    F: FnMut(),
{
    // Warm-up
    for _ in 0..5 {
        f();
    }
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..operations {
        f();
    }
    let duration = start.elapsed();
    
    let throughput = operations as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_millis() as f64 / operations as f64;
    
    BenchmarkResult {
        name: name.to_string(),
        operations,
        duration,
        throughput,
        latency_ms,
    }
}

fn run_distance_benchmarks(dimension: usize) -> Result<Vec<BenchmarkResult>> {
    println!("\n=== Distance Function Benchmarks (dim={}) ===", dimension);
    
    let mut results = Vec::new();
    let batch_size = 1000;
    let iterations = 100;
    
    let query = generate_random_vectors(1, dimension)[0].clone();
    let points: Vec<f32> = generate_random_vectors(batch_size, dimension)
        .into_iter()
        .flatten()
        .collect();
    let mut distances = vec![0.0; batch_size];
    
    // L2 Distance
    let l2_fn = diskann::create_distance_function(Distance::L2, dimension);
    let result = benchmark(
        &format!("L2 Distance ({}x{})", batch_size, dimension),
        iterations,
        || {
            l2_fn.batch_distance(&query, &points, &mut distances).unwrap();
        },
    );
    result.print();
    results.push(result);
    
    // Cosine Distance
    let cosine_fn = diskann::create_distance_function(Distance::Cosine, dimension);
    let result = benchmark(
        &format!("Cosine Distance ({}x{})", batch_size, dimension),
        iterations,
        || {
            cosine_fn.batch_distance(&query, &points, &mut distances).unwrap();
        },
    );
    result.print();
    results.push(result);
    
    // Inner Product
    let dot_fn = diskann::create_distance_function(Distance::InnerProduct, dimension);
    let result = benchmark(
        &format!("Inner Product ({}x{})", batch_size, dimension),
        iterations,
        || {
            dot_fn.batch_distance(&query, &points, &mut distances).unwrap();
        },
    );
    result.print();
    results.push(result);
    
    Ok(results)
}

fn run_index_build_benchmarks() -> Result<Vec<BenchmarkResult>> {
    println!("\n=== Index Build Benchmarks ===");
    
    let mut results = Vec::new();
    let configs = vec![
        (1000, 128, "Small"),
        (5000, 256, "Medium"),
        (10000, 768, "Large"),
    ];
    
    for (size, dim, label) in configs {
        let vectors = generate_random_vectors(size, dim);
        
        let result = benchmark(
            &format!("Build {} Index ({}x{})", label, size, dim),
            1,
            || {
                let _index = IndexBuilder::new()
                    .dimensions(dim)
                    .metric(Distance::L2)
                    .max_degree(64)
                    .search_list_size(100)
                    .build_from_vectors(vectors.clone())
                    .unwrap();
            },
        );
        result.print();
        results.push(result);
    }
    
    Ok(results)
}

fn run_search_benchmarks() -> Result<Vec<BenchmarkResult>> {
    println!("\n=== Search Benchmarks ===");
    
    let mut results = Vec::new();
    let size = 10000;
    let dim = 256;
    let k = 10;
    let num_queries = 1000;
    
    // Build index
    println!("Building index with {} vectors...", size);
    let vectors = generate_random_vectors(size, dim);
    let index = IndexBuilder::new()
        .dimensions(dim)
        .metric(Distance::L2)
        .max_degree(64)
        .search_list_size(100)
        .build_from_vectors(vectors)
        .unwrap();
    
    // Generate queries
    let queries = generate_random_vectors(num_queries, dim);
    
    // Single query benchmark
    let result = benchmark(
        &format!("Single Query Search (k={})", k),
        num_queries,
        || {
            let idx = rand::thread_rng().gen_range(0..queries.len());
            let _results = index.search(&queries[idx], k).unwrap();
        },
    );
    result.print();
    results.push(result);
    
    // Batch search benchmark
    let batch_size = 100;
    let result = benchmark(
        &format!("Batch Search ({} queries)", batch_size),
        num_queries / batch_size,
        || {
            let start_idx = rand::thread_rng().gen_range(0..queries.len() - batch_size);
            for i in 0..batch_size {
                let _results = index.search(&queries[start_idx + i], k).unwrap();
            }
        },
    );
    result.print();
    results.push(result);
    
    Ok(results)
}

fn run_dynamic_index_benchmarks() -> Result<Vec<BenchmarkResult>> {
    println!("\n=== Dynamic Index Benchmarks ===");
    
    let mut results = Vec::new();
    let initial_size = 5000;
    let dim = 256;
    let operations = 100;
    
    // Build initial index
    println!("Building dynamic index with {} vectors...", initial_size);
    let initial_vectors = generate_random_vectors(initial_size, dim);
    let mut index = DynamicIndex::build_from_vectors(
        initial_vectors,
        dim,
        Distance::L2,
        64,
        100,
        1.2,
    ).unwrap();
    
    // Insertion benchmark
    let new_vectors = generate_random_vectors(operations, dim);
    let result = benchmark(
        "Dynamic Insert",
        operations,
        || {
            let idx = rand::thread_rng().gen_range(0..new_vectors.len());
            let _id = index.insert(new_vectors[idx].clone()).unwrap();
        },
    );
    result.print();
    results.push(result);
    
    // Search on dynamic index
    let queries = generate_random_vectors(operations, dim);
    let result = benchmark(
        "Dynamic Search",
        operations,
        || {
            let idx = rand::thread_rng().gen_range(0..queries.len());
            let _results = index.search(&queries[idx], 10).unwrap();
        },
    );
    result.print();
    results.push(result);
    
    // Deletion benchmark
    let result = benchmark(
        "Dynamic Delete",
        operations / 2,
        || {
            let id = rand::thread_rng().gen_range(0..initial_size);
            let _ = index.delete(id);
        },
    );
    result.print();
    results.push(result);
    
    Ok(results)
}

fn print_summary(all_results: &HashMap<String, Vec<BenchmarkResult>>) {
    println!("\n=== Performance Summary ===");
    println!("{:-<80}", "");
    
    // Distance function summary
    if let Some(distance_results) = all_results.get("distance") {
        println!("\nDistance Functions (vectors/second):");
        for result in distance_results {
            let vectors_per_sec = result.throughput * 1000.0; // 1000 vectors per batch
            println!("  {:<30} {:>15.0} vectors/sec", 
                     result.name.split(" (").next().unwrap_or(&result.name), 
                     vectors_per_sec);
        }
    }
    
    // Index build summary
    if let Some(build_results) = all_results.get("build") {
        println!("\nIndex Build (vectors/second):");
        for result in build_results {
            // Extract vector count from name
            let parts: Vec<&str> = result.name.split(&['(', 'x'][..]).collect();
            if parts.len() >= 2 {
                if let Ok(count) = parts[1].parse::<f64>() {
                    let build_rate = count / result.duration.as_secs_f64();
                    println!("  {:<30} {:>15.0} vectors/sec", 
                             parts[0].trim(),
                             build_rate);
                }
            }
        }
    }
    
    // Search summary
    if let Some(search_results) = all_results.get("search") {
        println!("\nSearch Performance:");
        for result in search_results {
            println!("  {:<30} {:>15.0} queries/sec", 
                     result.name, 
                     result.throughput);
        }
    }
    
    println!("{:-<80}", "");
}

fn main() -> Result<()> {
    println!("DiskANN Performance Benchmark Suite");
    println!("===================================");
    
    // Enable logging to see which SIMD/GPU implementation is used
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    
    let mut all_results = HashMap::new();
    
    // Run distance benchmarks for different dimensions
    let mut distance_results = Vec::new();
    for dim in [128, 256, 768, 1024] {
        distance_results.extend(run_distance_benchmarks(dim)?);
    }
    all_results.insert("distance".to_string(), distance_results);
    
    // Run index build benchmarks
    let build_results = run_index_build_benchmarks()?;
    all_results.insert("build".to_string(), build_results);
    
    // Run search benchmarks
    let search_results = run_search_benchmarks()?;
    all_results.insert("search".to_string(), search_results);
    
    // Run dynamic index benchmarks
    let dynamic_results = run_dynamic_index_benchmarks()?;
    all_results.insert("dynamic".to_string(), dynamic_results);
    
    // Print summary
    print_summary(&all_results);
    
    println!("\nBenchmark complete!");
    
    Ok(())
}