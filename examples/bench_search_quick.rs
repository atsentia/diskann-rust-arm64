//! Quick search performance benchmark - runs in under 60 seconds
//! Tests search latency and throughput on small datasets

use anyhow::Result;
use diskann::{Distance, IndexBuilder};
use std::time::{Duration, Instant};
use console::style;

fn print_system_info() {
    println!("{}", style("=== System Information ===").bold().blue());
    println!("Platform: {}", std::env::consts::OS);
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("CPU: {} cores", num_cpus::get());
    
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    println!("Device: Qualcomm Snapdragon X (ARM64)");
    
    println!("{}", style("─".repeat(60)).dim());
}

fn main() -> Result<()> {
    println!("{}", style("DiskANN Quick Search Benchmark").bold().green());
    println!("{}", style("Lightweight benchmark - completes in <60 seconds").dim());
    println!();

    print_system_info();
    
    // Lightweight test parameters
    let num_vectors = 5000;  // Small dataset
    let num_queries = 100;   // Limited queries
    let dimensions = vec![128, 768];
    let k_values = vec![10, 50];
    
    println!("\n{}", style("=== Benchmark Configuration ===").bold().blue());
    println!("Dataset size: {} vectors", num_vectors);
    println!("Query count: {}", num_queries);
    println!("Dimensions: {:?}", dimensions);
    println!("K values: {:?}", k_values);
    println!("{}", style("─".repeat(60)).dim());

    for dim in dimensions {
        println!("\n{}", style(format!("=== Dimension: {} ===", dim)).bold().yellow());
        
        // Generate test data
        println!("Generating {} vectors...", num_vectors);
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        // Build index
        println!("Building index...");
        let start = Instant::now();
        let index = IndexBuilder::new()
            .dimensions(dim)
            .metric(Distance::L2)
            .max_degree(32)
            .search_list_size(100)
            .build_from_vectors(vectors)?;
        println!("Index built in {:?}", start.elapsed());
        
        // Test different K values
        println!("\n{:<5} {:<15} {:<15} {:<15} {:<15}", 
            "K", "Avg (μs)", "P50 (μs)", "P90 (μs)", "P99 (μs)");
        println!("{}", style("─".repeat(65)).dim());
        
        for &k in &k_values {
            let mut latencies = Vec::new();
            
            // Warm up
            for i in 0..10 {
                let _ = index.search(&queries[i % queries.len()], k)?;
            }
            
            // Measure
            for query in &queries {
                let start = Instant::now();
                let _ = index.search(query, k)?;
                latencies.push(start.elapsed());
            }
            
            // Sort for percentiles
            latencies.sort();
            
            let avg = latencies.iter().sum::<Duration>() / latencies.len() as u32;
            let p50 = latencies[latencies.len() / 2];
            let p90 = latencies[latencies.len() * 9 / 10];
            let p99 = latencies[latencies.len() * 99 / 100];
            
            println!("{:<5} {:<15.1} {:<15.1} {:<15.1} {:<15.1}", 
                k,
                avg.as_micros() as f64,
                p50.as_micros() as f64,
                p90.as_micros() as f64,
                p99.as_micros() as f64
            );
        }
        
        // Quick throughput test
        println!("\n{} Throughput Test (1 second):", style("►").cyan());
        let start = Instant::now();
        let mut count = 0;
        let one_second = Duration::from_secs(1);
        
        while start.elapsed() < one_second {
            let _ = index.search(&queries[count % queries.len()], 10)?;
            count += 1;
        }
        
        println!("  Queries per second: {}", count);
        println!("  Average latency: {:.2} μs", 
            (1_000_000.0 / count as f64));
    }
    
    println!("\n{}", style("✓ Quick benchmark completed").green());
    println!("Total runtime: {:?}", std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)?
        .as_secs() % 3600);
    
    Ok(())
}