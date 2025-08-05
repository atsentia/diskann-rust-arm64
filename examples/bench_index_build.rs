//! Focused benchmark for index building performance
//! Tests Vamana graph construction with various dataset sizes

use anyhow::Result;
use diskann::{Distance, IndexBuilder};
use std::time::Instant;
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
    println!("{}", style("DiskANN Index Building Benchmark").bold().green());
    println!("{}", style("Testing Vamana graph construction performance").dim());
    println!();

    print_system_info();
    
    // Test parameters
    let dimensions = vec![128, 768];  // Common embedding dimensions
    let dataset_sizes = vec![1000, 5000, 10000, 25000];
    let max_degree = 32;
    let search_list_size = 100;
    let alpha = 1.2;
    
    println!("\n{}", style("=== Benchmark Configuration ===").bold().blue());
    println!("Algorithm: Vamana (RobustPrune)");
    println!("Max degree (R): {}", max_degree);
    println!("Search list size (L): {}", search_list_size);
    println!("Alpha: {}", alpha);
    println!("Dataset sizes: {:?}", dataset_sizes);
    println!("Dimensions: {:?}", dimensions);
    println!("{}", style("─".repeat(60)).dim());

    for dim in dimensions {
        println!("\n{}", style(format!("=== Testing Dimension: {} ===", dim)).bold().yellow());
        println!("{:<10} {:<15} {:<15} {:<15} {:<15}", 
            "Vectors", "Build Time", "Vectors/sec", "MB/sec", "Memory (MB)");
        println!("{}", style("─".repeat(75)).dim());

        for &num_vectors in &dataset_sizes {
            // Generate random vectors
            let mut vectors = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                let vec: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
                vectors.push(vec);
            }
            
            // Measure memory before
            let memory_before = get_memory_usage();
            
            // Build index
            let start = Instant::now();
            let index = IndexBuilder::new()
                .dimensions(dim)
                .metric(Distance::L2)
                .max_degree(max_degree)
                .search_list_size(search_list_size)
                .alpha(alpha)
                .build_from_vectors(vectors.clone())?;
            let build_time = start.elapsed();
            
            // Measure memory after
            let memory_after = get_memory_usage();
            let memory_used_mb = (memory_after - memory_before) as f64 / 1_048_576.0;
            
            // Calculate metrics
            let vectors_per_sec = num_vectors as f64 / build_time.as_secs_f64();
            let mb_per_sec = (num_vectors * dim * 4) as f64 / 1_048_576.0 / build_time.as_secs_f64();
            
            println!("{:<10} {:<15.2?} {:<15.0} {:<15.2} {:<15.2}", 
                num_vectors, 
                build_time,
                vectors_per_sec,
                mb_per_sec,
                memory_used_mb
            );
        }
    }
    
    // Test different parameters impact on 10K vectors
    println!("\n{}", style("=== Parameter Impact Analysis ===").bold().blue());
    println!("Testing on 10,000 vectors, dimension 128");
    println!("{:<15} {:<15} {:<15} {:<15}", 
        "Max Degree", "Search Size", "Build Time", "Avg Degree");
    println!("{}", style("─".repeat(60)).dim());
    
    let test_vectors: Vec<Vec<f32>> = (0..10000)
        .map(|_| (0..128).map(|_| rand::random::<f32>()).collect())
        .collect();
    
    let test_params = vec![
        (16, 50),
        (32, 100),
        (64, 150),
        (128, 200),
    ];
    
    for (r, l) in test_params {
        let start = Instant::now();
        let index = IndexBuilder::new()
            .dimensions(128)
            .metric(Distance::L2)
            .max_degree(r)
            .search_list_size(l)
            .alpha(1.2)
            .build_from_vectors(test_vectors.clone())?;
        let build_time = start.elapsed();
        
        println!("{:<15} {:<15} {:<15.2?} {:<15}", 
            r, l, build_time, "N/A");
    }
    
    println!("\n{}", style("✓ Benchmark completed successfully").green());
    
    Ok(())
}

fn get_memory_usage() -> usize {
    // Placeholder - would use proper memory tracking in production
    0
}