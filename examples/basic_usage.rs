//! Basic usage example for DiskANN

use diskann::{Distance, IndexBuilder};
use rand::prelude::*;

fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DiskANN Rust Example - Basic Usage\n");
    
    // Generate some random vectors
    let dimension = 128;
    let num_vectors = 1000;
    let vectors = generate_random_vectors(num_vectors, dimension);
    
    println!("Building index with {} vectors of dimension {}...", num_vectors, dimension);
    
    // Build the index
    let start = std::time::Instant::now();
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .max_degree(32)
        .search_list_size(50)
        .alpha(1.2)
        .build_from_vectors(vectors.clone())?;
    
    let build_time = start.elapsed();
    println!("Index built in {:?}\n", build_time);
    
    // Print index information
    println!("Index Information:");
    println!("  Size: {} vectors", index.size());
    println!("  Dimension: {}", index.dimension());
    println!("  Metric: {:?}", index.metric());
    
    // Perform some searches
    println!("\nPerforming searches...");
    
    // Search with first few vectors
    for i in 0..3 {
        println!("\nSearching with vector {}", i);
        let query = &vectors[i];
        
        let start = std::time::Instant::now();
        let results = index.search(query, 5)?;
        let search_time = start.elapsed();
        
        println!("  Search completed in {:?}", search_time);
        println!("  Top 5 results:");
        
        for (idx, (id, distance)) in results.iter().enumerate() {
            println!("    {}. Vector {} - Distance: {:.4}", idx + 1, id, distance);
        }
    }
    
    // Create a random query
    println!("\nSearching with random query vector...");
    let random_query = generate_random_vectors(1, dimension).into_iter().next().unwrap();
    
    let start = std::time::Instant::now();
    let results = index.search(&random_query, 10)?;
    let search_time = start.elapsed();
    
    println!("  Search completed in {:?}", search_time);
    println!("  Top 10 results:");
    
    for (idx, (id, distance)) in results.iter().enumerate() {
        println!("    {}. Vector {} - Distance: {:.4}", idx + 1, id, distance);
    }
    
    // Performance summary
    let total_search_time = std::time::Duration::from_micros(100); // Placeholder
    let qps = 1_000_000.0 / search_time.as_micros() as f64;
    
    println!("\nPerformance Summary:");
    println!("  Build time: {:?}", build_time);
    println!("  Average search time: {:?}", search_time);
    println!("  Queries per second: {:.0}", qps);
    
    // Check for SIMD support
    println!("\nSystem Capabilities:");
    println!("  ARM64 NEON: {}", if diskann::has_neon_support() { "✓" } else { "✗" });
    println!("  x86-64 AVX2: {}", if diskann::has_avx2_support() { "✓" } else { "✗" });
    
    Ok(())
}