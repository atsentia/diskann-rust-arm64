//! Example demonstrating Microsoft DiskANN C++ API compatibility
//!
//! This example shows how to use the compatibility layer that matches
//! the C++ DiskANN API for easy migration.

use diskann::compat::{
    DiskANNIndex, BuildParams, SearchParams, 
    build_index, search_index,
    METRIC_L2, METRIC_IP, METRIC_COSINE,
};
use diskann::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("DiskANN C++ API Compatibility Example");
    println!("=====================================\n");
    
    // Generate test data
    let num_vectors = 10000;
    let dimension = 128;
    let num_queries = 100;
    
    println!("Generating {} vectors of dimension {}...", num_vectors, dimension);
    let vectors = diskann::utils::generate_random_vectors(num_vectors, dimension);
    let queries = diskann::utils::generate_random_vectors(num_queries, dimension);
    
    // Save data in fvecs format (C++ compatible)
    let data_file = "test_data.fvecs";
    let query_file = "test_queries.fvecs";
    diskann::formats::write_fvecs(data_file, &vectors)?;
    diskann::formats::write_fvecs(query_file, &queries)?;
    
    // Example 1: Build and search using function API (matches C++ build/search functions)
    example_function_api(data_file, query_file)?;
    
    // Example 2: Build and search using object API (matches C++ Index class)
    example_object_api(data_file, &queries)?;
    
    // Example 3: Build disk-based index
    example_disk_index(data_file, &queries)?;
    
    // Clean up
    std::fs::remove_file(data_file).ok();
    std::fs::remove_file(query_file).ok();
    
    println!("\nAll examples completed successfully!");
    
    Ok(())
}

/// Example using C++ style function API
fn example_function_api(data_file: &str, query_file: &str) -> Result<()> {
    println!("\n=== Example 1: Function API (C++ style) ===");
    
    // Build parameters (matches C++ BuildParameters)
    let params = BuildParams {
        num_threads: 0,        // 0 = use all available
        max_degree: 64,        // R in paper
        search_list_size: 100, // L in paper
        alpha: 1.2,
        ..Default::default()
    };
    
    // Build index
    let index_prefix = "test_index";
    println!("Building index...");
    let start = Instant::now();
    build_index(data_file, index_prefix, &params, METRIC_L2)?;
    println!("Index built in {:.2}s", start.elapsed().as_secs_f64());
    
    // Search parameters
    let search_params = SearchParams {
        search_list_size: 200,
        beamwidth: 2,
        reorder_data: true,
    };
    
    // Search index
    println!("Searching index...");
    let result_file = "search_results.bin";
    let start = Instant::now();
    search_index(
        index_prefix,
        query_file,
        result_file,
        10000,  // num_points
        128,    // dimension
        10,     // k
        &search_params,
        METRIC_L2,
    )?;
    println!("Search completed in {:.2}s", start.elapsed().as_secs_f64());
    
    // Clean up
    std::fs::remove_file(format!("{}.bin", index_prefix)).ok();
    std::fs::remove_file(result_file).ok();
    
    Ok(())
}

/// Example using C++ style object API
fn example_object_api(data_file: &str, queries: &[Vec<f32>]) -> Result<()> {
    println!("\n=== Example 2: Object API (C++ Index class) ===");
    
    // Create build parameters
    let params = BuildParams::default();
    
    // Build index from data file
    println!("Building index...");
    let index_prefix = "test_index_obj";
    let start = Instant::now();
    let index = DiskANNIndex::build(data_file, index_prefix, &params, METRIC_L2)?;
    println!("Index built in {:.2}s", start.elapsed().as_secs_f64());
    
    // Get index statistics
    let stats = index.get_stats();
    println!("Index stats:");
    println!("  Num points: {}", stats.num_points);
    println!("  Dimension: {}", stats.dimension);
    println!("  Avg degree: {:.2}", stats.graph_degree);
    println!("  Memory usage: {:.2} MB", stats.memory_usage as f64 / (1024.0 * 1024.0));
    
    // Search
    let search_params = SearchParams::default();
    let k = 10;
    let mut total_time = 0.0;
    
    println!("Searching {} queries...", queries.len());
    for query in queries {
        let mut neighbors = vec![0u32; k];
        let mut distances = vec![0.0f32; k];
        
        let start = Instant::now();
        let count = index.search(
            query,
            k,
            &search_params,
            &mut neighbors,
            &mut distances,
        )?;
        total_time += start.elapsed().as_secs_f64();
        
        // First query result
        if query == &queries[0] {
            println!("First query results (found {} neighbors):", count);
            for i in 0..count.min(5) as usize {
                println!("  Neighbor {}: id={}, dist={:.4}", 
                         i, neighbors[i], distances[i]);
            }
        }
    }
    
    let avg_time_ms = (total_time / queries.len() as f64) * 1000.0;
    println!("Average search time: {:.2} ms", avg_time_ms);
    println!("Queries per second: {:.0}", 1000.0 / avg_time_ms);
    
    // Clean up
    std::fs::remove_file(format!("{}.bin", index_prefix)).ok();
    
    Ok(())
}

/// Example using disk-based index
fn example_disk_index(data_file: &str, queries: &[Vec<f32>]) -> Result<()> {
    println!("\n=== Example 3: Disk-based Index (PQ) ===");
    
    // Build parameters for PQ index
    let params = BuildParams {
        use_pq_build: true,
        num_pq_chunks: 16,  // 128 / 8 = 16 chunks
        ..Default::default()
    };
    
    // Build PQ index
    println!("Building PQ index...");
    let index_prefix = "test_index_pq";
    let start = Instant::now();
    let index = DiskANNIndex::build(data_file, index_prefix, &params, METRIC_COSINE)?;
    println!("PQ index built in {:.2}s", start.elapsed().as_secs_f64());
    
    // Get index statistics
    let stats = index.get_stats();
    println!("PQ Index stats:");
    println!("  Memory usage: {:.2} MB", stats.memory_usage as f64 / (1024.0 * 1024.0));
    
    // Batch search
    let k = 5;
    let batch_size = 10;
    let search_params = SearchParams::default();
    
    // Flatten queries for batch search
    let query_batch: Vec<f32> = queries[0..batch_size]
        .iter()
        .flatten()
        .cloned()
        .collect();
    
    let mut neighbors = vec![0u32; batch_size * k];
    let mut distances = vec![0.0f32; batch_size * k];
    
    println!("Performing batch search ({} queries)...", batch_size);
    let start = Instant::now();
    index.batch_search(
        &query_batch,
        batch_size,
        queries[0].len(),
        k,
        &search_params,
        &mut neighbors,
        &mut distances,
    )?;
    let batch_time = start.elapsed().as_secs_f64();
    
    println!("Batch search completed in {:.2} ms", batch_time * 1000.0);
    println!("Average per query: {:.2} ms", batch_time * 1000.0 / batch_size as f64);
    
    // Show first result
    println!("First query results from batch:");
    for i in 0..k {
        println!("  Neighbor {}: id={}, dist={:.4}", 
                 i, neighbors[i], distances[i]);
    }
    
    // Clean up
    std::fs::remove_file(format!("{}.pq", index_prefix)).ok();
    
    Ok(())
}

/// Example showing concurrent usage
#[allow(dead_code)]
fn example_concurrent_api() -> Result<()> {
    use diskann::compat::ConcurrentIndex;
    use std::sync::Arc;
    use std::thread;
    
    println!("\n=== Example 4: Concurrent API ===");
    
    // Build an index
    let vectors = diskann::utils::generate_random_vectors(1000, 64);
    let index = diskann::IndexBuilder::new()
        .dimensions(64)
        .metric(diskann::Distance::L2)
        .build_from_vectors(vectors)?;
    
    // Wrap in DiskANN compatibility layer
    let diskann_index = DiskANNIndex::from(index);
    let concurrent_index = Arc::new(ConcurrentIndex::new(diskann_index));
    
    // Spawn multiple search threads
    let mut handles = vec![];
    
    for thread_id in 0..4 {
        let index_clone = Arc::clone(&concurrent_index);
        let handle = thread::spawn(move || {
            let query = vec![0.5; 64];
            let params = SearchParams::default();
            
            for i in 0..25 {
                let (neighbors, distances) = index_clone
                    .search(&query, 5, &params)
                    .unwrap();
                
                if i == 0 {
                    println!("Thread {} first result: {:?}", 
                             thread_id, neighbors[0]);
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Concurrent searches completed successfully!");
    
    Ok(())
}