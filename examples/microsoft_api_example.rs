//! Example using the Microsoft DiskANN Rust API compatibility layer
//!
//! This example shows how to use our pure Rust implementation through
//! the Microsoft DiskANN-compatible API.

use diskann::external::{
    create_inmem_index, create_disk_index,
    IndexConfiguration, IndexWriteParametersBuilder,
    Metric, ANNResult,
    utils::Timer,
    DiskIndexBuildParameters,
    index::DiskIndexStorage,
};

fn main() -> ANNResult<()> {
    env_logger::init();
    
    println!("Microsoft DiskANN Rust API Compatibility Example");
    println!("==============================================\n");
    
    // Example 1: Build and search in-memory index
    example_inmem_index()?;
    
    // Example 2: Build and search disk-based index
    example_disk_index()?;
    
    Ok(())
}

/// Example using in-memory index with Microsoft API
fn example_inmem_index() -> ANNResult<()> {
    println!("=== In-Memory Index Example ===\n");
    
    // Generate test data
    let num_points = 1000;
    let dimension = 128;
    let data_path = "test_data.fbin";
    
    // Create test data file
    create_test_data(data_path, num_points, dimension)?;
    
    // Build parameters (matching Microsoft API)
    let index_write_params = IndexWriteParametersBuilder::new(100, 64) // L=100, R=64
        .with_alpha(1.2)
        .with_saturate_graph(false)
        .with_num_threads(0) // Use all threads
        .build();
    
    // Index configuration
    let config = IndexConfiguration::new(
        Metric::L2,
        dimension,
        dimension, // aligned_dim same as dim for simplicity
        num_points,
        false, // use_pq_dist
        0,     // num_frozen_pts
        false, // use_opq
        0,     // num_pq_chunks
        1.0,   // growth_potential
        index_write_params,
    );
    
    // Create index using factory function (Microsoft-style)
    let mut index = create_inmem_index::<f32>(config)?;
    
    // Build index from data file
    let timer = Timer::new();
    println!("Building index from {}...", data_path);
    index.build(data_path, num_points)?;
    println!("Index built in {:.2}s", timer.elapsed().as_secs_f64());
    
    // Save index
    let index_path = "test_index";
    index.save(index_path)?;
    println!("Index saved to {}", index_path);
    
    // Search example
    let query = vec![0.5f32; dimension];
    let k = 10;
    let l_search = 100;
    let mut indices = vec![0u32; k];
    
    let timer = Timer::new();
    let num_found = index.search(&query, k, l_search, &mut indices)?;
    let search_time = timer.elapsed();
    
    println!("\nSearch results:");
    println!("Found {} neighbors in {:.3}ms", num_found, search_time.as_micros() as f64 / 1000.0);
    println!("Top 5 neighbors: {:?}", &indices[..5.min(num_found as usize)]);
    
    // Clean up
    std::fs::remove_file(data_path).ok();
    std::fs::remove_file(index_path).ok();
    
    Ok(())
}

/// Example using disk-based index with Microsoft API
fn example_disk_index() -> ANNResult<()> {
    println!("\n=== Disk-Based Index Example ===\n");
    
    // Generate larger test data
    let num_points = 10000;
    let dimension = 256;
    let data_path = "test_data_large.fbin";
    
    create_test_data(data_path, num_points, dimension)?;
    
    // Disk build parameters
    let disk_build_params = DiskIndexBuildParameters {
        search_list_size: 100,
        build_list_size: 100,
        max_degree: 64,
        build_pq_bytes: 32,
        use_opq: false,
    };
    
    // Index configuration
    let index_write_params = IndexWriteParametersBuilder::new(100, 64).build();
    let config = IndexConfiguration::new(
        Metric::Cosine,
        dimension,
        dimension, // aligned_dim
        num_points,
        true,  // use_pq_dist for disk index
        0,     // num_frozen_pts
        false, // use_opq
        32,    // num_pq_chunks
        1.0,   // growth_potential
        index_write_params,
    );
    
    // Create disk index
    let storage = DiskIndexStorage::default();
    let mut index = create_disk_index::<f32>(Some(disk_build_params), config, storage)?;
    
    // Build index
    let timer = Timer::new();
    println!("Building disk index...");
    let codebook_prefix = "test_disk_index";
    index.build(data_path, codebook_prefix)?;
    println!("Disk index built in {:.2}s", timer.elapsed().as_secs_f64());
    
    // Search
    let query = normalize_vector(vec![0.3f32; dimension]);
    let k = 10;
    let mut indices = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    
    let timer = Timer::new();
    let num_found = index.search(&query, k, &mut indices, &mut distances)?;
    let search_time = timer.elapsed();
    
    println!("\nDisk search results:");
    println!("Found {} neighbors in {:.3}ms", num_found, search_time.as_micros() as f64 / 1000.0);
    for i in 0..5.min(num_found as usize) {
        println!("  Neighbor {}: id={}, distance={:.4}", i, indices[i], distances[i]);
    }
    
    // Clean up
    std::fs::remove_file(data_path).ok();
    std::fs::remove_file(format!("{}.pq", codebook_prefix)).ok();
    
    Ok(())
}

/// Helper to create test data file
fn create_test_data(path: &str, num_points: usize, dimension: usize) -> ANNResult<()> {
    use diskann::formats::write_fvecs;
    use rand::Rng;
    
    // Generate random vectors
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..num_points)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect();
    
    // For .fbin files, write raw binary without header
    if path.ends_with(".fbin") {
        use std::fs::File;
        use byteorder::{LittleEndian, WriteBytesExt};
        
        let mut file = File::create(path)?;
        
        // Write raw float data (no header for .fbin)
        for vector in &vectors {
            for &value in vector {
                file.write_f32::<LittleEndian>(value)?;
            }
        }
    } else {
        // Use fvecs format for other files
        write_fvecs(path, &vectors)?;
    }
    
    Ok(())
}

/// Normalize a vector for cosine similarity
fn normalize_vector(mut vec: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vec {
            *x /= norm;
        }
    }
    vec
}

/// Example showing dynamic index operations
#[allow(dead_code)]
fn example_dynamic_operations() -> ANNResult<()> {
    println!("\n=== Dynamic Index Operations Example ===\n");
    
    let dimension = 128;
    let initial_points = 1000;
    let data_path = "test_dynamic.fbin";
    let insert_path = "test_insert.fbin";
    
    // Create initial data
    create_test_data(data_path, initial_points, dimension)?;
    create_test_data(insert_path, 100, dimension)?;
    
    // Build index
    let params = IndexWriteParametersBuilder::new(100, 64).build();
    let config = IndexConfiguration::new(
        Metric::L2,
        dimension,
        dimension, // aligned_dim
        initial_points + 1000, // Allow growth
        false, // use_pq_dist
        1,     // 1 frozen point for dynamic index
        false, // use_opq
        0,     // num_pq_chunks
        1.2,   // 20% growth potential
        params,
    );
    
    let mut index = create_inmem_index::<f32>(config)?;
    index.build(data_path, initial_points)?;
    
    // Insert new points
    println!("Inserting 100 new points...");
    index.insert(insert_path, 100)?;
    
    // Delete some points
    println!("Deleting points 10-19...");
    let ids_to_delete: Vec<u32> = (10..20).collect();
    index.soft_delete(ids_to_delete.clone(), ids_to_delete.len())?;
    
    println!("Dynamic operations completed!");
    
    // Clean up
    std::fs::remove_file(data_path).ok();
    std::fs::remove_file(insert_path).ok();
    
    Ok(())
}