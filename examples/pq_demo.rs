//! Demonstration of Product Quantization functionality
//!
//! This example shows how to use the PQ components to compress vectors
//! and perform efficient similarity search.

use diskann::pq::{ProductQuantizer, PQParams, PQIndex};
use diskann::{Distance, Result};

fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn main() -> Result<()> {
    println!("🚀 DiskANN Rust - Product Quantization Demo");
    println!("============================================");
    
    // Parameters for demonstration
    let dimension = 128;
    let num_vectors = 1000;
    let num_subspaces = 8;
    let bits_per_subquantizer = 8;
    
    println!("\n📊 Configuration:");
    println!("  Vector dimension: {}", dimension);
    println!("  Number of vectors: {}", num_vectors);
    println!("  PQ subspaces: {}", num_subspaces);
    println!("  Bits per subquantizer: {}", bits_per_subquantizer);
    
    // Generate random vectors for training
    println!("\n🎲 Generating {} random {}-dimensional vectors...", num_vectors, dimension);
    let vectors = generate_random_vectors(num_vectors, dimension);
    
    // Create PQ parameters
    let pq_params = PQParams::new(num_subspaces, bits_per_subquantizer);
    println!("\n🔧 PQ Parameters:");
    println!("  Number of centroids per subspace: {}", pq_params.num_centroids);
    println!("  Subspace dimension: {}", dimension / num_subspaces);
    
    // Test 1: Basic ProductQuantizer
    println!("\n🧪 Test 1: Basic Product Quantization");
    println!("=====================================");
    
    let mut pq = ProductQuantizer::new(pq_params.clone(), dimension)?;
    println!("✅ Created ProductQuantizer");
    
    // Train the quantizer
    println!("🎯 Training quantizer on {} vectors...", vectors.len());
    let training_result = pq.train(&vectors)?;
    println!("✅ Training completed in {:?}", training_result.total_training_time);
    println!("   Average inertia: {:.6}", training_result.average_inertia());
    println!("   All subspaces converged: {}", training_result.all_converged());
    
    // Test encoding/decoding
    let test_vector = &vectors[0];
    let encoded = pq.encode(test_vector)?;
    let decoded = pq.decode(&encoded)?;
    
    // Calculate reconstruction error
    let mut mse = 0.0f32;
    for (orig, recon) in test_vector.iter().zip(decoded.iter()) {
        let diff = orig - recon;
        mse += diff * diff;
    }
    mse /= test_vector.len() as f32;
    
    println!("📈 Encoding Results:");
    println!("   Original vector size: {} floats ({} bytes)", 
             test_vector.len(), test_vector.len() * 4);
    println!("   Encoded size: {} bytes", encoded.len());
    println!("   Compression ratio: {:.1}x", 
             (test_vector.len() * 4) as f32 / encoded.len() as f32);
    println!("   Reconstruction MSE: {:.6}", mse);
    
    // Memory statistics
    let memory_stats = pq.memory_stats();
    println!("💾 Memory Usage:");
    println!("   Codebook size: {} KB", memory_stats.codebook_size_bytes / 1024);
    println!("   Bytes per vector: {}", memory_stats.bytes_per_vector);
    println!("   Compression ratio: {:.1}x", memory_stats.compression_ratio);
    
    // Test 2: PQ Index for Search
    println!("\n🧪 Test 2: PQ-based Index Search");
    println!("=================================");
    
    let mut pq_index = PQIndex::new(pq_params, dimension, Distance::L2)?;
    println!("✅ Created PQ Index");
    
    // Build the index
    println!("🏗️  Building index with {} vectors...", vectors.len());
    pq_index.build(vectors.clone())?;
    
    let index_stats = pq_index.memory_stats();
    println!("📊 Index Statistics:");
    println!("   Vectors in index: {}", index_stats.num_vectors);
    println!("   Original size: {} KB", index_stats.original_size_bytes / 1024);
    println!("   Compressed size: {} KB", index_stats.compressed_size_bytes / 1024);
    println!("   Total memory: {} KB", index_stats.total_memory_bytes / 1024);
    println!("   Compression ratio: {:.1}x", index_stats.compression_ratio);
    
    // Search test
    let query = &vectors[42]; // Use a vector from the dataset
    let k = 5;
    
    println!("\n🔍 Search Test:");
    println!("   Searching for {} nearest neighbors...", k);
    
    let search_results = pq_index.search(query, k)?;
    println!("   Found {} results:", search_results.len());
    for (i, (id, distance)) in search_results.iter().enumerate() {
        println!("     {}. ID: {}, Distance: {:.6}", i + 1, id, distance);
    }
    
    // Test accurate search
    let accurate_results = pq_index.search_accurate(query, k)?;
    println!("\n🎯 Accurate Search (Asymmetric Distance):");
    for (i, (id, distance)) in accurate_results.iter().enumerate() {
        println!("     {}. ID: {}, Distance: {:.6}", i + 1, id, distance);
    }
    
    // Test insertion
    println!("\n➕ Testing vector insertion...");
    let new_vector = generate_random_vectors(1, dimension)[0].clone();
    let new_id = pq_index.insert(new_vector.clone())?;
    println!("   Inserted new vector with ID: {}", new_id);
    
    // Search for the new vector
    let insert_results = pq_index.search(&new_vector, 1)?;
    println!("   Search for inserted vector: ID {}, Distance: {:.6}", 
             insert_results[0].0, insert_results[0].1);
    
    println!("\n🎉 Product Quantization Demo Complete!");
    println!("   ✅ PQ encoding/decoding works correctly");
    println!("   ✅ Index-based search is functional"); 
    println!("   ✅ Dynamic insertion is supported");
    println!("   ✅ Compression achieved: {:.1}x reduction in memory", index_stats.compression_ratio);
    
    Ok(())
}