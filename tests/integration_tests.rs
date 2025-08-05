//! Integration tests for DiskANN implementation

use diskann::{Distance, IndexBuilder};
use approx::assert_relative_eq;

/// Generate synthetic vectors for testing
fn generate_clustered_vectors(
    num_clusters: usize,
    vectors_per_cluster: usize,
    dimension: usize,
    cluster_spread: f32,
) -> Vec<Vec<f32>> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    let mut vectors = Vec::new();
    
    for cluster_id in 0..num_clusters {
        // Generate cluster center
        let center: Vec<f32> = (0..dimension)
            .map(|_| rng.gen_range(-10.0..10.0))
            .collect();
        
        // Generate vectors around center
        for _ in 0..vectors_per_cluster {
            let vector: Vec<f32> = center
                .iter()
                .map(|&c| c + rng.gen_range(-cluster_spread..cluster_spread))
                .collect();
            vectors.push(vector);
        }
    }
    
    // Shuffle vectors
    vectors.shuffle(&mut rng);
    vectors
}

#[test]
fn test_basic_index_operations() {
    let dimension = 64;
    let vectors = generate_clustered_vectors(5, 20, dimension, 0.5);
    
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .max_degree(16)
        .search_list_size(32)
        .build_from_vectors(vectors.clone())
        .expect("Failed to build index");
    
    assert_eq!(index.size(), 100);
    assert_eq!(index.dimension(), dimension);
    assert_eq!(index.metric(), Distance::L2);
}

#[test]
fn test_search_accuracy() {
    let dimension = 32;
    let vectors = generate_clustered_vectors(10, 10, dimension, 0.3);
    
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .max_degree(16)
        .search_list_size(50)
        .build_from_vectors(vectors.clone())
        .expect("Failed to build index");
    
    // Search with vectors from the dataset
    for i in 0..5 {
        let results = index.search(&vectors[i], 5).expect("Search failed");
        
        // First result should be the vector itself
        assert_eq!(results[0].0, i);
        assert_relative_eq!(results[0].1, 0.0, epsilon = 1e-6);
        
        // Results should be in ascending order of distance
        for j in 1..results.len() {
            assert!(results[j].1 >= results[j-1].1);
        }
    }
}

#[test]
fn test_different_metrics() {
    let dimension = 16;
    let vectors = generate_clustered_vectors(3, 10, dimension, 0.5);
    
    // Test L2 distance
    let l2_index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .build_from_vectors(vectors.clone())
        .expect("Failed to build L2 index");
    
    // Test Cosine distance
    let cosine_index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::Cosine)
        .build_from_vectors(vectors.clone())
        .expect("Failed to build Cosine index");
    
    // Test Inner Product distance
    let ip_index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::InnerProduct)
        .build_from_vectors(vectors.clone())
        .expect("Failed to build IP index");
    
    // Results should be different for different metrics
    let query = &vectors[0];
    let l2_results = l2_index.search(query, 5).unwrap();
    let cosine_results = cosine_index.search(query, 5).unwrap();
    let ip_results = ip_index.search(query, 5).unwrap();
    
    // All should find the query vector first
    assert_eq!(l2_results[0].0, 0);
    assert_eq!(cosine_results[0].0, 0);
    assert_eq!(ip_results[0].0, 0);
}

#[test]
fn test_large_scale_index() {
    let dimension = 128;
    let num_vectors = 10_000;
    
    // Generate vectors in batches to avoid memory issues
    let mut all_vectors = Vec::with_capacity(num_vectors);
    for _ in 0..10 {
        let mut batch = generate_clustered_vectors(10, 100, dimension, 1.0);
        all_vectors.append(&mut batch);
    }
    
    let start = std::time::Instant::now();
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .max_degree(32)
        .search_list_size(100)
        .alpha(1.2)
        .build_from_vectors(all_vectors.clone())
        .expect("Failed to build large index");
    let build_time = start.elapsed();
    
    println!("Built index with {} vectors in {:?}", num_vectors, build_time);
    assert_eq!(index.size(), num_vectors);
    
    // Benchmark search performance
    let num_queries = 100;
    let mut total_search_time = std::time::Duration::new(0, 0);
    
    for i in 0..num_queries {
        let query = &all_vectors[i * 100]; // Sample queries
        let start = std::time::Instant::now();
        let _results = index.search(query, 10).expect("Search failed");
        total_search_time += start.elapsed();
    }
    
    let avg_search_time = total_search_time / num_queries as u32;
    let qps = 1_000_000.0 / avg_search_time.as_micros() as f64;
    
    println!("Average search time: {:?}", avg_search_time);
    println!("Queries per second: {:.0}", qps);
    
    // Should achieve reasonable performance
    assert!(qps > 1000.0, "QPS too low: {}", qps);
}

#[test]
fn test_edge_cases() {
    // Single vector
    let single_vector = vec![vec![1.0, 2.0, 3.0]];
    let index = IndexBuilder::new()
        .dimensions(3)
        .metric(Distance::L2)
        .build_from_vectors(single_vector.clone())
        .expect("Failed to build single vector index");
    
    let results = index.search(&single_vector[0], 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0);
    
    // High-dimensional vectors
    let high_dim = 1024;
    let vectors = generate_clustered_vectors(2, 5, high_dim, 0.1);
    let index = IndexBuilder::new()
        .dimensions(high_dim)
        .metric(Distance::L2)
        .max_degree(16)
        .build_from_vectors(vectors)
        .expect("Failed to build high-dim index");
    
    assert_eq!(index.dimension(), high_dim);
}

#[test]
fn test_recall_quality() {
    let dimension = 64;
    let vectors = generate_clustered_vectors(5, 100, dimension, 0.5);
    
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .max_degree(32)
        .search_list_size(100)
        .build_from_vectors(vectors.clone())
        .expect("Failed to build index");
    
    // Compute exact nearest neighbors for comparison
    let query = &vectors[0];
    let k = 10;
    
    // Exact search
    let mut exact_distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist = v.iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            (i, dist)
        })
        .collect();
    exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    exact_distances.truncate(k);
    
    // Approximate search
    let approx_results = index.search(query, k).unwrap();
    
    // Calculate recall
    let exact_ids: std::collections::HashSet<usize> = 
        exact_distances.iter().map(|(id, _)| *id).collect();
    let approx_ids: std::collections::HashSet<usize> = 
        approx_results.iter().map(|(id, _)| *id).collect();
    
    let intersection = exact_ids.intersection(&approx_ids).count();
    let recall = intersection as f32 / k as f32;
    
    println!("Recall@{}: {:.2}%", k, recall * 100.0);
    assert!(recall >= 0.9, "Recall too low: {}", recall); // Expect at least 90% recall
}

#[cfg(feature = "neon")]
#[test]
fn test_neon_consistency() {
    use diskann::distance::{neon::NeonDistance, scalar::ScalarDistance, DistanceFunction};
    
    let dimension = 128;
    let vectors = generate_clustered_vectors(2, 10, dimension, 1.0);
    
    let neon_calc = NeonDistance::new(Distance::L2, dimension);
    let scalar_calc = ScalarDistance::new(Distance::L2, dimension);
    
    // Compare results
    for i in 0..5 {
        for j in i+1..5 {
            let neon_dist = neon_calc.distance(&vectors[i], &vectors[j]).unwrap();
            let scalar_dist = scalar_calc.distance(&vectors[i], &vectors[j]).unwrap();
            
            assert_relative_eq!(neon_dist, scalar_dist, epsilon = 1e-5);
        }
    }
}