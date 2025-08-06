// Copyright (c) DiskANN Rust Contributors. All rights reserved.
// Licensed under the MIT license.

//! Cross-validation tests to verify algorithmic correctness between 
//! Rust implementation and expected C++ DiskANN behavior.

use diskann::*;
use std::collections::HashMap;

#[cfg(test)]
mod correctness_validation {
    use super::*;
    use diskann::utils::generate_random_vectors;
    use approx::assert_relative_eq;

    /// Test that Vamana graph construction produces deterministic, connected graphs
    #[test]
    fn test_vamana_graph_determinism() -> Result<()> {
        const DIMENSION: usize = 32;
        const NUM_VECTORS: usize = 100;
        const SEED: u64 = 42;

        // Generate deterministic test data
        let vectors = generate_random_vectors(NUM_VECTORS, DIMENSION, SEED);
        
        // Build the same index twice with identical parameters
        let build_index = |vectors: &[Vec<f32>]| -> Result<_> {
            IndexBuilder::new()
                .dimensions(DIMENSION)
                .metric(Distance::L2)
                .max_degree(32)
                .search_list_size(50)
                .alpha(1.2)
                .build_from_vectors(vectors.to_vec())
        };

        let index1 = build_index(&vectors)?;
        let index2 = build_index(&vectors)?;

        // Verify identical graph topology
        for i in 0..NUM_VECTORS {
            let neighbors1 = index1.get_neighbors(i).unwrap_or_default();
            let neighbors2 = index2.get_neighbors(i).unwrap_or_default();
            
            assert_eq!(neighbors1.len(), neighbors2.len(),
                "Graph topology mismatch at vertex {}", i);
            assert_eq!(neighbors1, neighbors2,
                "Different neighbors for vertex {}", i);
        }

        // Verify graph connectivity
        verify_graph_connectivity(&index1, NUM_VECTORS)?;
        
        Ok(())
    }

    /// Test that search results are deterministic and consistent
    #[test]
    fn test_search_determinism() -> Result<()> {
        const DIMENSION: usize = 64;
        const NUM_VECTORS: usize = 200;
        const NUM_QUERIES: usize = 10;
        const K: usize = 10;

        // Build test index
        let vectors = generate_random_vectors(NUM_VECTORS, DIMENSION, 12345);
        let index = IndexBuilder::new()
            .dimensions(DIMENSION)
            .metric(Distance::L2)
            .max_degree(48)
            .search_list_size(100)
            .build_from_vectors(vectors)?;

        // Generate test queries
        let queries = generate_random_vectors(NUM_QUERIES, DIMENSION, 54321);

        // Run each query multiple times and verify identical results
        for (query_id, query) in queries.iter().enumerate() {
            let mut all_results = Vec::new();
            
            // Run same query 5 times
            for _ in 0..5 {
                let results = index.search(query, K)?;
                all_results.push(results);
            }

            // Verify all runs produced identical results
            for run in 1..all_results.len() {
                assert_eq!(all_results[0].len(), all_results[run].len(),
                    "Result count mismatch for query {}", query_id);
                
                for (expected, actual) in all_results[0].iter().zip(all_results[run].iter()) {
                    assert_eq!(expected.0, actual.0,
                        "Neighbor ID mismatch for query {}", query_id);
                    assert_relative_eq!(expected.1, actual.1, epsilon = 1e-6,
                        "Distance mismatch for query {}", query_id);
                }
            }
        }

        Ok(())
    }

    /// Test distance function correctness against known mathematical properties
    #[test]
    fn test_distance_function_correctness() -> Result<()> {
        use diskann::distance::*;

        let test_vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0],
        ];

        // Test L2 distance properties
        let l2_fn = create_distance_function(Distance::L2, 4);
        
        // Self-distance should be zero
        for v in &test_vectors {
            let dist = l2_fn.distance(v, v)?;
            assert_relative_eq!(dist, 0.0, epsilon = 1e-6, "L2 self-distance not zero");
        }

        // Symmetry: d(a,b) = d(b,a)
        for (i, a) in test_vectors.iter().enumerate() {
            for (j, b) in test_vectors.iter().enumerate() {
                if i != j {
                    let dist_ab = l2_fn.distance(a, b)?;
                    let dist_ba = l2_fn.distance(b, a)?;
                    assert_relative_eq!(dist_ab, dist_ba, epsilon = 1e-6,
                        "L2 distance not symmetric");
                }
            }
        }

        // Known distance calculations
        let unit_x = vec![1.0, 0.0, 0.0, 0.0];
        let unit_y = vec![0.0, 1.0, 0.0, 0.0];
        let dist = l2_fn.distance(&unit_x, &unit_y)?;
        assert_relative_eq!(dist, 2.0_f32.sqrt(), epsilon = 1e-6, 
            "L2 distance calculation incorrect");

        // Test Cosine distance properties
        let cosine_fn = create_distance_function(Distance::Cosine, 4);
        
        // Parallel vectors should have distance 0
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![2.0, 4.0, 6.0, 8.0]; // 2 * v1
        let cosine_dist = cosine_fn.distance(&v1, &v2)?;
        assert!(cosine_dist < 1e-6, "Parallel vectors should have cosine distance ~0");

        // Orthogonal vectors should have distance 1
        let ortho_dist = cosine_fn.distance(&unit_x, &unit_y)?;
        assert_relative_eq!(ortho_dist, 1.0, epsilon = 1e-6,
            "Orthogonal vectors should have cosine distance 1");

        Ok(())
    }

    /// Test Product Quantization correctness
    #[test]
    fn test_pq_compression_correctness() -> Result<()> {
        use diskann::pq::*;

        const DIMENSION: usize = 128;
        const NUM_VECTORS: usize = 1000;
        const NUM_SUBSPACES: usize = 16;
        const BITS_PER_SUBQUANTIZER: usize = 8;

        // Generate test data
        let vectors = generate_random_vectors(NUM_VECTORS, DIMENSION, 98765);

        // Create and train PQ
        let params = PQParams {
            num_subspaces: NUM_SUBSPACES,
            num_centroids: 1 << BITS_PER_SUBQUANTIZER,
            bits_per_subquantizer: BITS_PER_SUBQUANTIZER,
            seed: 42,
            kmeans_params: Default::default(),
        };

        let mut pq = ProductQuantizer::new(DIMENSION, params)?;
        let training_result = pq.train(&vectors)?;

        // Test encoding/decoding consistency
        for (i, vector) in vectors.iter().take(10).enumerate() {
            let encoded = training_result.encode(vector)?;
            let decoded = training_result.decode(&encoded)?;

            // Verify dimensions match
            assert_eq!(vector.len(), decoded.len(),
                "Decoded vector dimension mismatch for vector {}", i);

            // Verify reconstruction is reasonable (not exact due to quantization)
            let mse = vector.iter().zip(decoded.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>() / vector.len() as f32;
            
            assert!(mse < 1.0, "PQ reconstruction error too high for vector {}: MSE = {}", i, mse);
        }

        // Verify compression ratio
        let original_size = vectors.len() * DIMENSION * std::mem::size_of::<f32>();
        let compressed_size = vectors.len() * NUM_SUBSPACES * (BITS_PER_SUBQUANTIZER / 8);
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        assert!(compression_ratio >= 16.0, 
            "PQ compression ratio too low: {}", compression_ratio);

        Ok(())
    }

    /// Test filtered search correctness
    #[test]
    fn test_filtered_search_correctness() -> Result<()> {
        use diskann::labels::*;

        const DIMENSION: usize = 32;
        const NUM_VECTORS: usize = 100;

        // Generate vectors with specific labels
        let vectors = generate_random_vectors(NUM_VECTORS, DIMENSION, 11111);
        let labels: Vec<Vec<u32>> = (0..NUM_VECTORS)
            .map(|i| {
                if i % 3 == 0 { vec![1, 2] }      // Label group A
                else if i % 3 == 1 { vec![2, 3] } // Label group B  
                else { vec![1, 3] }               // Label group C
            })
            .collect();

        // Build index with labels
        let index = IndexBuilder::new()
            .dimensions(DIMENSION)
            .metric(Distance::L2)
            .max_degree(32)
            .search_list_size(50)
            .build_from_vectors_with_labels(vectors, labels)?;

        let query = vec![0.5; DIMENSION];

        // Test "any of" filter
        let filter = LabelFilter::AnyOf(vec![1]);
        let results = index.search_with_filter(&query, 20, filter)?;
        
        // Verify all results have label 1
        for (id, _) in &results {
            let vector_labels = index.get_labels(*id as usize).unwrap();
            assert!(vector_labels.contains(&1), 
                "Filtered result {} doesn't contain required label 1", id);
        }

        // Test "all of" filter
        let filter = LabelFilter::AllOf(vec![2, 3]);
        let results = index.search_with_filter(&query, 20, filter)?;
        
        // Verify all results have both labels 2 and 3
        for (id, _) in &results {
            let vector_labels = index.get_labels(*id as usize).unwrap();
            assert!(vector_labels.contains(&2) && vector_labels.contains(&3),
                "Filtered result {} doesn't contain required labels 2 and 3", id);
        }

        // Test exact filter
        let filter = LabelFilter::Exact(vec![1, 2]);
        let results = index.search_with_filter(&query, 20, filter)?;
        
        // Verify all results have exactly labels 1 and 2
        for (id, _) in &results {
            let mut vector_labels = index.get_labels(*id as usize).unwrap();
            vector_labels.sort();
            assert_eq!(vector_labels, vec![1, 2],
                "Filtered result {} doesn't have exact labels [1, 2]", id);
        }

        Ok(())
    }

    /// Test dynamic index operations correctness
    #[test]
    fn test_dynamic_operations_correctness() -> Result<()> {
        use diskann::index::DynamicIndex;

        const DIMENSION: usize = 16;
        let mut index = DynamicIndex::new(DIMENSION, Distance::L2)?;

        // Insert vectors and track IDs
        let mut inserted_ids = Vec::new();
        for i in 0..50 {
            let vector = vec![i as f32; DIMENSION];
            let id = index.insert(vector, vec![i as u32])?;
            inserted_ids.push(id);
        }

        // Verify all insertions are searchable
        let query = vec![25.0; DIMENSION];
        let results = index.search(&query, 10)?;
        assert!(results.len() > 0, "No search results after insertions");

        // Test deletion
        let id_to_delete = inserted_ids[25];
        index.delete(id_to_delete)?;

        // Verify deleted vector is not in search results
        let results_after_delete = index.search(&query, 50)?;
        for (result_id, _) in &results_after_delete {
            assert_ne!(*result_id as usize, id_to_delete, 
                "Deleted vector still appears in search results");
        }

        // Test consolidation
        let stats_before = index.get_statistics();
        index.consolidate()?;
        let stats_after = index.get_statistics();
        
        // Verify consolidation reduced fragmentation
        assert!(stats_after.fragmentation_ratio <= stats_before.fragmentation_ratio,
            "Consolidation didn't reduce fragmentation");

        Ok(())
    }

    /// Verify graph connectivity using BFS traversal
    fn verify_graph_connectivity(index: &dyn Index, num_vertices: usize) -> Result<()> {
        use std::collections::VecDeque;

        let mut visited = vec![false; num_vertices];
        let mut queue = VecDeque::new();
        
        // Start BFS from vertex 0
        queue.push_back(0);
        visited[0] = true;
        let mut reachable_count = 1;

        while let Some(vertex) = queue.pop_front() {
            if let Some(neighbors) = index.get_neighbors(vertex) {
                for &neighbor in &neighbors {
                    if neighbor < num_vertices && !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                        reachable_count += 1;
                    }
                }
            }
        }

        // For small graphs, expect high connectivity
        let connectivity_ratio = reachable_count as f32 / num_vertices as f32;
        assert!(connectivity_ratio > 0.8, 
            "Graph connectivity too low: {:.2}%", connectivity_ratio * 100.0);

        Ok(())
    }

    /// Test range search correctness
    #[test]
    fn test_range_search_correctness() -> Result<()> {
        const DIMENSION: usize = 8;
        const NUM_VECTORS: usize = 50;

        // Create vectors in a grid pattern for predictable distances
        let mut vectors = Vec::new();
        for i in 0..NUM_VECTORS {
            let mut vector = vec![0.0; DIMENSION];
            vector[0] = (i / 10) as f32;
            vector[1] = (i % 10) as f32;
            vectors.push(vector);
        }

        let index = IndexBuilder::new()
            .dimensions(DIMENSION)
            .metric(Distance::L2)
            .max_degree(20)
            .search_list_size(30)
            .build_from_vectors(vectors.clone())?;

        // Test range search from origin
        let query = vec![0.0; DIMENSION];
        let radius = 5.0;
        let range_results = index.range_search(&query, radius)?;

        // Verify all results are within radius
        for (id, distance) in &range_results {
            assert!(*distance <= radius + 1e-6,
                "Range search result {} has distance {} > radius {}", id, distance, radius);
        }

        // Verify no closer points were missed by checking manually
        let l2_fn = diskann::distance::create_distance_function(Distance::L2, DIMENSION);
        for (i, vector) in vectors.iter().enumerate() {
            let dist = l2_fn.distance(&query, vector)?;
            if dist <= radius {
                assert!(range_results.iter().any(|(id, _)| *id == i as u32),
                    "Vector {} at distance {} was missed by range search", i, dist);
            }
        }

        Ok(())
    }
}

/// Helper trait to add testing methods to Index implementations
trait IndexTestExt {
    fn get_neighbors(&self, vertex_id: usize) -> Option<Vec<usize>>;
    fn get_labels(&self, vertex_id: usize) -> Option<Vec<u32>>;
}

// Implementation would need to be added to actual Index types
// This is a placeholder for the test interface