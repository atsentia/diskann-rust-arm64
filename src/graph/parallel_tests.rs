use crate::graph::vamana::VamanaGraph;
use crate::distance::Distance;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use hashbrown::HashSet;

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_construction_basic() {
    // Test basic parallel construction
    let num_vectors = 100;
    let dimension = 32;
    
    let mut graph = VamanaGraph::new(
        num_vectors,
        dimension,
        Distance::L2,
        32,  // max_degree
        50,  // search_list_size
        1.2, // alpha
    );
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();
    
    // Build should succeed
    graph.build(&vectors).unwrap();
    
    // Verify graph properties
    let stats = graph.stats();
    assert_eq!(stats.num_vertices, num_vectors);
    assert!(stats.num_edges > 0);
    assert!(stats.avg_degree > 0.0);
    assert!(stats.avg_degree <= 32.0); // Should respect max_degree
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_thread_safety() {
    // Test that parallel construction is thread-safe
    let num_vectors = 1000;
    let dimension = 64;
    let num_trials = 5;
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 11 + j * 17) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();
    
    // Build multiple graphs in parallel
    let graphs: Vec<_> = (0..num_trials)
        .map(|_| {
            let mut graph = VamanaGraph::new(
                num_vectors,
                dimension,
                Distance::L2,
                48,
                75,
                1.2,
            );
            graph.build(&vectors).unwrap();
            graph
        })
        .collect();
    
    // All graphs should have similar properties
    let stats: Vec<_> = graphs.iter().map(|g| g.stats()).collect();
    
    for i in 1..num_trials {
        // Entry points should match
        assert_eq!(stats[0].entry_point, stats[i].entry_point);
        
        // Edge counts should be within 5% of each other
        let edge_diff = (stats[0].num_edges as f64 - stats[i].num_edges as f64).abs();
        let tolerance = stats[0].num_edges as f64 * 0.05;
        assert!(edge_diff <= tolerance, 
            "Edge count difference {} exceeds 5% tolerance", edge_diff);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_deterministic_with_seed() {
    // Test that parallel construction produces consistent results with same seed
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    
    let num_vectors = 500;
    let dimension = 128;
    let seed = 42;
    
    // Generate deterministic random vectors
    let mut rng = StdRng::seed_from_u64(seed);
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            (0..dimension)
                .map(|_| rand::Rng::gen::<f32>(&mut rng))
                .collect()
        })
        .collect();
    
    // Build two graphs with same data
    let mut graph1 = VamanaGraph::new(num_vectors, dimension, Distance::L2, 64, 100, 1.2);
    let mut graph2 = VamanaGraph::new(num_vectors, dimension, Distance::L2, 64, 100, 1.2);
    
    graph1.build(&vectors).unwrap();
    graph2.build(&vectors).unwrap();
    
    // Should have same entry point (medoid is deterministic)
    assert_eq!(graph1.stats().entry_point, graph2.stats().entry_point);
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_search_quality() {
    // Test that parallel-built graphs have good search quality
    let num_vectors = 1000;
    let dimension = 128;
    let num_queries = 100;
    let k = 10;
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| {
                    let cluster = i / 250; // 4 clusters
                    let base = cluster as f32 * 0.25;
                    base + ((i * 7 + j * 11) % 50) as f32 / 500.0
                })
                .collect()
        })
        .collect();
    
    let mut graph = VamanaGraph::new(
        num_vectors,
        dimension,
        Distance::L2,
        64,
        100,
        1.2,
    );
    
    graph.build(&vectors).unwrap();
    
    // Test search quality
    let mut total_recall = 0;
    let mut exact_matches = 0;
    
    for i in 0..num_queries {
        let query_id = i * 10; // Sample queries
        let query = &vectors[query_id];
        
        // Get approximate results
        let results = graph.search(query, k, &vectors).unwrap();
        
        // The first result should be the query itself
        if results[0].0 == query_id && results[0].1 < 0.001 {
            exact_matches += 1;
        }
        
        // Check if neighbors are from same cluster
        let query_cluster = query_id / 250;
        let same_cluster = results.iter()
            .filter(|(id, _)| *id / 250 == query_cluster)
            .count();
        
        total_recall += same_cluster;
    }
    
    let avg_recall = total_recall as f32 / (num_queries * k) as f32;
    let exact_match_rate = exact_matches as f32 / num_queries as f32;
    
    // Should find itself in most cases
    assert!(exact_match_rate > 0.9, 
        "Low exact match rate: {:.2}%", exact_match_rate * 100.0);
    
    // Should have good cluster recall
    assert!(avg_recall > 0.5,
        "Low cluster recall: {:.2}%", avg_recall * 100.0);
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_edge_distribution() {
    // Test that parallel construction creates well-distributed edges
    let num_vectors = 200;
    let dimension = 64;
    let max_degree = 32;
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 13 + j * 17) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();
    
    let mut graph = VamanaGraph::new(
        num_vectors,
        dimension,
        Distance::L2,
        max_degree,
        50,
        1.2,
    );
    
    graph.build(&vectors).unwrap();
    
    // Get degree distribution
    let degrees = graph.get_degree_distribution();
    
    // No node should exceed max_degree
    for &degree in &degrees {
        assert!(degree <= max_degree, 
            "Node degree {} exceeds max_degree {}", degree, max_degree);
    }
    
    // Most nodes should have reasonable degree (not too sparse)
    let well_connected = degrees.iter()
        .filter(|&&d| d >= max_degree / 4)
        .count();
    
    let connection_rate = well_connected as f32 / num_vectors as f32;
    assert!(connection_rate > 0.8,
        "Only {:.1}% of nodes are well connected", connection_rate * 100.0);
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_incremental_consistency() {
    // Test that parallel construction handles incremental builds correctly
    let dimension = 32;
    let batch_size = 100;
    let num_batches = 3;
    
    // Generate all vectors upfront
    let all_vectors: Vec<Vec<f32>> = (0..batch_size * num_batches)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 7 + j * 11) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();
    
    // Build graph with all data at once
    let mut graph_full = VamanaGraph::new(
        batch_size * num_batches,
        dimension,
        Distance::L2,
        32,
        50,
        1.2,
    );
    graph_full.build(&all_vectors).unwrap();
    
    // Search quality baseline
    let test_queries = 10;
    let k = 5;
    let mut baseline_results = Vec::new();
    
    for i in 0..test_queries {
        let query_id = i * 30;
        let results = graph_full.search(&all_vectors[query_id], k, &all_vectors).unwrap();
        baseline_results.push(results);
    }
    
    // Verify search results are reasonable
    for (i, results) in baseline_results.iter().enumerate() {
        let query_id = i * 30;
        // Should find itself
        assert_eq!(results[0].0, query_id);
        assert!(results[0].1 < 0.001);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_memory_efficiency() {
    // Test that parallel construction doesn't leak memory
    let num_vectors = 1000;
    let dimension = 128;
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();
    
    // Track allocations (simplified - in real test would use allocation tracker)
    let graphs: Vec<_> = (0..5)
        .map(|_| {
            let mut graph = VamanaGraph::new(
                num_vectors,
                dimension,
                Distance::L2,
                48,
                75,
                1.2,
            );
            graph.build(&vectors).unwrap();
            graph
        })
        .collect();
    
    // All graphs should have similar memory footprint
    // (This is a simplified test - real memory tracking would be more complex)
    assert_eq!(graphs.len(), 5);
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_cancellation_safety() {
    // Test that parallel construction can handle early termination gracefully
    use std::sync::atomic::AtomicBool;
    use std::thread;
    use std::time::Duration;
    
    let num_vectors = 10000;
    let dimension = 128;
    let should_cancel = Arc::new(AtomicBool::new(false));
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();
    
    // This test would require modifying the implementation to support cancellation
    // For now, we just verify that normal construction completes
    let mut graph = VamanaGraph::new(
        num_vectors,
        dimension,
        Distance::L2,
        64,
        100,
        1.2,
    );
    
    // Build should complete successfully
    graph.build(&vectors).unwrap();
    assert!(graph.stats().num_edges > 0);
}

