// Comprehensive tests for O(n) medoid calculation
use diskann::graph::vamana::VamanaGraph;
use diskann::Distance;
use std::time::Instant;

#[test]
fn test_medoid_correctness_simple() {
    // Simple test case where medoid is obvious
    let vectors = vec![
        vec![0.0, 0.0],   // Far from center
        vec![10.0, 10.0], // Far from center
        vec![5.0, 5.0],   // At the center - should be medoid
        vec![4.0, 6.0],   // Close to center
        vec![6.0, 4.0],   // Close to center
    ];
    
    let graph = VamanaGraph::new(5, 2, Distance::L2, 16, 32, 1.2);
    let medoid = graph.find_medoid(&vectors).unwrap();
    
    // Point at index 2 [5.0, 5.0] should be the medoid
    // as it's closest to the centroid [5.0, 5.0]
    assert_eq!(medoid, 2, "Expected medoid to be the center point");
}

#[test]
fn test_medoid_performance_o_n() {
    // Test that medoid calculation is O(n) not O(n²)
    let sizes = vec![100, 1000, 10000];
    let mut times = Vec::new();
    
    for &size in &sizes {
        let vectors: Vec<Vec<f32>> = (0..size)
            .map(|_| (0..64).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        let graph = VamanaGraph::new(size, 64, Distance::L2, 32, 64, 1.2);
        
        let start = Instant::now();
        let _medoid = graph.find_medoid(&vectors).unwrap();
        let duration = start.elapsed();
        
        times.push(duration.as_micros());
        println!("Medoid calculation for {} vectors: {}μs", size, duration.as_micros());
    }
    
    // Check that time grows linearly (O(n))
    // If it was O(n²), time[2]/time[1] would be ~100, but for O(n) it should be ~10
    let ratio1 = times[1] as f64 / times[0] as f64;
    let ratio2 = times[2] as f64 / times[1] as f64;
    
    println!("Time ratios: {:.2}, {:.2}", ratio1, ratio2);
    
    // For O(n), ratios should be close to size ratios (10)
    // For O(n²), ratios would be close to size² ratios (100)
    assert!(ratio1 < 20.0, "First ratio too high: {}", ratio1);
    assert!(ratio2 < 20.0, "Second ratio too high: {}", ratio2);
}

#[test]
fn test_medoid_with_outliers() {
    // Test medoid is robust to outliers
    let mut vectors = vec![];
    
    // Cluster of points around origin
    for _ in 0..20 {
        vectors.push(vec![
            rand::random::<f32>() * 2.0,
            rand::random::<f32>() * 2.0,
        ]);
    }
    
    // Add outlier
    vectors.push(vec![100.0, 100.0]);
    
    let graph = VamanaGraph::new(21, 2, Distance::L2, 16, 32, 1.2);
    let medoid = graph.find_medoid(&vectors).unwrap();
    
    // Medoid should be from the main cluster, not the outlier
    assert_ne!(medoid, 20, "Medoid should not be the outlier");
    assert!(medoid < 20, "Medoid should be from main cluster");
}

#[test]
fn test_medoid_high_dimensional() {
    // Test in high dimensions (typical for embeddings)
    let dimension = 768; // Common embedding size
    let num_vectors = 100;
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dimension).map(|_| rand::random::<f32>()).collect())
        .collect();
    
    let graph = VamanaGraph::new(num_vectors, dimension, Distance::L2, 32, 64, 1.2);
    
    let start = Instant::now();
    let medoid = graph.find_medoid(&vectors).unwrap();
    let duration = start.elapsed();
    
    assert!(medoid < num_vectors, "Medoid should be valid index");
    assert!(duration.as_millis() < 100, "High-dim medoid too slow: {}ms", duration.as_millis());
}

#[test] 
fn test_medoid_different_metrics() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.577, 0.577, 0.577], // Normalized [1,1,1]
    ];
    
    // Test with L2 distance
    let graph_l2 = VamanaGraph::new(4, 3, Distance::L2, 16, 32, 1.2);
    let medoid_l2 = graph_l2.find_medoid(&vectors).unwrap();
    
    // Test with Cosine distance
    let graph_cos = VamanaGraph::new(4, 3, Distance::Cosine, 16, 32, 1.2);
    let medoid_cos = graph_cos.find_medoid(&vectors).unwrap();
    
    // Both should find point 3 as medoid (closest to centroid)
    assert_eq!(medoid_l2, 3, "L2 medoid incorrect");
    assert_eq!(medoid_cos, 3, "Cosine medoid incorrect");
}

#[test]
fn test_graph_build_with_o_n_medoid() {
    // Integration test: full graph build with O(n) medoid
    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..32).map(|_| rand::random::<f32>()).collect())
        .collect();
    
    let mut graph = VamanaGraph::new(1000, 32, Distance::L2, 24, 48, 1.2);
    
    let start = Instant::now();
    graph.build(&vectors).unwrap();
    let build_time = start.elapsed();
    
    // Should complete reasonably fast with O(n) medoid
    assert!(build_time.as_secs() < 10, "Build took too long: {:?}", build_time);
    
    // Verify graph quality
    let stats = graph.stats();
    assert_eq!(stats.num_vertices, 1000);
    assert!(stats.avg_degree > 5.0, "Graph too sparse: {}", stats.avg_degree);
    assert!(stats.avg_degree <= 24.0, "Graph too dense: {}", stats.avg_degree);
    
    // Test search works
    let results = graph.search(&vectors[0], 10, &vectors).unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(results[0].0, 0); // Should find self
}