use crate::graph::vamana::VamanaGraph;
use crate::distance::Distance;

#[test]
fn test_medoid_empty_vectors() {
    let graph = VamanaGraph::new(
        0,          // num_vertices
        128,        // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    let empty_vectors: Vec<Vec<f32>> = vec![];
    let result = graph.find_medoid(&empty_vectors);
    assert!(result.is_err());
}

#[test]
fn test_medoid_single_vector() {
    let graph = VamanaGraph::new(
        1,          // num_vertices
        128,        // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    let vectors = vec![vec![1.0f32; 128]];
    let medoid = graph.find_medoid(&vectors).unwrap();
    assert_eq!(medoid, 0);
}

#[test]
fn test_medoid_identical_vectors() {
    let graph = VamanaGraph::new(
        10,         // num_vertices
        128,        // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    // All vectors are identical
    let vectors: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0f32; 128]).collect();
    let medoid = graph.find_medoid(&vectors).unwrap();
    
    // Any vector can be the medoid when all are identical
    assert!(medoid < 10);
}

#[test]
fn test_medoid_known_case() {
    let graph = VamanaGraph::new(
        5,          // num_vertices
        2,          // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    // Create vectors in 2D for easy verification
    // Points: (0,0), (1,0), (0,1), (1,1), (0.5,0.5)
    let vectors = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5], // This is at the center
    ];
    
    let medoid = graph.find_medoid(&vectors).unwrap();
    
    // The centroid is at (0.5, 0.5), so vector 4 should be the medoid
    assert_eq!(medoid, 4);
}

#[test]
fn test_medoid_consistency() {
    // Test consistency by creating two identical graphs
    let graph1 = VamanaGraph::new(
        100,        // num_vertices
        128,        // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    let graph2 = VamanaGraph::new(
        100,        // num_vertices
        128,        // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    // Generate random test data
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            (0..128)
                .map(|j| ((i * 13 + j * 7) % 100) as f32 / 100.0)
                .collect()
        })
        .collect();
    
    // Both graphs should find the same medoid
    let medoid1 = graph1.find_medoid(&vectors).unwrap();
    let medoid2 = graph2.find_medoid(&vectors).unwrap();
    
    assert_eq!(medoid1, medoid2, 
        "Two identical graphs should return the same medoid");
}

#[test]
fn test_medoid_large_dataset() {
    let graph = VamanaGraph::new(
        1000,       // num_vertices
        128,        // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    // Generate larger dataset
    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            (0..128)
                .map(|j| {
                    let x = i as f32 / 1000.0;
                    let y = j as f32 / 128.0;
                    (x * x + y * y).sin()
                })
                .collect()
        })
        .collect();
    
    let medoid = graph.find_medoid(&vectors).unwrap();
    
    // Verify the medoid is a valid index
    assert!(medoid < 1000);
}

#[test]
fn test_medoid_numerical_stability() {
    let graph = VamanaGraph::new(
        3,          // num_vertices
        4,          // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    // Test with reasonable values to avoid numerical overflow
    let vectors = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![100.0, 100.0, 100.0, 100.0],
        vec![50.0, 50.0, 50.0, 50.0], // This should be the medoid
    ];
    
    let medoid = graph.find_medoid(&vectors).unwrap();
    
    // The middle vector should be closest to the centroid
    assert_eq!(medoid, 2);
}

#[test]
fn test_medoid_dimension_alignment() {
    // Test with non-aligned dimension (not multiple of 4)
    let graph = VamanaGraph::new(
        10,         // num_vertices
        127,        // dimension
        Distance::L2, // metric
        64,         // max_degree
        75,         // search_list_size
        1.2,        // alpha
    );
    
    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..127).map(|j| (i + j) as f32).collect())
        .collect();
    
    // Should use scalar implementation for non-aligned dimensions
    let medoid = graph.find_medoid(&vectors).unwrap();
    assert!(medoid < 10);
}