// Smoke test for parallel graph construction
use diskann::*;
use diskann::graph::vamana::VamanaGraph;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧪 Parallel Smoke Test - ARM64");
    
    // Small test - should be fast
    test_small_parallel()?;
    
    // Medium test - check parallel efficiency
    test_medium_parallel()?;
    
    println!("✅ All parallel smoke tests passed!");
    Ok(())
}

fn test_small_parallel() -> Result<()> {
    println!("\n📊 Small Test (1000 vectors × 16D)");
    
    let vectors = generate_test_vectors(1000, 16);
    let mut graph = VamanaGraph::new(1000, 16, Distance::L2, 16, 32, 1.2);
    
    let start = Instant::now();
    graph.build(&vectors)?;
    let duration = start.elapsed();
    
    let stats = graph.stats();
    println!("   Build time: {:.2}s ({:.0} vectors/sec)", 
            duration.as_secs_f64(), 
            1000.0 / duration.as_secs_f64());
    println!("   Graph stats: {} vertices, avg degree: {:.1}", 
            stats.num_vertices, stats.avg_degree);
    
    // Validate graph quality
    assert!(stats.avg_degree > 1.0, "Graph too sparse");
    assert!(duration.as_secs() < 10, "Build too slow for small dataset");
    
    Ok(())
}

fn test_medium_parallel() -> Result<()> {
    println!("\n📊 Medium Test (5000 vectors × 32D)");
    
    let vectors = generate_test_vectors(5000, 32);
    let mut graph = VamanaGraph::new(5000, 32, Distance::L2, 24, 48, 1.2);
    
    let start = Instant::now();
    graph.build(&vectors)?;
    let duration = start.elapsed();
    
    let stats = graph.stats();
    println!("   Build time: {:.2}s ({:.0} vectors/sec)", 
            duration.as_secs_f64(), 
            5000.0 / duration.as_secs_f64());
    println!("   Graph stats: {} vertices, avg degree: {:.1}", 
            stats.num_vertices, stats.avg_degree);
    
    // Validate graph quality
    assert!(stats.avg_degree > 2.0, "Graph too sparse");
    assert!(duration.as_secs() < 30, "Build too slow for medium dataset");
    
    // Test search performance
    let query = &vectors[0];
    let search_start = Instant::now();
    let results = graph.search(query, 10, &vectors)?;
    let search_time = search_start.elapsed();
    
    println!("   Search time: {:.1}μs, found {} results", 
            search_time.as_micros(), results.len());
    
    assert_eq!(results.len(), 10, "Should find k results");
    assert_eq!(results[0].0, 0, "Should find self as nearest");
    
    Ok(())
}

fn generate_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rand::random::<f32>()).collect())
        .collect()
}