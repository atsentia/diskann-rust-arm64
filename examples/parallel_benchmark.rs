//! Benchmark to compare sequential vs parallel graph construction
//! Designed to match C++ DiskANN's parallel performance testing

use diskann::graph::vamana::VamanaGraph;
use diskann::distance::Distance;
use std::time::Instant;

fn main() {
    println!("🚀 Parallel Graph Construction Benchmark - ARM64");
    println!("================================================");
    
    // Test sizes matching C++ benchmark
    let test_sizes = vec![1000, 10000, 25000];
    let dimension = 128;
    let max_degree = 64;
    let search_list_size = 75;
    let alpha = 1.2;
    
    // Get number of threads
    let num_threads = num_cpus::get();
    println!("🖥️  CPU Cores: {}", num_threads);
    println!("🧵 Rayon threads: {}", rayon::current_num_threads());
    
    for &num_vectors in &test_sizes {
        println!("\n📊 Testing with {} vectors × {} dimensions", num_vectors, dimension);
        println!("   Graph parameters: R={}, L={}, α={}", max_degree, search_list_size, alpha);
        
        // Generate random test data
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|i| {
                (0..dimension)
                    .map(|j| ((i * 13 + j * 7) % 100) as f32 / 100.0)
                    .collect()
            })
            .collect();
        
        // Test 1: Sequential build (by creating single-threaded pool)
        println!("\n   📋 Sequential Build:");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        
        let seq_graph = pool.install(|| {
            let mut graph = VamanaGraph::new(
                num_vectors,
                dimension,
                Distance::L2,
                max_degree,
                search_list_size,
                alpha,
            );
            
            let start = Instant::now();
            graph.build(&vectors).unwrap();
            let elapsed = start.elapsed();
            
            println!("      ⏱️  Time: {:.2}s", elapsed.as_secs_f64());
            println!("      📈 Rate: {:.0} vectors/sec", num_vectors as f64 / elapsed.as_secs_f64());
            
            (graph, elapsed)
        });
        
        // Test 2: Parallel build (default thread pool)
        println!("\n   🚀 Parallel Build ({} threads):", num_threads);
        let mut par_graph = VamanaGraph::new(
            num_vectors,
            dimension,
            Distance::L2,
            max_degree,
            search_list_size,
            alpha,
        );
        
        let start = Instant::now();
        par_graph.build(&vectors).unwrap();
        let par_elapsed = start.elapsed();
        
        println!("      ⏱️  Time: {:.2}s", par_elapsed.as_secs_f64());
        println!("      📈 Rate: {:.0} vectors/sec", num_vectors as f64 / par_elapsed.as_secs_f64());
        
        // Calculate speedup
        let speedup = seq_graph.1.as_secs_f64() / par_elapsed.as_secs_f64();
        println!("\n   ⚡ Speedup: {:.2}x", speedup);
        println!("   📊 Efficiency: {:.1}%", (speedup / num_threads as f64) * 100.0);
        
        // Verify both graphs have similar quality
        let seq_stats = seq_graph.0.stats();
        let par_stats = par_graph.stats();
        
        println!("\n   🔍 Graph Quality Comparison:");
        println!("      Sequential: {} edges, {:.1} avg degree", 
                seq_stats.num_edges, seq_stats.avg_degree);
        println!("      Parallel:   {} edges, {:.1} avg degree", 
                par_stats.num_edges, par_stats.avg_degree);
        
        // Test search quality
        let num_queries = 100.min(num_vectors / 10);
        let k = 10;
        let search_size = 50;
        
        let mut seq_recall_sum = 0.0;
        let mut par_recall_sum = 0.0;
        
        for i in 0..num_queries {
            let query = &vectors[i];
            
            let seq_results = seq_graph.0.search(query, k, search_size).unwrap();
            let par_results = par_graph.search(query, k, search_size).unwrap();
            
            // Calculate recall (how many results match)
            let seq_ids: hashbrown::HashSet<_> = seq_results.iter().map(|(id, _)| id).collect();
            let par_ids: hashbrown::HashSet<_> = par_results.iter().map(|(id, _)| id).collect();
            
            let overlap = seq_ids.intersection(&par_ids).count();
            let recall = overlap as f32 / k as f32;
            
            seq_recall_sum += 1.0; // Sequential is our baseline
            par_recall_sum += recall;
        }
        
        let avg_recall = par_recall_sum / num_queries as f32;
        println!("      🎯 Parallel recall vs sequential: {:.1}%", avg_recall * 100.0);
    }
    
    println!("\n✅ Benchmark completed!");
    
    // Compare with C++ reference
    println!("\n📊 C++ DiskANN Reference (25K vectors):");
    println!("   Sequential: 337 vectors/sec");
    println!("   Parallel (8 threads): 2,514 vectors/sec (7.5x speedup)");
}