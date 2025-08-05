// Quick disk benchmark test for ARM64
use diskann::*;
use diskann::index::disk::PQFlashIndex;
use diskann::Distance;
use std::time::Instant;

const TEST_VECTORS: usize = 25000; // Moderate size
const DIMENSION: usize = 128;

fn main() -> Result<()> {
    println!("🚀 Quick Disk Test - ARM64");
    println!("Testing {} vectors × {} dimensions", TEST_VECTORS, DIMENSION);
    
    // Generate test data
    println!("📊 Generating test vectors...");
    let start = Instant::now();
    let mut vectors = Vec::new();
    for _ in 0..TEST_VECTORS {
        let mut vec = Vec::with_capacity(DIMENSION);
        for _ in 0..DIMENSION {
            vec.push(rand::random::<f32>());
        }
        vectors.push(vec);
    }
    println!("✅ Generated in {:.2}s", start.elapsed().as_secs_f64());
    
    // Build index
    println!("🏗️  Building PQ Flash Index...");
    let build_start = Instant::now();
    
    let config = diskann::index::disk::PQFlashConfig {
        max_degree: 32,
        search_list_size: 64,
        pq_params: diskann::index::disk::PQParams {
            num_chunks: 8,
            bits_per_chunk: 8,
        },
        alpha: 1.2,
        num_threads: num_cpus::get(),
        use_reorder_data: true,
        beam_width: 4,
    };
    
    let index_path = "test_quick_disk.index";
    
    match PQFlashIndex::build_from_vectors(index_path, vectors, config) {
        Ok(index) => {
            let build_time = build_start.elapsed();
            println!("✅ Build completed in {:.2}s ({:.0} vectors/sec)", 
                    build_time.as_secs_f64(), 
                    TEST_VECTORS as f64 / build_time.as_secs_f64());
            
            // Quick search test
            println!("🔍 Testing search...");
            let query: Vec<f32> = (0..DIMENSION).map(|_| rand::random::<f32>()).collect();
            
            let search_start = Instant::now();
            let (results, _stats) = index.search(&query, 10, 50)?;
            let search_time = search_start.elapsed();
            
            println!("✅ Search completed in {:.1}μs, found {} results", 
                    search_time.as_micros(), results.len());
            
            println!("🎉 Test completed successfully!");
        }
        Err(e) => {
            println!("❌ Build failed: {:?}", e);
            return Err(e);
        }
    }
    
    Ok(())
}