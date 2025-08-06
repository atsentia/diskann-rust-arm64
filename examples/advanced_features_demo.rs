//! Advanced features demonstration
//!
//! This example showcases the new advanced features implemented for DiskANN Rust:
//! - NUMA awareness and thread pinning
//! - Advanced I/O with direct I/O support
//! - High-performance natural number containers
//! - Advanced performance monitoring and statistics

use diskann::{
    utils::{
        numa::{NumaConfig, get_numa_topology, init_numa_optimizations},
        containers::{NaturalNumberMap, NaturalNumberSet},
        stats::{PerformanceMonitor, PercentileStats, ThroughputCalculator},
    },
    io::direct::{DirectReader, AlignedBuffer, CachedDirectReader},
    Result,
};
use std::time::Duration;

fn main() -> Result<()> {
    env_logger::init();
    
    println!("🚀 DiskANN Rust Advanced Features Demo");
    println!("=====================================\n");

    // 1. NUMA Awareness Demo
    demo_numa_awareness()?;
    
    // 2. High-Performance Containers Demo
    demo_containers();
    
    // 3. Performance Statistics Demo
    demo_performance_stats()?;
    
    // 4. Direct I/O Demo (requires a test file)
    demo_direct_io()?;
    
    println!("\n✅ All advanced features demonstrated successfully!");
    Ok(())
}

fn demo_numa_awareness() -> Result<()> {
    println!("📊 NUMA Awareness Demo");
    println!("----------------------");
    
    // Get NUMA topology
    let topology = get_numa_topology()?;
    println!("NUMA Nodes: {}", topology.num_nodes);
    
    for (node_id, cores) in topology.cores_per_node.iter().enumerate() {
        println!("  Node {}: {} cores, {} MB memory", 
                 node_id, 
                 cores.len(),
                 topology.memory_per_node[node_id] / (1024 * 1024));
    }
    
    // Configure NUMA optimizations
    let numa_config = NumaConfig::default();
    println!("NUMA enabled: {}", numa_config.enabled);
    
    // Initialize NUMA optimizations
    init_numa_optimizations(&numa_config)?;
    
    println!("✅ NUMA topology detected and configured\n");
    Ok(())
}

fn demo_containers() {
    println!("📦 High-Performance Containers Demo");
    println!("-----------------------------------");
    
    // Natural Number Map Demo
    let mut nat_map = NaturalNumberMap::new();
    nat_map.insert(0, "zero");
    nat_map.insert(42, "answer");
    nat_map.insert(1000, "large");
    
    println!("NaturalNumberMap operations:");
    println!("  Size: {}", nat_map.len());
    println!("  Value at 42: {:?}", nat_map.get(42));
    println!("  Keys: {:?}", nat_map.keys().collect::<Vec<_>>());
    
    // Natural Number Set Demo
    let mut nat_set = NaturalNumberSet::new();
    nat_set.insert(1);
    nat_set.insert(3);
    nat_set.insert(7);
    nat_set.insert(15);
    nat_set.insert(31);
    
    println!("\nNaturalNumberSet operations:");
    println!("  Size: {}", nat_set.len());
    println!("  Contains 7: {}", nat_set.contains(7));
    println!("  Contains 8: {}", nat_set.contains(8));
    println!("  Values: {:?}", nat_set.iter().collect::<Vec<_>>());
    
    // Set operations
    let mut other_set = NaturalNumberSet::new();
    other_set.insert(3);
    other_set.insert(15);
    other_set.insert(63);
    
    let mut union_set = nat_set.clone();
    union_set.union(&other_set);
    println!("  Union size: {}", union_set.len());
    
    println!("✅ Container operations completed\n");
}

fn demo_performance_stats() -> Result<()> {
    println!("📈 Performance Statistics Demo");
    println!("------------------------------");
    
    // Create a performance monitor
    let mut monitor = PerformanceMonitor::new(1000);
    
    // Simulate some operations with timing
    for i in 0..50 {
        let start = std::time::Instant::now();
        
        // Simulate work with variable duration
        std::thread::sleep(Duration::from_micros(100 + (i % 20) * 10));
        
        let elapsed = start.elapsed();
        monitor.record_duration("search_time", elapsed);
        
        // Also record some synthetic metrics
        monitor.record("accuracy", 0.95 + (i as f64 * 0.001));
        monitor.record("memory_usage", 1000.0 + (i as f64 * 10.0));
    }
    
    // Generate and display report
    let report = monitor.report();
    println!("{}", report);
    
    // Throughput calculator demo
    let mut throughput = ThroughputCalculator::new(Duration::from_millis(100));
    
    for _ in 0..100 {
        throughput.record_operation();
        std::thread::sleep(Duration::from_micros(50));
    }
    
    println!("Throughput: {:.2} ops/sec", throughput.current_throughput());
    println!("Total operations: {}", throughput.total_operations());
    
    println!("✅ Performance monitoring demonstrated\n");
    Ok(())
}

fn demo_direct_io() -> Result<()> {
    println!("💾 Direct I/O Demo");
    println!("------------------");
    
    // Create a test file with aligned data
    let test_data = create_test_file()?;
    let file_size = std::fs::metadata(&test_data)?.len();
    println!("Created test file with {} bytes", file_size);
    
    // Test aligned buffer
    let mut buffer = AlignedBuffer::new(8192)?;
    println!("Aligned buffer created: {} bytes", buffer.len());
    
    // Verify alignment
    let ptr = buffer.as_slice().as_ptr();
    let alignment = ptr as usize % 4096;
    println!("Buffer alignment: {} (should be 0)", alignment);
    
    // Test direct reader (this will fall back to regular I/O on non-Linux)
    match DirectReader::open(&test_data) {
        Ok(mut reader) => {
            println!("Direct reader opened successfully");
            println!("File size: {} bytes", reader.size());
            
            // Test aligned read
            if let Err(e) = reader.read_aligned(0, buffer.as_mut_slice()) {
                println!("Direct read failed (expected on non-Linux): {}", e);
            } else {
                println!("Direct read succeeded");
            }
        },
        Err(e) => println!("Direct reader failed (expected on non-Linux): {}", e),
    }
    
    // Test cached reader
    match CachedDirectReader::open(&test_data, 10, 4096, 2) {
        Ok(mut cached_reader) => {
            println!("Cached direct reader opened successfully");
            
            let mut small_buffer = vec![0u8; 100];
            if let Ok(bytes_read) = cached_reader.read_at(0, &mut small_buffer) {
                println!("Cached read: {} bytes", bytes_read);
                
                let stats = cached_reader.cache_stats();
                println!("Cache stats: {} blocks in cache", stats.size);
            }
        },
        Err(e) => println!("Cached reader failed: {}", e),
    }
    
    // Clean up test file
    let _ = std::fs::remove_file(&test_data);
    
    println!("✅ Direct I/O features demonstrated\n");
    Ok(())
}

fn create_test_file() -> Result<std::path::PathBuf> {
    use std::io::Write;
    
    let temp_dir = tempfile::TempDir::new()?;
    let file_path = temp_dir.path().join("test_direct_io.bin");
    
    // Create 16KB of test data (4 sectors)
    let mut data = vec![0u8; 16384];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }
    
    let mut file = std::fs::File::create(&file_path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    
    // Prevent temp dir from being dropped
    std::mem::forget(temp_dir);
    
    Ok(file_path)
}