//! Focused benchmark for distance computation performance
//! Tests L2 distance calculation with various vector dimensions

use anyhow::Result;
use diskann::{Distance, create_distance_function};
use std::time::Instant;
use console::style;

fn print_system_info() {
    println!("{}", style("=== System Information ===").bold().blue());
    println!("Platform: {}", std::env::consts::OS);
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("CPU: {} cores", num_cpus::get());
    
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        println!("Device: Qualcomm Snapdragon X (ARM64)");
        // Check NPU availability (placeholder - actual implementation would check Windows ML)
        if false { // QualcommDistance not exposed in public API
            println!("NPU Status: {} Available", style("✓").green());
        } else {
            println!("NPU Status: {} Not Available", style("✗").red());
        }
    }
    
    // Check SIMD availability
    #[cfg(target_arch = "aarch64")]
    println!("SIMD: ARM64 NEON");
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("SIMD: x86-64 AVX-512");
        } else if is_x86_feature_detected!("avx2") {
            println!("SIMD: x86-64 AVX2");
        } else if is_x86_feature_detected!("sse4.2") {
            println!("SIMD: x86-64 SSE4.2");
        } else {
            println!("SIMD: Basic x86-64");
        }
    }
    
    println!("{}", style("─".repeat(60)).dim());
}

fn main() -> Result<()> {
    println!("{}", style("DiskANN Distance Computation Benchmark").bold().green());
    println!("{}", style("Testing L2 distance performance across dimensions").dim());
    println!();

    print_system_info();
    
    // Test parameters
    let dimensions = vec![128, 256, 512, 768, 1024, 1536];
    let num_iterations = 100_000;
    
    println!("\n{}", style("=== Benchmark Configuration ===").bold().blue());
    println!("Distance metric: L2 (Euclidean)");
    println!("Iterations per dimension: {}", num_iterations.to_string().as_str());
    println!("Dimensions to test: {:?}", dimensions);
    println!("{}", style("─".repeat(60)).dim());

    // Generate random vectors for testing
    println!("\n{}", style("Generating test vectors...").yellow());
    let mut test_vectors = Vec::new();
    for &dim in &dimensions {
        let vec1: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let vec2: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        test_vectors.push((dim, vec1, vec2));
    }

    println!("\n{}", style("=== Running Benchmarks ===").bold().blue());
    println!("{:<10} {:<15} {:<15} {:<15} {:<20}", 
        "Dimension", "Time/Op (ns)", "Ops/Second", "GB/s", "Implementation");
    println!("{}", style("─".repeat(80)).dim());

    for (dim, vec1, vec2) in &test_vectors {
        let distance_fn = create_distance_function(Distance::L2, *dim);
        
        // Warm up
        for _ in 0..1000 {
            let _ = distance_fn.distance(vec1, vec2)?;
        }
        
        // Measure
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = distance_fn.distance(vec1, vec2)?;
        }
        let elapsed = start.elapsed();
        
        // Calculate metrics
        let ns_per_op = elapsed.as_nanos() as f64 / num_iterations as f64;
        let ops_per_sec = 1_000_000_000.0 / ns_per_op;
        let bytes_per_op = (dim * 2 * 4) as f64; // 2 vectors * 4 bytes per f32
        let gb_per_sec = (bytes_per_op * ops_per_sec) / 1_000_000_000.0;
        
        // Determine implementation used
        let impl_name = if cfg!(all(target_os = "windows", target_arch = "aarch64")) 
            && false /* QualcommDistance::is_available() */ {
            "NPU/DirectML"
        } else if cfg!(target_arch = "aarch64") {
            "ARM64 NEON"
        } else if cfg!(target_arch = "x86_64") {
            "x86-64 SIMD"
        } else {
            "Portable SIMD"
        };
        
        println!("{:<10} {:<15.2} {:<15.0} {:<15.2} {:<20}", 
            dim, ns_per_op, ops_per_sec, gb_per_sec, impl_name);
    }
    
    println!("\n{}", style("=== Comparison with Scalar Implementation ===").bold().blue());
    println!("{:<10} {:<15} {:<15}", "Dimension", "Scalar (ns)", "SIMD Speedup");
    println!("{}", style("─".repeat(40)).dim());
    
    // Compare with scalar implementation for reference
    for (dim, vec1, vec2) in &test_vectors {
        // Scalar implementation
        let scalar_start = Instant::now();
        for _ in 0..num_iterations/10 { // Less iterations for slower scalar
            let mut sum = 0.0f32;
            for i in 0..*dim {
                let diff = vec1[i] - vec2[i];
                sum += diff * diff;
            }
            let _ = sum.sqrt();
        }
        let scalar_elapsed = scalar_start.elapsed();
        let scalar_ns = scalar_elapsed.as_nanos() as f64 / (num_iterations/10) as f64;
        
        // SIMD implementation
        let distance_fn = create_distance_function(Distance::L2, *dim);
        let simd_start = Instant::now();
        for _ in 0..num_iterations {
            let _ = distance_fn.distance(vec1, vec2)?;
        }
        let simd_elapsed = simd_start.elapsed();
        let simd_ns = simd_elapsed.as_nanos() as f64 / num_iterations as f64;
        
        let speedup = scalar_ns / simd_ns;
        
        println!("{:<10} {:<15.2} {:<15.2}x", dim, scalar_ns, speedup);
    }
    
    println!("\n{}", style("✓ Benchmark completed successfully").green());
    
    Ok(())
}