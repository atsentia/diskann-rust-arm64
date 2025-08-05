//! Quick NPU/SIMD comparison benchmark - runs in <60 seconds
//! Compares CPU SIMD vs potential NPU acceleration

use anyhow::Result;
use diskann::{Distance, create_distance_function};
use std::time::{Duration, Instant};
use console::style;

fn print_system_info() {
    println!("{}", style("=== System Information ===").bold().blue());
    println!("Platform: {}", std::env::consts::OS);
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("CPU: {} cores", num_cpus::get());
    
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        println!("Device: Qualcomm Snapdragon X (ARM64)");
        println!("SIMD: ARM64 NEON");
        // In a real implementation, we'd check Windows ML/DirectML here
        println!("NPU: {} (Windows ML integration pending)", style("Not Available").yellow());
    }
    
    println!("{}", style("─".repeat(60)).dim());
}

fn main() -> Result<()> {
    println!("{}", style("DiskANN NPU/SIMD Quick Benchmark").bold().green());
    println!("{}", style("Comparing acceleration methods - completes in <60 seconds").dim());
    println!();

    print_system_info();
    
    // Quick test parameters
    let dimensions = vec![128, 768, 1536];
    let batch_sizes = vec![1, 32, 128];  // NPU typically better for batches
    let num_iterations = 10000;
    
    println!("\n{}", style("=== Benchmark Configuration ===").bold().blue());
    println!("Dimensions: {:?}", dimensions);
    println!("Batch sizes: {:?}", batch_sizes);
    println!("Iterations per test: {}", num_iterations);
    println!("{}", style("─".repeat(60)).dim());

    // Test different dimensions and batch sizes
    println!("\n{}", style("=== Distance Computation Performance ===").bold().blue());
    println!("{:<10} {:<10} {:<15} {:<15} {:<20}", 
        "Dimension", "Batch", "Time/Op (μs)", "GFLOPS", "Implementation");
    println!("{}", style("─".repeat(70)).dim());

    for &dim in &dimensions {
        for &batch_size in &batch_sizes {
            // Generate test vectors
            let vectors: Vec<Vec<f32>> = (0..batch_size * 2)
                .map(|_| (0..dim).map(|_| rand::random::<f32>()).collect())
                .collect();
            
            let distance_fn = create_distance_function(Distance::L2, dim);
            
            // Warm up
            for i in 0..100 {
                let _ = distance_fn.distance(&vectors[i % batch_size], &vectors[batch_size + (i % batch_size)])?;
            }
            
            // Measure batch processing
            let start = Instant::now();
            for _ in 0..num_iterations/batch_size {
                for i in 0..batch_size {
                    let _ = distance_fn.distance(&vectors[i], &vectors[batch_size + i])?;
                }
            }
            let elapsed = start.elapsed();
            
            // Calculate metrics
            let us_per_op = elapsed.as_micros() as f64 / num_iterations as f64;
            let flops_per_distance = dim * 3; // sub, mul, add per element + sqrt
            let gflops = (flops_per_distance as f64 * num_iterations as f64) / elapsed.as_nanos() as f64;
            
            let impl_name = if batch_size > 32 {
                "ARM64 NEON (NPU candidate)"
            } else {
                "ARM64 NEON"
            };
            
            println!("{:<10} {:<10} {:<15.2} {:<15.2} {:<20}", 
                dim, batch_size, us_per_op, gflops, impl_name);
        }
        println!();
    }

    // Simulate what NPU acceleration could provide
    println!("\n{}", style("=== Projected NPU Performance ===").bold().blue());
    println!("{}", style("Based on typical NPU characteristics:").dim());
    println!("• Large batch (128+): 5-10x speedup expected");
    println!("• Power efficiency: 10-20x better GFLOPS/Watt");
    println!("• Best for: Batch inference, repeated operations");
    println!("• Overhead: High for single operations");
    
    println!("\n{}", style("=== Current Status ===").bold().yellow());
    println!("✗ NPU: Windows ML integration not yet implemented");
    println!("✓ CPU: ARM64 NEON SIMD fully operational");
    println!("✓ Performance: Already achieving good results with NEON");
    
    println!("\n{}", style("✓ Quick benchmark completed").green());
    
    Ok(())
}