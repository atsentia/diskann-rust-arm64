//! Quick timing verification test
use std::time::Instant;

fn main() {
    println!("🔍 Timing Verification Test");
    
    // Simulate exactly what our benchmark does
    let query_count = 1000;
    
    // Time 1000 very fast operations
    let start = Instant::now();
    
    for _i in 0..query_count {
        // Simulate very fast operation (just some math)
        let _result = (42.0f32 * 3.14159f32).sqrt();
    }
    
    let elapsed = start.elapsed();
    let qps = query_count as f64 / elapsed.as_secs_f64();
    let avg_latency_microseconds = (elapsed.as_secs_f64() * 1_000_000.0) / query_count as f64;
    
    println!("📊 Results:");
    println!("   Queries: {}", query_count);
    println!("   Total time: {:.9}s ({:.3}ms)", elapsed.as_secs_f64(), elapsed.as_secs_f64() * 1000.0);
    println!("   QPS: {:.0}", qps);
    println!("   Avg latency: {:.1}μs", avg_latency_microseconds);
    
    // Manual verification
    println!("\n🧮 Manual Verification:");
    println!("   1000 queries / {:.6}s = {:.0} QPS", elapsed.as_secs_f64(), qps);
    println!("   {:.6}s / 1000 queries = {:.3}ms per query = {:.1}μs per query", 
             elapsed.as_secs_f64(), 
             (elapsed.as_secs_f64() * 1000.0) / query_count as f64,
             avg_latency_microseconds);
    
    if qps > 1_000_000.0 {
        println!("⚠️  WARNING: QPS over 1M - this means sub-microsecond operations!");
    }
}