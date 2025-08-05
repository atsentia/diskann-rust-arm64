//! Performance metrics and measurement utilities
//!
//! This module provides accurate performance measurement following Rust best practices.

use std::time::{Duration, Instant};
use std::hint::black_box;

/// Timer for measuring elapsed time with minimal overhead
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    #[inline]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    /// Get elapsed time since timer creation
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    /// Get elapsed time in microseconds
    #[inline]
    pub fn elapsed_micros(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1_000_000.0
    }
}

/// Performance metrics collector
#[derive(Debug, Default)]
pub struct Metrics {
    samples: Vec<Duration>,
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a timing sample
    pub fn add_sample(&mut self, duration: Duration) {
        self.samples.push(duration);
    }
    
    /// Measure a function execution time
    #[inline]
    pub fn measure<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = Timer::new();
        let result = f();
        self.add_sample(timer.elapsed());
        result
    }
    
    /// Get statistics from collected samples
    pub fn stats(&self) -> Stats {
        if self.samples.is_empty() {
            return Stats::default();
        }
        
        let mut sorted = self.samples.clone();
        sorted.sort();
        
        let count = sorted.len();
        let sum: Duration = sorted.iter().sum();
        let mean = sum / count as u32;
        
        let p50 = sorted[count / 2];
        let p90 = sorted[count * 9 / 10];
        let p99 = sorted[count * 99 / 100];
        let min = sorted[0];
        let max = sorted[count - 1];
        
        Stats {
            count,
            mean,
            min,
            max,
            p50,
            p90,
            p99,
        }
    }
}

/// Statistical summary of timing measurements
#[derive(Debug, Default)]
pub struct Stats {
    pub count: usize,
    pub mean: Duration,
    pub min: Duration,
    pub max: Duration,
    pub p50: Duration,
    pub p90: Duration,
    pub p99: Duration,
}

impl Stats {
    /// Convert to queries per second
    pub fn qps(&self) -> f64 {
        if self.mean.as_secs_f64() > 0.0 {
            1.0 / self.mean.as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// Pretty print statistics
    pub fn print(&self, label: &str) {
        println!("{} Statistics:", label);
        println!("  Count: {}", self.count);
        println!("  Mean: {:?}", self.mean);
        println!("  Min: {:?}", self.min);
        println!("  Max: {:?}", self.max);
        println!("  P50: {:?}", self.p50);
        println!("  P90: {:?}", self.p90);
        println!("  P99: {:?}", self.p99);
        println!("  QPS: {:.0}", self.qps());
    }
}

/// Benchmark a function with warmup and multiple iterations
pub fn benchmark<F, R>(name: &str, warmup: usize, iterations: usize, mut f: F) -> Stats
where
    F: FnMut() -> R,
{
    // Warmup
    for _ in 0..warmup {
        black_box(f());
    }
    
    // Actual measurement
    let mut metrics = Metrics::new();
    for _ in 0..iterations {
        metrics.measure(|| black_box(f()));
    }
    
    let stats = metrics.stats();
    stats.print(name);
    stats
}

/// Measure throughput for batch operations
pub fn measure_throughput<F>(
    name: &str,
    batch_size: usize,
    duration_secs: u64,
    mut f: F,
) -> f64
where
    F: FnMut(),
{
    let start = Instant::now();
    let target_duration = Duration::from_secs(duration_secs);
    let mut operations = 0;
    
    while start.elapsed() < target_duration {
        f();
        operations += batch_size;
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    let throughput = operations as f64 / elapsed;
    
    println!("{} Throughput: {:.0} ops/sec", name, throughput);
    throughput
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_timer() {
        let timer = Timer::new();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(20));
    }
    
    #[test]
    fn test_metrics() {
        let mut metrics = Metrics::new();
        
        for i in 1..=10 {
            metrics.add_sample(Duration::from_millis(i));
        }
        
        let stats = metrics.stats();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, Duration::from_millis(1));
        assert_eq!(stats.max, Duration::from_millis(10));
        assert_eq!(stats.p50, Duration::from_millis(5));
    }
    
    #[test]
    fn test_benchmark() {
        let stats = benchmark("test", 5, 10, || {
            thread::sleep(Duration::from_micros(100));
            42
        });
        
        assert_eq!(stats.count, 10);
        assert!(stats.mean >= Duration::from_micros(100));
    }
}