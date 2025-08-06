//! Advanced performance statistics and monitoring
//!
//! This module provides detailed performance tracking and percentile statistics
//! similar to `percentile_stats.h` from the C++ DiskANN implementation.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Performance statistics tracker with percentile calculations
#[derive(Debug, Clone)]
pub struct PercentileStats {
    name: String,
    samples: VecDeque<f64>,
    max_samples: usize,
    sum: f64,
    sum_squares: f64,
    min_value: f64,
    max_value: f64,
    count: u64,
}

impl PercentileStats {
    /// Create a new percentile statistics tracker
    pub fn new(name: &str, max_samples: usize) -> Self {
        Self {
            name: name.to_string(),
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            sum: 0.0,
            sum_squares: 0.0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            count: 0,
        }
    }

    /// Add a new sample
    pub fn add_sample(&mut self, value: f64) {
        // Remove oldest sample if at capacity
        if self.samples.len() >= self.max_samples {
            if let Some(old_value) = self.samples.pop_front() {
                self.sum -= old_value;
                self.sum_squares -= old_value * old_value;
                // Recalculate min/max since we removed a sample
                self.recalculate_min_max();
            }
        }

        // Add new sample
        self.samples.push_back(value);
        self.sum += value;
        self.sum_squares += value * value;
        
        // Update min/max with new value
        if self.samples.len() == 1 {
            // First sample
            self.min_value = value;
            self.max_value = value;
        } else {
            self.min_value = self.min_value.min(value);
            self.max_value = self.max_value.max(value);
        }
        
        self.count += 1;
    }

    /// Recalculate min and max values from current samples
    fn recalculate_min_max(&mut self) {
        if self.samples.is_empty() {
            self.min_value = f64::INFINITY;
            self.max_value = f64::NEG_INFINITY;
        } else {
            self.min_value = self.samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            self.max_value = self.samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }
    }

    /// Get the mean value
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f64
        }
    }

    /// Get the standard deviation
    pub fn std_dev(&self) -> f64 {
        if self.samples.len() < 2 {
            0.0
        } else {
            let n = self.samples.len() as f64;
            let variance = (self.sum_squares - (self.sum * self.sum) / n) / (n - 1.0);
            variance.max(0.0).sqrt()
        }
    }

    /// Get the minimum value
    pub fn min(&self) -> f64 {
        self.min_value
    }

    /// Get the maximum value
    pub fn max(&self) -> f64 {
        self.max_value
    }

    /// Get a percentile value (0.0 to 100.0)
    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut sorted_samples: Vec<f64> = self.samples.iter().copied().collect();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (p / 100.0) * (sorted_samples.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            sorted_samples[lower_index]
        } else {
            let weight = index - lower_index as f64;
            sorted_samples[lower_index] * (1.0 - weight) + sorted_samples[upper_index] * weight
        }
    }

    /// Get common percentiles (50th, 90th, 95th, 99th)
    pub fn common_percentiles(&self) -> CommonPercentiles {
        CommonPercentiles {
            p50: self.percentile(50.0),
            p90: self.percentile(90.0),
            p95: self.percentile(95.0),
            p99: self.percentile(99.0),
        }
    }

    /// Get the number of samples
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get the total count of all samples ever added
    pub fn total_count(&self) -> u64 {
        self.count
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
        self.sum = 0.0;
        self.sum_squares = 0.0;
        self.min_value = f64::INFINITY;
        self.max_value = f64::NEG_INFINITY;
        self.count = 0;
    }

    /// Get a summary of statistics
    pub fn summary(&self) -> StatsSummary {
        StatsSummary {
            name: self.name.clone(),
            count: self.sample_count(),
            total_count: self.total_count(),
            mean: self.mean(),
            std_dev: self.std_dev(),
            min: self.min(),
            max: self.max(),
            percentiles: self.common_percentiles(),
        }
    }
}

/// Common percentile values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonPercentiles {
    pub p50: f64,  // Median
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Summary of statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSummary {
    pub name: String,
    pub count: usize,
    pub total_count: u64,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: CommonPercentiles,
}

/// Timer for measuring operation durations
#[derive(Debug)]
pub struct Timer {
    start_time: Instant,
    name: String,
}

impl Timer {
    /// Create a new timer
    pub fn new(name: &str) -> Self {
        Self {
            start_time: Instant::now(),
            name: name.to_string(),
        }
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_micros(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_millis(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Restart the timer
    pub fn restart(&mut self) {
        self.start_time = Instant::now();
    }
}

/// RAII timer that automatically records duration to statistics
pub struct ScopedTimer<'a> {
    timer: Timer,
    stats: &'a mut PercentileStats,
}

impl<'a> ScopedTimer<'a> {
    /// Create a new scoped timer
    pub fn new(name: &str, stats: &'a mut PercentileStats) -> Self {
        Self {
            timer: Timer::new(name),
            stats,
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        let elapsed_micros = self.timer.elapsed_micros() as f64;
        self.stats.add_sample(elapsed_micros);
    }
}

/// Performance monitor for tracking multiple metrics
#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics: hashbrown::HashMap<String, PercentileStats>,
    max_samples_per_metric: usize,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(max_samples_per_metric: usize) -> Self {
        Self {
            metrics: hashbrown::HashMap::new(),
            max_samples_per_metric,
        }
    }

    /// Get or create a metric
    pub fn get_or_create_metric(&mut self, name: &str) -> &mut PercentileStats {
        self.metrics.entry(name.to_string()).or_insert_with(|| {
            PercentileStats::new(name, self.max_samples_per_metric)
        })
    }

    /// Record a sample for a metric
    pub fn record(&mut self, metric_name: &str, value: f64) {
        self.get_or_create_metric(metric_name).add_sample(value);
    }

    /// Record elapsed time in microseconds
    pub fn record_duration(&mut self, metric_name: &str, duration: Duration) {
        let micros = duration.as_micros() as f64;
        self.record(metric_name, micros);
    }

    /// Create a scoped timer for a metric
    pub fn scoped_timer(&mut self, metric_name: &str) -> ScopedTimer {
        let stats = self.get_or_create_metric(metric_name);
        ScopedTimer::new(metric_name, stats)
    }

    /// Get statistics for a metric
    pub fn get_stats(&self, metric_name: &str) -> Option<StatsSummary> {
        self.metrics.get(metric_name).map(|stats| stats.summary())
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }

    /// Get summaries for all metrics
    pub fn all_summaries(&self) -> Vec<StatsSummary> {
        self.metrics.values().map(|stats| stats.summary()).collect()
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        for stats in self.metrics.values_mut() {
            stats.clear();
        }
    }

    /// Clear a specific metric
    pub fn clear_metric(&mut self, metric_name: &str) {
        if let Some(stats) = self.metrics.get_mut(metric_name) {
            stats.clear();
        }
    }

    /// Generate a performance report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Performance Report ===\n");
        
        for summary in self.all_summaries() {
            report.push_str(&format!("\nMetric: {}\n", summary.name));
            report.push_str(&format!("  Samples: {} (total: {})\n", summary.count, summary.total_count));
            report.push_str(&format!("  Mean: {:.2} μs\n", summary.mean));
            report.push_str(&format!("  Std Dev: {:.2} μs\n", summary.std_dev));
            report.push_str(&format!("  Min: {:.2} μs\n", summary.min));
            report.push_str(&format!("  Max: {:.2} μs\n", summary.max));
            report.push_str(&format!("  P50: {:.2} μs\n", summary.percentiles.p50));
            report.push_str(&format!("  P90: {:.2} μs\n", summary.percentiles.p90));
            report.push_str(&format!("  P95: {:.2} μs\n", summary.percentiles.p95));
            report.push_str(&format!("  P99: {:.2} μs\n", summary.percentiles.p99));
        }
        
        report
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new(10000) // Default to 10k samples per metric
    }
}

/// Macro for easy timing of code blocks
#[macro_export]
macro_rules! time_block {
    ($monitor:expr, $metric:expr, $block:expr) => {{
        let _timer = $monitor.scoped_timer($metric);
        $block
    }};
}

/// Throughput calculator for measuring operations per second
#[derive(Debug)]
pub struct ThroughputCalculator {
    operation_count: u64,
    start_time: Instant,
    last_report_time: Instant,
    report_interval: Duration,
}

impl ThroughputCalculator {
    /// Create a new throughput calculator
    pub fn new(report_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            operation_count: 0,
            start_time: now,
            last_report_time: now,
            report_interval,
        }
    }

    /// Record an operation
    pub fn record_operation(&mut self) {
        self.operation_count += 1;
    }

    /// Record multiple operations
    pub fn record_operations(&mut self, count: u64) {
        self.operation_count += count;
    }

    /// Get current throughput (operations per second)
    pub fn current_throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.operation_count as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get throughput since last report
    pub fn throughput_since_last_report(&mut self) -> (f64, Duration) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_report_time);
        let operations_since_last = self.operation_count; // Simplified for demo
        
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            operations_since_last as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        
        self.last_report_time = now;
        (throughput, elapsed)
    }

    /// Check if it's time for a throughput report
    pub fn should_report(&self) -> bool {
        self.last_report_time.elapsed() >= self.report_interval
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.operation_count = 0;
        self.start_time = now;
        self.last_report_time = now;
    }

    /// Get total operations recorded
    pub fn total_operations(&self) -> u64 {
        self.operation_count
    }

    /// Get total elapsed time
    pub fn total_elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_percentile_stats_basic() {
        let mut stats = PercentileStats::new("test", 1000);
        
        // Add some samples
        for i in 1..=100 {
            stats.add_sample(i as f64);
        }
        
        assert_eq!(stats.sample_count(), 100);
        assert_eq!(stats.total_count(), 100);
        assert_eq!(stats.min(), 1.0);
        assert_eq!(stats.max(), 100.0);
        
        // Mean should be ~50.5
        let mean = stats.mean();
        assert!((mean - 50.5).abs() < 0.1);
        
        // Median should be ~50.5
        let median = stats.percentile(50.0);
        assert!((median - 50.5).abs() < 1.0);
        
        // 99th percentile should be ~99
        let p99 = stats.percentile(99.0);
        assert!((p99 - 99.0).abs() < 1.0);
    }

    #[test]
    fn test_percentile_stats_capacity_limit() {
        let mut stats = PercentileStats::new("test", 10);
        
        // Add more samples than capacity
        for i in 1..=20 {
            stats.add_sample(i as f64);
        }
        
        // Should only keep last 10 samples
        assert_eq!(stats.sample_count(), 10);
        assert_eq!(stats.total_count(), 20);
        
        // Min should be from recent samples
        assert!(stats.min() >= 11.0);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        thread::sleep(Duration::from_millis(10));
        
        let elapsed = timer.elapsed_millis();
        assert!(elapsed >= 10);
        assert!(elapsed < 100); // Should be reasonably close
    }

    #[test]
    fn test_scoped_timer() {
        let mut stats = PercentileStats::new("test", 100);
        
        {
            let _timer = ScopedTimer::new("test", &mut stats);
            thread::sleep(Duration::from_millis(5));
        }
        
        assert_eq!(stats.sample_count(), 1);
        let sample = stats.mean();
        assert!(sample >= 5000.0); // At least 5ms in microseconds
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new(100);
        
        // Record some metrics
        monitor.record("search_time", 1000.0);
        monitor.record("search_time", 1500.0);
        monitor.record("build_time", 5000.0);
        
        let search_stats = monitor.get_stats("search_time").unwrap();
        assert_eq!(search_stats.count, 2);
        assert_eq!(search_stats.mean, 1250.0);
        
        let build_stats = monitor.get_stats("build_time").unwrap();
        assert_eq!(build_stats.count, 1);
        assert_eq!(build_stats.mean, 5000.0);
        
        // Test report generation
        let report = monitor.report();
        assert!(report.contains("Performance Report"));
        assert!(report.contains("search_time"));
        assert!(report.contains("build_time"));
    }

    #[test]
    fn test_throughput_calculator() {
        let mut calc = ThroughputCalculator::new(Duration::from_millis(100));
        
        // Record some operations
        calc.record_operations(100);
        thread::sleep(Duration::from_millis(10));
        
        let throughput = calc.current_throughput();
        assert!(throughput > 0.0);
        
        assert_eq!(calc.total_operations(), 100);
        assert!(calc.total_elapsed().as_millis() >= 10);
    }

    #[test]
    fn test_common_percentiles() {
        let mut stats = PercentileStats::new("test", 1000);
        
        // Add samples from 1 to 1000
        for i in 1..=1000 {
            stats.add_sample(i as f64);
        }
        
        let percentiles = stats.common_percentiles();
        
        // Check that percentiles are in reasonable ranges
        assert!((percentiles.p50 - 500.5).abs() < 1.0);
        assert!((percentiles.p90 - 900.5).abs() < 1.0);
        assert!((percentiles.p95 - 950.5).abs() < 1.0);
        assert!((percentiles.p99 - 990.5).abs() < 1.0);
    }

    #[test]
    fn test_stats_summary_serialization() {
        let mut stats = PercentileStats::new("test", 100);
        stats.add_sample(100.0);
        stats.add_sample(200.0);
        
        let summary = stats.summary();
        
        // Test JSON serialization
        let json = serde_json::to_string(&summary).unwrap();
        let deserialized: StatsSummary = serde_json::from_str(&json).unwrap();
        
        assert_eq!(summary.name, deserialized.name);
        assert_eq!(summary.count, deserialized.count);
        assert!((summary.mean - deserialized.mean).abs() < 0.001);
    }
}