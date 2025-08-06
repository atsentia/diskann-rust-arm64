//! Comprehensive DiskANN Rust vs C++ Parity Testing Framework
//! 
//! This module implements the exhaustive test plan for validating the Rust DiskANN
//! implementation against the original Microsoft C++ DiskANN library.
//! 
//! The tests are organized into three tiers:
//! - Tier 1: Foundational Parity & Correctness Analysis
//! - Tier 2: Advanced Capabilities & Robustness Testing  
//! - Tier 3: Granular Performance & Efficiency Benchmarking

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, RwLock, Barrier};
use std::thread;
use std::path::{Path, PathBuf};
use std::process::Command;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Serialize, Deserialize};

pub mod tier1_foundational;
pub mod tier2_robustness;  
pub mod tier3_performance;
pub mod cpp_integration;
pub mod test_data;
pub mod comparison_framework;
pub mod reporting;

/// Global configuration for comprehensive parity testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityTestConfig {
    /// Path to the C++ DiskANN reference implementation
    pub cpp_diskann_path: PathBuf,
    /// Global random seed for deterministic testing
    pub global_seed: u64,
    /// Tolerance for floating-point comparisons
    pub float_tolerance: f64,
    /// Timeout for individual test operations
    pub test_timeout: Duration,
    /// Whether to run expensive performance tests
    pub run_performance_tests: bool,
    /// Whether to run stress tests
    pub run_stress_tests: bool,
    /// Maximum memory usage for testing (in bytes)
    pub max_memory_usage: Option<usize>,
}

impl Default for ParityTestConfig {
    fn default() -> Self {
        Self {
            cpp_diskann_path: PathBuf::from("/tmp/diskann_cpp_reference"),
            global_seed: 42,
            float_tolerance: 1e-6,
            test_timeout: Duration::from_secs(300), // 5 minutes per test
            run_performance_tests: std::env::var("RUN_PERF_TESTS").is_ok(),
            run_stress_tests: std::env::var("RUN_STRESS_TESTS").is_ok(),
            max_memory_usage: None,
        }
    }
}

/// Results from comparing Rust and C++ implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Test identifier
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Detailed comparison metrics
    pub metrics: ComparisonMetrics,
    /// Error message if test failed
    pub error_message: Option<String>,
    /// Duration of the test
    pub duration: Duration,
}

/// Detailed metrics from implementation comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// Algorithm correctness metrics
    pub correctness: CorrectnessMetrics,
    /// Performance comparison metrics
    pub performance: PerformanceMetrics,
    /// Resource usage metrics
    pub resources: ResourceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessMetrics {
    /// Whether results are bitwise identical
    pub results_identical: bool,
    /// Maximum difference in distance calculations
    pub max_distance_difference: f64,
    /// Number of different neighbor results
    pub neighbor_differences: usize,
    /// Graph topology similarity score (0.0 = completely different, 1.0 = identical)
    pub graph_similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Rust implementation performance
    pub rust_performance: PerformanceData,
    /// C++ implementation performance
    pub cpp_performance: PerformanceData,
    /// Performance ratio (rust/cpp)
    pub performance_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    /// Operation throughput (operations per second)
    pub throughput: f64,
    /// Average latency
    pub avg_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// 99th percentile latency
    pub p99_latency: Duration,
    /// Peak memory usage
    pub peak_memory: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Memory usage comparison
    pub memory_usage_ratio: f64,
    /// Disk I/O operations comparison
    pub disk_io_ratio: f64,
    /// CPU utilization comparison
    pub cpu_utilization_ratio: f64,
}

/// Framework for executing comprehensive parity tests
pub struct ParityTestFramework {
    config: ParityTestConfig,
    results: Vec<ComparisonResult>,
}

impl ParityTestFramework {
    pub fn new(config: ParityTestConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Execute all tiers of parity testing
    pub fn run_comprehensive_tests(&mut self) -> Result<Vec<ComparisonResult>> {
        log::info!("Starting comprehensive DiskANN parity testing");
        
        // Verify C++ reference is available
        self.verify_cpp_reference()?;
        
        // Tier 1: Foundational tests (must pass)
        log::info!("Running Tier 1: Foundational Parity Tests");
        let tier1_results = self.run_tier1_tests()?;
        self.results.extend(tier1_results);
        
        // Only proceed if Tier 1 passes
        let tier1_passed = self.results.iter().all(|r| r.passed);
        if !tier1_passed {
            return Err(anyhow!("Tier 1 foundational tests failed - cannot proceed"));
        }
        
        // Tier 2: Robustness tests (should pass)
        log::info!("Running Tier 2: Robustness and Edge Case Tests");
        let tier2_results = self.run_tier2_tests()?;
        self.results.extend(tier2_results);
        
        // Tier 3: Performance tests (may have acceptable differences)
        if self.config.run_performance_tests {
            log::info!("Running Tier 3: Performance Benchmarking");
            let tier3_results = self.run_tier3_tests()?;
            self.results.extend(tier3_results);
        }
        
        log::info!("Comprehensive parity testing completed");
        Ok(self.results.clone())
    }

    /// Verify C++ reference implementation is available and built
    fn verify_cpp_reference(&self) -> Result<()> {
        let cpp_path = &self.config.cpp_diskann_path;
        
        if !cpp_path.exists() {
            return Err(anyhow!(
                "C++ DiskANN reference not found at {:?}. Please run setup_cpp_reference.sh",
                cpp_path
            ));
        }
        
        // Check if executables exist
        let required_binaries = ["build_memory_index", "search_memory_index", "build_disk_index"];
        for binary in &required_binaries {
            let binary_path = cpp_path.join("build").join(binary);
            if !binary_path.exists() {
                return Err(anyhow!(
                    "Required C++ binary {:?} not found. Please build C++ DiskANN",
                    binary_path
                ));
            }
        }
        
        log::info!("C++ DiskANN reference verified at {:?}", cpp_path);
        Ok(())
    }

    fn run_tier1_tests(&self) -> Result<Vec<ComparisonResult>> {
        tier1_foundational::run_all_tier1_tests(&self.config)
    }

    fn run_tier2_tests(&self) -> Result<Vec<ComparisonResult>> {
        tier2_robustness::run_all_tier2_tests(&self.config)
    }

    fn run_tier3_tests(&self) -> Result<Vec<ComparisonResult>> {
        tier3_performance::run_all_tier3_tests(&self.config)
    }

    /// Generate comprehensive test report
    pub fn generate_report(&self) -> Result<String> {
        reporting::generate_comprehensive_report(&self.results)
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> TestSummary {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;
        
        let avg_performance_ratio = if !self.results.is_empty() {
            self.results.iter()
                .map(|r| r.metrics.performance.performance_ratio)
                .sum::<f64>() / self.results.len() as f64
        } else {
            0.0
        };

        TestSummary {
            total_tests,
            passed_tests,
            failed_tests,
            success_rate: passed_tests as f64 / total_tests as f64,
            avg_performance_ratio,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f64,
    pub avg_performance_ratio: f64,
}

/// Utility functions for test execution
pub mod utils {
    use super::*;
    
    /// Execute a test operation with timeout
    pub fn execute_with_timeout<F, T>(
        operation: F,
        timeout: Duration,
        operation_name: &str,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();
        
        let handle = thread::spawn(move || {
            let result = operation();
            let _ = sender.send(result);
        });
        
        match receiver.recv_timeout(timeout) {
            Ok(result) => {
                handle.join().map_err(|_| anyhow!("Thread panicked"))?;
                result
            }
            Err(_) => {
                // Note: we can't actually kill the thread, but we can abandon it
                Err(anyhow!("Operation '{}' timed out after {:?}", operation_name, timeout))
            }
        }
    }

    /// Compare floating-point values with tolerance
    pub fn compare_floats(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    /// Generate deterministic random vectors
    pub fn generate_random_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        
        (0..count)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.gen::<f32>())
                    .collect()
            })
            .collect()
    }

    /// Measure operation duration and return result with timing
    pub fn time_operation<F, T>(operation: F) -> (T, Duration)
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Calculate percentiles from a sorted list of durations
    pub fn calculate_percentile(sorted_durations: &[Duration], percentile: f64) -> Duration {
        if sorted_durations.is_empty() {
            return Duration::from_secs(0);
        }
        
        let index = (sorted_durations.len() as f64 * percentile / 100.0) as usize;
        let index = index.min(sorted_durations.len() - 1);
        sorted_durations[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = ParityTestConfig::default();
        assert_eq!(config.global_seed, 42);
        assert_eq!(config.float_tolerance, 1e-6);
    }

    #[test]
    fn test_framework_creation() {
        let config = ParityTestConfig::default();
        let framework = ParityTestFramework::new(config);
        assert_eq!(framework.results.len(), 0);
    }

    #[test]
    fn test_utils_generate_random_vectors() {
        let vectors = utils::generate_random_vectors(10, 5, 42);
        assert_eq!(vectors.len(), 10);
        assert_eq!(vectors[0].len(), 5);
        
        // Test determinism
        let vectors2 = utils::generate_random_vectors(10, 5, 42);
        assert_eq!(vectors, vectors2);
    }

    #[test]
    fn test_utils_compare_floats() {
        assert!(utils::compare_floats(1.0, 1.0000001, 1e-6));
        assert!(!utils::compare_floats(1.0, 1.1, 1e-6));
    }

    #[test]
    fn test_utils_time_operation() {
        let (result, duration) = utils::time_operation(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
    }
}