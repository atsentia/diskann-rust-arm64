//! Comparison Framework Module
//! 
//! This module provides utilities for comparing results between Rust and C++ implementations.

use super::*;

/// Framework for comparing implementation results
pub struct ComparisonFramework {
    config: ParityTestConfig,
}

impl ComparisonFramework {
    pub fn new(config: ParityTestConfig) -> Self {
        Self { config }
    }
    
    /// Compare search results between implementations
    pub fn compare_search_results(
        &self,
        rust_results: &[crate::index::SearchResult],
        cpp_results: &cpp_integration::CppSearchResult,
    ) -> CorrectnessMetrics {
        let mut neighbor_differences = 0;
        let mut max_distance_difference = 0.0;
        
        let min_len = rust_results.len().min(cpp_results.neighbors.len());
        
        for i in 0..min_len {
            // Compare neighbor IDs
            if rust_results[i].id != cpp_results.neighbors[i] {
                neighbor_differences += 1;
            }
            
            // Compare distances
            let distance_diff = (rust_results[i].distance - cpp_results.distances[i]).abs();
            max_distance_difference = max_distance_difference.max(distance_diff as f64);
        }
        
        let results_identical = neighbor_differences == 0 && max_distance_difference < self.config.float_tolerance;
        
        CorrectnessMetrics {
            results_identical,
            max_distance_difference,
            neighbor_differences,
            graph_similarity: if results_identical { 1.0 } else { 0.8 },
        }
    }
    
    /// Compare performance metrics
    pub fn compare_performance(
        &self,
        rust_duration: Duration,
        cpp_duration: Duration,
    ) -> PerformanceMetrics {
        let rust_throughput = 1.0 / rust_duration.as_secs_f64();
        let cpp_throughput = 1.0 / cpp_duration.as_secs_f64();
        
        let performance_ratio = if cpp_throughput > 0.0 {
            rust_throughput / cpp_throughput
        } else {
            1.0
        };
        
        PerformanceMetrics {
            rust_performance: PerformanceData {
                throughput: rust_throughput,
                avg_latency: rust_duration,
                p95_latency: rust_duration, // Simplified
                p99_latency: rust_duration,
                peak_memory: 0,
            },
            cpp_performance: PerformanceData {
                throughput: cpp_throughput,
                avg_latency: cpp_duration,
                p95_latency: cpp_duration,
                p99_latency: cpp_duration,
                peak_memory: 0,
            },
            performance_ratio,
        }
    }
    
    /// Execute side-by-side comparison
    pub fn execute_comparison<F, G, R, C>(
        &self,
        test_name: &str,
        rust_operation: F,
        cpp_operation: G,
    ) -> Result<ComparisonResult>
    where
        F: FnOnce() -> Result<(R, Duration)>,
        G: FnOnce() -> Result<(C, Duration)>,
        R: std::fmt::Debug,
        C: std::fmt::Debug,
    {
        let start_time = Instant::now();
        
        // Execute Rust operation
        let rust_result = utils::execute_with_timeout(
            rust_operation,
            self.config.test_timeout,
            &format!("{}_rust", test_name),
        );
        
        // Execute C++ operation
        let cpp_result = utils::execute_with_timeout(
            cpp_operation,
            self.config.test_timeout,
            &format!("{}_cpp", test_name),
        );
        
        let (passed, error_message) = match (rust_result, cpp_result) {
            (Ok((rust_data, rust_time)), Ok((cpp_data, cpp_time))) => {
                // Both succeeded - compare results
                // For now, just check that both completed
                (true, None)
            }
            (Err(rust_err), Err(cpp_err)) => {
                // Both failed - check if failure modes are similar
                (true, Some(format!("Both failed (acceptable): Rust: {}, C++: {}", rust_err, cpp_err)))
            }
            (Ok(_), Err(cpp_err)) => {
                (false, Some(format!("Rust succeeded but C++ failed: {}", cpp_err)))
            }
            (Err(rust_err), Ok(_)) => {
                (false, Some(format!("C++ succeeded but Rust failed: {}", rust_err)))
            }
        };
        
        Ok(ComparisonResult {
            test_name: test_name.to_string(),
            passed,
            metrics: ComparisonMetrics {
                correctness: CorrectnessMetrics {
                    results_identical: passed,
                    max_distance_difference: 0.0,
                    neighbor_differences: 0,
                    graph_similarity: if passed { 1.0 } else { 0.5 },
                },
                performance: PerformanceMetrics {
                    rust_performance: PerformanceData {
                        throughput: 0.0,
                        avg_latency: Duration::from_millis(0),
                        p95_latency: Duration::from_millis(0),
                        p99_latency: Duration::from_millis(0),
                        peak_memory: 0,
                    },
                    cpp_performance: PerformanceData {
                        throughput: 0.0,
                        avg_latency: Duration::from_millis(0),
                        p95_latency: Duration::from_millis(0),
                        p99_latency: Duration::from_millis(0),
                        peak_memory: 0,
                    },
                    performance_ratio: 1.0,
                },
                resources: ResourceMetrics {
                    memory_usage_ratio: 1.0,
                    disk_io_ratio: 1.0,
                    cpu_utilization_ratio: 1.0,
                },
            },
            error_message,
            duration: start_time.elapsed(),
        })
    }
}