//! Reporting Module
//! 
//! This module generates comprehensive reports from parity test results.

use super::*;

/// Generate a comprehensive HTML report from test results
pub fn generate_comprehensive_report(results: &[ComparisonResult]) -> Result<String> {
    let mut report = String::new();
    
    report.push_str("# DiskANN Rust vs C++ Comprehensive Parity Report\n\n");
    report.push_str(&format!("**Generated**: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    report.push_str(&format!("**Total Tests**: {}\n\n", results.len()));
    
    // Executive Summary
    let passed_tests = results.iter().filter(|r| r.passed).count();
    let failed_tests = results.len() - passed_tests;
    let success_rate = passed_tests as f64 / results.len() as f64;
    
    report.push_str("## Executive Summary\n\n");
    report.push_str(&format!("- **Total Tests**: {}\n", results.len()));
    report.push_str(&format!("- **Passed**: {} ({:.1}%)\n", passed_tests, success_rate * 100.0));
    report.push_str(&format!("- **Failed**: {} ({:.1}%)\n", failed_tests, (1.0 - success_rate) * 100.0));
    
    if success_rate >= 0.95 {
        report.push_str("- **Overall Status**: ✅ **EXCELLENT** - Implementation ready for production\n");
    } else if success_rate >= 0.85 {
        report.push_str("- **Overall Status**: ⚠️ **GOOD** - Minor issues to address\n");
    } else if success_rate >= 0.70 {
        report.push_str("- **Overall Status**: ⚠️ **MODERATE** - Several issues need attention\n");
    } else {
        report.push_str("- **Overall Status**: ❌ **POOR** - Major issues require resolution\n");
    }
    
    report.push_str("\n");
    
    // Tier-by-tier breakdown
    report.push_str("## Results by Test Tier\n\n");
    
    for tier_prefix in &["tier1_", "tier2_", "tier3_"] {
        let tier_results: Vec<_> = results.iter()
            .filter(|r| r.test_name.starts_with(tier_prefix))
            .collect();
            
        if !tier_results.is_empty() {
            let tier_name = match *tier_prefix {
                "tier1_" => "Tier 1: Foundational Parity",
                "tier2_" => "Tier 2: Robustness Testing",
                "tier3_" => "Tier 3: Performance Benchmarking",
                _ => "Other Tests",
            };
            
            let tier_passed = tier_results.iter().filter(|r| r.passed).count();
            let tier_total = tier_results.len();
            let tier_rate = tier_passed as f64 / tier_total as f64;
            
            report.push_str(&format!("### {}\n", tier_name));
            report.push_str(&format!("- **Tests**: {}/{} passed ({:.1}%)\n", tier_passed, tier_total, tier_rate * 100.0));
            
            if *tier_prefix == "tier1_" && tier_rate < 1.0 {
                report.push_str("- **⚠️ CRITICAL**: Tier 1 failures indicate fundamental implementation issues\n");
            }
            
            report.push_str("\n");
        }
    }
    
    // Detailed test results
    report.push_str("## Detailed Test Results\n\n");
    
    for result in results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        report.push_str(&format!("### {} - {}\n", status, result.test_name));
        report.push_str(&format!("- **Duration**: {:?}\n", result.duration));
        
        if let Some(error) = &result.error_message {
            report.push_str(&format!("- **Error**: {}\n", error));
        }
        
        // Add performance metrics if available
        let perf = &result.metrics.performance;
        if perf.rust_performance.throughput > 0.0 {
            report.push_str(&format!("- **Rust Throughput**: {:.2} ops/sec\n", perf.rust_performance.throughput));
        }
        if perf.rust_performance.avg_latency > Duration::from_millis(0) {
            report.push_str(&format!("- **Rust Latency**: {:?}\n", perf.rust_performance.avg_latency));
        }
        if perf.performance_ratio != 1.0 {
            report.push_str(&format!("- **Performance Ratio**: {:.2}x\n", perf.performance_ratio));
        }
        
        report.push_str("\n");
    }
    
    // Recommendations
    report.push_str("## Recommendations\n\n");
    
    let failed_results: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    
    if failed_results.is_empty() {
        report.push_str("🎉 **All tests passed!** The Rust implementation shows excellent parity with the C++ reference.\n\n");
        report.push_str("**Next Steps:**\n");
        report.push_str("- Deploy to production environments\n");
        report.push_str("- Monitor performance in real-world workloads\n");
        report.push_str("- Consider additional optimization opportunities\n");
    } else {
        report.push_str("**Priority Actions:**\n\n");
        
        // Categorize failures
        let tier1_failures: Vec<_> = failed_results.iter()
            .filter(|r| r.test_name.starts_with("tier1_"))
            .collect();
            
        if !tier1_failures.is_empty() {
            report.push_str("🚨 **CRITICAL - Tier 1 Failures:**\n");
            for failure in tier1_failures {
                report.push_str(&format!("- Fix `{}`: {}\n", 
                    failure.test_name,
                    failure.error_message.as_ref().unwrap_or(&"Unknown error".to_string())
                ));
            }
            report.push_str("\n");
        }
        
        let tier2_failures: Vec<_> = failed_results.iter()
            .filter(|r| r.test_name.starts_with("tier2_"))
            .collect();
            
        if !tier2_failures.is_empty() {
            report.push_str("⚠️ **HIGH PRIORITY - Tier 2 Failures:**\n");
            for failure in tier2_failures {
                report.push_str(&format!("- Address `{}`: {}\n", 
                    failure.test_name,
                    failure.error_message.as_ref().unwrap_or(&"Unknown error".to_string())
                ));
            }
            report.push_str("\n");
        }
        
        let tier3_failures: Vec<_> = failed_results.iter()
            .filter(|r| r.test_name.starts_with("tier3_"))
            .collect();
            
        if !tier3_failures.is_empty() {
            report.push_str("📊 **OPTIMIZATION - Tier 3 Failures:**\n");
            for failure in tier3_failures {
                report.push_str(&format!("- Optimize `{}`: {}\n", 
                    failure.test_name,
                    failure.error_message.as_ref().unwrap_or(&"Unknown error".to_string())
                ));
            }
            report.push_str("\n");
        }
    }
    
    // Performance summary
    let performance_ratios: Vec<f64> = results.iter()
        .map(|r| r.metrics.performance.performance_ratio)
        .filter(|&ratio| ratio != 1.0 && ratio > 0.0)
        .collect();
        
    if !performance_ratios.is_empty() {
        let avg_ratio = performance_ratios.iter().sum::<f64>() / performance_ratios.len() as f64;
        
        report.push_str("## Performance Summary\n\n");
        report.push_str(&format!("- **Average Performance Ratio**: {:.2}x vs C++\n", avg_ratio));
        
        if avg_ratio >= 0.9 {
            report.push_str("- **Performance Status**: ✅ Excellent performance parity\n");
        } else if avg_ratio >= 0.75 {
            report.push_str("- **Performance Status**: ⚠️ Good performance with room for optimization\n");
        } else {
            report.push_str("- **Performance Status**: ❌ Performance gaps need attention\n");
        }
        
        report.push_str("\n");
    }
    
    // Footer
    report.push_str("---\n");
    report.push_str("*Report generated by DiskANN Comprehensive Parity Testing Framework*\n");
    
    Ok(report)
}

/// Generate a simple text summary
pub fn generate_summary(results: &[ComparisonResult]) -> String {
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let rate = passed as f64 / total as f64;
    
    format!(
        "Parity Test Summary: {}/{} passed ({:.1}%) - {}",
        passed,
        total,
        rate * 100.0,
        if rate >= 0.95 { "EXCELLENT" } else if rate >= 0.85 { "GOOD" } else if rate >= 0.70 { "MODERATE" } else { "POOR" }
    )
}