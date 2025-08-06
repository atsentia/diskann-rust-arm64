#!/bin/bash
# Comprehensive DiskANN Rust vs C++ Parity Testing Runner
# This script executes the complete test suite for validating Rust implementation
# against the C++ reference implementation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Function to display help
show_help() {
    cat << EOF
DiskANN Comprehensive Parity Testing

Usage: $0 [options]

Options:
    --tier1-only          Run only Tier 1 foundational tests (fast)
    --no-perf            Skip performance tests (Tier 3)
    --no-stress          Skip stress tests
    --cpp-path PATH      Path to C++ DiskANN installation
    --timeout SECONDS    Timeout for individual tests (default: 300)
    --output-dir DIR     Directory for test results (default: test_results)
    --verbose            Enable verbose output
    --help               Show this help message

Test Tiers:
    Tier 1: Foundational Parity & Correctness Analysis (Critical)
    Tier 2: Advanced Capabilities & Robustness Testing  
    Tier 3: Granular Performance & Efficiency Benchmarking

Environment Variables:
    CPP_DISKANN_PATH     Path to C++ DiskANN reference implementation
    RUN_PERF_TESTS       Set to 1 to enable performance tests
    RUN_STRESS_TESTS     Set to 1 to enable stress tests
    RUST_LOG             Set logging level (e.g., debug, info)

Examples:
    $0                           # Run all tests
    $0 --tier1-only             # Quick validation
    $0 --no-stress --verbose    # Skip stress tests with verbose output
    CPP_DISKANN_PATH=/custom/path $0  # Use custom C++ installation

EOF
}

# Parse command line arguments
TIER1_ONLY=false
NO_PERF=false
NO_STRESS=false
CPP_PATH=""
TIMEOUT=300
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier1-only)
            TIER1_ONLY=true
            shift
            ;;
        --no-perf)
            NO_PERF=true
            shift
            ;;
        --no-stress)
            NO_STRESS=true
            shift
            ;;
        --cpp-path)
            CPP_PATH="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --output-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Setup environment
setup_environment() {
    log_step "Setting up test environment"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Set C++ path if provided
    if [ -n "$CPP_PATH" ]; then
        export CPP_DISKANN_PATH="$CPP_PATH"
    fi
    
    # Set test flags based on options
    if [ "$NO_PERF" = false ] && [ "$TIER1_ONLY" = false ]; then
        export RUN_PERF_TESTS=1
    fi
    
    if [ "$NO_STRESS" = false ] && [ "$TIER1_ONLY" = false ]; then
        export RUN_STRESS_TESTS=1
    fi
    
    # Set logging level
    if [ "$VERBOSE" = true ]; then
        export RUST_LOG=debug
    else
        export RUST_LOG=info
    fi
    
    log_info "Environment configured:"
    log_info "  - Results directory: $RESULTS_DIR"
    log_info "  - C++ DiskANN path: ${CPP_DISKANN_PATH:-default}"
    log_info "  - Performance tests: ${RUN_PERF_TESTS:-disabled}"
    log_info "  - Stress tests: ${RUN_STRESS_TESTS:-disabled}"
    log_info "  - Test timeout: ${TIMEOUT}s"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites"
    
    # Check if Rust project builds
    cd "$PROJECT_ROOT"
    if ! cargo check --quiet; then
        log_error "Rust project does not compile. Please fix build errors first."
        exit 1
    fi
    
    # Check C++ reference if performance tests are enabled
    if [ "${RUN_PERF_TESTS:-}" = "1" ]; then
        if [ -z "${CPP_DISKANN_PATH:-}" ]; then
            log_warning "C++ DiskANN path not set. Performance comparisons will be limited."
            log_info "Run: ./scripts/setup_cpp_reference.sh to set up C++ reference"
        elif [ ! -d "${CPP_DISKANN_PATH}" ]; then
            log_error "C++ DiskANN path does not exist: ${CPP_DISKANN_PATH}"
            log_info "Run: ./scripts/setup_cpp_reference.sh to set up C++ reference"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Run the comprehensive test suite
run_tests() {
    log_step "Running comprehensive parity tests"
    
    cd "$PROJECT_ROOT"
    
    local test_args=""
    local test_name="comprehensive_parity_tests"
    
    if [ "$TIER1_ONLY" = true ]; then
        test_name="comprehensive_parity_tests::test_tier1_foundational"
        log_info "Running Tier 1 foundational tests only"
    else
        log_info "Running complete test suite (all tiers)"
    fi
    
    # Run tests with timeout and capture output
    local test_output_file="$RESULTS_DIR/test_output_$TIMESTAMP.log"
    local test_results_file="$RESULTS_DIR/test_results_$TIMESTAMP.json"
    
    log_info "Test output will be saved to: $test_output_file"
    
    if timeout "${TIMEOUT}s" cargo test --release --test "$test_name" -- --nocapture 2>&1 | tee "$test_output_file"; then
        log_success "Tests completed successfully"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_error "Tests timed out after ${TIMEOUT}s"
        else
            log_error "Tests failed with exit code $exit_code"
        fi
        return $exit_code
    fi
}

# Run specific test categories
run_api_tests() {
    log_step "Running API parameter parity tests"
    
    cd "$PROJECT_ROOT"
    timeout 60s cargo test --release --test comprehensive_parity_tests::test_api_parameter_parity -- --nocapture
}

run_distance_tests() {
    log_step "Running distance metric precision tests"
    
    cd "$PROJECT_ROOT"
    timeout 60s cargo test --release --test comprehensive_parity_tests::test_distance_precision -- --nocapture
}

run_performance_tests() {
    if [ "${RUN_PERF_TESTS:-}" != "1" ]; then
        log_info "Performance tests disabled (use --enable-perf or set RUN_PERF_TESTS=1)"
        return 0
    fi
    
    log_step "Running performance benchmark tests"
    
    cd "$PROJECT_ROOT"
    timeout 300s cargo test --release --test comprehensive_parity_tests::test_performance_benchmark -- --nocapture
}

run_stress_tests() {
    if [ "${RUN_STRESS_TESTS:-}" != "1" ]; then
        log_info "Stress tests disabled (use --enable-stress or set RUN_STRESS_TESTS=1)"
        return 0
    fi
    
    log_step "Running concurrent stress tests"
    
    cd "$PROJECT_ROOT"
    timeout 180s cargo test --release --test comprehensive_parity_tests::test_concurrent_stress -- --nocapture
}

# Generate test report
generate_report() {
    log_step "Generating test report"
    
    local report_file="$RESULTS_DIR/parity_report_$TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# DiskANN Rust vs C++ Parity Test Report

**Generated**: $(date)
**Test Suite**: Comprehensive Parity Testing
**Configuration**: 
- Tier 1 Only: $TIER1_ONLY
- Performance Tests: ${RUN_PERF_TESTS:-disabled}
- Stress Tests: ${RUN_STRESS_TESTS:-disabled}
- C++ Reference: ${CPP_DISKANN_PATH:-not configured}

## Test Results

EOF

    # Extract test results from output if available
    local latest_output=$(ls -t "$RESULTS_DIR"/test_output_*.log 2>/dev/null | head -1)
    if [ -n "$latest_output" ] && [ -f "$latest_output" ]; then
        echo "### Test Output Summary" >> "$report_file"
        echo '```' >> "$report_file"
        tail -20 "$latest_output" >> "$report_file"
        echo '```' >> "$report_file"
        echo >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF
## Summary

This report shows the results of comprehensive parity testing between the Rust DiskANN implementation and the Microsoft C++ reference implementation.

### Test Tiers

1. **Tier 1: Foundational Parity** - Critical correctness tests that must pass
2. **Tier 2: Robustness Testing** - Edge cases and advanced features  
3. **Tier 3: Performance Benchmarking** - Performance comparison and optimization

### Next Steps

Based on the test results above:
- Address any Tier 1 failures immediately (blocking issues)
- Review Tier 2 failures for robustness improvements
- Use Tier 3 results to guide performance optimization

EOF

    log_success "Report generated: $report_file"
}

# Display final summary
display_summary() {
    echo
    echo "🎯 Comprehensive Parity Testing Summary"
    echo "======================================="
    echo "Test Run ID: $TIMESTAMP"
    echo "Results Directory: $RESULTS_DIR"
    echo
    
    # Check if tests passed by looking at the most recent output
    local latest_output=$(ls -t "$RESULTS_DIR"/test_output_*.log 2>/dev/null | head -1)
    if [ -n "$latest_output" ] && [ -f "$latest_output" ]; then
        if grep -q "test result: ok" "$latest_output"; then
            echo "✅ Overall Status: PASSED"
        elif grep -q "test result: FAILED" "$latest_output"; then
            echo "❌ Overall Status: FAILED"
        else
            echo "⚠️  Overall Status: INCOMPLETE"
        fi
        echo
        
        # Show test counts if available
        if grep -q "test result:" "$latest_output"; then
            echo "Test Results:"
            grep "test result:" "$latest_output" | tail -1
            echo
        fi
    fi
    
    echo "Available artifacts:"
    echo "  - Test output: $(ls -t "$RESULTS_DIR"/test_output_*.log 2>/dev/null | head -1 | xargs basename)"
    echo "  - Test report: $(ls -t "$RESULTS_DIR"/parity_report_*.md 2>/dev/null | head -1 | xargs basename)"
    echo
    echo "To run individual test categories:"
    echo "  cargo test --test comprehensive_parity_tests::test_tier1_foundational"
    echo "  cargo test --test comprehensive_parity_tests::test_api_parameter_parity"
    echo "  cargo test --test comprehensive_parity_tests::test_distance_precision"
    echo
}

# Main execution
main() {
    echo "🚀 DiskANN Comprehensive Parity Testing"
    echo "========================================"
    echo
    
    setup_environment
    check_prerequisites
    
    local overall_status=0
    
    if [ "$TIER1_ONLY" = true ]; then
        # Quick validation mode
        log_info "🏃 Quick Validation Mode - Tier 1 Tests Only"
        if ! run_tests; then
            overall_status=1
        fi
    else
        # Full test suite
        log_info "🔬 Full Test Suite Mode - All Tiers"
        
        # Run main comprehensive tests
        if ! run_tests; then
            overall_status=1
        fi
        
        # Run additional specific tests if main tests passed
        if [ $overall_status -eq 0 ]; then
            run_api_tests || true
            run_distance_tests || true
            run_performance_tests || true
            run_stress_tests || true
        fi
    fi
    
    generate_report
    display_summary
    
    if [ $overall_status -eq 0 ]; then
        log_success "All tests completed successfully! 🎉"
    else
        log_error "Some tests failed. Please review the results."
    fi
    
    exit $overall_status
}

# Run main function
main "$@"