#!/bin/bash

# DiskANN Rust Benchmark Runner
# Runs comprehensive performance benchmarks on M2 ARM64

set -e

# Configuration
OUTPUT_DIR="examples/runs/macM2arm64"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAX_TIME=60  # Maximum time per benchmark in seconds
LOG_FILE="${OUTPUT_DIR}/benchmark_run_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run benchmark with timeout
run_benchmark() {
    local name="$1"
    local example="$2"
    local description="$3"
    
    print_status "Running $description..."
    
    local start_time=$(date +%s)
    local output_file="${OUTPUT_DIR}/${name}_${TIMESTAMP}.log"
    
    # Run with timeout (note: timeout command may not be available on macOS)
    if command -v timeout >/dev/null 2>&1; then
        if timeout ${MAX_TIME}s cargo run --release --example "$example" > "$output_file" 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            print_success "$description completed in ${duration}s"
            echo "Results saved to: $output_file"
        else
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                print_warning "$description timed out after ${MAX_TIME}s"
            else
                print_error "$description failed with exit code $exit_code"
            fi
        fi
    else
        # Fallback without timeout (macOS doesn't have timeout by default)
        if cargo run --release --example "$example" > "$output_file" 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            print_success "$description completed in ${duration}s"
            echo "Results saved to: $output_file"
        else
            print_error "$description failed"
        fi
    fi
    
    echo ""
}

# Function to get system info
get_system_info() {
    echo "=== System Information ===" | tee -a "$LOG_FILE"
    echo "Date: $(date)" | tee -a "$LOG_FILE"
    echo "Platform: $(uname -m)" | tee -a "$LOG_FILE"
    echo "OS: $(uname -s) $(uname -r)" | tee -a "$LOG_FILE"
    
    # macOS specific info
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v sysctl >/dev/null 2>&1; then
            echo "CPU: $(sysctl -n machdep.cpu.brand_string)" | tee -a "$LOG_FILE"
            echo "CPU Cores: $(sysctl -n hw.ncpu)" | tee -a "$LOG_FILE"
            echo "Memory: $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB" | tee -a "$LOG_FILE"
        fi
    fi
    
    # Rust info
    echo "Rust: $(rustc --version)" | tee -a "$LOG_FILE"
    echo "Cargo: $(cargo --version)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Main execution
main() {
    echo -e "${BLUE}======================================"
    echo -e "  DiskANN Rust Benchmark Suite"
    echo -e "  Platform: macOS M2 ARM64"
    echo -e "  Timestamp: $TIMESTAMP"
    echo -e "======================================${NC}"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "Cargo.toml" ]; then
        print_error "Please run this script from the DiskANN project root directory"
        exit 1
    fi
    
    # Check if Rust is available
    if ! command -v cargo >/dev/null 2>&1; then
        print_error "Cargo not found. Please install Rust and Cargo."
        exit 1
    fi
    
    # Create main log file
    get_system_info
    
    print_status "Building release version..."
    if cargo build --release --examples; then
        print_success "Build completed"
    else
        print_error "Build failed"
        exit 1
    fi
    echo ""
    
    # Run comprehensive benchmark suite
    run_benchmark "comprehensive" "benchmark_suite" "Comprehensive Benchmark Suite"
    
    # Run individual benchmarks
    print_status "Running individual benchmarks..."
    
    run_benchmark "simd" "simd_benchmark" "SIMD Distance Functions"
    
    run_benchmark "performance" "performance_benchmark" "General Performance Test"
    
    run_benchmark "memory_scaling" "memory_scaling" "Memory Scaling Analysis"
    
    run_benchmark "cold_disk" "cold_disk_benchmark" "Cold Disk Performance"
    
    run_benchmark "gpu_vs_cpu" "gpu_vs_cpu" "GPU vs CPU Comparison"
    
    run_benchmark "real_world" "real_world_datasets" "Real-world Dataset Tests"
    
    run_benchmark "pq_demo" "pq_demo" "Product Quantization Demo"
    
    run_benchmark "microsoft_api" "microsoft_api_example" "Microsoft API Compatibility"
    
    # Generate summary
    echo -e "${BLUE}======================================"
    echo -e "  Benchmark Summary"
    echo -e "======================================${NC}"
    
    echo "All benchmark results saved in: $OUTPUT_DIR"
    echo "Files created:"
    ls -la "$OUTPUT_DIR"/*"$TIMESTAMP"* 2>/dev/null || echo "No output files found"
    
    echo ""
    echo "To analyze results:"
    echo "  cat $OUTPUT_DIR/*$TIMESTAMP*.log"
    echo ""
    echo "To compare with previous runs:"
    echo "  ls $OUTPUT_DIR/"
    
    print_success "Benchmark suite completed!"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -t, --timeout  Set timeout per benchmark (default: 60s)"
    echo "  -o, --output   Set output directory (default: examples/runs/macM2arm64)"
    echo "  --list         List available benchmarks without running"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all benchmarks with default settings"
    echo "  $0 -t 120            # Run with 2-minute timeout per benchmark"
    echo "  $0 --output /tmp/bench  # Save results to /tmp/bench"
}

# Function to list available benchmarks
list_benchmarks() {
    echo "Available benchmarks:"
    echo "  benchmark_suite      - Comprehensive benchmark suite"
    echo "  simd_benchmark      - SIMD distance function performance"
    echo "  performance_benchmark - General performance tests"
    echo "  memory_scaling      - Memory usage scaling analysis"
    echo "  cold_disk_benchmark - Cold disk I/O performance"
    echo "  gpu_vs_cpu         - GPU vs CPU performance comparison"
    echo "  real_world_datasets - Real-world dataset performance"
    echo "  pq_demo            - Product quantization demonstration"
    echo "  microsoft_api_example - Microsoft API compatibility test"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--timeout)
            MAX_TIME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --list)
            list_benchmarks
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main