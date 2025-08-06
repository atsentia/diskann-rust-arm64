#!/bin/bash
# Setup script for C++ DiskANN reference implementation
# This script clones, builds, and prepares the Microsoft DiskANN C++ implementation
# for comparison with the Rust implementation.

set -e

# Configuration
CPP_DIR="/tmp/diskann_cpp_reference"
DISKANN_REPO="https://github.com/microsoft/DiskANN.git"
DISKANN_COMMIT="main" # Use latest main branch by default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for required tools
    local required_tools=("git" "cmake" "make" "g++")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and try again."
        exit 1
    fi
    
    # Check for sufficient disk space (at least 1GB)
    local available_space=$(df /tmp | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 1048576 ]; then # 1GB in KB
        log_warning "Low disk space in /tmp. DiskANN build may fail."
    fi
    
    log_success "System requirements check passed"
}

# Function to clone the DiskANN repository
clone_diskann() {
    log_info "Cloning Microsoft DiskANN repository..."
    
    if [ -d "$CPP_DIR" ]; then
        log_warning "Directory $CPP_DIR already exists"
        read -p "Remove existing directory and re-clone? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$CPP_DIR"
        else
            log_info "Using existing directory"
            return 0
        fi
    fi
    
    git clone "$DISKANN_REPO" "$CPP_DIR"
    cd "$CPP_DIR"
    
    # Checkout specific commit if requested
    if [ "$DISKANN_COMMIT" != "main" ]; then
        log_info "Checking out commit: $DISKANN_COMMIT"
        git checkout "$DISKANN_COMMIT"
    fi
    
    log_success "DiskANN repository cloned successfully"
}

# Function to build DiskANN
build_diskann() {
    log_info "Building DiskANN C++ implementation..."
    
    cd "$CPP_DIR"
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with CMake
    log_info "Configuring with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          -DUNIT_TEST=ON \
          ..
    
    # Build
    log_info "Compiling (this may take several minutes)..."
    local num_cores=$(nproc 2>/dev/null || echo 4)
    make -j"$num_cores"
    
    log_success "DiskANN built successfully"
}

# Function to verify the build
verify_build() {
    log_info "Verifying DiskANN build..."
    
    cd "$CPP_DIR/build"
    
    # List of expected executables
    local expected_binaries=(
        "apps/build_memory_index"
        "apps/search_memory_index"
        "apps/build_disk_index"
        "apps/search_disk_index"
        "apps/build_stitched_index"
        "apps/search_stitched_index"
    )
    
    local missing_binaries=()
    
    for binary in "${expected_binaries[@]}"; do
        if [ ! -f "$binary" ]; then
            missing_binaries+=("$binary")
        fi
    done
    
    if [ ${#missing_binaries[@]} -ne 0 ]; then
        log_error "Missing expected binaries:"
        for binary in "${missing_binaries[@]}"; do
            echo "  - $binary"
        done
        return 1
    fi
    
    # Test one of the binaries
    log_info "Testing build_memory_index binary..."
    if ! ./apps/build_memory_index --help &>/dev/null; then
        log_warning "build_memory_index binary may not be working correctly"
        return 1
    fi
    
    log_success "Build verification passed"
}

# Function to create test data for comparison
create_test_data() {
    log_info "Creating test datasets for comparison..."
    
    local test_data_dir="$CPP_DIR/test_data"
    mkdir -p "$test_data_dir"
    
    # Create a small test dataset using the data generator
    cd "$CPP_DIR/build"
    
    # Check if we have the random data generator
    if [ -f "tests/generate_random_data" ]; then
        log_info "Generating random test data..."
        ./tests/generate_random_data --data_type float --output_file "$test_data_dir/random_1k_128d.fvecs" --num_points 1000 --dimension 128
        ./tests/generate_random_data --data_type float --output_file "$test_data_dir/random_queries_100_128d.fvecs" --num_points 100 --dimension 128
        log_success "Test data created"
    else
        log_warning "Random data generator not found, skipping test data creation"
    fi
}

# Function to create wrapper scripts for easy access
create_wrappers() {
    log_info "Creating wrapper scripts..."
    
    local wrapper_dir="$SCRIPT_DIR/cpp_wrappers"
    mkdir -p "$wrapper_dir"
    
    # Create wrapper for build_memory_index
    cat > "$wrapper_dir/cpp_build_memory_index.sh" << EOF
#!/bin/bash
# Wrapper for C++ DiskANN build_memory_index
cd "$CPP_DIR/build"
./apps/build_memory_index "\$@"
EOF
    
    # Create wrapper for search_memory_index
    cat > "$wrapper_dir/cpp_search_memory_index.sh" << EOF
#!/bin/bash
# Wrapper for C++ DiskANN search_memory_index
cd "$CPP_DIR/build"
./apps/search_memory_index "\$@"
EOF
    
    # Create wrapper for build_disk_index
    cat > "$wrapper_dir/cpp_build_disk_index.sh" << EOF
#!/bin/bash
# Wrapper for C++ DiskANN build_disk_index
cd "$CPP_DIR/build"
./apps/build_disk_index "\$@"
EOF
    
    # Create wrapper for search_disk_index
    cat > "$wrapper_dir/cpp_search_disk_index.sh" << EOF
#!/bin/bash
# Wrapper for C++ DiskANN search_disk_index
cd "$CPP_DIR/build"
./apps/search_disk_index "\$@"
EOF
    
    # Make all wrappers executable
    chmod +x "$wrapper_dir"/*.sh
    
    log_success "Wrapper scripts created in $wrapper_dir"
}

# Function to run basic smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    cd "$CPP_DIR/build"
    
    # Test help output for key binaries
    local test_binaries=("apps/build_memory_index" "apps/search_memory_index")
    
    for binary in "${test_binaries[@]}"; do
        log_info "Testing $binary..."
        if ! timeout 10s ./"$binary" --help &>/dev/null; then
            log_warning "$binary may not be working correctly"
        else
            log_success "$binary is working"
        fi
    done
}

# Function to display setup summary
display_summary() {
    log_success "C++ DiskANN reference setup completed successfully!"
    echo
    echo "Setup Summary:"
    echo "  - C++ DiskANN location: $CPP_DIR"
    echo "  - Build directory: $CPP_DIR/build"
    echo "  - Wrapper scripts: $SCRIPT_DIR/cpp_wrappers/"
    echo "  - Test data: $CPP_DIR/test_data/ (if created)"
    echo
    echo "Available executables:"
    cd "$CPP_DIR/build"
    find apps/ -type f -executable | head -10
    echo
    echo "To run comprehensive parity tests:"
    echo "  cd $SCRIPT_DIR"
    echo "  cargo test --test comprehensive_parity --release"
    echo
    echo "Environment variables for testing:"
    echo "  export CPP_DISKANN_PATH=\"$CPP_DIR\""
    echo "  export RUN_PERF_TESTS=1  # Enable performance tests"
    echo "  export RUN_STRESS_TESTS=1  # Enable stress tests"
}

# Function to handle cleanup on error
cleanup_on_error() {
    log_error "Setup failed. Cleaning up..."
    if [ -d "$CPP_DIR" ] && [ "$1" != "keep" ]; then
        read -p "Remove incomplete installation at $CPP_DIR? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$CPP_DIR"
        fi
    fi
}

# Main execution
main() {
    echo "🚀 DiskANN C++ Reference Setup"
    echo "=============================="
    echo
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --commit)
                DISKANN_COMMIT="$2"
                shift 2
                ;;
            --dir)
                CPP_DIR="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --commit COMMIT  Use specific git commit (default: main)"
                echo "  --dir DIR        Install to specific directory (default: /tmp/diskann_cpp_reference)"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Set up error handling
    trap 'cleanup_on_error' ERR
    
    # Execute setup steps
    check_requirements
    clone_diskann
    build_diskann
    verify_build
    create_test_data
    create_wrappers
    run_smoke_tests
    
    # Success!
    display_summary
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi