# DiskANN GPU Acceleration Examples

This directory contains examples demonstrating GPU acceleration features in DiskANN-RS.

## Examples

### 1. GPU vs CPU Performance Comparison (`gpu_vs_cpu.rs`)

Demonstrates the performance difference between GPU and CPU for various batch sizes.

```bash
cargo run --release --features "cuda" --example gpu_vs_cpu
```

**Key insights:**
- Shows performance crossover points
- Demonstrates when to use GPU vs CPU
- Includes throughput measurements

### 2. Multi-GPU Usage (`multi_gpu.rs`)

Shows how to utilize multiple GPUs for parallel batch processing.

```bash
cargo run --release --features "cuda" --example multi_gpu
```

**Features:**
- Parallel processing across multiple GPUs
- Thread-based GPU distribution
- Result consistency verification

### 3. Platform-Specific GPU Optimization (`platform_specific_gpu.rs`)

Platform-specific examples for optimal GPU usage.

```bash
# macOS with Metal
cargo run --release --features "metal" --example platform_specific_gpu

# Windows with CUDA or DirectML
cargo run --release --features "cuda,directml" --example platform_specific_gpu

# Linux with CUDA or ROCm
cargo run --release --features "cuda,rocm" --example platform_specific_gpu
```

**Platforms covered:**
- Apple Metal with Neural Engine detection
- NVIDIA CUDA on Windows/Linux
- AMD ROCm on Linux
- Qualcomm Snapdragon X on Windows ARM64

### 4. Batch Size Optimization (`batch_optimization.rs`)

Interactive tool to find optimal batch sizes for your hardware.

```bash
cargo run --release --features "all-gpu" --example batch_optimization
```

**Features:**
- Tests multiple batch sizes
- Measures throughput and latency
- Provides memory usage estimates
- Recommends optimal configurations

## Building Examples

### Basic CPU-only build
```bash
cargo build --release --examples
```

### With specific GPU support
```bash
# NVIDIA CUDA
cargo build --release --features "cuda" --examples

# Apple Metal
cargo build --release --features "metal" --examples

# Cross-platform WebGPU
cargo build --release --features "webgpu" --examples

# All GPU backends
cargo build --release --features "all-gpu" --examples
```

## Performance Guidelines

Based on our benchmarks:

1. **Batch Size < 32**: CPU SIMD is optimal
   - Lower latency
   - No GPU transfer overhead
   - Efficient cache utilization

2. **Batch Size 32-256**: Transition zone
   - Performance depends on dimension
   - Test both CPU and GPU

3. **Batch Size > 256**: GPU is optimal
   - 10-100x speedup possible
   - Massive parallelism benefits
   - Worth the transfer overhead

4. **Dimension Impact**:
   - Larger dimensions (>512) benefit more from GPU
   - Smaller dimensions (<128) may favor CPU

## Hardware Requirements

### NVIDIA CUDA
- CUDA 11.0 or later
- Compute capability 6.0+
- Linux or Windows

### Apple Metal
- macOS 10.13+
- Any Mac with Metal support
- M-series chips have Neural Engine

### AMD ROCm
- ROCm 5.0+
- Linux only
- GFX900+ GPUs

### WebGPU
- Any modern GPU
- Cross-platform
- Requires WebGPU-capable drivers

## Troubleshooting

### GPU not detected
```bash
# Check available features
cargo run --example gpu_vs_cpu 2>&1 | grep "Using"

# Enable debug logging
RUST_LOG=debug cargo run --example gpu_vs_cpu
```

### Performance issues
1. Ensure release build (`--release`)
2. Check batch size is appropriate
3. Verify GPU drivers are up to date
4. Monitor GPU utilization

### Memory errors
- Reduce batch size
- Check available GPU memory
- Use smaller dimensions for testing