# Benchmarks

This directory contains performance benchmarks for comparing kernel implementations.

## GEMM Benchmark

The `bench_gemm.py` script benchmarks the GEMM (General Matrix Multiplication) kernel, comparing the Triton implementation against PyTorch's native `torch.matmul`.

### Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- Triton

### Usage

Run the benchmark:

```bash
python benchmark/bench_gemm.py
```

This will:
1. Run a simple performance comparison for different matrix sizes
2. Benchmark different block size configurations
3. Generate a detailed performance plot (`gemm-square-performance.png`)

### Benchmark Components

#### Simple Performance Comparison

Compares PyTorch and Triton implementations for square matrices of various sizes (512, 1024, 2048, 4096), showing execution time and speedup.

#### Block Size Configuration

Tests different block size configurations to find the optimal settings:
- (64, 64, 16)
- (64, 64, 32)
- (128, 128, 32)
- (128, 128, 64)
- (256, 128, 32)
- (256, 256, 64)

#### Performance Plots

Generates detailed TFLOPS plots using `triton.testing.perf_report` for visual comparison across different matrix sizes.

### Benchmark Utilities

The benchmarks use Triton's testing utilities:

- `triton.testing.do_bench()`: Measures kernel execution time with proper warmup and multiple runs
- `triton.testing.perf_report()`: Decorator for generating performance comparison plots
- `triton.testing.Benchmark()`: Configuration for parameterized benchmarking

### Output

Example output:

```
GEMM Kernel Benchmark
================================================================================
Device: NVIDIA A100-SXM4-40GB
CUDA Version: 11.8
PyTorch Version: 2.0.0

================================================================================
Simple Performance Comparison
================================================================================

Size       PyTorch (ms)    Triton (ms)     Speedup   
-------------------------------------------------------
512        0.245           0.198           1.24x
1024       1.542           1.234           1.25x
2048       11.234          9.876           1.14x
4096       87.654          82.123          1.07x

================================================================================
Benchmarking different block size configurations
================================================================================
Block ( 64,  64, 16):  89.234 ms,  1.54 TFLOPS
Block ( 64,  64, 32):  85.123 ms,  1.61 TFLOPS
Block (128, 128, 32):  82.456 ms,  1.66 TFLOPS
Block (128, 128, 64):  81.234 ms,  1.69 TFLOPS
Block (256, 128, 32):  83.567 ms,  1.64 TFLOPS
Block (256, 256, 64):  80.123 ms,  1.71 TFLOPS

Best configuration: Block (256, 256, 64)
Performance: 80.123 ms, 1.71 TFLOPS
```

### Notes

- Benchmarks require a CUDA-capable GPU
- Performance may vary depending on GPU architecture and CUDA version
- The script uses `triton.testing.do_bench()` which includes proper warmup and statistical measurements
- TFLOPS (Tera Floating Point Operations Per Second) is calculated as: (2 * M * N * K) / (time_in_seconds * 10^12)
