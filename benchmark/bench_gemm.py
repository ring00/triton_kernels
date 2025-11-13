"""
Performance benchmark for GEMM kernel implementations.

Compares the performance of Triton GEMM kernel against PyTorch's native matmul
using triton.testing utilities.
"""

import torch
import triton

from triton_kernels.kernels.gemm import gemm_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as x-axis
        x_vals=[128 * i for i in range(2, 33)],  # Different matrix sizes
        line_arg="provider",  # Argument name whose value corresponds to a different line
        line_vals=["triton", "torch"],  # Possible values for `line_arg`
        line_names=["Triton", "PyTorch"],  # Label names for the lines
        styles=[("blue", "-"), ("green", "-")],  # Line styles
        ylabel="TFLOPS",  # Label for y-axis
        plot_name="gemm-performance",  # Name for the plot
        args={},  # Values for function arguments not in `x_names` and `line_arg`
    )
)
def benchmark_square_gemm(M, N, K, provider):
    """
    Benchmark square matrix multiplication for different providers.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Number of columns in A / rows in B
        provider: Which implementation to use ('triton' or 'torch')
    """
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Quantiles for performance measurement
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm_triton(a, b), quantiles=quantiles
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Calculate TFLOPS: (2 * M * N * K) / (time_in_seconds * 10^12)
    def tflops(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return tflops(ms), tflops(max_ms), tflops(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="gemm-square-performance",
        args={},
    )
)
def benchmark_square_only(size, provider):
    """
    Benchmark square matrix multiplication (M=N=K).

    Args:
        size: Matrix dimension (size x size)
        provider: Which implementation to use ('triton' or 'torch')
    """
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm_triton(a, b), quantiles=quantiles
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    def tflops(ms):
        return 2 * size * size * size * 1e-12 / (ms * 1e-3)

    return tflops(ms), tflops(max_ms), tflops(min_ms)


def benchmark_block_sizes():
    """
    Benchmark different block size configurations for Triton GEMM.
    """
    print("\n" + "=" * 80)
    print("Benchmarking different block size configurations")
    print("=" * 80)

    M, N, K = 4096, 4096, 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    block_configs = [
        (64, 64, 16),
        (64, 64, 32),
        (128, 128, 32),
        (128, 128, 64),
        (256, 128, 32),
        (256, 256, 64),
    ]

    results = []
    for block_m, block_n, block_k in block_configs:

        def benchmark_fn(bm=block_m, bn=block_n, bk=block_k):
            return gemm_triton(a, b, block_size_m=bm, block_size_n=bn, block_size_k=bk)

        ms = triton.testing.do_bench(benchmark_fn)
        tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
        results.append((block_m, block_n, block_k, ms, tflops))
        print(
            f"Block ({block_m:3d}, {block_n:3d}, {block_k:2d}): "
            f"{ms:6.3f} ms, {tflops:6.2f} TFLOPS"
        )

    # Find best configuration
    best = max(results, key=lambda x: x[4])
    print(f"\nBest configuration: Block ({best[0]}, {best[1]}, {best[2]})")
    print(f"Performance: {best[3]:.3f} ms, {best[4]:.2f} TFLOPS")


def run_simple_benchmark():
    """
    Run a simple benchmark comparing PyTorch and Triton implementations.
    """
    print("\n" + "=" * 80)
    print("Simple Performance Comparison")
    print("=" * 80)

    sizes = [512, 1024, 2048, 4096]

    print(f"\n{'Size':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    for size in sizes:
        a = torch.randn((size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)

        # Benchmark PyTorch
        torch_ms = triton.testing.do_bench(lambda a=a, b=b: torch.matmul(a, b))

        # Benchmark Triton
        triton_ms = triton.testing.do_bench(lambda a=a, b=b: gemm_triton(a, b))

        speedup = torch_ms / triton_ms
        print(f"{size:<10} {torch_ms:<15.3f} {triton_ms:<15.3f} {speedup:<10.2f}x")


if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Benchmarks require a GPU.")
        exit(1)

    print("GEMM Kernel Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Run simple benchmark
    run_simple_benchmark()

    # Benchmark block sizes
    benchmark_block_sizes()

    # Generate performance plots
    print("\n" + "=" * 80)
    print("Generating detailed performance plots...")
    print("=" * 80)

    # Run square matrix benchmark
    benchmark_square_only.run(print_data=True, show_plots=False, save_path=".")

    print("\nBenchmark completed!")
    print("Performance plot saved as: gemm-square-performance.png")
