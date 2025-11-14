"""
Performance benchmark for packed merge and split operations.

This benchmark compares PyTorch and Triton implementations across
different tensor sizes and batch configurations.
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from triton_kernels import (
    packed_merge_torch,
    packed_merge_triton,
    packed_split_torch,
    packed_split_triton,
)


def benchmark_merge(
    vid, txt, vid_lengths, txt_lengths, n_iters=100, warmup=10, use_triton=True
):
    """Benchmark merge operation."""
    if use_triton:
        func = packed_merge_triton
    else:
        func = packed_merge_torch

    # Warmup
    for _ in range(warmup):
        _ = func(vid, txt, vid_lengths, txt_lengths)

    if vid.is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = func(vid, txt, vid_lengths, txt_lengths)
    if vid.is_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms


def benchmark_split(
    x, vid_lengths, txt_lengths, n_iters=100, warmup=10, use_triton=True
):
    """Benchmark split operation."""
    if use_triton:
        func = packed_split_triton
    else:
        func = packed_split_torch

    # Warmup
    for _ in range(warmup):
        _ = func(x, vid_lengths, txt_lengths)

    if x.is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = func(x, vid_lengths, txt_lengths)
    if x.is_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms


def run_benchmarks():
    """Run benchmarks for different configurations."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return

    print("=" * 80)
    print("Packed Operations Benchmark")
    print("=" * 80)

    device = "cuda"
    dtype = torch.float16
    n_iters = 100
    warmup = 10

    # Test configurations: (batch_size, avg_vid_len, avg_txt_len, hidden_dim)
    configs = [
        (4, 32, 16, 64),
        (8, 64, 32, 128),
        (16, 128, 64, 256),
        (32, 256, 128, 512),
        (64, 512, 256, 1024),
    ]

    results = []

    for batch_size, avg_vid_len, avg_txt_len, h in configs:
        print(
            f"\nConfig: batch={batch_size}, vid_len={avg_vid_len}, "
            f"txt_len={avg_txt_len}, h={h}"
        )

        # Generate varying lengths around average
        vid_lengths = [
            max(1, avg_vid_len + (i % 3 - 1) * avg_vid_len // 4)
            for i in range(batch_size)
        ]
        txt_lengths = [
            max(1, avg_txt_len + (i % 3 - 1) * avg_txt_len // 4)
            for i in range(batch_size)
        ]

        total_vid = sum(vid_lengths)
        total_txt = sum(txt_lengths)
        total_merged = total_vid + total_txt

        # Create tensors
        vid = torch.randn(total_vid, h, dtype=dtype, device=device)
        txt = torch.randn(total_txt, h, dtype=dtype, device=device)
        merged = torch.randn(total_merged, h, dtype=dtype, device=device)

        # Benchmark merge
        try:
            time_merge_pytorch = benchmark_merge(
                vid.cpu().to(torch.float32),
                txt.cpu().to(torch.float32),
                vid_lengths,
                txt_lengths,
                n_iters=n_iters,
                warmup=warmup,
                use_triton=False,
            )
            time_merge_triton = benchmark_merge(
                vid,
                txt,
                vid_lengths,
                txt_lengths,
                n_iters=n_iters,
                warmup=warmup,
                use_triton=True,
            )
            speedup_merge = time_merge_pytorch / time_merge_triton

            print(
                f"  Merge - PyTorch: {time_merge_pytorch:.4f} ms, "
                f"Triton: {time_merge_triton:.4f} ms, "
                f"Speedup: {speedup_merge:.2f}x"
            )
        except Exception as e:
            print(f"  Merge benchmark failed: {e}")
            time_merge_pytorch = time_merge_triton = speedup_merge = 0

        # Benchmark split
        try:
            time_split_pytorch = benchmark_split(
                merged.cpu().to(torch.float32),
                vid_lengths,
                txt_lengths,
                n_iters=n_iters,
                warmup=warmup,
                use_triton=False,
            )
            time_split_triton = benchmark_split(
                merged,
                vid_lengths,
                txt_lengths,
                n_iters=n_iters,
                warmup=warmup,
                use_triton=True,
            )
            speedup_split = time_split_pytorch / time_split_triton

            print(
                f"  Split - PyTorch: {time_split_pytorch:.4f} ms, "
                f"Triton: {time_split_triton:.4f} ms, "
                f"Speedup: {speedup_split:.2f}x"
            )
        except Exception as e:
            print(f"  Split benchmark failed: {e}")
            time_split_pytorch = time_split_triton = speedup_split = 0

        results.append(
            {
                "batch_size": batch_size,
                "avg_vid_len": avg_vid_len,
                "avg_txt_len": avg_txt_len,
                "hidden_dim": h,
                "total_elements": total_merged * h,
                "merge_pytorch_ms": time_merge_pytorch,
                "merge_triton_ms": time_merge_triton,
                "merge_speedup": speedup_merge,
                "split_pytorch_ms": time_split_pytorch,
                "split_triton_ms": time_split_triton,
                "split_speedup": speedup_split,
            }
        )

    # Create results dataframe
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print(df.to_string(index=False))

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Packed Operations Performance Comparison", fontsize=16)

    # Plot 1: Merge time comparison
    ax = axes[0, 0]
    x = range(len(df))
    ax.plot(x, df["merge_pytorch_ms"], marker="o", label="PyTorch", linewidth=2)
    ax.plot(x, df["merge_triton_ms"], marker="s", label="Triton", linewidth=2)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Merge Operation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{r['batch_size']}" for _, r in df.iterrows()], rotation=45)

    # Plot 2: Split time comparison
    ax = axes[0, 1]
    ax.plot(x, df["split_pytorch_ms"], marker="o", label="PyTorch", linewidth=2)
    ax.plot(x, df["split_triton_ms"], marker="s", label="Triton", linewidth=2)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Split Operation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{r['batch_size']}" for _, r in df.iterrows()], rotation=45)

    # Plot 3: Merge speedup
    ax = axes[1, 0]
    ax.bar(x, df["merge_speedup"], alpha=0.7, color="green")
    ax.axhline(y=1.0, color="r", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Speedup")
    ax.set_title("Merge Operation Speedup (Triton vs PyTorch)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{r['batch_size']}" for _, r in df.iterrows()], rotation=45)

    # Plot 4: Split speedup
    ax = axes[1, 1]
    ax.bar(x, df["split_speedup"], alpha=0.7, color="blue")
    ax.axhline(y=1.0, color="r", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Speedup")
    ax.set_title("Split Operation Speedup (Triton vs PyTorch)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{r['batch_size']}" for _, r in df.iterrows()], rotation=45)

    plt.tight_layout()
    plt.savefig("packed_ops_benchmark.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'packed_ops_benchmark.png'")

    print("\n" + "=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
