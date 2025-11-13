"""
Example demonstrating GEMM kernel usage.

This script shows how to use both PyTorch reference and Triton implementations
of the GEMM kernel.
"""

import torch

from triton_kernels import gemm_torch, gemm_triton


def main():
    print("GEMM Kernel Example")
    print("=" * 50)

    # Matrix dimensions
    M, K, N = 256, 128, 256
    print(f"\nMatrix dimensions: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")

    # Create random input matrices
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)

    # PyTorch reference implementation
    print("\n1. PyTorch Reference Implementation:")
    c_torch = gemm_torch(a, b)
    print(f"   Output shape: {c_torch.shape}")
    print(f"   Output dtype: {c_torch.dtype}")

    # Verify correctness
    expected = torch.matmul(a, b)
    is_correct = torch.allclose(c_torch, expected)
    print(f"   Matches torch.matmul: {is_correct}")

    # Triton implementation (requires CUDA)
    if torch.cuda.is_available():
        print("\n2. Triton Optimized Implementation:")
        a_gpu = a.to(device="cuda", dtype=torch.float16)
        b_gpu = b.to(device="cuda", dtype=torch.float16)

        c_triton = gemm_triton(a_gpu, b_gpu)
        print(f"   Output shape: {c_triton.shape}")
        print(f"   Output dtype: {c_triton.dtype}")

        # Verify correctness (allow for FP16 precision differences)
        c_torch_gpu = c_torch.to(device="cuda", dtype=torch.float16)
        is_correct = torch.allclose(c_triton, c_torch_gpu, rtol=1e-2, atol=1e-2)
        print(f"   Matches PyTorch result: {is_correct}")

        # Test with different block sizes
        print("\n3. Testing different block sizes:")
        block_configs = [
            (64, 64, 16),
            (128, 128, 32),
            (256, 256, 64),
        ]
        for block_m, block_n, block_k in block_configs:
            c_triton = gemm_triton(
                a_gpu,
                b_gpu,
                block_size_m=block_m,
                block_size_n=block_n,
                block_size_k=block_k,
            )
            is_correct = torch.allclose(c_triton, c_torch_gpu, rtol=1e-2, atol=1e-2)
            print(
                f"   Block size ({block_m}, {block_n}, {block_k}): {'✓' if is_correct else '✗'}"
            )
    else:
        print("\n2. Triton Optimized Implementation:")
        print("   CUDA not available - Triton kernel skipped")

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
