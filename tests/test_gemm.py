"""
Comprehensive unit tests for GEMM kernel implementations.
"""

import pytest
import torch

from triton_kernels.gemm import gemm_torch, gemm_triton


class TestGEMMTorch:
    """Test cases for PyTorch reference GEMM implementation."""

    def test_basic_multiplication(self):
        """Test basic matrix multiplication."""
        M, K, N = 64, 32, 48
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)

        c = gemm_torch(a, b)

        assert c.shape == (M, N)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected)

    def test_square_matrices(self):
        """Test multiplication of square matrices."""
        N = 128
        a = torch.randn(N, N, dtype=torch.float32)
        b = torch.randn(N, N, dtype=torch.float32)

        c = gemm_torch(a, b)

        assert c.shape == (N, N)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected)

    def test_small_matrices(self):
        """Test with very small matrices."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        c = gemm_torch(a, b)

        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert torch.allclose(c, expected)

    def test_identity_matrix(self):
        """Test multiplication with identity matrix."""
        N = 64
        a = torch.randn(N, N, dtype=torch.float32)
        identity = torch.eye(N, dtype=torch.float32)

        c = gemm_torch(a, identity)

        assert torch.allclose(c, a)

    def test_zero_matrix(self):
        """Test multiplication with zero matrix."""
        M, K, N = 32, 64, 48
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.zeros(K, N, dtype=torch.float32)

        c = gemm_torch(a, b)

        assert torch.allclose(c, torch.zeros(M, N))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGEMMTriton:
    """Test cases for Triton GEMM implementation."""

    def test_basic_multiplication(self):
        """Test basic matrix multiplication with Triton."""
        M, K, N = 128, 64, 96
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        c = gemm_triton(a, b)

        assert c.shape == (M, N)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    def test_square_matrices(self):
        """Test multiplication of square matrices with Triton."""
        N = 256
        a = torch.randn(N, N, dtype=torch.float16, device="cuda")
        b = torch.randn(N, N, dtype=torch.float16, device="cuda")

        c = gemm_triton(a, b)

        assert c.shape == (N, N)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    def test_non_square_matrices(self):
        """Test with non-square matrices."""
        M, K, N = 512, 256, 128
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        c = gemm_triton(a, b)

        assert c.shape == (M, N)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    def test_different_block_sizes(self):
        """Test with different block size configurations."""
        M, K, N = 256, 128, 256
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        # Test different block size configurations
        block_configs = [
            (64, 64, 16),
            (128, 128, 32),
            (256, 256, 64),
        ]

        for block_m, block_n, block_k in block_configs:
            c = gemm_triton(
                a, b, block_size_m=block_m, block_size_n=block_n, block_size_k=block_k
            )
            assert c.shape == (M, N)
            expected = torch.matmul(a, b)
            assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    def test_small_matrices(self):
        """Test with small matrices."""
        M, K, N = 32, 32, 32
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        c = gemm_triton(a, b)

        assert c.shape == (M, N)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    def test_large_matrices(self):
        """Test with large matrices."""
        M, K, N = 1024, 512, 1024
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        c = gemm_triton(a, b)

        assert c.shape == (M, N)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    def test_identity_matrix(self):
        """Test multiplication with identity matrix."""
        N = 256
        a = torch.randn(N, N, dtype=torch.float16, device="cuda")
        identity = torch.eye(N, dtype=torch.float16, device="cuda")

        c = gemm_triton(a, identity)

        assert torch.allclose(c, a, rtol=1e-2, atol=1e-2)

    def test_zero_matrix(self):
        """Test multiplication with zero matrix."""
        M, K, N = 128, 256, 128
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.zeros(K, N, dtype=torch.float16, device="cuda")

        c = gemm_triton(a, b)

        assert torch.allclose(
            c,
            torch.zeros(M, N, dtype=torch.float16, device="cuda"),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_incompatible_dimensions(self):
        """Test that incompatible dimensions raise an error."""
        M, K, N = 64, 32, 48
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(N, N, dtype=torch.float16, device="cuda")  # Wrong shape

        with pytest.raises(AssertionError, match="Incompatible dimensions"):
            gemm_triton(a, b)

    def test_non_contiguous_input(self):
        """Test that non-contiguous inputs raise an error."""
        M, K, N = 64, 32, 48
        a = torch.randn(K, M, dtype=torch.float16, device="cuda").T  # Non-contiguous
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        with pytest.raises(AssertionError, match="Matrix A must be contiguous"):
            gemm_triton(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGEMMComparison:
    """Test cases comparing PyTorch and Triton implementations."""

    def test_torch_vs_triton(self):
        """Test that Triton and PyTorch implementations produce similar results."""
        M, K, N = 256, 128, 256
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)

        # PyTorch result
        c_torch = gemm_torch(a, b)

        # Triton result
        a_gpu = a.to(device="cuda", dtype=torch.float16)
        b_gpu = b.to(device="cuda", dtype=torch.float16)
        c_triton = gemm_triton(a_gpu, b_gpu)

        # Compare (allowing for float16 precision differences)
        c_torch_gpu = c_torch.to(device="cuda", dtype=torch.float16)
        assert torch.allclose(c_triton, c_torch_gpu, rtol=1e-2, atol=1e-2)
