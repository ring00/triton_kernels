"""
Custom Triton kernels with PyTorch reference implementations.

This package provides optimized Triton kernels for common operations,
each with a reference PyTorch implementation for correctness validation.
"""

from triton_kernels.kernels.gemm import gemm_torch, gemm_triton

__version__ = "0.1.0"
__all__ = ["gemm_torch", "gemm_triton"]
