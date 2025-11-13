# Triton Kernels

A collection of custom Triton kernels with PyTorch reference implementations and comprehensive unit tests.

## Overview

This project provides optimized Triton kernels for common GPU operations. Each kernel includes:
- A reference PyTorch implementation for correctness validation
- An optimized Triton kernel implementation
- Comprehensive unit tests

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/ring00/triton_kernels.git
cd triton_kernels

# Install with UV
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- CUDA-capable GPU (for Triton kernels)

## Available Kernels

### GEMM (General Matrix Multiplication)

Matrix multiplication kernel with both PyTorch reference and Triton implementations.

**Usage:**

```python
import torch
from triton_kernels import gemm_torch, gemm_triton

# Create input matrices
M, K, N = 256, 128, 256
a = torch.randn(M, K, dtype=torch.float32)
b = torch.randn(K, N, dtype=torch.float32)

# PyTorch reference implementation
c_torch = gemm_torch(a, b)

# Triton optimized implementation (requires CUDA)
a_gpu = a.to(device='cuda', dtype=torch.float16)
b_gpu = b.to(device='cuda', dtype=torch.float16)
c_triton = gemm_triton(a_gpu, b_gpu)
```

**Parameters:**
- `a`: Input tensor of shape (M, K)
- `b`: Input tensor of shape (K, N)
- `block_size_m`: Block size for M dimension (default: 128)
- `block_size_n`: Block size for N dimension (default: 128)
- `block_size_k`: Block size for K dimension (default: 32)

## Development

### Code Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting:

```bash
# Format code
ruff format .

# Run linter
ruff check .

# Fix linting issues automatically
ruff check --fix .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=triton_kernels --cov-report=html

# Run only CPU tests (no CUDA required)
pytest -m "not cuda"

# Run specific test file
pytest tests/test_gemm.py
```

### Running Examples

```bash
# Run the GEMM example
python examples/gemm_example.py
```

The example demonstrates:
- Basic usage of both PyTorch and Triton implementations
- Matrix multiplication with different sizes
- Testing different block size configurations (when CUDA is available)

### Project Structure

```
triton_kernels/
├── triton_kernels/          # Main package
│   ├── __init__.py
│   └── kernels/             # Kernel implementations
│       ├── __init__.py
│       └── gemm.py          # GEMM kernel
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_gemm.py        # GEMM tests
├── examples/                # Example scripts
│   └── gemm_example.py     # GEMM usage example
├── pyproject.toml          # Project configuration
├── setup.py                # Setup script
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass
2. Code is formatted with Ruff
3. New kernels include both PyTorch reference and Triton implementations
4. Comprehensive unit tests are provided

## License

MIT License - see LICENSE file for details