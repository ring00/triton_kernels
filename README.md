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

### Packed Operations (Merge and Split)

Packed merge and split operations for handling variable-length sequences, commonly used in multi-modal models.

**Usage:**

```python
import torch
from triton_kernels import (
    packed_merge_torch,
    packed_merge_triton,
    packed_split_torch,
    packed_split_triton,
)

# Create input tensors
vid = torch.randn(10, 64, dtype=torch.float32)
txt = torch.randn(6, 64, dtype=torch.float32)
vid_lengths = [4, 3, 3]
txt_lengths = [2, 2, 2]

# PyTorch reference implementations
merged = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)
vid_split, txt_split = packed_split_torch(merged, vid_lengths, txt_lengths)

# Triton optimized implementations (requires CUDA)
vid_gpu = vid.to(device='cuda', dtype=torch.float16)
txt_gpu = txt.to(device='cuda', dtype=torch.float16)
merged_triton = packed_merge_triton(vid_gpu, txt_gpu, vid_lengths, txt_lengths)
vid_triton, txt_triton = packed_split_triton(merged_triton, vid_lengths, txt_lengths)
```

**Parameters for packed_merge:**
- `vid`: Video tensor of shape (total_vid_length, h)
- `txt`: Text tensor of shape (total_txt_length, h) or None
- `vid_lengths`: List of lengths for video segments
- `txt_lengths`: List of lengths for text segments or None
- `block_size_h`: Block size for hidden dimension (Triton only, default: 128)

**Parameters for packed_split:**
- `x`: Input tensor of shape (total_length, h)
- `vid_lengths`: List of lengths for video segments
- `txt_lengths`: List of lengths for text segments or None
- `vid_padding`: Padding size for video output (default: 0)
- `txt_padding`: Padding size for text output (default: 0)
- `block_size_h`: Block size for hidden dimension (Triton only, default: 128)

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

# Run the packed operations example
python examples/packed_ops_example.py
```

The examples demonstrate:
- Basic usage of both PyTorch and Triton implementations
- Matrix multiplication with different sizes (GEMM)
- Packed merge and split operations for variable-length sequences
- Testing different block size configurations (when CUDA is available)

### Running Benchmarks

```bash
# Run the GEMM benchmark (requires CUDA)
python benchmark/bench_gemm.py

# Run the packed operations benchmark (requires CUDA)
python benchmark/bench_packed_ops.py
```

The benchmarks:
- Compare performance of Triton vs PyTorch implementations
- Test different configurations and tensor sizes
- Generate performance plots and speedup metrics
- Require a CUDA-capable GPU

See [benchmark/README.md](benchmark/README.md) for detailed information.

### Project Structure

```
triton_kernels/
├── src/triton_kernels/      # Main package
│   ├── __init__.py
│   ├── gemm.py              # GEMM kernel
│   └── packed_ops.py        # Packed merge/split operations
├── tests/                   # Unit tests
│   ├── test_gemm.py         # GEMM tests
│   └── test_packed_ops.py   # Packed operations tests
├── examples/                # Example scripts
│   ├── gemm_example.py      # GEMM usage example
│   └── packed_ops_example.py # Packed operations example
├── benchmark/               # Performance benchmarks
│   ├── bench_gemm.py        # GEMM benchmark
│   └── bench_packed_ops.py  # Packed operations benchmark
├── pyproject.toml           # Project configuration
├── setup.py                 # Setup script
└── README.md                # This file
```

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass
2. Code is formatted with Ruff
3. New kernels include both PyTorch reference and Triton implementations
4. Comprehensive unit tests are provided

## License

MIT License - see LICENSE file for details