# Copilot Agent Instructions for triton_kernels

## Repository Overview

**triton_kernels** is a Python library providing custom Triton GPU kernels with PyTorch reference implementations for matrix operations.

- **Type**: Python package (~10 source files)
- **Language**: Python 3.10+ (DO NOT support < 3.10)
- **Frameworks**: PyTorch 2.7+, Triton 3.0+, CUDA (GPU)
- **Build**: pip/uv with pyproject.toml
- **Testing**: pytest with pytest-cov
- **Linting**: ruff

## Installation (2-3 minutes)

**ALWAYS use editable installation**:

```bash
# Recommended: UV
pip install uv && uv pip install -e ".[dev]"

# Alternative: pip
pip install -e ".[dev]"
```

**Critical**: Always use `[dev]` extras for pytest, pytest-cov, and ruff.

## Validation Workflow (Run in This Order)

1. **Format**: `ruff format .` (must run before other checks)
2. **Lint**: `ruff check .` or `ruff check --fix .`
3. **Test**: `pytest` or `pytest -v`
   - CPU only (no GPU): `pytest -m "not cuda"` (5 tests pass, ~1-2s)
   - With coverage: `pytest --cov=triton_kernels --cov-report=term-missing`
4. **Examples** (optional): `python examples/gemm_example.py`

**Common Issues**:
- `pytest: command not found` → Run `pip install -e ".[dev]"`
- "CUDA not available" → EXPECTED without GPU. CPU tests (5) pass, CUDA tests (11) skip.
- Import errors → Use editable install: `pip install -e ".[dev]"`

## Project Structure

```
triton_kernels/
├── src/triton_kernels/       # Main package
│   ├── __init__.py           # Exports: gemm_torch, gemm_triton
│   └── gemm.py               # GEMM implementations
├── tests/test_gemm.py        # 16 tests (5 CPU, 11 CUDA)
├── examples/gemm_example.py  # Usage demo
├── benchmark/bench_gemm.py   # Performance benchmarks
├── pyproject.toml            # All config (deps, ruff, pytest)
├── setup.py                  # Minimal (defers to pyproject.toml)
└── README.md                 # Documentation
```

**Key Files**:
- **pyproject.toml**: ALL project config (dependencies, ruff, pytest settings)
- **src/triton_kernels/gemm.py**: PyTorch reference + Triton kernel (float16 output, block size params)
- **tests/test_gemm.py**: 3 classes (TestGEMMTorch, TestGEMMTriton, TestGEMMComparison)

**Pattern**: Each kernel has `<name>_torch` (PyTorch reference) and `<name>_triton` (optimized, CUDA-only)

## Code Standards

**Ruff** (configured in pyproject.toml): 88 char line length, Python 3.10+, double quotes, spaces
- Format: `ruff format .` (ALWAYS run before committing)
- Lint: `ruff check .` or `ruff check --fix .`
- Both must pass with no errors

**Testing**:
- Use pytest with descriptive test names: `Test<Component>`, `test_<description>`
- Mark CUDA tests: `@pytest.mark.skipif(not torch.cuda.is_available())`
- Triton kernels match PyTorch within rtol=1e-2, atol=1e-2

**Docstrings**: Google/NumPy style for modules, classes, functions

## Dependencies (from pyproject.toml)

**Core**: torch ≥2.7.0, triton ≥3.0.0, numpy ≥1.24.4, matplotlib ≥3.10.7, pandas ≥2.3.3  
**Dev**: pytest ≥7.0.0, pytest-cov ≥4.0.0, ruff ≥0.1.0  
**Note**: CUDA libs (nvidia-*) auto-installed with torch/triton. CPU tests work without GPU.

## Pre-Commit Checklist (No CI/CD exists)

1. `ruff format .` → "8 files left unchanged"
2. `ruff check .` → "All checks passed!"
3. `pytest -m "not cuda"` → "5 passed, 11 skipped"
4. Optional: `pytest` (all tests), `python examples/gemm_example.py`

## Making Changes

### Add New Kernel
1. Create `src/triton_kernels/<kernel>.py` with `<kernel>_torch` and `<kernel>_triton`
2. Export in `__init__.py`: `from .<kernel> import ...` + update `__all__`
3. Create `tests/test_<kernel>.py` with `Test<Kernel>Torch` and `Test<Kernel>Triton` classes
4. Run: `ruff format . && ruff check . && pytest`
5. Optional: Add `examples/<kernel>_example.py` and `benchmark/bench_<kernel>.py`

### Fix Bugs
1. Write failing test → 2. Fix code → 3. `pytest tests/test_<file>.py -v` → 4. `ruff format . && ruff check .` → 5. `pytest`

### Update Dependencies
Edit `pyproject.toml` → Reinstall `pip install -e ".[dev]"` → (UV: `uv lock`) → `pytest`

## Key Implementation Details

**pyproject.toml**: DO NOT modify `requires-python` (≥3.10) or `build-backend` (uv_build)

**gemm.py**: Triton kernel uses `@triton.jit`, meta-params (BLOCK_SIZE_M/N/K), outputs float16, validates contiguity/shape

**test_gemm.py**: 16 tests across 3 classes, CUDA tests auto-skip without GPU, uses `torch.allclose(rtol=1e-2, atol=1e-2)`

## Final Note

These instructions are validated and complete. Only search if: (1) instructions incomplete for your task, (2) encounter undocumented error, (3) need code details not here. **This repo is small (~10 files, most <200 lines) - prefer reading source files directly.**
