"""
Modulation operations for tensor segments.

This module provides:
1. Reference PyTorch implementations for correctness validation
2. Optimized Triton kernel implementations
"""

import torch
import triton
import triton.language as tl


def scale_shift_modulate_torch(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of scale-shift modulation.

    Performs: y[segment] = x[segment] * (1 + scale[i]) + shift[i]
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate
        scale: Scale values for each segment
        shift: Shift values for each segment
        lengths: List of segment lengths
        indices: Optional list of indices into scale/shift tensors.
                If None, uses range(len(lengths))

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    scale = 1 + scale

    y = torch.empty_like(x)
    offset = 0
    for i, length in zip(indices, lengths, strict=False):
        if length > 0:
            y[offset : offset + length] = (
                x[offset : offset + length] * scale[i] + shift[i]
            )
            offset += length

    y[offset:] = 0

    return y


@triton.jit
def _scale_shift_modulate_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    lengths_ptr,
    indices_ptr,
    num_segments,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for scale-shift modulation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Find which segment each element belongs to
    cumsum = 0
    for seg_idx in range(num_segments):
        idx = tl.load(indices_ptr + seg_idx)
        length = tl.load(lengths_ptr + seg_idx)

        if length > 0:
            mask = (offsets >= cumsum) & (offsets < cumsum + length)
            scale_val = tl.load(scale_ptr + idx)
            shift_val = tl.load(shift_ptr + idx)

            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y_vals = x_vals * (1.0 + scale_val) + shift_val
            tl.store(y_ptr + offsets, y_vals, mask=mask)

            cumsum += length

    # Zero out remaining elements
    total_length = tl.load(lengths_ptr + num_segments - 1)
    for i in range(num_segments - 1):
        total_length += tl.load(lengths_ptr + i)

    mask = offsets >= total_length
    tl.store(y_ptr + offsets, 0.0, mask=mask)


def scale_shift_modulate_triton(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Triton implementation of scale-shift modulation.

    Performs: y[segment] = x[segment] * (1 + scale[i]) + shift[i]
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate (must be CUDA tensor)
        scale: Scale values for each segment
        shift: Shift values for each segment
        lengths: List of segment lengths
        indices: Optional list of indices into scale/shift tensors.
                If None, uses range(len(lengths))
        block_size: Block size for the kernel (default: 256)

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    # Convert lists to tensors
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=x.device)
    indices_tensor = torch.tensor(indices, dtype=torch.int32, device=x.device)

    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = (triton.cdiv(n_elements, block_size),)
    _scale_shift_modulate_kernel[grid](
        x,
        scale,
        shift,
        y,
        lengths_tensor,
        indices_tensor,
        len(lengths),
        BLOCK_SIZE=block_size,
    )

    return y


def scale_modulate_torch(
    x: torch.Tensor,
    scale: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of scale modulation.

    Performs: y[segment] = x[segment] * (1 + scale[i])
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate
        scale: Scale values for each segment
        lengths: List of segment lengths
        indices: Optional list of indices into scale tensor.
                If None, uses range(len(lengths))

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    scale = 1 + scale

    y = torch.empty_like(x)
    offset = 0
    for i, length in zip(indices, lengths, strict=False):
        if length > 0:
            y[offset : offset + length] = x[offset : offset + length] * scale[i]
            offset += length

    y[offset:] = 0

    return y


@triton.jit
def _scale_modulate_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    lengths_ptr,
    indices_ptr,
    num_segments,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for scale modulation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    cumsum = 0
    for seg_idx in range(num_segments):
        idx = tl.load(indices_ptr + seg_idx)
        length = tl.load(lengths_ptr + seg_idx)

        if length > 0:
            mask = (offsets >= cumsum) & (offsets < cumsum + length)
            scale_val = tl.load(scale_ptr + idx)

            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y_vals = x_vals * (1.0 + scale_val)
            tl.store(y_ptr + offsets, y_vals, mask=mask)

            cumsum += length

    total_length = tl.load(lengths_ptr + num_segments - 1)
    for i in range(num_segments - 1):
        total_length += tl.load(lengths_ptr + i)

    mask = offsets >= total_length
    tl.store(y_ptr + offsets, 0.0, mask=mask)


def scale_modulate_triton(
    x: torch.Tensor,
    scale: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Triton implementation of scale modulation.

    Performs: y[segment] = x[segment] * (1 + scale[i])
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate (must be CUDA tensor)
        scale: Scale values for each segment
        lengths: List of segment lengths
        indices: Optional list of indices into scale tensor.
                If None, uses range(len(lengths))
        block_size: Block size for the kernel (default: 256)

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=x.device)
    indices_tensor = torch.tensor(indices, dtype=torch.int32, device=x.device)

    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = (triton.cdiv(n_elements, block_size),)
    _scale_modulate_kernel[grid](
        x,
        scale,
        lengths_tensor,
        indices_tensor,
        len(lengths),
        BLOCK_SIZE=block_size,
    )

    return y


def gate_modulate_torch(
    x: torch.Tensor,
    gate: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of gate modulation.

    Performs: y[segment] = x[segment] * gate[i]
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate
        gate: Gate values for each segment
        lengths: List of segment lengths
        indices: Optional list of indices into gate tensor.
                If None, uses range(len(lengths))

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    y = torch.empty_like(x)
    offset = 0
    for i, length in zip(indices, lengths, strict=False):
        if length > 0:
            y[offset : offset + length] = x[offset : offset + length] * gate[i]
            offset += length

    y[offset:] = 0

    return y


@triton.jit
def _gate_modulate_kernel(
    x_ptr,
    gate_ptr,
    y_ptr,
    lengths_ptr,
    indices_ptr,
    num_segments,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for gate modulation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    cumsum = 0
    for seg_idx in range(num_segments):
        idx = tl.load(indices_ptr + seg_idx)
        length = tl.load(lengths_ptr + seg_idx)

        if length > 0:
            mask = (offsets >= cumsum) & (offsets < cumsum + length)
            gate_val = tl.load(gate_ptr + idx)

            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y_vals = x_vals * gate_val
            tl.store(y_ptr + offsets, y_vals, mask=mask)

            cumsum += length

    total_length = tl.load(lengths_ptr + num_segments - 1)
    for i in range(num_segments - 1):
        total_length += tl.load(lengths_ptr + i)

    mask = offsets >= total_length
    tl.store(y_ptr + offsets, 0.0, mask=mask)


def gate_modulate_triton(
    x: torch.Tensor,
    gate: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Triton implementation of gate modulation.

    Performs: y[segment] = x[segment] * gate[i]
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate (must be CUDA tensor)
        gate: Gate values for each segment
        lengths: List of segment lengths
        indices: Optional list of indices into gate tensor.
                If None, uses range(len(lengths))
        block_size: Block size for the kernel (default: 256)

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=x.device)
    indices_tensor = torch.tensor(indices, dtype=torch.int32, device=x.device)

    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = (triton.cdiv(n_elements, block_size),)
    _gate_modulate_kernel[grid](
        x,
        gate,
        y,
        lengths_tensor,
        indices_tensor,
        len(lengths),
        BLOCK_SIZE=block_size,
    )

    return y


def tanh_modulate_torch(
    x: torch.Tensor,
    scale: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of tanh-scaled modulation.

    Performs: y[segment] = x[segment] * tanh(scale[i])
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate
        scale: Scale values for each segment (will be passed through tanh)
        lengths: List of segment lengths
        indices: Optional list of indices into scale tensor.
                If None, uses range(len(lengths))

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    scale = scale.tanh()

    y = torch.empty_like(x)
    offset = 0
    for i, length in zip(indices, lengths, strict=False):
        if length > 0:
            y[offset : offset + length] = x[offset : offset + length] * scale[i]
            offset += length

    y[offset:] = 0

    return y


@triton.jit
def _tanh_modulate_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    lengths_ptr,
    indices_ptr,
    num_segments,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for tanh modulation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    cumsum = 0
    for seg_idx in range(num_segments):
        idx = tl.load(indices_ptr + seg_idx)
        length = tl.load(lengths_ptr + seg_idx)

        if length > 0:
            mask = (offsets >= cumsum) & (offsets < cumsum + length)
            scale_val = tl.load(scale_ptr + idx)
            tanh_scale = tl.libdevice.tanh(scale_val)

            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y_vals = x_vals * tanh_scale
            tl.store(y_ptr + offsets, y_vals, mask=mask)

            cumsum += length

    total_length = tl.load(lengths_ptr + num_segments - 1)
    for i in range(num_segments - 1):
        total_length += tl.load(lengths_ptr + i)

    mask = offsets >= total_length
    tl.store(y_ptr + offsets, 0.0, mask=mask)


def tanh_modulate_triton(
    x: torch.Tensor,
    scale: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Triton implementation of tanh-scaled modulation.

    Performs: y[segment] = x[segment] * tanh(scale[i])
    for each segment defined by lengths and indices.

    Args:
        x: Input tensor to modulate (must be CUDA tensor)
        scale: Scale values for each segment (will be passed through tanh)
        lengths: List of segment lengths
        indices: Optional list of indices into scale tensor.
                If None, uses range(len(lengths))
        block_size: Block size for the kernel (default: 256)

    Returns:
        Modulated tensor with same shape as x
    """
    if not indices:
        indices = list(range(len(lengths)))

    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=x.device)
    indices_tensor = torch.tensor(indices, dtype=torch.int32, device=x.device)

    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = (triton.cdiv(n_elements, block_size),)
    _tanh_modulate_kernel[grid](
        x,
        scale,
        y,
        lengths_tensor,
        indices_tensor,
        len(lengths),
        BLOCK_SIZE=block_size,
    )

    return y
