"""
Modulation operations for tensor segments.

This module provides various modulation operations that apply
scale, shift, gate, or tanh transformations to segments of tensors
based on provided lengths and indices.
"""

import torch


def scale_shift_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Apply scale and shift modulation to tensor segments.

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


def scale_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Apply scale modulation to tensor segments.

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


def gate_modulate(
    x: torch.Tensor,
    gate: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Apply gate modulation to tensor segments.

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


def tanh_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    lengths: list[int],
    indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Apply tanh-scaled modulation to tensor segments.

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
