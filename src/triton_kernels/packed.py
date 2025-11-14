"""
Packed tensor operations for merging and splitting tensor segments.

This module provides utilities for working with packed tensors,
where multiple segments of varying lengths are concatenated together.
"""

import torch


def packed_merge(
    vid: torch.Tensor,
    txt: torch.Tensor | None,
    vid_lengths: list[int],
    txt_lengths: list[int] | None,
) -> torch.Tensor:
    """
    Merge two tensors based on the provided lengths.

    Interleaves segments from vid and txt tensors according to
    their respective length lists.

    Args:
        vid: Query tensor for the first segment of shape (s, h)
        txt: Query tensor for the second segment of shape (s', h).
            Can be None if no text segments are present.
        vid_lengths: List of lengths for the video segments
        txt_lengths: List of lengths for the text segments.
            Can be None if no text segments are present.

    Returns:
        Merged tensor containing interleaved segments from both inputs
    """
    segments = []
    vid_offset = 0
    txt_offset = 0

    if txt is not None and txt_lengths:
        for vid_length, txt_length in zip(vid_lengths, txt_lengths, strict=True):
            if vid_length > 0:
                segments.append(vid[vid_offset : vid_offset + vid_length])
                vid_offset += vid_length
            if txt_length > 0:
                segments.append(txt[txt_offset : txt_offset + txt_length])
                txt_offset += txt_length
    else:
        for vid_length in vid_lengths:
            if vid_length > 0:
                segments.append(vid[vid_offset : vid_offset + vid_length])
                vid_offset += vid_length

    return torch.cat(segments, dim=0)


def packed_split(
    x: torch.Tensor,
    vid_lengths: list[int],
    txt_lengths: list[int] | None,
    vid_padding: int = 0,
    txt_padding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Split the input tensor into two segments based on the provided lengths.

    Separates an interleaved tensor into two separate tensors (vid and txt)
    according to the provided length lists.

    Args:
        x: Input tensor of shape (s, h)
        vid_lengths: List of lengths for the first segment
        txt_lengths: List of lengths for the second segment.
            Can be None if no text segments are present.
        vid_padding: Padding size for the first segment (default: 0)
        txt_padding: Padding size for the second segment (default: 0)

    Returns:
        Tuple of (vid_tensor, txt_tensor) where txt_tensor is None if
        txt_lengths was None
    """
    vid_segments = []
    txt_segments = []
    offset = 0
    if txt_lengths:
        for vid_length, txt_length in zip(vid_lengths, txt_lengths, strict=True):
            if vid_length > 0:
                segment_h = x[offset : offset + vid_length]
                offset += vid_length
                vid_segments.append(segment_h)
            if txt_length > 0:
                segment_e = x[offset : offset + txt_length]
                offset += txt_length
                txt_segments.append(segment_e)
    else:
        for vid_length in vid_lengths:
            if vid_length > 0:
                segment_h = x[offset : offset + vid_length]
                offset += vid_length
                vid_segments.append(segment_h)

    if vid_padding > 0:
        vid_segments.append(
            torch.zeros(vid_padding, *x.shape[1:], dtype=x.dtype, device=x.device)
        )
    if txt_padding > 0:
        txt_segments.append(
            torch.zeros(txt_padding, *x.shape[1:], dtype=x.dtype, device=x.device)
        )

    return (
        torch.cat(vid_segments, dim=0),
        torch.cat(txt_segments, dim=0) if txt_segments else None,
    )
