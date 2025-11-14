"""
Packed merge and split operations.

This module provides:
1. Reference PyTorch implementations for correctness validation
2. Optimized Triton kernel implementations

These operations are useful for merging and splitting packed sequences,
commonly used in multi-modal models where different modalities (e.g., video and text)
need to be processed together or separately.
"""

import torch
import triton
import triton.language as tl


def packed_merge_torch(
    vid: torch.Tensor,
    txt: torch.Tensor | None,
    vid_lengths: list[int],
    txt_lengths: list[int] | None,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of packed merge operation.

    Merges two tensors based on the provided lengths, interleaving segments
    from both tensors according to the length lists.

    Args:
        vid: Query tensor for the first segment of shape (s, h).
        txt: Query tensor for the second segment of shape (s', h) or None.
        vid_lengths: List of lengths for the video segments.
        txt_lengths: List of lengths for the text segments or None.

    Returns:
        Merged tensor of shape (total_length, h) where total_length is the
        sum of all segment lengths.
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


def packed_split_torch(
    x: torch.Tensor,
    vid_lengths: list[int],
    txt_lengths: list[int] | None,
    vid_padding: int = 0,
    txt_padding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reference PyTorch implementation of packed split operation.

    Splits the input tensor into two segments based on the provided lengths,
    with optional padding for each segment.

    Args:
        x: Input tensor of shape (s, h).
        vid_lengths: List of lengths for the first segment.
        txt_lengths: List of lengths for the second segment or None.
        vid_padding: Padding size for the first segment.
        txt_padding: Padding size for the second segment.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: Two tensors corresponding
        to the split segments. The second tensor is None if txt_lengths is None.
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


@triton.jit
def _packed_merge_kernel(
    # Input pointers
    vid_ptr,
    txt_ptr,
    out_ptr,
    # Offset arrays
    vid_offsets_ptr,
    txt_offsets_ptr,
    out_offsets_ptr,
    # Lengths arrays
    vid_lengths_ptr,
    txt_lengths_ptr,
    # Dimensions
    n_segments: tl.constexpr,
    h: tl.constexpr,
    has_txt: tl.constexpr,
    # Block size
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Triton kernel for packed merge operation.

    Each program processes one segment at a time, copying data from
    the appropriate input tensor to the output tensor.
    """
    pid = tl.program_id(axis=0)

    # Determine which segment this program is processing
    segment_idx = pid // 2 if has_txt else pid
    is_txt_segment = (pid % 2 == 1) if has_txt else False

    if segment_idx >= n_segments:
        return

    # Load the length for this segment
    if is_txt_segment:
        length = tl.load(txt_lengths_ptr + segment_idx)
        in_offset = tl.load(txt_offsets_ptr + segment_idx)
        in_ptr = txt_ptr
    else:
        length = tl.load(vid_lengths_ptr + segment_idx)
        in_offset = tl.load(vid_offsets_ptr + segment_idx)
        in_ptr = vid_ptr

    if length <= 0:
        return

    out_offset = tl.load(out_offsets_ptr + pid)

    # Process each row
    for row in range(length):
        # Load from input
        h_offsets = tl.arange(0, BLOCK_SIZE_H)
        mask = h_offsets < h

        in_addr = in_ptr + (in_offset + row) * h + h_offsets
        out_addr = out_ptr + (out_offset + row) * h + h_offsets

        data = tl.load(in_addr, mask=mask, other=0.0)
        tl.store(out_addr, data, mask=mask)


@triton.jit
def _packed_split_kernel(
    # Input/Output pointers
    x_ptr,
    vid_ptr,
    txt_ptr,
    # Offset arrays
    x_offsets_ptr,
    vid_offsets_ptr,
    txt_offsets_ptr,
    # Lengths arrays
    vid_lengths_ptr,
    txt_lengths_ptr,
    # Dimensions
    n_segments: tl.constexpr,
    h: tl.constexpr,
    has_txt: tl.constexpr,
    # Block size
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Triton kernel for packed split operation.

    Each program processes one segment at a time, copying data from
    the input tensor to the appropriate output tensor.
    """
    pid = tl.program_id(axis=0)

    # Determine which segment this program is processing
    segment_idx = pid // 2 if has_txt else pid
    is_txt_segment = (pid % 2 == 1) if has_txt else False

    if segment_idx >= n_segments:
        return

    # Load the length for this segment
    if is_txt_segment:
        length = tl.load(txt_lengths_ptr + segment_idx)
        out_offset = tl.load(txt_offsets_ptr + segment_idx)
        out_ptr = txt_ptr
    else:
        length = tl.load(vid_lengths_ptr + segment_idx)
        out_offset = tl.load(vid_offsets_ptr + segment_idx)
        out_ptr = vid_ptr

    if length <= 0:
        return

    x_offset = tl.load(x_offsets_ptr + pid)

    # Process each row
    for row in range(length):
        # Load from input
        h_offsets = tl.arange(0, BLOCK_SIZE_H)
        mask = h_offsets < h

        x_addr = x_ptr + (x_offset + row) * h + h_offsets
        out_addr = out_ptr + (out_offset + row) * h + h_offsets

        data = tl.load(x_addr, mask=mask, other=0.0)
        tl.store(out_addr, data, mask=mask)


def packed_merge_triton(
    vid: torch.Tensor,
    txt: torch.Tensor | None,
    vid_lengths: list[int],
    txt_lengths: list[int] | None,
    block_size_h: int = 128,
) -> torch.Tensor:
    """
    Triton implementation of packed merge operation.

    Merges two tensors based on the provided lengths, interleaving segments
    from both tensors according to the length lists.

    Args:
        vid: Query tensor for the first segment of shape (s, h) on CUDA.
        txt: Query tensor for the second segment of shape (s', h) on CUDA or None.
        vid_lengths: List of lengths for the video segments.
        txt_lengths: List of lengths for the text segments or None.
        block_size_h: Block size for the hidden dimension (default: 128).

    Returns:
        Merged tensor of shape (total_length, h) where total_length is the
        sum of all segment lengths.
    """
    assert vid.is_cuda, "Input tensor must be on CUDA"
    assert vid.is_contiguous(), "Input tensor must be contiguous"
    if txt is not None:
        assert txt.is_cuda, "Text tensor must be on CUDA"
        assert txt.is_contiguous(), "Text tensor must be contiguous"
        assert len(vid_lengths) == len(txt_lengths), "Length lists must match"

    h = vid.shape[1]
    device = vid.device
    dtype = vid.dtype

    # Calculate total output length and build offset arrays
    has_txt = txt is not None and txt_lengths is not None
    segments_info = []
    vid_offset = 0
    txt_offset = 0
    out_offset = 0

    if has_txt:
        for vid_len, txt_len in zip(vid_lengths, txt_lengths, strict=True):
            if vid_len > 0:
                segments_info.append(
                    ("vid", vid_len, vid_offset, out_offset)
                )  # type, length, in_offset, out_offset
                vid_offset += vid_len
                out_offset += vid_len
            if txt_len > 0:
                segments_info.append(("txt", txt_len, txt_offset, out_offset))
                txt_offset += txt_len
                out_offset += txt_len
    else:
        for vid_len in vid_lengths:
            if vid_len > 0:
                segments_info.append(("vid", vid_len, vid_offset, out_offset))
                vid_offset += vid_len
                out_offset += vid_len

    total_length = out_offset

    # Create output tensor
    output = torch.empty((total_length, h), device=device, dtype=dtype)

    if total_length == 0:
        return output

    # Build offset arrays for kernel
    n_programs = len(segments_info)
    out_offsets = torch.zeros(n_programs, dtype=torch.int32, device=device)
    vid_offsets_list = []
    txt_offsets_list = []
    vid_lengths_list = []
    txt_lengths_list = []

    program_idx = 0
    for seg_type, length, in_offset, out_off in segments_info:
        out_offsets[program_idx] = out_off
        if seg_type == "vid":
            vid_offsets_list.append(in_offset)
            vid_lengths_list.append(length)
            if has_txt:
                txt_offsets_list.append(0)
                txt_lengths_list.append(0)
        else:  # txt
            txt_offsets_list.append(in_offset)
            txt_lengths_list.append(length)
            vid_offsets_list.append(0)
            vid_lengths_list.append(0)
        program_idx += 1

    vid_offsets = torch.tensor(vid_offsets_list, dtype=torch.int32, device=device)
    vid_lengths_t = torch.tensor(vid_lengths_list, dtype=torch.int32, device=device)

    if has_txt:
        txt_offsets = torch.tensor(txt_offsets_list, dtype=torch.int32, device=device)
        txt_lengths_t = torch.tensor(txt_lengths_list, dtype=torch.int32, device=device)
    else:
        # Create dummy tensors for the case without text
        txt_offsets = torch.zeros(1, dtype=torch.int32, device=device)
        txt_lengths_t = torch.zeros(1, dtype=torch.int32, device=device)
        txt = torch.empty((0, h), device=device, dtype=dtype)

    # Launch kernel
    n_segments = len(vid_lengths)
    grid = (n_programs,)

    _packed_merge_kernel[grid](
        vid,
        txt,
        output,
        vid_offsets,
        txt_offsets,
        out_offsets,
        vid_lengths_t,
        txt_lengths_t,
        n_segments=n_segments,
        h=h,
        has_txt=has_txt,
        BLOCK_SIZE_H=triton.next_power_of_2(h),
    )

    return output


def packed_split_triton(
    x: torch.Tensor,
    vid_lengths: list[int],
    txt_lengths: list[int] | None,
    vid_padding: int = 0,
    txt_padding: int = 0,
    block_size_h: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Triton implementation of packed split operation.

    Splits the input tensor into two segments based on the provided lengths,
    with optional padding for each segment.

    Args:
        x: Input tensor of shape (s, h) on CUDA.
        vid_lengths: List of lengths for the first segment.
        txt_lengths: List of lengths for the second segment or None.
        vid_padding: Padding size for the first segment.
        txt_padding: Padding size for the second segment.
        block_size_h: Block size for the hidden dimension (default: 128).

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: Two tensors corresponding
        to the split segments. The second tensor is None if txt_lengths is None.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    if txt_lengths is not None:
        assert len(vid_lengths) == len(txt_lengths), "Length lists must match"

    h = x.shape[1]
    device = x.device
    dtype = x.dtype

    has_txt = txt_lengths is not None
    segments_info = []
    vid_offset_out = 0
    txt_offset_out = 0
    x_offset = 0

    # Build segment information
    if has_txt:
        for vid_len, txt_len in zip(vid_lengths, txt_lengths, strict=True):
            if vid_len > 0:
                segments_info.append(
                    ("vid", vid_len, x_offset, vid_offset_out)
                )  # type, length, x_offset, out_offset
                x_offset += vid_len
                vid_offset_out += vid_len
            if txt_len > 0:
                segments_info.append(("txt", txt_len, x_offset, txt_offset_out))
                x_offset += txt_len
                txt_offset_out += txt_len
    else:
        for vid_len in vid_lengths:
            if vid_len > 0:
                segments_info.append(("vid", vid_len, x_offset, vid_offset_out))
                x_offset += vid_len
                vid_offset_out += vid_len

    # Calculate output sizes
    total_vid_length = vid_offset_out + vid_padding
    total_txt_length = txt_offset_out + txt_padding if has_txt else 0

    # Create output tensors
    vid_out = torch.zeros((total_vid_length, h), device=device, dtype=dtype)
    txt_out = (
        torch.zeros((total_txt_length, h), device=device, dtype=dtype)
        if has_txt
        else None
    )

    if len(segments_info) == 0:
        return vid_out, txt_out

    # Build offset arrays for kernel
    n_programs = len(segments_info)
    x_offsets = torch.zeros(n_programs, dtype=torch.int32, device=device)
    vid_offsets_list = []
    txt_offsets_list = []
    vid_lengths_list = []
    txt_lengths_list = []

    program_idx = 0
    for seg_type, length, x_off, out_off in segments_info:
        x_offsets[program_idx] = x_off
        if seg_type == "vid":
            vid_offsets_list.append(out_off)
            vid_lengths_list.append(length)
            if has_txt:
                txt_offsets_list.append(0)
                txt_lengths_list.append(0)
        else:  # txt
            txt_offsets_list.append(out_off)
            txt_lengths_list.append(length)
            vid_offsets_list.append(0)
            vid_lengths_list.append(0)
        program_idx += 1

    vid_offsets = torch.tensor(vid_offsets_list, dtype=torch.int32, device=device)
    vid_lengths_t = torch.tensor(vid_lengths_list, dtype=torch.int32, device=device)

    if has_txt:
        txt_offsets = torch.tensor(txt_offsets_list, dtype=torch.int32, device=device)
        txt_lengths_t = torch.tensor(txt_lengths_list, dtype=torch.int32, device=device)
        if txt_out is None:
            txt_out = torch.empty((0, h), device=device, dtype=dtype)
    else:
        # Create dummy tensors for the case without text
        txt_offsets = torch.zeros(1, dtype=torch.int32, device=device)
        txt_lengths_t = torch.zeros(1, dtype=torch.int32, device=device)
        txt_out = torch.empty((0, h), device=device, dtype=dtype)

    # Launch kernel
    n_segments = len(vid_lengths)
    grid = (n_programs,)

    _packed_split_kernel[grid](
        x,
        vid_out,
        txt_out,
        x_offsets,
        vid_offsets,
        txt_offsets,
        vid_lengths_t,
        txt_lengths_t,
        n_segments=n_segments,
        h=h,
        has_txt=has_txt,
        BLOCK_SIZE_H=triton.next_power_of_2(h),
    )

    return vid_out, txt_out if has_txt else None
