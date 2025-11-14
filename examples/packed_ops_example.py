"""
Example usage of packed merge and split operations.

This example demonstrates:
1. Basic usage of PyTorch reference implementations
2. Usage with Triton implementations (when CUDA is available)
3. Round-trip merge and split operations
"""

import torch

from triton_kernels import (
    packed_merge_torch,
    packed_merge_triton,
    packed_split_torch,
    packed_split_triton,
)


def main():
    print("=" * 80)
    print("Packed Merge and Split Operations Example")
    print("=" * 80)

    # Create sample data
    h = 64  # hidden dimension
    vid_lengths = [4, 3, 5]
    txt_lengths = [2, 3, 1]

    total_vid = sum(vid_lengths)
    total_txt = sum(txt_lengths)

    vid = torch.randn(total_vid, h, dtype=torch.float32)
    txt = torch.randn(total_txt, h, dtype=torch.float32)

    print("\nInput tensors:")
    print(f"  Video tensor shape: {vid.shape}")
    print(f"  Text tensor shape: {txt.shape}")
    print(f"  Video lengths: {vid_lengths}")
    print(f"  Text lengths: {txt_lengths}")

    # PyTorch reference implementations
    print("\n" + "-" * 80)
    print("PyTorch Reference Implementations")
    print("-" * 80)

    merged = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)
    print(f"\nMerged tensor shape: {merged.shape}")
    print(f"Expected shape: ({sum(vid_lengths) + sum(txt_lengths)}, {h})")

    vid_split, txt_split = packed_split_torch(merged, vid_lengths, txt_lengths)
    print("\nAfter split:")
    print(f"  Video tensor shape: {vid_split.shape}")
    print(f"  Text tensor shape: {txt_split.shape}")

    # Verify round-trip
    vid_match = torch.allclose(vid, vid_split, rtol=1e-5, atol=1e-5)
    txt_match = torch.allclose(txt, txt_split, rtol=1e-5, atol=1e-5)
    print("\nRound-trip verification:")
    print(f"  Video matches: {vid_match}")
    print(f"  Text matches: {txt_match}")

    # Example with padding
    print("\n" + "-" * 80)
    print("Split with Padding Example")
    print("-" * 80)

    vid_padded, txt_padded = packed_split_torch(
        merged, vid_lengths, txt_lengths, vid_padding=3, txt_padding=2
    )
    print("\nWith padding (vid_padding=3, txt_padding=2):")
    print(f"  Video tensor shape: {vid_padded.shape} (original: {total_vid})")
    print(f"  Text tensor shape: {txt_padded.shape} (original: {total_txt})")

    # Example without text
    print("\n" + "-" * 80)
    print("Video-only Example")
    print("-" * 80)

    merged_vid_only = packed_merge_torch(vid, None, vid_lengths, None)
    print(f"\nMerged video-only tensor shape: {merged_vid_only.shape}")

    vid_only_split, txt_none = packed_split_torch(merged_vid_only, vid_lengths, None)
    print("After split:")
    print(f"  Video tensor shape: {vid_only_split.shape}")
    print(f"  Text tensor: {txt_none}")

    # Triton implementations (only if CUDA is available)
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Triton Implementations (CUDA)")
        print("=" * 80)

        vid_gpu = vid.to(device="cuda", dtype=torch.float16)
        txt_gpu = txt.to(device="cuda", dtype=torch.float16)

        merged_triton = packed_merge_triton(vid_gpu, txt_gpu, vid_lengths, txt_lengths)
        print(f"\nTriton merged tensor shape: {merged_triton.shape}")

        vid_triton, txt_triton = packed_split_triton(
            merged_triton, vid_lengths, txt_lengths
        )
        print("Triton split:")
        print(f"  Video tensor shape: {vid_triton.shape}")
        print(f"  Text tensor shape: {txt_triton.shape}")

        # Compare with PyTorch reference
        merged_ref = merged.to(device="cuda", dtype=torch.float16)
        match = torch.allclose(merged_triton, merged_ref, rtol=1e-2, atol=1e-2)
        print(f"\nTriton vs PyTorch reference match: {match}")

        # Verify round-trip with Triton
        vid_match_triton = torch.allclose(vid_triton, vid_gpu, rtol=1e-2, atol=1e-2)
        txt_match_triton = torch.allclose(txt_triton, txt_gpu, rtol=1e-2, atol=1e-2)
        print("\nTriton round-trip verification:")
        print(f"  Video matches: {vid_match_triton}")
        print(f"  Text matches: {txt_match_triton}")
    else:
        print("\n" + "=" * 80)
        print("CUDA not available - skipping Triton implementations")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
