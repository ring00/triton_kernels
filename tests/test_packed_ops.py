"""
Comprehensive unit tests for packed operations.
"""

import pytest
import torch

from triton_kernels.packed_ops import (
    packed_merge_torch,
    packed_merge_triton,
    packed_split_torch,
    packed_split_triton,
)


class TestPackedMergeTorch:
    """Test cases for PyTorch reference packed merge implementation."""

    def test_basic_merge_with_text(self):
        """Test basic merge operation with both video and text."""
        vid = torch.randn(10, 64, dtype=torch.float32)
        txt = torch.randn(6, 64, dtype=torch.float32)
        vid_lengths = [4, 3, 3]
        txt_lengths = [2, 2, 2]

        result = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

        assert result.shape == (16, 64)
        # Verify segments are in correct order
        torch.testing.assert_close(result[0:4], vid[0:4])
        torch.testing.assert_close(result[4:6], txt[0:2])
        torch.testing.assert_close(result[6:9], vid[4:7])
        torch.testing.assert_close(result[9:11], txt[2:4])
        torch.testing.assert_close(result[11:14], vid[7:10])
        torch.testing.assert_close(result[14:16], txt[4:6])

    def test_merge_without_text(self):
        """Test merge operation with only video."""
        vid = torch.randn(10, 64, dtype=torch.float32)
        vid_lengths = [4, 3, 3]

        result = packed_merge_torch(vid, None, vid_lengths, None)

        assert result.shape == (10, 64)
        torch.testing.assert_close(result, vid)

    def test_merge_with_zero_lengths(self):
        """Test merge with some zero-length segments."""
        vid = torch.randn(7, 64, dtype=torch.float32)
        txt = torch.randn(4, 64, dtype=torch.float32)
        vid_lengths = [4, 0, 3]
        txt_lengths = [2, 2, 0]

        result = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

        assert result.shape == (11, 64)
        torch.testing.assert_close(result[0:4], vid[0:4])
        torch.testing.assert_close(result[4:6], txt[0:2])
        torch.testing.assert_close(result[6:8], txt[2:4])
        torch.testing.assert_close(result[8:11], vid[4:7])

    def test_merge_single_segment(self):
        """Test merge with single segment."""
        vid = torch.randn(5, 32, dtype=torch.float32)
        txt = torch.randn(3, 32, dtype=torch.float32)
        vid_lengths = [5]
        txt_lengths = [3]

        result = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

        assert result.shape == (8, 32)
        torch.testing.assert_close(result[0:5], vid)
        torch.testing.assert_close(result[5:8], txt)

    def test_merge_different_hidden_sizes(self):
        """Test merge with different hidden dimensions."""
        for h in [32, 64, 128, 256]:
            vid = torch.randn(8, h, dtype=torch.float32)
            txt = torch.randn(4, h, dtype=torch.float32)
            vid_lengths = [4, 4]
            txt_lengths = [2, 2]

            result = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

            assert result.shape == (12, h)


class TestPackedSplitTorch:
    """Test cases for PyTorch reference packed split implementation."""

    def test_basic_split_with_text(self):
        """Test basic split operation with both video and text."""
        x = torch.randn(16, 64, dtype=torch.float32)
        vid_lengths = [4, 3, 3]
        txt_lengths = [2, 2, 2]

        vid, txt = packed_split_torch(x, vid_lengths, txt_lengths)

        assert vid.shape == (10, 64)
        assert txt.shape == (6, 64)
        torch.testing.assert_close(vid[0:4], x[0:4])
        torch.testing.assert_close(txt[0:2], x[4:6])
        torch.testing.assert_close(vid[4:7], x[6:9])
        torch.testing.assert_close(txt[2:4], x[9:11])
        torch.testing.assert_close(vid[7:10], x[11:14])
        torch.testing.assert_close(txt[4:6], x[14:16])

    def test_split_without_text(self):
        """Test split operation with only video."""
        x = torch.randn(10, 64, dtype=torch.float32)
        vid_lengths = [4, 3, 3]

        vid, txt = packed_split_torch(x, vid_lengths, None)

        assert vid.shape == (10, 64)
        assert txt is None
        torch.testing.assert_close(vid, x)

    def test_split_with_padding(self):
        """Test split with padding."""
        x = torch.randn(10, 64, dtype=torch.float32)
        vid_lengths = [4, 3, 3]
        txt_lengths = [0, 0, 0]

        vid, txt = packed_split_torch(
            x, vid_lengths, txt_lengths, vid_padding=2, txt_padding=3
        )

        assert vid.shape == (12, 64)
        assert txt.shape == (3, 64)
        torch.testing.assert_close(vid[0:10], x)
        torch.testing.assert_close(vid[10:12], torch.zeros(2, 64))
        torch.testing.assert_close(txt, torch.zeros(3, 64))

    def test_split_with_zero_lengths(self):
        """Test split with some zero-length segments."""
        x = torch.randn(11, 64, dtype=torch.float32)
        vid_lengths = [4, 0, 3]
        txt_lengths = [2, 2, 0]

        vid, txt = packed_split_torch(x, vid_lengths, txt_lengths)

        assert vid.shape == (7, 64)
        assert txt.shape == (4, 64)

    def test_split_single_segment(self):
        """Test split with single segment."""
        x = torch.randn(8, 32, dtype=torch.float32)
        vid_lengths = [5]
        txt_lengths = [3]

        vid, txt = packed_split_torch(x, vid_lengths, txt_lengths)

        assert vid.shape == (5, 32)
        assert txt.shape == (3, 32)
        torch.testing.assert_close(vid, x[0:5])
        torch.testing.assert_close(txt, x[5:8])


class TestPackedMergeSplitRoundTrip:
    """Test cases for merge-split round-trip consistency."""

    def test_roundtrip_with_text(self):
        """Test that merge followed by split recovers original tensors."""
        vid_orig = torch.randn(10, 64, dtype=torch.float32)
        txt_orig = torch.randn(6, 64, dtype=torch.float32)
        vid_lengths = [4, 3, 3]
        txt_lengths = [2, 2, 2]

        merged = packed_merge_torch(vid_orig, txt_orig, vid_lengths, txt_lengths)
        vid, txt = packed_split_torch(merged, vid_lengths, txt_lengths)

        torch.testing.assert_close(vid, vid_orig)
        torch.testing.assert_close(txt, txt_orig)

    def test_roundtrip_without_text(self):
        """Test that merge followed by split recovers original tensor."""
        vid_orig = torch.randn(10, 64, dtype=torch.float32)
        vid_lengths = [4, 3, 3]

        merged = packed_merge_torch(vid_orig, None, vid_lengths, None)
        vid, txt = packed_split_torch(merged, vid_lengths, None)

        torch.testing.assert_close(vid, vid_orig)
        assert txt is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedMergeTriton:
    """Test cases for Triton packed merge implementation."""

    def test_basic_merge_with_text(self):
        """Test basic merge operation with both video and text."""
        vid = torch.randn(10, 64, dtype=torch.float16, device="cuda")
        txt = torch.randn(6, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 3, 3]
        txt_lengths = [2, 2, 2]

        result = packed_merge_triton(vid, txt, vid_lengths, txt_lengths)

        assert result.shape == (16, 64)
        # Compare with PyTorch reference
        vid_cpu = vid.cpu().to(torch.float32)
        txt_cpu = txt.cpu().to(torch.float32)
        expected = packed_merge_torch(vid_cpu, txt_cpu, vid_lengths, txt_lengths)
        torch.testing.assert_close(
            result.cpu().to(torch.float32), expected, rtol=1e-2, atol=1e-2
        )

    def test_merge_without_text(self):
        """Test merge operation with only video."""
        vid = torch.randn(10, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 3, 3]

        result = packed_merge_triton(vid, None, vid_lengths, None)

        assert result.shape == (10, 64)
        vid_cpu = vid.cpu().to(torch.float32)
        expected = packed_merge_torch(vid_cpu, None, vid_lengths, None)
        torch.testing.assert_close(
            result.cpu().to(torch.float32), expected, rtol=1e-2, atol=1e-2
        )

    def test_merge_with_zero_lengths(self):
        """Test merge with some zero-length segments."""
        vid = torch.randn(7, 64, dtype=torch.float16, device="cuda")
        txt = torch.randn(4, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 0, 3]
        txt_lengths = [2, 2, 0]

        result = packed_merge_triton(vid, txt, vid_lengths, txt_lengths)

        assert result.shape == (11, 64)
        vid_cpu = vid.cpu().to(torch.float32)
        txt_cpu = txt.cpu().to(torch.float32)
        expected = packed_merge_torch(vid_cpu, txt_cpu, vid_lengths, txt_lengths)
        torch.testing.assert_close(
            result.cpu().to(torch.float32), expected, rtol=1e-2, atol=1e-2
        )

    def test_merge_different_hidden_sizes(self):
        """Test merge with different hidden dimensions."""
        for h in [32, 64, 128, 256]:
            vid = torch.randn(8, h, dtype=torch.float16, device="cuda")
            txt = torch.randn(4, h, dtype=torch.float16, device="cuda")
            vid_lengths = [4, 4]
            txt_lengths = [2, 2]

            result = packed_merge_triton(vid, txt, vid_lengths, txt_lengths)

            assert result.shape == (12, h)
            vid_cpu = vid.cpu().to(torch.float32)
            txt_cpu = txt.cpu().to(torch.float32)
            expected = packed_merge_torch(vid_cpu, txt_cpu, vid_lengths, txt_lengths)
            torch.testing.assert_close(
                result.cpu().to(torch.float32), expected, rtol=1e-2, atol=1e-2
            )

    def test_merge_large_tensors(self):
        """Test merge with large tensors."""
        vid = torch.randn(256, 128, dtype=torch.float16, device="cuda")
        txt = torch.randn(128, 128, dtype=torch.float16, device="cuda")
        vid_lengths = [64, 64, 64, 64]
        txt_lengths = [32, 32, 32, 32]

        result = packed_merge_triton(vid, txt, vid_lengths, txt_lengths)

        assert result.shape == (384, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedSplitTriton:
    """Test cases for Triton packed split implementation."""

    def test_basic_split_with_text(self):
        """Test basic split operation with both video and text."""
        x = torch.randn(16, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 3, 3]
        txt_lengths = [2, 2, 2]

        vid, txt = packed_split_triton(x, vid_lengths, txt_lengths)

        assert vid.shape == (10, 64)
        assert txt.shape == (6, 64)
        # Compare with PyTorch reference
        x_cpu = x.cpu().to(torch.float32)
        vid_expected, txt_expected = packed_split_torch(x_cpu, vid_lengths, txt_lengths)
        torch.testing.assert_close(
            vid.cpu().to(torch.float32), vid_expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            txt.cpu().to(torch.float32), txt_expected, rtol=1e-2, atol=1e-2
        )

    def test_split_without_text(self):
        """Test split operation with only video."""
        x = torch.randn(10, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 3, 3]

        vid, txt = packed_split_triton(x, vid_lengths, None)

        assert vid.shape == (10, 64)
        assert txt is None
        x_cpu = x.cpu().to(torch.float32)
        vid_expected, _ = packed_split_torch(x_cpu, vid_lengths, None)
        torch.testing.assert_close(
            vid.cpu().to(torch.float32), vid_expected, rtol=1e-2, atol=1e-2
        )

    def test_split_with_padding(self):
        """Test split with padding."""
        x = torch.randn(10, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 3, 3]
        txt_lengths = [0, 0, 0]

        vid, txt = packed_split_triton(
            x, vid_lengths, txt_lengths, vid_padding=2, txt_padding=3
        )

        assert vid.shape == (12, 64)
        assert txt.shape == (3, 64)
        # Check that padding is zeros
        torch.testing.assert_close(
            vid[10:12].cpu(), torch.zeros(2, 64), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(txt.cpu(), torch.zeros(3, 64), rtol=1e-2, atol=1e-2)

    def test_split_with_zero_lengths(self):
        """Test split with some zero-length segments."""
        x = torch.randn(11, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 0, 3]
        txt_lengths = [2, 2, 0]

        vid, txt = packed_split_triton(x, vid_lengths, txt_lengths)

        assert vid.shape == (7, 64)
        assert txt.shape == (4, 64)

    def test_split_different_hidden_sizes(self):
        """Test split with different hidden dimensions."""
        for h in [32, 64, 128, 256]:
            x = torch.randn(12, h, dtype=torch.float16, device="cuda")
            vid_lengths = [4, 4]
            txt_lengths = [2, 2]

            vid, txt = packed_split_triton(x, vid_lengths, txt_lengths)

            assert vid.shape == (8, h)
            assert txt.shape == (4, h)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedMergeSplitTritonRoundTrip:
    """Test cases for Triton merge-split round-trip consistency."""

    def test_roundtrip_with_text(self):
        """Test that merge followed by split recovers original tensors."""
        vid_orig = torch.randn(10, 64, dtype=torch.float16, device="cuda")
        txt_orig = torch.randn(6, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 3, 3]
        txt_lengths = [2, 2, 2]

        merged = packed_merge_triton(vid_orig, txt_orig, vid_lengths, txt_lengths)
        vid, txt = packed_split_triton(merged, vid_lengths, txt_lengths)

        torch.testing.assert_close(vid, vid_orig, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(txt, txt_orig, rtol=1e-2, atol=1e-2)

    def test_roundtrip_without_text(self):
        """Test that merge followed by split recovers original tensor."""
        vid_orig = torch.randn(10, 64, dtype=torch.float16, device="cuda")
        vid_lengths = [4, 3, 3]

        merged = packed_merge_triton(vid_orig, None, vid_lengths, None)
        vid, txt = packed_split_triton(merged, vid_lengths, None)

        torch.testing.assert_close(vid, vid_orig, rtol=1e-2, atol=1e-2)
        assert txt is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedOpsComparison:
    """Test cases comparing PyTorch and Triton implementations."""

    def test_merge_torch_vs_triton(self):
        """Test that Triton and PyTorch merge implementations produce similar results."""
        vid = torch.randn(20, 128, dtype=torch.float32)
        txt = torch.randn(12, 128, dtype=torch.float32)
        vid_lengths = [5, 7, 8]
        txt_lengths = [4, 4, 4]

        # PyTorch result
        result_torch = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

        # Triton result
        vid_gpu = vid.to(device="cuda", dtype=torch.float16)
        txt_gpu = txt.to(device="cuda", dtype=torch.float16)
        result_triton = packed_merge_triton(vid_gpu, txt_gpu, vid_lengths, txt_lengths)

        # Compare (allowing for float16 precision differences)
        result_torch_gpu = result_torch.to(device="cuda", dtype=torch.float16)
        torch.testing.assert_close(
            result_triton, result_torch_gpu, rtol=1e-2, atol=1e-2
        )

    def test_split_torch_vs_triton(self):
        """Test that Triton and PyTorch split implementations produce similar results."""
        x = torch.randn(32, 128, dtype=torch.float32)
        vid_lengths = [5, 7, 8]
        txt_lengths = [4, 4, 4]

        # PyTorch result
        vid_torch, txt_torch = packed_split_torch(x, vid_lengths, txt_lengths)

        # Triton result
        x_gpu = x.to(device="cuda", dtype=torch.float16)
        vid_triton, txt_triton = packed_split_triton(x_gpu, vid_lengths, txt_lengths)

        # Compare (allowing for float16 precision differences)
        vid_torch_gpu = vid_torch.to(device="cuda", dtype=torch.float16)
        txt_torch_gpu = txt_torch.to(device="cuda", dtype=torch.float16)
        torch.testing.assert_close(vid_triton, vid_torch_gpu, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(txt_triton, txt_torch_gpu, rtol=1e-2, atol=1e-2)
