"""
Comprehensive unit tests for packed tensor kernel implementations.
"""

import pytest
import torch

from triton_kernels.packed import (
    packed_merge_torch,
    packed_merge_triton,
    packed_split_torch,
    packed_split_triton,
)


class TestPackedMergeTorch:
    """Test cases for PyTorch reference packed merge implementation."""

    def test_basic_merge_with_txt(self):
        """Test basic merge with both vid and txt tensors."""
        vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        txt = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
        vid_lengths = [2, 1]
        txt_lengths = [1, 1]

        result = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

        # Expected: [vid[0:2], txt[0:1], vid[2:3], txt[1:2]]
        # = [[1,2], [3,4], [7,8], [5,6], [9,10]]
        expected = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [5.0, 6.0], [9.0, 10.0]]
        )
        assert torch.allclose(result, expected)

    def test_merge_without_txt(self):
        """Test merge with only vid tensor."""
        vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        vid_lengths = [2, 1]

        result = packed_merge_torch(vid, None, vid_lengths, None)

        # Expected: [vid[0:2], vid[2:3]]
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert torch.allclose(result, expected)

    def test_merge_1d_tensors(self):
        """Test merge with 1D tensors."""
        vid = torch.tensor([1.0, 2.0, 3.0, 4.0])
        txt = torch.tensor([5.0, 6.0])
        vid_lengths = [2, 2]
        txt_lengths = [1, 1]

        result = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

        expected = torch.tensor([1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
        assert torch.allclose(result, expected)

    def test_merge_3d_tensors(self):
        """Test merge with 3D tensors."""
        vid = torch.randn(4, 2, 3)
        txt = torch.randn(2, 2, 3)
        vid_lengths = [2, 2]
        txt_lengths = [1, 1]

        result = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)

        # Verify shape
        assert result.shape == (6, 2, 3)

        # Verify segments
        assert torch.allclose(result[0:2], vid[0:2])
        assert torch.allclose(result[2:3], txt[0:1])
        assert torch.allclose(result[3:5], vid[2:4])
        assert torch.allclose(result[5:6], txt[1:2])


class TestPackedSplitTorch:
    """Test cases for PyTorch reference packed split implementation."""

    def test_basic_split_with_txt(self):
        """Test basic split with txt segments."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        vid_lengths = [2, 1]
        txt_lengths = [1, 1]

        vid, txt = packed_split_torch(x, vid_lengths, txt_lengths)

        # Expected vid: [[1,2], [3,4], [7,8]]
        # Expected txt: [[5,6], [9,10]]
        expected_vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
        expected_txt = torch.tensor([[5.0, 6.0], [9.0, 10.0]])
        assert torch.allclose(vid, expected_vid)
        assert torch.allclose(txt, expected_txt)

    def test_split_without_txt(self):
        """Test split with only vid segments."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        vid_lengths = [2, 1]

        vid, txt = packed_split_torch(x, vid_lengths, None)

        expected_vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert torch.allclose(vid, expected_vid)
        assert txt is None

    def test_split_with_vid_padding(self):
        """Test split with vid padding."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        vid_lengths = [2]
        txt_lengths = None
        vid_padding = 1

        vid, txt = packed_split_torch(
            x, vid_lengths, txt_lengths, vid_padding=vid_padding
        )

        # Expected: [[1,2], [3,4], [0,0]]
        assert vid.shape == (3, 2)
        assert torch.allclose(vid[0:2], x)
        assert torch.allclose(vid[2:3], torch.zeros(1, 2))
        assert txt is None

    def test_split_1d_tensor(self):
        """Test split with 1D tensor."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        vid_lengths = [2, 2]
        txt_lengths = [1, 1]

        vid, txt = packed_split_torch(x, vid_lengths, txt_lengths)

        expected_vid = torch.tensor([1.0, 2.0, 4.0, 5.0])
        expected_txt = torch.tensor([3.0, 6.0])
        assert torch.allclose(vid, expected_vid)
        assert torch.allclose(txt, expected_txt)


class TestPackedRoundTripTorch:
    """Test cases for round-trip operations (merge -> split -> merge)."""

    def test_roundtrip_with_txt(self):
        """Test that split(merge(vid, txt)) == (vid, txt)."""
        vid = torch.randn(5, 3)
        txt = torch.randn(3, 3)
        vid_lengths = [2, 3]
        txt_lengths = [1, 2]

        merged = packed_merge_torch(vid, txt, vid_lengths, txt_lengths)
        vid_recovered, txt_recovered = packed_split_torch(
            merged, vid_lengths, txt_lengths
        )

        assert torch.allclose(vid_recovered, vid)
        assert torch.allclose(txt_recovered, txt)

    def test_roundtrip_without_txt(self):
        """Test that split(merge(vid, None)) == (vid, None)."""
        vid = torch.randn(4, 2)
        vid_lengths = [2, 2]

        merged = packed_merge_torch(vid, None, vid_lengths, None)
        vid_recovered, txt_recovered = packed_split_torch(merged, vid_lengths, None)

        assert torch.allclose(vid_recovered, vid)
        assert txt_recovered is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestPackedMergeTriton:
    """Test cases for Triton packed merge implementation."""

    def test_basic_merge_with_txt(self):
        """Test basic merge with both vid and txt tensors with Triton."""
        vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda")
        txt = torch.tensor([[7.0, 8.0], [9.0, 10.0]], device="cuda")
        vid_lengths = [2, 1]
        txt_lengths = [1, 1]

        result = packed_merge_triton(vid, txt, vid_lengths, txt_lengths)

        expected = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [5.0, 6.0], [9.0, 10.0]], device="cuda"
        )
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)

    def test_matches_torch(self):
        """Test that Triton matches PyTorch reference."""
        vid = torch.randn(5, 3, device="cuda")
        txt = torch.randn(3, 3, device="cuda")
        vid_lengths = [2, 3]
        txt_lengths = [1, 2]

        result_torch = packed_merge_torch(
            vid.cpu(), txt.cpu(), vid_lengths, txt_lengths
        ).cuda()
        result_triton = packed_merge_triton(vid, txt, vid_lengths, txt_lengths)

        assert torch.allclose(result_torch, result_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestPackedSplitTriton:
    """Test cases for Triton packed split implementation."""

    def test_basic_split_with_txt(self):
        """Test basic split with txt segments with Triton."""
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device="cuda"
        )
        vid_lengths = [2, 1]
        txt_lengths = [1, 1]

        vid, txt = packed_split_triton(x, vid_lengths, txt_lengths)

        expected_vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]], device="cuda")
        expected_txt = torch.tensor([[5.0, 6.0], [9.0, 10.0]], device="cuda")
        assert torch.allclose(vid, expected_vid, rtol=1e-2, atol=1e-2)
        assert torch.allclose(txt, expected_txt, rtol=1e-2, atol=1e-2)

    def test_matches_torch(self):
        """Test that Triton matches PyTorch reference."""
        x = torch.randn(8, 3, device="cuda")
        vid_lengths = [2, 3]
        txt_lengths = [1, 2]

        vid_torch, txt_torch = packed_split_torch(x.cpu(), vid_lengths, txt_lengths)
        vid_triton, txt_triton = packed_split_triton(x, vid_lengths, txt_lengths)

        assert torch.allclose(vid_torch.cuda(), vid_triton, rtol=1e-2, atol=1e-2)
        assert torch.allclose(txt_torch.cuda(), txt_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestPackedComparison:
    """Comparison tests between PyTorch and Triton implementations."""

    def test_roundtrip_triton(self):
        """Test round-trip with Triton implementations."""
        vid = torch.randn(5, 3, device="cuda")
        txt = torch.randn(3, 3, device="cuda")
        vid_lengths = [2, 3]
        txt_lengths = [1, 2]

        # Merge with Triton
        merged = packed_merge_triton(vid, txt, vid_lengths, txt_lengths)

        # Split with Triton
        vid_recovered, txt_recovered = packed_split_triton(
            merged, vid_lengths, txt_lengths
        )

        assert torch.allclose(vid_recovered, vid, rtol=1e-2, atol=1e-2)
        assert torch.allclose(txt_recovered, txt, rtol=1e-2, atol=1e-2)
