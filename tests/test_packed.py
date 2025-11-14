"""
Unit tests for packed tensor operations.
"""

import torch

from triton_kernels.packed import packed_merge, packed_split


class TestPackedMerge:
    """Test cases for packed_merge function."""

    def test_basic_merge_with_txt(self):
        """Test basic merge with both vid and txt tensors."""
        vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        txt = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
        vid_lengths = [2, 1]
        txt_lengths = [1, 1]

        result = packed_merge(vid, txt, vid_lengths, txt_lengths)

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

        result = packed_merge(vid, None, vid_lengths, None)

        # Expected: [vid[0:2], vid[2:3]]
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert torch.allclose(result, expected)

    def test_merge_with_zero_lengths(self):
        """Test merge with zero-length segments."""
        vid = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        txt = torch.tensor([[5.0, 6.0]])
        vid_lengths = [1, 0, 1]
        txt_lengths = [0, 1, 0]

        result = packed_merge(vid, txt, vid_lengths, txt_lengths)

        # Expected: [vid[0:1], txt[0:1], vid[1:2]]
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_merge_1d_tensors(self):
        """Test merge with 1D tensors."""
        vid = torch.tensor([1.0, 2.0, 3.0, 4.0])
        txt = torch.tensor([5.0, 6.0])
        vid_lengths = [2, 2]
        txt_lengths = [1, 1]

        result = packed_merge(vid, txt, vid_lengths, txt_lengths)

        expected = torch.tensor([1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
        assert torch.allclose(result, expected)

    def test_merge_3d_tensors(self):
        """Test merge with 3D tensors."""
        vid = torch.randn(4, 2, 3)
        txt = torch.randn(2, 2, 3)
        vid_lengths = [2, 2]
        txt_lengths = [1, 1]

        result = packed_merge(vid, txt, vid_lengths, txt_lengths)

        # Verify shape
        assert result.shape == (6, 2, 3)

        # Verify segments
        assert torch.allclose(result[0:2], vid[0:2])
        assert torch.allclose(result[2:3], txt[0:1])
        assert torch.allclose(result[3:5], vid[2:4])
        assert torch.allclose(result[5:6], txt[1:2])

    def test_merge_empty_txt(self):
        """Test merge with txt as None but txt_lengths as empty list."""
        vid = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        vid_lengths = [1, 1]

        result = packed_merge(vid, None, vid_lengths, [])

        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_merge_all_zero_vid_lengths(self):
        """Test merge with all zero vid lengths."""
        vid = torch.tensor([[1.0, 2.0]])
        txt = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        vid_lengths = [0, 0]
        txt_lengths = [1, 1]

        result = packed_merge(vid, txt, vid_lengths, txt_lengths)

        expected = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        assert torch.allclose(result, expected)


class TestPackedSplit:
    """Test cases for packed_split function."""

    def test_basic_split_with_txt(self):
        """Test basic split with txt segments."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        vid_lengths = [2, 1]
        txt_lengths = [1, 1]

        vid, txt = packed_split(x, vid_lengths, txt_lengths)

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

        vid, txt = packed_split(x, vid_lengths, None)

        expected_vid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert torch.allclose(vid, expected_vid)
        assert txt is None

    def test_split_with_vid_padding(self):
        """Test split with vid padding."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        vid_lengths = [2]
        txt_lengths = None
        vid_padding = 1

        vid, txt = packed_split(x, vid_lengths, txt_lengths, vid_padding=vid_padding)

        # Expected: [[1,2], [3,4], [0,0]]
        assert vid.shape == (3, 2)
        assert torch.allclose(vid[0:2], x)
        assert torch.allclose(vid[2:3], torch.zeros(1, 2))
        assert txt is None

    def test_split_with_txt_padding(self):
        """Test split with txt padding."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        vid_lengths = [2]
        txt_lengths = [1]
        txt_padding = 2

        vid, txt = packed_split(x, vid_lengths, txt_lengths, txt_padding=txt_padding)

        expected_vid = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        # Expected txt: [[5,6], [0,0], [0,0]]
        assert torch.allclose(vid, expected_vid)
        assert txt.shape == (3, 2)
        assert torch.allclose(txt[0:1], torch.tensor([[5.0, 6.0]]))
        assert torch.allclose(txt[1:3], torch.zeros(2, 2))

    def test_split_with_both_padding(self):
        """Test split with both vid and txt padding."""
        x = torch.tensor([[1.0], [2.0], [3.0]])
        vid_lengths = [1]
        txt_lengths = [1]
        vid_padding = 1
        txt_padding = 1

        vid, txt = packed_split(x, vid_lengths, txt_lengths, vid_padding, txt_padding)

        assert vid.shape == (2, 1)
        assert txt.shape == (2, 1)
        assert torch.allclose(vid[0:1], torch.tensor([[1.0]]))
        assert torch.allclose(vid[1:2], torch.zeros(1, 1))
        assert torch.allclose(txt[0:1], torch.tensor([[2.0]]))
        assert torch.allclose(txt[1:2], torch.zeros(1, 1))

    def test_split_with_zero_lengths(self):
        """Test split with zero-length segments."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        vid_lengths = [1, 0, 1]
        txt_lengths = [0, 1, 0]

        vid, txt = packed_split(x, vid_lengths, txt_lengths)

        # Expected vid: [[1,2], [5,6]]
        # Expected txt: [[3,4]]
        expected_vid = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        expected_txt = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(vid, expected_vid)
        assert torch.allclose(txt, expected_txt)

    def test_split_1d_tensor(self):
        """Test split with 1D tensor."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        vid_lengths = [2, 2]
        txt_lengths = [1, 1]

        vid, txt = packed_split(x, vid_lengths, txt_lengths)

        expected_vid = torch.tensor([1.0, 2.0, 4.0, 5.0])
        expected_txt = torch.tensor([3.0, 6.0])
        assert torch.allclose(vid, expected_vid)
        assert torch.allclose(txt, expected_txt)

    def test_split_3d_tensor(self):
        """Test split with 3D tensor."""
        x = torch.randn(6, 2, 3)
        vid_lengths = [2, 2]
        txt_lengths = [1, 1]

        vid, txt = packed_split(x, vid_lengths, txt_lengths)

        # Verify shapes
        assert vid.shape == (4, 2, 3)
        assert txt.shape == (2, 2, 3)

        # Verify segments
        assert torch.allclose(vid[0:2], x[0:2])
        assert torch.allclose(txt[0:1], x[2:3])
        assert torch.allclose(vid[2:4], x[3:5])
        assert torch.allclose(txt[1:2], x[5:6])

    def test_split_all_zero_vid_lengths(self):
        """Test split with all zero vid lengths and padding."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        vid_lengths = [0, 0]
        txt_lengths = [1, 1]

        # With all zero lengths and no padding, vid_segments would be empty
        # and torch.cat would fail. We need padding to make this work.
        vid_with_padding, txt_with_padding = packed_split(
            x, vid_lengths, txt_lengths, vid_padding=1
        )

        assert vid_with_padding.shape == (1, 2)
        assert torch.allclose(vid_with_padding, torch.zeros(1, 2))
        expected_txt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(txt_with_padding, expected_txt)


class TestPackedRoundTrip:
    """Test cases for round-trip operations (merge -> split -> merge)."""

    def test_roundtrip_with_txt(self):
        """Test that split(merge(vid, txt)) == (vid, txt)."""
        vid = torch.randn(5, 3)
        txt = torch.randn(3, 3)
        vid_lengths = [2, 3]
        txt_lengths = [1, 2]

        merged = packed_merge(vid, txt, vid_lengths, txt_lengths)
        vid_recovered, txt_recovered = packed_split(merged, vid_lengths, txt_lengths)

        assert torch.allclose(vid_recovered, vid)
        assert torch.allclose(txt_recovered, txt)

    def test_roundtrip_without_txt(self):
        """Test that split(merge(vid, None)) == (vid, None)."""
        vid = torch.randn(4, 2)
        vid_lengths = [2, 2]

        merged = packed_merge(vid, None, vid_lengths, None)
        vid_recovered, txt_recovered = packed_split(merged, vid_lengths, None)

        assert torch.allclose(vid_recovered, vid)
        assert txt_recovered is None

    def test_roundtrip_with_zero_lengths(self):
        """Test round-trip with zero-length segments."""
        vid = torch.randn(3, 2)
        txt = torch.randn(2, 2)
        vid_lengths = [1, 0, 2]
        txt_lengths = [0, 1, 1]

        merged = packed_merge(vid, txt, vid_lengths, txt_lengths)
        vid_recovered, txt_recovered = packed_split(merged, vid_lengths, txt_lengths)

        assert torch.allclose(vid_recovered, vid)
        assert torch.allclose(txt_recovered, txt)
