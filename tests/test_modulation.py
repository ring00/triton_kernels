"""
Unit tests for modulation operations.
"""

import torch

from triton_kernels.modulation import (
    gate_modulate,
    scale_modulate,
    scale_shift_modulate,
    tanh_modulate,
)


class TestScaleShiftModulate:
    """Test cases for scale_shift_modulate function."""

    def test_basic_operation(self):
        """Test basic scale and shift modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        scale = torch.tensor([0.5, 1.0])
        shift = torch.tensor([1.0, 2.0])
        lengths = [3, 2]

        y = scale_shift_modulate(x, scale, shift, lengths)

        # First segment: [1,2,3] * (1+0.5) + 1 = [2.5, 4.0, 5.5]
        # Second segment: [4,5] * (1+1.0) + 2 = [10.0, 12.0]
        # Remaining: [0]
        expected = torch.tensor([2.5, 4.0, 5.5, 10.0, 12.0, 0.0])
        assert torch.allclose(y, expected)

    def test_with_indices(self):
        """Test modulation with custom indices."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor([0.0, 1.0, 2.0])
        shift = torch.tensor([0.0, 1.0, 2.0])
        lengths = [2, 2]
        indices = [1, 2]

        y = scale_shift_modulate(x, scale, shift, lengths, indices)

        # First segment: [1,2] * (1+1.0) + 1 = [3.0, 5.0]
        # Second segment: [3,4] * (1+2.0) + 2 = [11.0, 14.0]
        expected = torch.tensor([3.0, 5.0, 11.0, 14.0])
        assert torch.allclose(y, expected)

    def test_zero_lengths(self):
        """Test with zero-length segments."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor([0.5, 1.0, 0.0])
        shift = torch.tensor([1.0, 2.0, 0.0])
        lengths = [2, 0, 2]

        y = scale_shift_modulate(x, scale, shift, lengths)

        # First segment: [1,2] * 1.5 + 1 = [2.5, 4.0]
        # Second segment: skipped (length 0)
        # Third segment: [3,4] * 1.0 + 0 = [3.0, 4.0]
        expected = torch.tensor([2.5, 4.0, 3.0, 4.0])
        assert torch.allclose(y, expected)

    def test_multidimensional_tensor(self):
        """Test with multidimensional tensor."""
        x = torch.randn(6, 4)
        scale = torch.randn(2, 4)
        shift = torch.randn(2, 4)
        lengths = [3, 2]

        y = scale_shift_modulate(x, scale, shift, lengths)

        # Verify first segment
        expected_first = x[:3] * (1 + scale[0]) + shift[0]
        assert torch.allclose(y[:3], expected_first)

        # Verify second segment
        expected_second = x[3:5] * (1 + scale[1]) + shift[1]
        assert torch.allclose(y[3:5], expected_second)

        # Verify zero padding
        assert torch.allclose(y[5:], torch.zeros(1, 4))

    def test_empty_indices(self):
        """Test with empty indices parameter."""
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor([0.5, 1.0])
        shift = torch.tensor([1.0, 2.0])
        lengths = [2, 1]

        y = scale_shift_modulate(x, scale, shift, lengths, None)

        # First segment: [1,2] * (1+0.5) + 1 = [2.5, 4.0]
        # Second segment: [3] * (1+1.0) + 2 = [8.0]
        expected = torch.tensor([2.5, 4.0, 8.0])
        assert torch.allclose(y, expected)


class TestScaleModulate:
    """Test cases for scale_modulate function."""

    def test_basic_operation(self):
        """Test basic scale modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        scale = torch.tensor([0.5, 1.0])
        lengths = [3, 1]

        y = scale_modulate(x, scale, lengths)

        # First segment: [1,2,3] * (1+0.5) = [1.5, 3.0, 4.5]
        # Second segment: [4] * (1+1.0) = [8.0]
        # Remaining: [0]
        expected = torch.tensor([1.5, 3.0, 4.5, 8.0, 0.0])
        assert torch.allclose(y, expected)

    def test_with_indices(self):
        """Test modulation with custom indices."""
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor([0.0, 1.0, 2.0])
        lengths = [2, 1]
        indices = [2, 1]

        y = scale_modulate(x, scale, lengths, indices)

        # First segment: [1,2] * (1+2.0) = [3.0, 6.0]
        # Second segment: [3] * (1+1.0) = [6.0]
        expected = torch.tensor([3.0, 6.0, 6.0])
        assert torch.allclose(y, expected)

    def test_multidimensional_tensor(self):
        """Test with multidimensional tensor."""
        x = torch.randn(5, 3)
        scale = torch.randn(2, 3)
        lengths = [3, 2]

        y = scale_modulate(x, scale, lengths)

        # Verify first segment
        expected_first = x[:3] * (1 + scale[0])
        assert torch.allclose(y[:3], expected_first)

        # Verify second segment
        expected_second = x[3:5] * (1 + scale[1])
        assert torch.allclose(y[3:5], expected_second)


class TestGateModulate:
    """Test cases for gate_modulate function."""

    def test_basic_operation(self):
        """Test basic gate modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        gate = torch.tensor([0.5, 2.0])
        lengths = [2, 1]

        y = gate_modulate(x, gate, lengths)

        # First segment: [1,2] * 0.5 = [0.5, 1.0]
        # Second segment: [3] * 2.0 = [6.0]
        # Remaining: [0]
        expected = torch.tensor([0.5, 1.0, 6.0, 0.0])
        assert torch.allclose(y, expected)

    def test_with_indices(self):
        """Test modulation with custom indices."""
        x = torch.tensor([1.0, 2.0, 3.0])
        gate = torch.tensor([0.0, 1.0, 0.5])
        lengths = [2, 1]
        indices = [2, 0]

        y = gate_modulate(x, gate, lengths, indices)

        # First segment: [1,2] * 0.5 = [0.5, 1.0]
        # Second segment: [3] * 0.0 = [0.0]
        expected = torch.tensor([0.5, 1.0, 0.0])
        assert torch.allclose(y, expected)

    def test_zero_gate(self):
        """Test with zero gate values."""
        x = torch.tensor([1.0, 2.0, 3.0])
        gate = torch.tensor([0.0, 0.0])
        lengths = [2, 1]

        y = gate_modulate(x, gate, lengths)

        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(y, expected)

    def test_multidimensional_tensor(self):
        """Test with multidimensional tensor."""
        x = torch.randn(4, 2)
        gate = torch.randn(2, 2)
        lengths = [2, 2]

        y = gate_modulate(x, gate, lengths)

        # Verify first segment
        expected_first = x[:2] * gate[0]
        assert torch.allclose(y[:2], expected_first)

        # Verify second segment
        expected_second = x[2:4] * gate[1]
        assert torch.allclose(y[2:4], expected_second)


class TestTanhModulate:
    """Test cases for tanh_modulate function."""

    def test_basic_operation(self):
        """Test basic tanh modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor([0.0, 1.0])
        lengths = [2, 1]

        y = tanh_modulate(x, scale, lengths)

        # First segment: [1,2] * tanh(0.0) = [0.0, 0.0]
        # Second segment: [3] * tanh(1.0) ≈ [3 * 0.7616]
        # Remaining: [0]
        expected = torch.tensor([0.0, 0.0, 3.0 * torch.tanh(torch.tensor(1.0)), 0.0])
        assert torch.allclose(y, expected)

    def test_with_indices(self):
        """Test modulation with custom indices."""
        x = torch.tensor([2.0, 4.0])
        scale = torch.tensor([0.0, 2.0, -2.0])
        lengths = [1, 1]
        indices = [1, 2]

        y = tanh_modulate(x, scale, lengths, indices)

        # First segment: [2] * tanh(2.0)
        # Second segment: [4] * tanh(-2.0)
        expected = torch.tensor(
            [2.0 * torch.tanh(torch.tensor(2.0)), 4.0 * torch.tanh(torch.tensor(-2.0))]
        )
        assert torch.allclose(y, expected)

    def test_large_values(self):
        """Test with large scale values (tanh saturation)."""
        x = torch.tensor([1.0, 2.0])
        scale = torch.tensor([10.0, -10.0])
        lengths = [1, 1]

        y = tanh_modulate(x, scale, lengths)

        # tanh(10) ≈ 1.0, tanh(-10) ≈ -1.0
        expected = torch.tensor(
            [
                1.0 * torch.tanh(torch.tensor(10.0)),
                2.0 * torch.tanh(torch.tensor(-10.0)),
            ]
        )
        assert torch.allclose(y, expected, atol=1e-6)

    def test_multidimensional_tensor(self):
        """Test with multidimensional tensor."""
        x = torch.randn(3, 2)
        scale = torch.randn(2, 2)
        lengths = [2, 1]

        y = tanh_modulate(x, scale, lengths)

        # Verify first segment
        expected_first = x[:2] * scale[0].tanh()
        assert torch.allclose(y[:2], expected_first)

        # Verify second segment
        expected_second = x[2:3] * scale[1].tanh()
        assert torch.allclose(y[2:3], expected_second)
