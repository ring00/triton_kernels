"""
Comprehensive unit tests for modulation kernel implementations.
"""

import pytest
import torch

from triton_kernels.modulation import (
    gate_modulate_torch,
    gate_modulate_triton,
    scale_modulate_torch,
    scale_modulate_triton,
    scale_shift_modulate_torch,
    scale_shift_modulate_triton,
    tanh_modulate_torch,
    tanh_modulate_triton,
)


class TestScaleShiftModulateTorch:
    """Test cases for PyTorch reference scale-shift modulate implementation."""

    def test_basic_operation(self):
        """Test basic scale and shift modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        scale = torch.tensor([0.5, 1.0])
        shift = torch.tensor([1.0, 2.0])
        lengths = [3, 2]

        y = scale_shift_modulate_torch(x, scale, shift, lengths)

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

        y = scale_shift_modulate_torch(x, scale, shift, lengths, indices)

        # First segment: [1,2] * (1+1.0) + 1 = [3.0, 5.0]
        # Second segment: [3,4] * (1+2.0) + 2 = [11.0, 14.0]
        expected = torch.tensor([3.0, 5.0, 11.0, 14.0])
        assert torch.allclose(y, expected)

    def test_multidimensional_tensor(self):
        """Test with multidimensional tensor."""
        x = torch.randn(6, 4)
        scale = torch.randn(2, 4)
        shift = torch.randn(2, 4)
        lengths = [3, 2]

        y = scale_shift_modulate_torch(x, scale, shift, lengths)

        # Verify first segment
        expected_first = x[:3] * (1 + scale[0]) + shift[0]
        assert torch.allclose(y[:3], expected_first)

        # Verify second segment
        expected_second = x[3:5] * (1 + scale[1]) + shift[1]
        assert torch.allclose(y[3:5], expected_second)

        # Verify zero padding
        assert torch.allclose(y[5:], torch.zeros(1, 4))


class TestScaleModulateTorch:
    """Test cases for PyTorch reference scale modulate implementation."""

    def test_basic_operation(self):
        """Test basic scale modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        scale = torch.tensor([0.5, 1.0])
        lengths = [3, 1]

        y = scale_modulate_torch(x, scale, lengths)

        # First segment: [1,2,3] * (1+0.5) = [1.5, 3.0, 4.5]
        # Second segment: [4] * (1+1.0) = [8.0]
        # Remaining: [0]
        expected = torch.tensor([1.5, 3.0, 4.5, 8.0, 0.0])
        assert torch.allclose(y, expected)

    def test_multidimensional_tensor(self):
        """Test with multidimensional tensor."""
        x = torch.randn(5, 3)
        scale = torch.randn(2, 3)
        lengths = [3, 2]

        y = scale_modulate_torch(x, scale, lengths)

        # Verify first segment
        expected_first = x[:3] * (1 + scale[0])
        assert torch.allclose(y[:3], expected_first)

        # Verify second segment
        expected_second = x[3:5] * (1 + scale[1])
        assert torch.allclose(y[3:5], expected_second)


class TestGateModulateTorch:
    """Test cases for PyTorch reference gate modulate implementation."""

    def test_basic_operation(self):
        """Test basic gate modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        gate = torch.tensor([0.5, 2.0])
        lengths = [2, 1]

        y = gate_modulate_torch(x, gate, lengths)

        # First segment: [1,2] * 0.5 = [0.5, 1.0]
        # Second segment: [3] * 2.0 = [6.0]
        # Remaining: [0]
        expected = torch.tensor([0.5, 1.0, 6.0, 0.0])
        assert torch.allclose(y, expected)

    def test_zero_gate(self):
        """Test with zero gate values."""
        x = torch.tensor([1.0, 2.0, 3.0])
        gate = torch.tensor([0.0, 0.0])
        lengths = [2, 1]

        y = gate_modulate_torch(x, gate, lengths)

        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(y, expected)


class TestTanhModulateTorch:
    """Test cases for PyTorch reference tanh modulate implementation."""

    def test_basic_operation(self):
        """Test basic tanh modulation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor([0.0, 1.0])
        lengths = [2, 1]

        y = tanh_modulate_torch(x, scale, lengths)

        # First segment: [1,2] * tanh(0.0) = [0.0, 0.0]
        # Second segment: [3] * tanh(1.0) ≈ [3 * 0.7616]
        # Remaining: [0]
        expected = torch.tensor([0.0, 0.0, 3.0 * torch.tanh(torch.tensor(1.0)), 0.0])
        assert torch.allclose(y, expected)

    def test_large_values(self):
        """Test with large scale values (tanh saturation)."""
        x = torch.tensor([1.0, 2.0])
        scale = torch.tensor([10.0, -10.0])
        lengths = [1, 1]

        y = tanh_modulate_torch(x, scale, lengths)

        # tanh(10) ≈ 1.0, tanh(-10) ≈ -1.0
        expected = torch.tensor(
            [
                1.0 * torch.tanh(torch.tensor(10.0)),
                2.0 * torch.tanh(torch.tensor(-10.0)),
            ]
        )
        assert torch.allclose(y, expected, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestScaleShiftModulateTriton:
    """Test cases for Triton scale-shift modulate implementation."""

    def test_basic_operation(self):
        """Test basic scale and shift modulation with Triton."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        scale = torch.tensor([0.5, 1.0], device="cuda")
        shift = torch.tensor([1.0, 2.0], device="cuda")
        lengths = [3, 2]

        y = scale_shift_modulate_triton(x, scale, shift, lengths)

        expected = torch.tensor([2.5, 4.0, 5.5, 10.0, 12.0, 0.0], device="cuda")
        assert torch.allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_matches_torch(self):
        """Test that Triton matches PyTorch reference."""
        x = torch.randn(100, device="cuda")
        scale = torch.randn(3, device="cuda")
        shift = torch.randn(3, device="cuda")
        lengths = [30, 40, 20]

        y_torch = scale_shift_modulate_torch(
            x.cpu(), scale.cpu(), shift.cpu(), lengths
        ).cuda()
        y_triton = scale_shift_modulate_triton(x, scale, shift, lengths)

        assert torch.allclose(y_torch, y_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestScaleModulateTriton:
    """Test cases for Triton scale modulate implementation."""

    def test_basic_operation(self):
        """Test basic scale modulation with Triton."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        scale = torch.tensor([0.5, 1.0], device="cuda")
        lengths = [3, 1]

        y = scale_modulate_triton(x, scale, lengths)

        expected = torch.tensor([1.5, 3.0, 4.5, 8.0, 0.0], device="cuda")
        assert torch.allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_matches_torch(self):
        """Test that Triton matches PyTorch reference."""
        x = torch.randn(100, device="cuda")
        scale = torch.randn(3, device="cuda")
        lengths = [30, 40, 20]

        y_torch = scale_modulate_torch(x.cpu(), scale.cpu(), lengths).cuda()
        y_triton = scale_modulate_triton(x, scale, lengths)

        assert torch.allclose(y_torch, y_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestGateModulateTriton:
    """Test cases for Triton gate modulate implementation."""

    def test_basic_operation(self):
        """Test basic gate modulation with Triton."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        gate = torch.tensor([0.5, 2.0], device="cuda")
        lengths = [2, 1]

        y = gate_modulate_triton(x, gate, lengths)

        expected = torch.tensor([0.5, 1.0, 6.0, 0.0], device="cuda")
        assert torch.allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_matches_torch(self):
        """Test that Triton matches PyTorch reference."""
        x = torch.randn(100, device="cuda")
        gate = torch.randn(3, device="cuda")
        lengths = [30, 40, 20]

        y_torch = gate_modulate_torch(x.cpu(), gate.cpu(), lengths).cuda()
        y_triton = gate_modulate_triton(x, gate, lengths)

        assert torch.allclose(y_torch, y_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestTanhModulateTriton:
    """Test cases for Triton tanh modulate implementation."""

    def test_basic_operation(self):
        """Test basic tanh modulation with Triton."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        scale = torch.tensor([0.0, 1.0], device="cuda")
        lengths = [2, 1]

        y = tanh_modulate_triton(x, scale, lengths)

        expected = torch.tensor(
            [0.0, 0.0, 3.0 * torch.tanh(torch.tensor(1.0)), 0.0], device="cuda"
        )
        assert torch.allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_matches_torch(self):
        """Test that Triton matches PyTorch reference."""
        x = torch.randn(100, device="cuda")
        scale = torch.randn(3, device="cuda")
        lengths = [30, 40, 20]

        y_torch = tanh_modulate_torch(x.cpu(), scale.cpu(), lengths).cuda()
        y_triton = tanh_modulate_triton(x, scale, lengths)

        assert torch.allclose(y_torch, y_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.cuda
class TestModulationComparison:
    """Comparison tests between PyTorch and Triton implementations."""

    def test_all_modulations_match(self):
        """Test that all modulation functions produce consistent results."""
        x = torch.randn(50, device="cuda")
        scale = torch.randn(2, device="cuda")
        shift = torch.randn(2, device="cuda")
        gate = torch.randn(2, device="cuda")
        lengths = [20, 25]

        # Test scale_shift_modulate
        y1_torch = scale_shift_modulate_torch(
            x.cpu(), scale.cpu(), shift.cpu(), lengths
        ).cuda()
        y1_triton = scale_shift_modulate_triton(x, scale, shift, lengths)
        assert torch.allclose(y1_torch, y1_triton, rtol=1e-2, atol=1e-2)

        # Test scale_modulate
        y2_torch = scale_modulate_torch(x.cpu(), scale.cpu(), lengths).cuda()
        y2_triton = scale_modulate_triton(x, scale, lengths)
        assert torch.allclose(y2_torch, y2_triton, rtol=1e-2, atol=1e-2)

        # Test gate_modulate
        y3_torch = gate_modulate_torch(x.cpu(), gate.cpu(), lengths).cuda()
        y3_triton = gate_modulate_triton(x, gate, lengths)
        assert torch.allclose(y3_torch, y3_triton, rtol=1e-2, atol=1e-2)

        # Test tanh_modulate
        y4_torch = tanh_modulate_torch(x.cpu(), scale.cpu(), lengths).cuda()
        y4_triton = tanh_modulate_triton(x, scale, lengths)
        assert torch.allclose(y4_torch, y4_triton, rtol=1e-2, atol=1e-2)
