# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Common mask test
"""

import pytest
import torch
from protmyth.modules.common.mask import masked_mean


@pytest.mark.parametrize(
    "value, mask, dim, eps, return_masked, expected_mean, expected_masked",
    [
        # Test case 1: Simple case with 1D tensor
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            -1,
            1e-10,
            False,
            torch.tensor(2.0),  # Expected mean
            None  # Expected masked value not returned
        ),
        # Test case 2: 2D tensor with return_masked=True
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
            0,
            1e-10,
            True,
            torch.tensor([1.0, 3.0]),  # Expected mean
            torch.tensor([[1.0, 1.0], [0.0, 2.0]])  # Expected masked value
        ),
        # Test case 3: Different dimensions
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            1,
            1e-10,
            False,
            torch.tensor([1.5, 3.5]),  # Expected mean
            None  # Expected masked value not returned
        ),
    ]
)
def test_masked_mean(value, mask, dim, eps, return_masked, expected_mean, expected_masked):
    """Test the masked_mean function with various inputs.

    Parameters
    ----------
    value : torch.Tensor
        The input tensor to compute the mean from.
    mask : torch.Tensor
        The binary mask tensor of the same shape as `value`.
    dim : int or tuple of int
        The dimensions along which the mean is computed.
    eps : float
        A small constant for numerical stability.
    return_masked : bool
        Whether to return the masked value alongside the mean.
    expected_mean : torch.Tensor
        The expected mean result.
    expected_masked : torch.Tensor or None
        The expected masked tensor result, if `return_masked` is True.
    """
    result = masked_mean(value, mask, dim, eps, return_masked)

    if return_masked:
        mean_value, masked_value = result
        assert torch.allclose(mean_value, expected_mean, atol=1e-6), "Mean value mismatch"
        assert torch.allclose(masked_value, expected_masked, atol=1e-6), "Masked value mismatch"
    else:
        assert isinstance(result, torch.Tensor), "Result should be a torch.Tensor"
        assert torch.allclose(result, expected_mean, atol=1e-6), "Mean value mismatch"
