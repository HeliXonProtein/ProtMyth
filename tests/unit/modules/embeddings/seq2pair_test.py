# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

'''
"""Common layers test
"""

import pytest
import torch
from typing import Union
from jaxtyping import Float
from protmyth.modules.common.layers import DistanceToBins  # Adjust the import path as necessary


@pytest.mark.parametrize(
    "dist_min, dist_max, num_bins, below_bin, above_bin, dist_input, one_hot, expected_output",
    [
        # Test case 1: Basic binning with below and above bins
        (
            0.0, 10.0, 7, True, True,
            torch.tensor([[-1.0, 0.5, 2.5, 5.0, 7.5, 10.0, 12.0]]),
            False,
            torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        ),
        # Test case 2: No below bin, with above bin
        (
            0.0, 10.0, 6, False, True,
            torch.tensor([[-1.0, 0.5, 2.5, 5.0, 7.5, 10.0, 12.0]]),
            False,
            torch.tensor([[0, 0, 1, 2, 3, 4, 5]])
        ),
        # Test case 3: With below bin, no above bin
        (
            0.0, 10.0, 6, True, False,
            torch.tensor([[-1.0, 0.5, 2.5, 5.0, 7.5, 10.0, 12.0]]),
            False,
            torch.tensor([[0, 1, 2, 3, 4, 5, 5]])
        ),
        # Test case 4: No below bin, no above bin
        (
            0.0, 10.0, 5, False, False,
            torch.tensor([[-1.0, 0.5, 2.5, 5.0, 7.5, 10.0, 12.0]]),
            False,
            torch.tensor([[0, 0, 1, 2, 3, 4, 4]])
        ),
        # Test case 5: One-hot encoding with below and above bins
        (
            0.0, 10.0, 7, True, True,
            torch.tensor([[-1.0, 0.5, 2.5, 5.0, 7.5, 10.0, 12.0]]),
            True,
            torch.tensor([
                [[1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]]
            ])
        ),
        # Add more test cases as needed
    ]
)
def test_distance_to_bins(
    dist_min: float,
    dist_max: float,
    num_bins: int,
    below_bin: bool,
    above_bin: bool,
    dist_input: Float[torch.Tensor, "batch seq_len"],
    one_hot: bool,
    expected_output: Union[Float[torch.Tensor, "batch seq_len"],
                           Float[torch.Tensor, "batch seq_len num_bins"]]
) -> None:
    """
    Test the DistanceToBins module with various configurations.

    Parameters
    ----------
    dist_min : float
        The lower bound of the distance range.
    dist_max : float
        The upper bound of the distance range.
    num_bins : int
        The number of bins to divide the distance range into.
    below_bin : bool
        Whether to include a bin for values below `dist_min`.
    above_bin : bool
        Whether to include a bin for values above `dist_max`.
    dist_input : Float[torch.Tensor, "batch seq_len"]
        Input tensor containing distances to be discretized.
    one_hot : bool
        Whether to return a one-hot encoded tensor.
    expected_output : torch.Tensor
        The expected output tensor.
    """
    device = torch.device('cpu')

    # Initialize the DistanceToBins module
    module = DistanceToBins(
        dist_min=dist_min,
        dist_max=dist_max,
        num_bins=num_bins,
        below_bin=below_bin,
        above_bin=above_bin,
        device=device
    )

    # Move inputs to the correct device
    dist_input = dist_input.to(device)
    expected_output = expected_output.to(device)

    # Perform the forward pass
    output = module(dist_input, one_hot=one_hot)

    # Assert the output matches the expected output
    assert torch.equal(output, expected_output), f"Expected output {expected_output}, but got {output}"
'''