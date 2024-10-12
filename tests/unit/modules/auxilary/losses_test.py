# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Common layers test
"""

import pytest
import torch
from typing import Union
from jaxtyping import Float
from protmyth.modules.auxilary.losses import RobertaLMHead  # Adjust the import path as necessary


@pytest.mark.parametrize(
    "embed_dim, output_dim, weight, features, expected_output",
    [
        # Test case 1: basic dim=1 test
        (
            1, 1,
            torch.tensor([[0, 1, 2, 3, 4, 5, 6]]),
            torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        ),
        # Add more test cases as needed
    ]
)
def test_RobertaLMHead(
    embed_dim: int,
    output_dim: int,
    weight: Float[torch.Tensor, "...Z w_dim"],
    features: Float[torch.Tensor, "...Z f_dim"],
    expected_output: Float[torch.Tensor, "...Z w_dim"]
) -> None:
    """
    Test the RobertaLMHead module with various configurations.

    Parameters
    ----------
    expected_output : torch.Tensor
        The expected output tensor.
    """
    device = torch.device('cpu')

    # Initialize the DistanceToBins module
    module = RobertaLMHead(
        embed_dim=embed_dim,
        output_dim=output_dim,
        weight=weight
    )

    # Move inputs to the correct device
    features = features.to(device)
    expected_output = expected_output.to(device)

    # Perform the forward pass
    output = module(features)

    # Assert the output matches the expected output
    assert torch.equal(output, expected_output), f"Expected output {expected_output}, but got {output}"
