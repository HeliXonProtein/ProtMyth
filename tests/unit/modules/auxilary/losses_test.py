# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""losses test
"""

import pytest
import torch
from jaxtyping import Float
from protmyth.modules.auxilary.losses import RobertaLMHead, ContactPredictionHead  # Adjust the import path as necessary


@pytest.mark.parametrize(
    "weight, embed_dim, output_dim, features, expected_output",
    [
        # Test case 1: basic dim=1 test
        (
            torch.tensor([[0.0]]),
            1, 1,
            torch.tensor([0.0]),
            torch.tensor([0.0])
        ),
        # Add more test cases as needed
    ]
)
def test_RobertaLMHead(
    embed_dim: int,
    output_dim: int,
    weight: Float[torch.Tensor, "..."],
    features: Float[torch.Tensor, "..."],
    expected_output: Float[torch.Tensor, "..."]
) -> None:
    """Test the RobertaLMHead module with various configurations.

    Parameters
    ----------
    embed_dim: int,
    output_dim: int,
    weight: Float[torch.Tensor, "..."],
    features: Float[torch.Tensor, "..."],
    expected_output: Float[torch.Tensor, "..."]
    """
    device = torch.device('cpu')

    # Initialize the RobertaLMHead module
    module = RobertaLMHead(
        weight=weight,
        embed_dim=embed_dim,
        output_dim=output_dim,
    )

    # Move inputs to the correct device
    features = features.to(device)
    expected_output = expected_output.to(device)

    # Perform the forward pass
    output = module(features)

    # Assert the output matches the expected output
    assert torch.equal(output, expected_output), f"Expected output {expected_output}, but got {output}"


@pytest.mark.parametrize(
    "in_features, prepend_bos, append_eos, bias, eos_idx, tokens, attentions, expected_output",
    [
        # Test case 1: basic dim=1 test
        (
            1, False, False, False, 0,
            torch.tensor([2.0]),
            torch.tensor([[[[[1.0]]]]]),
            torch.tensor([[[0.5]]])
        ),
        # Add more test cases as needed
    ]
)
def test_ContactPredictionHead(
    in_features: int,
    prepend_bos: bool,
    append_eos: bool,
    bias: bool,
    eos_idx: int,
    tokens: Float[torch.Tensor, "..."],
    attentions: Float[torch.Tensor, "..."],
    expected_output: Float[torch.Tensor, "..."]
) -> None:
    """Test the ContactPredictionHead module with various configurations.

    Parameters
    ----------
    in_features: int,
    prepend_bos: int,
    append_eos: bool,
    bias: bool,
    eos_idx: Optional[int],
    tokens: Float[torch.Tensor, "..."],
    attentions: Float[torch.Tensor, "..."],
    expected_output: Float[torch.Tensor, "..."]
    """
    device = torch.device('cpu')

    # Initialize the ContactPredictionHead module
    module = ContactPredictionHead(
        in_features=in_features,
        prepend_bos=prepend_bos,
        append_eos=append_eos,
        bias=bias,
        eos_idx=eos_idx
    )

    # Move inputs to the correct device
    tokens = tokens.to(device)
    attentions = attentions.to(device)
    expected_output = expected_output.to(device)

    # Perform the forward pass
    output = module(tokens, attentions)

    # Assert the output matches the expected output
    assert torch.equal(output, expected_output), f"Expected output {expected_output}, but got {output}"
