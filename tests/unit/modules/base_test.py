# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import torch
from torch import nn
import pytest
from jaxtyping import Float
from typing import Tuple
from protmyth.modules.base import BaseModule

# Type alias for a floating-point tensor with arbitrary dimensions
_ForwardReturnType = Float[torch.Tensor, "..."]


class LinearTestModule(BaseModule):
    """Helper class for testing the BaseModule class.

    Parameters
    ----------
    input_embed : int
        Number of features in the input.
    output_embed : int
        Number of features in the output.
    """

    def __init__(self, input_embed: int, output_embed: int):
        super().__init__()
        self.linear = nn.Linear(input_embed, output_embed)

    def forward(self, x: _ForwardReturnType) -> _ForwardReturnType:
        """Forward pass of the LinearTestModule.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The transformed output tensor.
        """
        return self.linear(x)


@pytest.mark.parametrize(
    "batch_shape, input_embed, output_embed",
    [
        ((4, 8), 24, 32),
        ((31,), 17, 8),
    ],
)
def test_linear_base_module(batch_shape: Tuple[int, ...],
                            input_embed: int,
                            output_embed: int) -> None:
    """Test the linear transformation of the LinearTestModule.

    Parameters
    ----------
    batch_shape : Tuple[int, ...]
        Shape representing the batch dimensions, excluding the feature dimension.
    input_embed : int
        Number of input features.
    output_embed : int
        Number of output features.
    """
    # Create an instance of the module
    module = LinearTestModule(input_embed, output_embed)

    # Check if the created module is a subclass of BaseModule
    assert isinstance(module, BaseModule), "Expected LinearTestModule to inherit from BaseModule."

    # Generate a random input tensor with the specified shape
    x = torch.randn(*batch_shape, input_embed)

    # Perform forward pass
    output = module.forward(x)

    # Determine the expected output shape
    expected_shape = (*batch_shape, output_embed)

    # Validate the shape of the output
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
