# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains implementations of ESM1 and ESM1b layer normalization
for the ProtMyth framework.

Layer normalization is a technique to normalize the inputs across the features
for each example in a batch, which helps stabilize and accelerate training.

Classes:
    - ESM1LayerNorm: Implements layer normalization for the ESM1 model.
    - ESM1bLayerNorm: Implements layer normalization for the ESM1b model.
"""

import torch
from torch import nn
from jaxtyping import Float
from protmyth.modules.base import BaseModule
from typing import Sequence
from typing import Optional


class ESM1LayerNorm(BaseModule):
    """Layer normalization for ESM1 model.

    Parameters
    ----------
    hidden_size : int or Sequence[int]
        Size of the hidden layer.
    eps : float, optional
        A small value added to variance to avoid division by zero (default is 1e-12).
    affine : bool, optional
        If True, the module has learnable affine parameters (default is True).
    """

    def __init__(self, hidden_size: int | Sequence[int], eps: float = 1e-12, affine: bool = True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.hidden_size))
            self.bias = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, *args: Float[torch.Tensor, "batch ..."], **kwargs) -> Float[torch.Tensor, "batch ..."]:
        """Perform forward pass of the layer normalization.

        Parameters
        ----------
        args : torch.Tensor
            Input tensor of shape (batch_size, ... , hidden_size).
        **kwargs: Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        # Compute mean and variance across specified dimensions
        x = args[0]
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)

        if self.affine:
            x = (self.weight * x) + self.bias
        return x


class ESM1bLayerNorm(BaseModule):
    """Layer normalization for ESM1b model.

    Parameters
    ----------
    hidden_size : int or Sequence[int]
        Size of the hidden layer.
    eps : float, optional
        A small value added to variance to avoid division by zero (default is 1e-12).
    affine : bool, optional
        If True, the module has learnable affine parameters (default is True).
    """

    def __init__(self, hidden_size: int | Sequence[int], eps: float = 1e-12, affine: bool = True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)

        self.weight: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.hidden_size))
            self.bias = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, *args: Float[torch.Tensor, "batch ..."], **kwargs) -> Float[torch.Tensor, "batch ..."]:
        """Perform forward pass of the layer normalization.

        Parameters
        ----------
        args : torch.Tensor
            Input tensor of shape (batch_size, ... , hidden_size).
        **kwargs: Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        # Compute mean and variance across specified dimensions
        x = args[0]
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)

        if self.affine:
            x = (self.weight * x) + self.bias
        return x
