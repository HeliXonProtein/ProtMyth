# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains common layers used in ProtMyth."""

import torch
import einops
from jaxtyping import Float
from torch.nn import functional as F
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module
from typing import Union


@register_module("common")
class DistanceToBins(BaseModule[Float[torch.Tensor, "..."]]):
    """Module for discretizing distances into bins.

    This module converts continuous distance values into discrete bins, which can
    be useful for tasks like histogram-based feature extraction or quantization
    in neural networks. It also handles values below and above the specified range.
    """

    def __init__(self,
                 dist_min: float,
                 dist_max: float,
                 num_bins: int,
                 below_bin: bool = True,
                 above_bin: bool = True,
                 device: torch.device = torch.device('cpu')) -> None:
        """Initialize the DistanceToBins module.

        Parameters
        ----------
        dist_min : float
            The lower bound of the distance range.
        dist_max : float
            The upper bound of the distance range.
        num_bins : int
            The number of bins to divide the distance range into.
        below_bin : bool, optional
            If True, include a bin for values below `dist_min`.
        above_bin : bool, optional
            If True, include a bin for values above `dist_max`.
        device : torch.device, optional
            The device to store the breakpoints on.
        """
        super().__init__()
        self.num_bins = num_bins
        self.below_bin = below_bin
        self.above_bin = above_bin

        # Adjust the number of bins to account for below and above bins
        breaks = torch.linspace(dist_min, dist_max, num_bins - below_bin - above_bin + 1, device=device)
        # Register the breaks as a buffer
        self.register_buffer("breaks", breaks)

    def forward(self,
                dist: Float[torch.Tensor, "... seq_len"],
                one_hot: bool = False,
                ) -> Union[Float[torch.Tensor, "... seq_len"],
                           Float[torch.Tensor, "... seq_len num_bins"]]:
        """Discretize a tensor of distances into bins.

        Parameters
        ----------
        dist : Float[torch.Tensor, "... seq_len"]
            A tensor containing the distances to be discretized.
        one_hot : bool, optional
            If True, return a one-hot encoded tensor representing the bins.
            If False, return a tensor of bin indices.

        Returns
        -------
        torch.Tensor
            A tensor representing the discretized distances. If `one_hot` is True,
            the output is a one-hot encoded tensor with shape (..., seq_len, num_bins).
            Otherwise, it is a tensor of bin indices with shape (..., seq_len).
        """
        # Reshape distances to allow broadcasting with breaks
        dist_reshaped = einops.rearrange(dist, "... -> ... 1")

        # Compute bin indices by comparing distances with breakpoints
        dist_bin = einops.reduce(dist_reshaped > self.breaks, '... b -> ...', 'sum').long()

        # Assert type for mypy test
        assert isinstance(self.breaks, torch.Tensor), \
            f"Expected breaks to be torch.Tensor, but got {type(self.breaks).__name__}"

        # Adjust for below and above bins
        if not self.below_bin:
            dist_bin = torch.where(dist < self.breaks[0],
                                   torch.tensor(0, device=dist.device),
                                   dist_bin - 1)
        if not self.above_bin:
            dist_bin = torch.where(dist >= self.breaks[-1],
                                   torch.tensor(self.num_bins - 1, device=dist.device),
                                   dist_bin)
        if one_hot:
            # Convert bin indices to one-hot encoding
            dist_bin = F.one_hot(dist_bin, self.num_bins)

        return dist_bin
