# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains common layers used in ProtMyth."""

import einops
import torch
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F

from protmyth.modules import base, register

@register.register_module("common")
class DistanceToBins(base.BaseModule[Float[torch.Tensor, "..."]]):
    """Module for discretizing distance to bins."""

    def __init__(self,
                 dist_min: float,
                 dist_max: float,
                 num_bins: int) -> None:
        """
        Initialize the DistanceToBins module.

        Parameters
        ----------
        dist_min : float
            Lower bound of bins.
        dist_max : float
            Upper bound of bins.
        num_bins : int
            Number of bins.
        module_config : config.ModuleConfig, optional
            Configuration for the module, by default None.
        """
        super().__init__()
        self.num_bins = num_bins
        breaks = torch.linspace(dist_min, dist_max, num_bins - 1, device=self.device)
        self.register_buffer("breaks", breaks)

    def forward(self,
                dist: Float[torch.Tensor, "..."],
                one_hot: bool = False
                ) -> Float[torch.Tensor, "..."]:
        """
        Discretize a tensor along a given dimension.

        Parameters
        ----------
        dist : Float[torch.Tensor, "..."]
            Tensor to discretize.
        dim : int, optional
            Dimension along which to discretize, should be -1 or dist.shape[dim] = 1.
            If dim = -1 and dist.shape[-1] != 1, dist will be unsqueezed to have a dimension of length 1.
        one_hot : bool, optional
            Whether to return a one-hot distogram or a long tensor of bin indices.

        Returns
        -------
        torch.Tensor
            A one-hot distogram if one_hot is True, otherwise a long tensor of bin indices.

        Raises
        ------
        ValueError
            If dim != -1 and dist.shape[dim] != 1.
        """
        if not isinstance(self.breaks, torch.Tensor):
            raise RuntimeError("breaks is not a Tensor!")

        dist_reshaped = einops.rearrange(dist, "... -> ... 1")
        dist_bin = einops.reduce(dist_reshaped > self.breaks, '... b -> ...', 'sum').long()

        if one_hot:
            dist_bin = F.one_hot(dist_bin, self.num_bins)

        return dist_bin