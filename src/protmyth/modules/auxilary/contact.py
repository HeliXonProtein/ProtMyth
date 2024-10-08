# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Auxilary head like contact
"""

import torch
from torch import nn
import einops

from jaxtyping import Float
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


@register_module("auxilary")
class ContactPredictHead(BaseModule[Float[torch.Tensor, "..."]]):
    """Predict distogram logits of pairwise residues.

    Before contact loss.
    """

    def __init__(
        self,
        d_pair: int,
        dist_min: float,
        dist_max: float,
        num_bins: int,
        module_config: config.ModuleConfig | None = None,
    ):
        """
        Args:
            d_pair: Dimension of pair features.
            dist_min: Lower bound of distogram bins.
            dist_max: Upper bound of distogram bins.
            num_bins: Number of distogram bins.
        """
        super().__init__()
        self.linear_logits = nn.Linear(d_pair, num_bins)
        self.distogram_fn = common.layers.DistanceToBins(dist_min, dist_max, num_bins, module_config)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair: Pair features, (B, N, N).

        Returns:
            Un-normalized logits of predicted distogram, (B, N, N, num_bins).
        """
        half_logits = self.linear_logits(pair)
        logits = half_logits + einops.rearrange(half_logits, "... i j d -> ... j i d")
        return logits
