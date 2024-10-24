# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Auxilary losses like bert loss
"""

import torch
from torch import nn
import torch.nn.functional as F

import math
from jaxtyping import Float

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


def gelu(
        x: torch.Tensor
        ) -> torch.Tensor:
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    Parameters
    ----------
    x: Float[torch.Tensor, "...Z f_dim"]
        A tensor containing the features.

    Returns
    -------
    torch.Tensor
        A tensor after gelu
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@register_module("auxilary")
class RobertaLMHead(BaseModule[Float[torch.Tensor, "..."]]):
    """Head for masked language modeling."""

    def __init__(
            self,
            weight: Float[torch.Tensor, "..."],
            embed_dim: int = 1280,
            output_dim: int = 33,
            ) -> None:
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
            self,
            features: Float[torch.Tensor, "..."]
            ) -> Float[torch.Tensor, "..."]:
        """Implementation of the forward function.

        Parameters
        ----------
        features: Float[torch.Tensor, "..."]
            A tensor containing the features.

        Returns
        -------
        torch.Tensor
            A tensor after robertahead
        """
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
