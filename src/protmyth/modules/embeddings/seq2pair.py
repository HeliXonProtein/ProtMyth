# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!


"""embeddings embeddings
"""

import torch
from torch import nn
import torch.nn.functional as F

from jaxtyping import Float, Bool
import einops

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module



@register_module("embeddings")
class RelativePositionalEmbedding(BaseModule[Float[torch.Tensor, "..."]]):
    """
    A module for generating relative positional embeddings used in attention mechanisms.

    Parameters
    ----------
    att_embed_dim : int
        The dimension for the attention embeddings.
    relpos_len : int, optional
        The maximum relative position length, by default 32.

    Attributes
    ----------
    pair_activations : nn.Linear
        Linear layer to process the relative position features.
    """

    def __init__(self, att_embed_dim: int, relpos_len: int = 32) -> None:
        super().__init__()
        self.relpos_len = relpos_len
        self.pair_activations = nn.Linear(2 * relpos_len + 1, att_embed_dim)

    def forward(self,
                Z: Float[torch.Tensor, "... Z z_dim"]
                ) -> Float[torch.Tensor, "... Z embed_dim"]:
        """
        Forward pass for the RelativePositionalEmbedding.

        Parameters
        ----------
        Z : torch.Tensor
            Input tensor of size [..., Z, z_dim].

        Returns
        -------
        torch.Tensor
            Tensor with relative positional embeddings.
        """

        # Assuming this is part of a method within a class
        B, L = Z.shape[:2]

        # Create relative position indices
        position_indices = torch.arange(L, device=Z.device)
        di = position_indices.unsqueeze(0)  # Shape: [1, L]
        dj = position_indices.unsqueeze(1)  # Shape: [L, 1]
        d = di - dj  # Shape: [L, L]

        # Clip the relative positions to the specified range
        d = torch.clamp(d, -self.relpos_len, self.relpos_len)
        d = self.relpos_len - d  # Offset to maintain consistency with AlphaFold

        # Create one-hot encoded relative position features
        relpos_onehot = F.one_hot(d, num_classes=2 * self.relpos_len + 1).float()

        # Expand the one-hot encoded features to match batch size
        relpos_feat = einops.repeat(relpos_onehot, 'l1 l2 c -> b l1 l2 c', b=B).contiguous()

        return self.pair_activations(relpos_feat)



@register_module("embeddings")
class PairEmbedder(BaseModule[Float[torch.Tensor, "..."]]):
    """
    A module for embedding pairs using attention mechanisms.

    Parameters
    ----------
    n_head : int
        Number of attention heads.
    att_embed_dim : int
        Dimension of the attention embeddings.

    Attributes
    ----------
    layernorm : nn.LayerNorm
        Layer normalization applied to the pairs.
    linear : nn.Linear
        Linear transformation of the normalized pairs.
    dropout : nn.Dropout
        Dropout layer applied after the linear transformation.
    """

    def __init__(self, n_head: int, att_embed_dim: int) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(n_head)
        self.linear = nn.Linear(n_head, att_embed_dim)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self,
                pair: Float[torch.Tensor, "... Z z_dim"]
                ) -> Float[torch.Tensor, "... Z out_dim"]:
        """
        Forward pass for the PairEmbedder.

        Parameters
        ----------
        pair : torch.Tensor
            Input tensor of pair features with size [..., Z, z_dim].

        Returns
        -------
        torch.Tensor
            Output tensor with embedded pairs.
        """
        z = self.layernorm(pair)
        z = self.linear(z)
        z = self.dropout(z)
        return z