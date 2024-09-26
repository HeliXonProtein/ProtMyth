# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""seq2node embeddings, this module contains all positional embeddings in 1 dimension
SinusoidalPositionalEmbedding,
LearnedPositionalEmbedding
"""

from protmyth.modules.register import register_module

import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Optional


@register_module("embeddings")
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.

    Parameters
    ----------
    num_embeddings : int
        The number of embeddings.
    embedding_dim : int
        The dimension of each embedding vector.
    padding_idx : int
        The index used for padding.

    Attributes
    ----------
    max_positions : int
        The maximum number of positions.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int]):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self,
                input: Int[torch.Tensor, "B L"],
                ) -> Float[torch.Tensor, "B L embed"]:
        """
        Forward pass for the LearnedPositionalEmbedding.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of size [batch_size, sequence_length].

        Returns
        -------
        torch.Tensor
            Output tensor with positional embeddings.
        """
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "  
                f"sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


@register_module("embeddings")
class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module generates sinusoidal positional embeddings.

    Parameters
    ----------
    embed_dim : int
        The dimension of each embedding vector.
    padding_idx : int
        The index used for padding.
    learned : bool, optional
        Whether to use learned embeddings, by default False.

    Attributes
    ----------
    embed_dim : int
        The dimension of each embedding vector.
    padding_idx : int
        The index used for padding.
    weights : Optional[torch.Tensor]
        The precomputed sinusoidal weights.
    """

    def __init__(self,
                 embed_dim: int,
                 padding_idx: int,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self,
                x: Int[torch.Tensor, "... L"],
                ) -> Float[torch.Tensor, "... L embed_dim"]:
        """
        Forward pass for the SinusoidalPositionalEmbedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of size [batch_size, sequence_length].

        Returns
        -------
        torch.Tensor
            Output tensor with sinusoidal positional embeddings.
        """
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return einops.rearrange(self.weights.index_select(0, positions.view(-1)), '(b s) d -> b s d', b=bsz).detach()

    def make_positions(self,
                       x: Int[torch.Tensor, "B L"],
                       ) -> Int[torch.Tensor, "B L"]:
        """
        Create position indices for the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of size [batch_size, sequence_length].

        Returns
        -------
        torch.Tensor
            Tensor containing position indices.
        """
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self,
                      num_embeddings: int,
                      ) -> Float[torch.Tensor, "L embed_dim"]:
        """
        Generate sinusoidal embeddings.

        Parameters
        ----------
        num_embeddings : int
            The number of embeddings to generate.

        Returns
        -------
        torch.Tensor
            Sinusoidal embeddings.
        """
        # Assuming this is part of a method within a class
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)

        # Create the exponential scale
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb_scale)

        # Create the position indices and compute the sinusoidal embeddings
        positions = torch.arange(num_embeddings, dtype=torch.float)
        emb = einops.rearrange(positions, 'n -> n 1') * einops.rearrange(emb, 'd -> 1 d')

        # Concatenate sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # If embed_dim is odd, zero pad the last dimension
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

            # Zero out the embeddings at the padding index, if specified
        if self.padding_idx is not None:
            emb = einops.rearrange(emb, 'n d -> n d')
            emb[self.padding_idx, :] = 0

        return emb