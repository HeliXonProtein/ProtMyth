# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""seq2node embeddings, this module contains all positional embeddings in 1 dimension
SinusoidalPositionalEmbedding,
LearnedPositionalEmbedding
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from jaxtyping import Float, Bool
from typing import Optional
import torchviz
from graphviz import Digraph
import einops
from collections.abc import Sequence

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module

@register_module("seq2node")
class SinusoidalPositionalEmbedding(BaseModule[Float[torch.Tensor, "..."]]):
    """SinusoidalPositionalEmbedding
    """
    def __init__(
            self, 
            embed_dim: int, 
            padding_idx: int, 
            learned: bool=False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(
            self, 
            x: Float[torch.Tensor, "... X x_dim"],
    ) -> Float[torch.Tensor, "... X out_dim"]:
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

    def make_positions(
            self,
            x: Float[torch.Tensor, "... X x_dim"], 
    ) -> Float[torch.Tensor, "... X out_dim"]:
        mask = x.ne(self.padding_idx)
        range_buf = (
            torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        )
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(
            self, 
            num_embeddings: int,
    ) -> Float[torch.Tensor, "... Q out_dim"]:
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb
    
    def make_graph(
        self,
        batch_dims: Sequence[int],
        device: torch.device,
    ) -> Digraph:
        """Make a graph of the attention module.
        Returns:
            Output Digraph: the graph of the Relative_Positional_Embedding module with random initialization.
        """
        z_data = torch.randn(list(batch_dims) + [1], device=device)
        output = self.forward(z_data)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))


@register_module("seq2node")
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            padding_idx: int
    ) -> None:
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(
            self, 
            input: Float[torch.Tensor, "... Q q_dim"],
        ) -> Float[torch.Tensor, "... Q out_dim"]:
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
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

    def make_graph(
        self,
        batch_dims: Sequence[int],
        device: torch.device,
    ) -> Digraph:
        """Make a graph of the attention module.
        Returns:
            Output Digraph: the graph of the Relative_Positional_Embedding module with random initialization.
        """
        z_data = torch.randn(list(batch_dims) + [1], device=device)
        output = self.forward(z_data)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))


@register_module("seq2node")
class NodeEmbedder(BaseModule[Float[torch.Tensor, "..."]]):
    def __init__(
            self, 
            feat_dim: int, 
            embed_dim: int
    ) -> None:
        super(NodeEmbedder, self).__init__()
        self.preprocess_feat = nn.Linear(feat_dim, embed_dim)

    def forward(
            self, 
            node_feat: Float[torch.Tensor, "... Q q_dim"],
    ) -> Float[torch.Tensor, "... Q q_dim"]:
        node_embed = self.preprocess_feat(node_feat)
        return node_embed

    def make_graph(
        self,
        batch_dims: Sequence[int],
        device: torch.device,
    ) -> Digraph:
        """Make a graph of the attention module.
        Returns:
            Output Digraph: the graph of the Relative_Positional_Embedding module with random initialization.
        """
        z_data = torch.randn(list(batch_dims) + [self.feat_dim], device=device)
        output = self.forward(z_data)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))