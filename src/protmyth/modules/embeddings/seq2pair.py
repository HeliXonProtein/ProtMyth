# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!


"""seq2pair embeddings
"""

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

@register_module("seq2pair")
class Relative_Positional_Embedding(BaseModule[Float[torch.Tensor, "..."]]):
    def __init__(
                self, 
                att_embed_dim: int,
                relpos_len: int=32,
    ) -> None:
        super(Relative_Positional_Embedding, self).__init__()
        self.relpos_len = relpos_len
        self.pair_activations = nn.Linear(2 * relpos_len + 1, att_embed_dim)

    def forward(
            self,
            Z: Float[torch.Tensor, "... Z z_dim"],
    ) -> Float[torch.Tensor, "... Z #z_dim"]:
        with torch.no_grad():
            B, L = Z.shape[:2]
            di = torch.arange(0, L, layout=Z.layout, device=Z.device).unsqueeze(0)
            dj = torch.arange(0, L, layout=Z.layout, device=Z.device).unsqueeze(1)
            d = di - dj
            d[d > self.relpos_len] = self.relpos_len
            d[d < -self.relpos_len] = -self.relpos_len
            d = 32 - d  # to keep same with alphafold

            relpos_onehot = torch.eye(2 * self.relpos_len + 1)[d]
            relpos_feat = relpos_onehot.expand(B, -1, -1, -1).contiguous()
        return self.pair_activations(relpos_feat)
    
    def make_graph(
        self,
        batch_dims: Sequence[int],
        device: torch.device,
    ) -> Digraph:
        """Make a graph of the attention module.
        Returns:
            Output Digraph: the graph of the Relative_Positional_Embedding module with random initialization.
        """
        z_data = torch.randn(list(batch_dims) + [self.relpos_len], device=device)
        output = self.forward(z_data)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))


@register_module("seq2pair")
class PairEmbedder(BaseModule[Float[torch.Tensor, "..."]]):
    def __init__(self, 
                n_head: int, 
                att_embed_dim: int
    ) -> None:
        super(PairEmbedder, self).__init__()
        self.layernorm = nn.LayerNorm(n_head)
        self.linear = nn.Linear(n_head, att_embed_dim)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, 
                pair: Float[torch.Tensor, "... Z z_dim"],
    ) -> Float[torch.Tensor, "... Z out_dim"]:
        z = self.layernorm(pair)
        z = self.linear(z)
        z = self.dropout(z)
        return z

    def make_graph(
        self,
        batch_dims: Sequence[int],
        device: torch.device,
    ) -> Digraph:
        """Make a graph of the attention module.
        Returns:
            Output Digraph: the graph of the Relative_Positional_Embedding module with random initialization.
        """
        z_data = torch.randn(list(batch_dims) + [self.n_head], device=device)
        output = self.forward(z_data)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))