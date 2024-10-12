# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains attention mechanisms used in ProtMyth. Based on the attention mechanism used,
we divide the attention mechanism into three categories: (https://arxiv.org/pdf/2203.14263)

1. Feature-Related:
    a. Multiplicity.
    b. Levels.
    c. Representations.
2. General:
    a. Scoring.
    b. Alignment.
    c. Dimensionality.
3. Query-Related:
    a. Type.
    b. Multiplicity.

We will implement the following attention mechanisms and transform them into protein based modules:

1. Place holder.
"""

import einops
import torch
from torch import nn
from jaxtyping import Float, Bool

from typing import Optional
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


@register_module("common")
class Attention(BaseModule[Float[torch.Tensor, "..."]]):
    """Common attention module, now it supports:
        1.Multi-head
        2.qk scaling
        3.attention mask
    """

    def __init__(
        self,
        q_dim: int = 32,
        kv_dim: int = 32,
        c: int = 8,
        n_head: int = 4,
        out_dim: int = 32,
        use_bias: bool = False,
        gating: bool = True,
    ) -> None:
        """Attention initialization.

        Args:
            q_dim: Size of input query features.
            kv_dim: Size of input key and value features.
            c: Size of channels per head.
            n_head: Number of heads.
            out_dim: Size of output features.
            use_bias: Whether to apply bias to qkv linear.
            gating: Whether to apply a sigmoid gating for output.
        """
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.c = c
        self.n_head = n_head
        self.out_dim = out_dim
        self.gating = gating

        self.q_linear = nn.Linear(
            in_features=q_dim,
            out_features=n_head * c,
            bias=use_bias,
        )
        self.kv_linear = nn.Linear(
            in_features=kv_dim,
            out_features=n_head * c * 2,
            bias=use_bias,
        )

        if gating:
            self.gating_linear = nn.Linear(
                in_features=q_dim,
                out_features=n_head * c,
                bias=True,
            )

        self.output_linear = nn.Linear(
            in_features=n_head * c,
            out_features=out_dim,
            bias=True,
        )

    def _qk_scale(
        self,
    ) -> Float[torch.Tensor, "..."]:
        """scaling for attention logits
        """
        return self.c ** 0.5

    def forward(
        self,
        q_data: Float[torch.Tensor, "... Q q_dim"],
        kv_data: Float[torch.Tensor, "... K kv_dim"],
        attn_mask: Optional[Bool[torch.Tensor, "... #Q #K"]] = None,
    ) -> Float[torch.Tensor, "... Q out_dim"]:
        """Forward process of multi-head attention

        Args:
            q_data: Query features, (..., Q, q_dim).
            kv_data: Key and value features, (..., K, kv_dim).
            attn_mask: Attention mask with size (..., Q/1, K/1), or None
        Returns:
            Output features, (..., Q, out_dim).
        """
        q = einops.rearrange(self.q_linear(q_data), '... Q (H C) -> ... Q H C', C=self.c)
        k, v = einops.rearrange(self.kv_linear(kv_data), '... K (split H C) -> split ... K H C', split=2, C=self.c)

        logits = torch.einsum("...qhc,...khc->...hqk", q, k) * self._qk_scale()

        if attn_mask is not None:
            logits = torch.where(attn_mask.unsqueeze(-3), logits, -1e9)

        weights = nn.functional.softmax(logits, dim=-1)
        weighted_avg = torch.einsum("...hqk,...khc->...qhc", weights, v)
        weighted_avg = einops.rearrange(weighted_avg, '... Q H C -> ... Q (H C)')

        if self.gating:
            gate_values = torch.sigmoid(self.gating_linear(q_data))
            weighted_avg = weighted_avg * gate_values

        output = self.output_linear(weighted_avg)

        return output
